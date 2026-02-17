#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import logging
from collections import defaultdict
from copy import deepcopy
import json_repair
import pandas as pd
import trio

from graphrag_new.query_analyze_prompt import PROMPTS
from graphrag_new.utils import get_entity_type2sampels, get_llm_cache, set_llm_cache, get_relation
from rag_new.utils import num_tokens_from_string, get_float
from rag_new.utils.doc_store_conn import OrderByExpr

from rag_new.nlp.search import Dealer, index_name
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class KGSearchConfig:
    max_entities_from_query: int = 10
    fusion_alpha: float = 0.45
    fusion_beta: float = 0.35
    fusion_gamma: float = 0.20
    hub_penalty_weight: float = 0.10  # جریمه هاب برای GiG/GcG


class KGSearch(Dealer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cfg = KGSearchConfig()
        self._ty2ents_cache: Dict[Tuple[str, str], Any] = {}
    def _chat(self, llm_bdl, system, history, gen_conf):
        response = get_llm_cache(llm_bdl.llm_name, system, history, gen_conf)
        if response:
            return response
        response = llm_bdl.chat(system, history, gen_conf)
        if response.find("**ERROR**") >= 0:
            raise Exception(response)
        set_llm_cache(llm_bdl.llm_name, system, response, history, gen_conf)
        return response

    def query_rewrite(self, llm, question, idxnms, kb_ids):
        """استخراج توکن‌ها و تحلیل سوال با استفاده از LLM"""
        # کش نمونه‌ها برای کاهش هزینه trio.run
        cache_key = ("|".join(sorted(map(str, idxnms or []))), "|".join(sorted(map(str, kb_ids or []))))
        if cache_key in self._ty2ents_cache:
            ty2ents = self._ty2ents_cache[cache_key]
        else:
            ty2ents = trio.run(lambda: get_entity_type2sampels(idxnms, kb_ids))
            self._ty2ents_cache[cache_key] = ty2ents
        hint_prompt = PROMPTS["minirag_query2kwd"].format(query=question,
                                                          TYPE_POOL=json.dumps(ty2ents, ensure_ascii=False, indent=2))
        result = self._chat(llm, hint_prompt, [{"role": "user", "content": "Output:"}], {})
        try:
            keywords_data = json_repair.loads(result)
            type_keywords = keywords_data.get("answer_type_keywords", [])
            # افزایش سقف به‌صورت تطبیقی
            ent_limit = min(self._cfg.max_entities_from_query, max(5, len(str(question).split()) // 2))
            entities_from_query = keywords_data.get("entities_from_query", [])[:ent_limit]
            return type_keywords, entities_from_query
        except json_repair.JSONDecodeError:
            try:
                result = result.replace(hint_prompt[:-1], '').replace('user', '').replace('model', '').strip()
                result = '{' + result.split('{')[1].split('}')[0] + '}'
                keywords_data = json_repair.loads(result)
                type_keywords = keywords_data.get("answer_type_keywords", [])
                ent_limit = min(self._cfg.max_entities_from_query, max(5, len(str(question).split()) // 2))
                entities_from_query = keywords_data.get("entities_from_query", [])[:ent_limit]
                return type_keywords, entities_from_query
            except Exception as e:
                logging.exception(f"JSON parsing error: {result} -> {e}")
                # تلاش دوباره ساختاریافته (درخواست JSON دقیق)
                try:
                    reask = "Return a strict JSON with keys 'answer_type_keywords' (list) and 'entities_from_query' (list) only."
                    result2 = self._chat(llm, reask, [{"role": "user", "content": str(question)}], {})
                    data2 = json_repair.loads(result2)
                    type_keywords = data2.get("answer_type_keywords", [])
                    ent_limit = min(self._cfg.max_entities_from_query, max(5, len(str(question).split()) // 2))
                    entities_from_query = data2.get("entities_from_query", [])[:ent_limit]
                    return type_keywords, entities_from_query
                except Exception:
                    # بازگشت امن
                    return [], []

    # --- Enhanced helpers for Hybrid New (entity locking + fusion ranking) ---
    def _detect_gene_cancer(self, question: str, entities_from_query: list[str]) -> bool:
        ql = (question or "").lower()
        has_cancer = any(k in ql for k in ["cancer", "tumor", "malignancy", "carcinoma", "sarcoma", "leukemia", "lymphoma"])
        has_gene_token = any(e.lower() in ["tp53", "brca1", "brca2", "egfr", "kras", "pik3ca"] for e in entities_from_query or [])
        return has_cancer and has_gene_token

    def _canonicalize_entity(self, token: str) -> str | None:
        if not token:
            return None
        t = token.strip().lower()
        canonical_map = {
            # Genes (HGNC-style common symbols)
            "tp53": "TP53",
            "p53": "TP53",
            "tumor protein p53": "TP53",
            "brca1": "BRCA1",
            "brca2": "BRCA2",
            "egfr": "EGFR",
            "kras": "KRAS",
            "pik3ca": "PIK3CA",
        }
        return canonical_map.get(t)

    def _schema_boost_for_entity(self, ent_id: str, question: str) -> float:
        # Lightweight heuristic: prioritize Gene/Disease/Pathway terms appearing in question
        ql = (question or "").lower()
        boost = 0.0
        if not ent_id:
            return boost
        ent_lower = ent_id.lower()
        # Direct name match boost
        if ent_lower in ql:
            boost += 0.2
        # Domain boosts
        if any(k in ql for k in ["cancer", "tumor", "malignancy"]):
            if any(g in ent_lower for g in ["tp53", "brca", "egfr", "kras", "pik3ca"]):
                boost += 0.15
        if any(k in ql for k in ["pathway", "signaling", "dna damage", "apoptosis", "cell cycle", "go:"]):
            boost += 0.1
        return boost

    def _fusion_score_entity(self, ent_id: str, ent_info: dict, question: str, core_entity: str | None,
                              alpha: float = 0.45, beta: float = 0.35, gamma: float = 0.20, hub_w: float = 0.10) -> float:
        sim = get_float(ent_info.get("sim", 0))
        pr = get_float(ent_info.get("pagerank", 0))
        schema = self._schema_boost_for_entity(ent_id, question)
        # hub-penalty بر اساس تعداد همسایه‌های N-hop اگر موجود باشد
        try:
            deg_proxy = len(ent_info.get("n_hop_ents", []) or [])
        except Exception:
            deg_proxy = 0
        import math
        hub_pen = math.log(1 + max(0, deg_proxy))
        score = alpha * sim + beta * pr + gamma * schema - hub_w * hub_pen
        if core_entity and ent_id == core_entity:
            score += 1.0  # hard boost for locked core
        return score

    def _ent_info_from_(self, es_res, sim_thr=0.3):
        """استخراج اطلاعات موجودیت‌ها از نتایج Elasticsearch"""
        res = {}
        flds = ["content_with_weight", "_score", "entity_kwd", "rank_flt", "n_hop_with_weight"]
        es_res = self.dataStore.getFields(es_res, flds)
        for _, ent in es_res.items():
            for f in list(ent.keys()):
                if f in flds and ent.get(f) is None:
                    ent.pop(f, None)
            if get_float(ent.get("_score", 0)) < sim_thr:
                continue
            ent_kwd = ent.get("entity_kwd")
            if ent_kwd is None:
                continue
            if isinstance(ent_kwd, list) and len(ent_kwd) > 0:
                ent_kwd = ent_kwd[0]
            res[ent_kwd] = {
                "sim": get_float(ent.get("_score", 0)),
                "pagerank": get_float(ent.get("rank_flt", 0)),
                "n_hop_ents": json.loads(ent.get("n_hop_with_weight", "[]")),
                "description": ent.get("content_with_weight", "{}")
            }
        return res

    def _relation_info_from_(self, es_res, sim_thr=0.3, allow_metaedges: Optional[List[str]] = None, deny_metaedges: Optional[List[str]] = None):
        """استخراج اطلاعات روابط از نتایج Elasticsearch"""
        res = {}
        es_res = self.dataStore.getFields(es_res, ["content_with_weight", "_score", "from_entity_kwd", "to_entity_kwd",
                                                   "weight_int"])
        for _, ent in es_res.items():
            if get_float(ent["_score"]) < sim_thr:
                continue
            f = ent.get("from_entity_kwd")
            t = ent.get("to_entity_kwd")
            if f is None or t is None:
                continue
            if isinstance(f, list) and f:
                f = f[0]
            if isinstance(t, list) and t:
                t = t[0]
            meta = get_relation(f, t) or "related"
            if allow_metaedges and meta not in allow_metaedges:
                continue
            if deny_metaedges and meta in deny_metaedges:
                continue
            edge_id = f"Edge::{meta}::{f}__{t}"
            res[(f, t, meta)] = {
                "edge_id": edge_id,
                "from": f,
                "to": t,
                "metaedge": meta,
                "sim": get_float(ent["_score"]),
                "weight": get_float(ent.get("weight_int", 1)),
                "description": ent.get("content_with_weight", "{}"),
                "unbiased": None
            }
        return res

    def get_relevant_ents_by_keywords(self, keywords, filters, idxnms, kb_ids, emb_mdl, sim_thr=0.3, N=56):
        """بازیابی موجودیت‌ها بر اساس کلمات کلیدی"""
        if not keywords:
            return {}
        
        query_vector = self.get_vector(" ".join(keywords), emb_mdl, N, sim_thr)
        es_res = self.dataStore.search(["entity_kwd", "content_with_weight", "_score", "rank_flt", "n_hop_with_weight"],
                                      [query_vector], filters, [], None, 0, N, idxnms, kb_ids)
        return self._ent_info_from_(es_res, sim_thr)

    def get_relevant_relations_by_txt(self, txt, filters, idxnms, kb_ids, emb_mdl, sim_thr=0.3, N=56, allow_metaedges: Optional[List[str]] = None, deny_metaedges: Optional[List[str]] = None):
        """بازیابی روابط بر اساس متن"""
        if not txt:
            return {}
        
        query_vector = self.get_vector(txt, emb_mdl, N, sim_thr)
        es_res = self.dataStore.search(["from_entity_kwd", "to_entity_kwd", "content_with_weight", "_score", "weight_int"],
                                      [query_vector], filters, [], None, 0, N, idxnms, kb_ids)
        return self._relation_info_from_(es_res, sim_thr, allow_metaedges, deny_metaedges)

    # ---------------- Intent Router (Hetionet schema) ----------------
    def _detect_intent_config(self, question: str) -> Dict[str, Any]:
        q = (question or "").lower()
        cfg = {
            "intent": "general",
            "allow": [],
            "deny": ["DrD", "CrC"],
            "end_type": None,
            "hop_limit": 2,
        }
        resembles = any(k in q for k in ["resembles", "similar", "similarity", "alike"])
        # Gene→Gene
        if any(k in q for k in ["covary", "covaries", "coexpression", "co-expression", "هم‌واریانس", "هم‌بروز", "هم‌تغییر"]):
            cfg.update({"intent": "G-G_covary", "allow": ["GcG"], "end_type": "Gene", "hop_limit": 1})
        elif any(k in q for k in ["interaction", "interacts", "ppi", "تعامل"]):
            cfg.update({"intent": "G-G_interact", "allow": ["GiG"], "end_type": "Gene", "hop_limit": 1})
        elif any(k in q for k in ["regulates", "regulation", "تنظیم"]):
            cfg.update({"intent": "G-G_regulates", "allow": ["Gr>G"], "end_type": "Gene", "hop_limit": 1})
        # Disease→Drug/Class
        elif any(k in q for k in ["treats", "treatment", "therapy", "therapeutic", "درمان", "پالیتیو"]):
            cfg.update({"intent": "D→(C|PC)", "allow": ["CtD", "CpD", "PCiC", "CbG"], "end_type": "Compound|Pharmacologic Class", "hop_limit": 3})
        # Gene→Disease
        elif any(k in q for k in ["disease", "associated", "association", "بیماری"]):
            cfg.update({"intent": "G→D", "allow": ["DaG"], "end_type": "Disease", "hop_limit": 2})
        # Drug side-effect
        if any(k in q for k in ["side effect", "adverse", "عوارض"]):
            cfg.update({"intent": "C→SE", "allow": ["CcSE"], "end_type": "Side Effect", "hop_limit": 1})
        # Memberships
        if any(k in q for k in ["pathway", "biological process", "molecular function", "go:"]):
            cfg.update({"intent": "G↔(PW|BP|MF)", "allow": ["GpPW", "GpBP", "GpMF"], "end_type": "Gene|PW|BP|MF", "hop_limit": 1})
        if resembles:
            cfg["deny"] = [m for m in cfg["deny"] if m not in ("DrD", "CrC")]
        return cfg

    def get_relevant_ents_by_types(self, types, filters, idxnms, kb_ids, N=56):
        """بازیابی موجودیت‌ها بر اساس نوع"""
        if not types:
            return {}
        
        type_filters = deepcopy(filters)
        type_filters["entity_type"] = types
        es_res = self.dataStore.search(["entity_kwd", "content_with_weight", "_score", "rank_flt", "n_hop_with_weight"],
                                      [], type_filters, [], None, 0, N, idxnms, kb_ids)
        return self._ent_info_from_(es_res, 0.0)

    def retrieval(self, question: str,
               tenant_ids: str | list[str],
               kb_ids: list[str],
               emb_mdl,
               llm,
               max_token: int = 8196,
               ent_topn: int = 6,
               rel_topn: int = 6,
               comm_topn: int = 1,
               ent_sim_threshold: float = 0.3,
               rel_sim_threshold: float = 0.3,
                  **kwargs
               ):
        """الگوریتم اصلی بازیابی GraphRAG"""
        
        # تنظیم وزن‌ها/پیکربندی از kwargs
        self._cfg.fusion_alpha = float(kwargs.get("fusion_alpha", self._cfg.fusion_alpha))
        self._cfg.fusion_beta = float(kwargs.get("fusion_beta", self._cfg.fusion_beta))
        self._cfg.fusion_gamma = float(kwargs.get("fusion_gamma", self._cfg.fusion_gamma))
        self._cfg.hub_penalty_weight = float(kwargs.get("hub_penalty_weight", self._cfg.hub_penalty_weight))
        self._cfg.max_entities_from_query = int(kwargs.get("max_entities_from_query", self._cfg.max_entities_from_query))

        # مرحله 1: تحلیل سوال و استخراج توکن‌ها
        idxnms = [index_name(tenant_ids)]
        type_keywords, entities_from_query = self.query_rewrite(llm, question, idxnms, kb_ids)
        
        # مرحله 2: بازیابی موجودیت‌ها
        filters = {"kb_id": kb_ids}
        ents_by_keywords = self.get_relevant_ents_by_keywords(entities_from_query, filters, idxnms, kb_ids, emb_mdl, ent_sim_threshold, ent_topn)
        ents_by_types = self.get_relevant_ents_by_types(type_keywords, filters, idxnms, kb_ids, ent_topn)
        
        # ترکیب نتایج موجودیت‌ها
        all_entities = {}
        for ent_id, ent_info in ents_by_keywords.items():
            all_entities[ent_id] = ent_info
        for ent_id, ent_info in ents_by_types.items():
            if ent_id not in all_entities:
                all_entities[ent_id] = ent_info
            else:
                # ترکیب امتیازات
                all_entities[ent_id]["sim"] = max(all_entities[ent_id]["sim"], ent_info["sim"])

        # قفل هسته برای سناریوهای ژن-سرطان (Entity Locking)
        core_entity = None
        try:
            if self._detect_gene_cancer(question, entities_from_query):
                # تلاش برای پیدا کردن ژن اصلی از روی توکن‌ها
                for tok in entities_from_query:
                    canon = self._canonicalize_entity(tok)
                    if canon:
                        core_entity = canon
                        break
                # اگر توکن‌ها نبودند ولی خود سوال حاوی ژن است
                if core_entity is None:
                    core_from_q = self._canonicalize_entity(question)
                    if core_from_q:
                        core_entity = core_from_q
                # تزریق/تقویت نود هسته
                if core_entity is not None:
                    if core_entity not in all_entities:
                        all_entities[core_entity] = {"sim": 1.0, "pagerank": 1.0, "description": "{}"}
                    else:
                        # تقویت موجود
                        all_entities[core_entity]["sim"] = max(1.0, get_float(all_entities[core_entity].get("sim", 0)))
                        all_entities[core_entity]["pagerank"] = max(1.0, get_float(all_entities[core_entity].get("pagerank", 0)))
        except Exception:
            core_entity = None

        # رنک‌ ترکیبی (Dense+PR+Schema) برای موجودیت‌ها
        alpha, beta, gamma = self._cfg.fusion_alpha, self._cfg.fusion_beta, self._cfg.fusion_gamma
        fused_entities = []
        for ent_id, ent_info in all_entities.items():
            fused_score = self._fusion_score_entity(ent_id, ent_info, question, core_entity, alpha, beta, gamma, self._cfg.hub_penalty_weight)
            fused_entities.append((ent_id, {**ent_info, "fusion_score": fused_score, "schema_boost": self._schema_boost_for_entity(ent_id, question), "tags": []}))

        fused_entities.sort(key=lambda x: x[1]["fusion_score"], reverse=True)
        ranked_entities_list = fused_entities[:ent_topn]

        # اگر core_entity وجود داشت ولی در top-k نبود، تزریق سخت
        if core_entity and not any(ent_id == core_entity for ent_id, _ in ranked_entities_list):
            ranked_entities_list = [(core_entity, {"sim": 1.0, "pagerank": 1.0, "fusion_score": 1.0, "schema_boost": 0.0, "description": "{}", "tags": ["core_locked", "injected"]})] + ranked_entities_list[:-1]
        else:
            # برچسب core_locked
            ranked_entities_list = [(eid, {**info, "tags": list(set((info.get("tags") or []) + (["core_locked"] if eid == core_entity else [])))}) for eid, info in ranked_entities_list]

        ranked_entities = dict(ranked_entities_list)

        # مرحله 3: بازیابی روابط
        intent_cfg = self._detect_intent_config(question)
        relations = self.get_relevant_relations_by_txt(question, filters, idxnms, kb_ids, emb_mdl, rel_sim_threshold, rel_topn, allow_metaedges=intent_cfg.get("allow"), deny_metaedges=intent_cfg.get("deny"))
        
        # مرحله 4: بازیابی جامعه‌ها (Communities)
        communities = self._community_retrieval_(list(all_entities.keys()), filters, kb_ids, idxnms, comm_topn, max_token)
        
        # مرحله 5: رتبه‌بندی نهایی
        ranked_relations_items = sorted(relations.items(), key=lambda x: x[1]["sim"], reverse=True)[:rel_topn]
        ranked_relations = [ri[1] for ri in ranked_relations_items]
        
        return {
            "entities": [
                {"id": eid, "label": eid, "type": None, **info}
                for eid, info in ranked_entities.items()
            ],
            "relations": ranked_relations,
            "paths": [],  # می‌توان بر اساس گراف واقعی پر کرد
            "communities": communities,
            "analysis": {
                "intent": intent_cfg.get("intent"),
                "allowlist": intent_cfg.get("allow"),
                "denylist": intent_cfg.get("deny"),
                "hop_limit": intent_cfg.get("hop_limit"),
                "type_keywords": type_keywords,
                "entities_from_query": entities_from_query,
                "gene_cancer_intent": self._detect_gene_cancer(question, entities_from_query),
                "core_entity": core_entity,
                "fusion_weights": {"alpha": alpha, "beta": beta, "gamma": gamma, "hub_penalty_weight": self._cfg.hub_penalty_weight}
            }
        }

    def _community_retrieval_(self, entities, condition, kb_ids, idxnms, topn, max_token):
        """بازیابی جامعه‌ها (Communities)"""
        ## Community retrieval
        if not entities:
            return []
        
        # اینجا می‌توانید الگوریتم‌های مختلف برای یافتن جامعه‌ها پیاده‌سازی کنید
        # برای مثال: Louvain, Label Propagation, etc.
        return [] 
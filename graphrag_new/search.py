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


class KGSearch(Dealer):
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
        ty2ents = trio.run(lambda: get_entity_type2sampels(idxnms, kb_ids))
        hint_prompt = PROMPTS["minirag_query2kwd"].format(query=question,
                                                          TYPE_POOL=json.dumps(ty2ents, ensure_ascii=False, indent=2))
        result = self._chat(llm, hint_prompt, [{"role": "user", "content": "Output:"}], {})
        try:
            keywords_data = json_repair.loads(result)
            type_keywords = keywords_data.get("answer_type_keywords", [])
            entities_from_query = keywords_data.get("entities_from_query", [])[:5]
            return type_keywords, entities_from_query
        except json_repair.JSONDecodeError:
            try:
                result = result.replace(hint_prompt[:-1], '').replace('user', '').replace('model', '').strip()
                result = '{' + result.split('{')[1].split('}')[0] + '}'
                keywords_data = json_repair.loads(result)
                type_keywords = keywords_data.get("answer_type_keywords", [])
                entities_from_query = keywords_data.get("entities_from_query", [])[:5]
                return type_keywords, entities_from_query
            except Exception as e:
                logging.exception(f"JSON parsing error: {result} -> {e}")
                raise e

    def _ent_info_from_(self, es_res, sim_thr=0.3):
        """استخراج اطلاعات موجودیت‌ها از نتایج Elasticsearch"""
        res = {}
        flds = ["content_with_weight", "_score", "entity_kwd", "rank_flt", "n_hop_with_weight"]
        es_res = self.dataStore.getFields(es_res, flds)
        for _, ent in es_res.items():
            for f in flds:
                if f in ent and ent[f] is None:
                    del ent[f]
            if get_float(ent.get("_score", 0)) < sim_thr:
                continue
            if isinstance(ent["entity_kwd"], list):
                ent["entity_kwd"] = ent["entity_kwd"][0]
            res[ent["entity_kwd"]] = {
                "sim": get_float(ent.get("_score", 0)),
                "pagerank": get_float(ent.get("rank_flt", 0)),
                "n_hop_ents": json.loads(ent.get("n_hop_with_weight", "[]")),
                "description": ent.get("content_with_weight", "{}")
            }
        return res

    def _relation_info_from_(self, es_res, sim_thr=0.3):
        """استخراج اطلاعات روابط از نتایج Elasticsearch"""
        res = {}
        es_res = self.dataStore.getFields(es_res, ["content_with_weight", "_score", "from_entity_kwd", "to_entity_kwd",
                                                   "weight_int"])
        for _, ent in es_res.items():
            if get_float(ent["_score"]) < sim_thr:
                continue
            f, t = sorted([ent["from_entity_kwd"], ent["to_entity_kwd"]])
            if isinstance(f, list):
                f = f[0]
            if isinstance(t, list):
                t = t[0]
            res[(f, t)] = {
                "sim": get_float(ent["_score"]),
                "weight": get_float(ent.get("weight_int", 1)),
                "description": ent.get("content_with_weight", "{}")
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

    def get_relevant_relations_by_txt(self, txt, filters, idxnms, kb_ids, emb_mdl, sim_thr=0.3, N=56):
        """بازیابی روابط بر اساس متن"""
        if not txt:
            return {}
        
        query_vector = self.get_vector(txt, emb_mdl, N, sim_thr)
        es_res = self.dataStore.search(["from_entity_kwd", "to_entity_kwd", "content_with_weight", "_score", "weight_int"],
                                      [query_vector], filters, [], None, 0, N, idxnms, kb_ids)
        return self._relation_info_from_(es_res, sim_thr)

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
        
        # مرحله 3: بازیابی روابط
        relations = self.get_relevant_relations_by_txt(question, filters, idxnms, kb_ids, emb_mdl, rel_sim_threshold, rel_topn)
        
        # مرحله 4: بازیابی جامعه‌ها (Communities)
        communities = self._community_retrieval_(list(all_entities.keys()), filters, kb_ids, idxnms, comm_topn, max_token)
        
        # مرحله 5: رتبه‌بندی نهایی
        ranked_entities = sorted(all_entities.items(), key=lambda x: x[1]["sim"], reverse=True)[:ent_topn]
        ranked_relations = sorted(relations.items(), key=lambda x: x[1]["sim"], reverse=True)[:rel_topn]
        
        return {
            "entities": dict(ranked_entities),
            "relations": dict(ranked_relations),
            "communities": communities,
            "query_analysis": {
                "type_keywords": type_keywords,
                "entities_from_query": entities_from_query
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
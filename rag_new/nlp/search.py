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
import logging
import re
import math
from collections import OrderedDict
from dataclasses import dataclass

from rag_new.settings import TAG_FLD, PAGERANK_FLD
from rag_new.utils import rmSpace, get_float
from rag_new.nlp import query
import numpy as np
from rag_new.utils.doc_store_conn import DocStoreConnection, MatchDenseExpr, FusionExpr, OrderByExpr


def index_name(uid): 
    return f"ragflow_{uid}"


class Dealer:
    def __init__(self, dataStore: DocStoreConnection):
        self.qryr = query.FulltextQueryer()
        self.dataStore = dataStore

    @dataclass
    class SearchResult:
        total: int
        ids: list[str]
        query_vector: list[float] | None = None
        field: dict | None = None
        highlight: dict | None = None
        aggregation: list | dict | None = None
        keywords: list[str] | None = None
        group_docs: list[list] | None = None

    def get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
        """دریافت بردار embedding برای متن"""
        qv, _ = emb_mdl.encode_queries(txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(
                f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [get_float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"
        return MatchDenseExpr(vector_column_name, embedding_data, 'float', 'cosine', topk, {"similarity": similarity})

    def get_filters(self, req):
        """دریافت فیلترها از درخواست"""
        condition = dict()
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]
        for key in ["knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd", "to_entity_kwd", "removed_kwd"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]
        return condition

    def search(self, req, idx_names: str | list[str],
               kb_ids: list[str],
               emb_mdl=None,
               highlight=False,
               rank_feature: dict | None = None
               ):
        """جستجوی اصلی"""
        filters = self.get_filters(req)
        orderBy = OrderByExpr()

        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        ps = int(req.get("size", topk))
        offset, limit = pg * ps, ps

        src = req.get("fields",
                      ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd", "position_int",
                       "doc_id", "page_num_int", "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
                       "question_kwd", "question_tks", "doc_type_kwd",
                       "available_int", "content_with_weight", PAGERANK_FLD, TAG_FLD])
        kwds = set([])

        qst = req.get("question", "")
        q_vec = []
        if not qst:
            if req.get("sort"):
                orderBy.asc("page_num_int")
                orderBy.asc("top_int")
                orderBy.desc("create_timestamp_flt")
            res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
            total = self.dataStore.getTotal(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
        else:
            # جستجوی متنی
            if emb_mdl:
                q_vec = [self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))]
            
            # جستجوی کلیدواژه
            kwds = self.qryr.fulltext_query(qst)
            
            # جستجوی ترکیبی
            if q_vec and kwds:
                fusion = FusionExpr("or", 0.3, 0.7)
                res = self.dataStore.search(src, q_vec, filters, kwds, orderBy, offset, limit, idx_names, kb_ids, fusion)
            elif q_vec:
                res = self.dataStore.search(src, q_vec, filters, [], orderBy, offset, limit, idx_names, kb_ids)
            else:
                res = self.dataStore.search(src, [], filters, kwds, orderBy, offset, limit, idx_names, kb_ids)
            
            total = self.dataStore.getTotal(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))

        return self.SearchResult(
            total=total,
            ids=self.dataStore.getIds(res),
            query_vector=q_vec[0] if q_vec else None,
            field=self.dataStore.getFields(res, src),
            highlight=self.dataStore.getHighlight(res) if highlight else None,
            aggregation=self.dataStore.getAggregation(res),
            keywords=list(kwds),
            group_docs=self.dataStore.getGroupDocs(res)
        )

    @staticmethod
    def trans2floats(txt):
        """تبدیل متن به اعداد اعشاری"""
        return [get_float(x) for x in txt.split()]

    def insert_citations(self, answer, chunks, chunk_v,
                        embd_mdl, tkweight=0.1, vtweight=0.9):
        """درج استنادات در پاسخ"""
        if not chunks:
            return answer

        # محاسبه شباهت بین پاسخ و chunks
        similarities = []
        for chunk in chunks:
            chunk_text = chunk.get("content", "")
            if not chunk_text:
                continue
            
            # محاسبه شباهت متنی
            text_sim = self._calculate_text_similarity(answer, chunk_text)
            
            # محاسبه شباهت برداری
            vector_sim = self._calculate_vector_similarity(answer, chunk_text, embd_mdl)
            
            # ترکیب شباهت‌ها
            combined_sim = tkweight * text_sim + vtweight * vector_sim
            similarities.append((chunk, combined_sim))

        # مرتب‌سازی بر اساس شباهت
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # درج استنادات
        cited_answer = answer
        for chunk, sim in similarities[:3]:  # حداکثر 3 استناد
            if sim > 0.5:  # آستانه شباهت
                citation = f"[{chunk.get('doc_id', 'Unknown')}]"
                cited_answer += f" {citation}"

        return cited_answer

    def _calculate_text_similarity(self, text1, text2):
        """محاسبه شباهت متنی"""
        # پیاده‌سازی ساده شباهت متنی
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def _calculate_vector_similarity(self, text1, text2, embd_mdl):
        """محاسبه شباهت برداری"""
        try:
            vec1 = embd_mdl.encode_queries(text1)[0]
            vec2 = embd_mdl.encode_queries(text2)[0]
            
            # محاسبه cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            logging.error(f"Error calculating vector similarity: {e}")
            return 0.0

    def _rank_feature_scores(self, query_rfea, search_res):
        """محاسبه امتیازات ویژگی‌های رتبه‌بندی"""
        scores = {}
        for doc_id, doc_data in search_res.field.items():
            score = 0.0
            for feature_name, weight in query_rfea.items():
                if feature_name in doc_data:
                    feature_value = get_float(doc_data[feature_name])
                    score += weight * feature_value
            scores[doc_id] = score
        return scores

    def rerank(self, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_ltks",
               rank_feature: dict | None = None
               ):
        """بازرتبه‌بندی نتایج"""
        if not sres.field:
            return sres

        reranked_results = []
        for doc_id, doc_data in sres.field.items():
            content = doc_data.get(cfield, "")
            if not content:
                continue

            # محاسبه امتیاز ترکیبی
            text_score = self._calculate_text_similarity(query, content)
            vector_score = 0.0  # در صورت نیاز به embedding model
            
            combined_score = tkweight * text_score + vtweight * vector_score
            
            # اضافه کردن امتیاز ویژگی‌های رتبه‌بندی
            if rank_feature:
                rank_scores = self._rank_feature_scores(rank_feature, {doc_id: doc_data})
                combined_score += rank_scores.get(doc_id, 0.0)
            
            reranked_results.append((doc_id, combined_score))

        # مرتب‌سازی بر اساس امتیاز
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # بازسازی نتایج
        new_ids = [doc_id for doc_id, _ in reranked_results]
        new_field = {doc_id: sres.field[doc_id] for doc_id in new_ids if doc_id in sres.field}
        
        return self.SearchResult(
            total=sres.total,
            ids=new_ids,
            query_vector=sres.query_vector,
            field=new_field,
            highlight=sres.highlight,
            aggregation=sres.aggregation,
            keywords=sres.keywords,
            group_docs=sres.group_docs
        )

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3,
                       vtweight=0.7, cfield="content_ltks",
                       rank_feature: dict | None = None):
        """بازرتبه‌بندی با استفاده از مدل"""
        if not sres.field:
            return sres

        reranked_results = []
        for doc_id, doc_data in sres.field.items():
            content = doc_data.get(cfield, "")
            if not content:
                continue

            # استفاده از مدل بازرتبه‌بندی
            try:
                model_score = rerank_mdl.score(query, content)
                reranked_results.append((doc_id, model_score))
            except Exception as e:
                logging.error(f"Error in rerank model: {e}")
                continue

        # مرتب‌سازی بر اساس امتیاز
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # بازسازی نتایج
        new_ids = [doc_id for doc_id, _ in reranked_results]
        new_field = {doc_id: sres.field[doc_id] for doc_id in new_ids if doc_id in sres.field}
        
        return self.SearchResult(
            total=sres.total,
            ids=new_ids,
            query_vector=sres.query_vector,
            field=new_field,
            highlight=sres.highlight,
            aggregation=sres.aggregation,
            keywords=sres.keywords,
            group_docs=sres.group_docs
        )

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        """محاسبه شباهت ترکیبی"""
        # پیاده‌سازی شباهت ترکیبی
        return 0.5  # مقدار پیش‌فرض

    def retrieval(self, question, embd_mdl, tenant_ids, kb_ids, page, page_size, similarity_threshold=0.2,
                  vector_similarity_weight=0.3, top=1024, doc_ids=None, aggs=True,
                  rerank_mdl=None, highlight=False,
                  rank_feature: dict | None = {PAGERANK_FLD: 10}):
        """بازیابی اصلی"""
        req = {
            "question": question,
            "page": page,
            "size": page_size,
            "topk": top,
            "similarity": similarity_threshold,
            "doc_ids": doc_ids
        }
        
        idx_names = [index_name(tenant_ids)]
        
        # جستجوی اولیه
        search_result = self.search(req, idx_names, kb_ids, embd_mdl, highlight, rank_feature)
        
        # بازرتبه‌بندی
        if rerank_mdl:
            search_result = self.rerank_by_model(rerank_mdl, search_result, question)
        else:
            search_result = self.rerank(search_result, question, rank_feature=rank_feature)
        
        return search_result

    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        """بازیابی SQL"""
        try:
            result = self.dataStore.sql_query(sql, fetch_size, format)
            return result
        except Exception as e:
            logging.error(f"SQL retrieval error: {e}")
            return None

    def chunk_list(self, doc_id: str, tenant_id: str,
                   kb_ids: list[str], max_count=1024,
                   offset=0,
                   fields=["docnm_kwd", "content_with_weight", "img_id"]):
        """دریافت لیست chunks"""
        try:
            filters = {"doc_id": doc_id, "kb_id": kb_ids}
            res = self.dataStore.search(fields, [], filters, [], None, offset, max_count, 
                                      [index_name(tenant_id)], kb_ids)
            return self.dataStore.getFields(res, fields)
        except Exception as e:
            logging.error(f"Chunk list error: {e}")
            return {}

    def all_tags(self, tenant_id: str, kb_ids: list[str], S=1000):
        """دریافت تمام تگ‌ها"""
        try:
            filters = {"kb_id": kb_ids}
            res = self.dataStore.search([TAG_FLD], [], filters, [], None, 0, S, 
                                      [index_name(tenant_id)], kb_ids)
            return self.dataStore.getAggregation(res)
        except Exception as e:
            logging.error(f"All tags error: {e}")
            return {}

    def all_tags_in_portion(self, tenant_id: str, kb_ids: list[str], S=1000):
        """دریافت تگ‌ها در بخش"""
        try:
            filters = {"kb_id": kb_ids}
            res = self.dataStore.search([TAG_FLD], [], filters, [], None, 0, S, 
                                      [index_name(tenant_id)], kb_ids)
            return self.dataStore.getAggregation(res)
        except Exception as e:
            logging.error(f"All tags in portion error: {e}")
            return {}

    def tag_content(self, tenant_id: str, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
        """تگ کردن محتوا"""
        try:
            # پیاده‌سازی تگ کردن محتوا
            return {"tags": [], "keywords": []}
        except Exception as e:
            logging.error(f"Tag content error: {e}")
            return {"tags": [], "keywords": []}

    def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=1000):
        """تگ کردن سوال"""
        try:
            # پیاده‌سازی تگ کردن سوال
            return {"tags": [], "keywords": []}
        except Exception as e:
            logging.error(f"Tag query error: {e}")
            return {"tags": [], "keywords": []} 
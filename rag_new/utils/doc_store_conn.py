# -*- coding: utf-8 -*-
"""
Document Store Connection - اتصال به دیتابیس اسناد
"""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class DocStoreConnection(ABC):
    """کلاس پایه برای اتصال به دیتابیس اسناد"""
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def search(self, fields: List[str], vectors: List, filters: Dict, keywords: List, 
               order_by, offset: int, limit: int, index_names: List[str], kb_ids: List[str], 
               fusion=None) -> Any:
        """جستجو در دیتابیس"""
        pass
    
    @abstractmethod
    def getTotal(self, result) -> int:
        """دریافت تعداد کل نتایج"""
        pass
    
    @abstractmethod
    def getIds(self, result) -> List[str]:
        """دریافت شناسه‌های نتایج"""
        pass
    
    @abstractmethod
    def getFields(self, result, fields: List[str]) -> Dict[str, Dict]:
        """دریافت فیلدهای نتایج"""
        pass
    
    @abstractmethod
    def getHighlight(self, result) -> Dict[str, Dict]:
        """دریافت highlight نتایج"""
        pass
    
    @abstractmethod
    def getAggregation(self, result) -> Dict[str, Any]:
        """دریافت aggregation نتایج"""
        pass
    
    @abstractmethod
    def getGroupDocs(self, result) -> List[List]:
        """دریافت گروه‌بندی اسناد"""
        pass
    
    @abstractmethod
    def sql_query(self, sql: str, fetch_size: int, format: str) -> Any:
        """اجرای کوئری SQL"""
        pass


class MatchDenseExpr:
    """عبارت جستجوی برداری متراکم"""
    
    def __init__(self, column_name: str, vector: List[float], data_type: str, 
                 metric: str, topk: int, params: Dict[str, Any]):
        self.column_name = column_name
        self.vector = vector
        self.data_type = data_type
        self.metric = metric
        self.topk = topk
        self.params = params


class FusionExpr:
    """عبارت ترکیب نتایج"""
    
    def __init__(self, method: str, weight1: float, weight2: float):
        self.method = method  # "or", "and", "weighted"
        self.weight1 = weight1
        self.weight2 = weight2


class OrderByExpr:
    """عبارت مرتب‌سازی"""
    
    def __init__(self):
        self.orders = []
    
    def asc(self, field: str):
        """مرتب‌سازی صعودی"""
        self.orders.append((field, "asc"))
        return self
    
    def desc(self, field: str):
        """مرتب‌سازی نزولی"""
        self.orders.append((field, "desc"))
        return self


class MockDocStoreConnection(DocStoreConnection):
    """کلاس Mock برای تست"""
    
    def __init__(self):
        super().__init__()
        self.mock_data = {}
    
    def search(self, fields: List[str], vectors: List, filters: Dict, keywords: List, 
               order_by, offset: int, limit: int, index_names: List[str], kb_ids: List[str], 
               fusion=None):
        """جستجوی Mock"""
        # پیاده‌سازی ساده برای تست
        return {
            "total": 0,
            "ids": [],
            "fields": {},
            "highlights": {},
            "aggregations": {},
            "group_docs": []
        }
    
    def getTotal(self, result) -> int:
        return result.get("total", 0)
    
    def getIds(self, result) -> List[str]:
        return result.get("ids", [])
    
    def getFields(self, result, fields: List[str]) -> Dict[str, Dict]:
        return result.get("fields", {})
    
    def getHighlight(self, result) -> Dict[str, Dict]:
        return result.get("highlights", {})
    
    def getAggregation(self, result) -> Dict[str, Any]:
        return result.get("aggregations", {})
    
    def getGroupDocs(self, result) -> List[List]:
        return result.get("group_docs", [])
    
    def sql_query(self, sql: str, fetch_size: int, format: str) -> Any:
        return {"results": [], "total": 0}


class ElasticsearchConnection(DocStoreConnection):
    """اتصال به Elasticsearch"""
    
    def __init__(self, hosts: List[str], **kwargs):
        super().__init__()
        try:
            from elasticsearch import Elasticsearch
            self.es = Elasticsearch(hosts, **kwargs)
        except ImportError:
            self.logger.error("Elasticsearch library not installed")
            self.es = None
    
    def search(self, fields: List[str], vectors: List, filters: Dict, keywords: List, 
               order_by, offset: int, limit: int, index_names: List[str], kb_ids: List[str], 
               fusion=None):
        """جستجو در Elasticsearch"""
        if not self.es:
            return MockDocStoreConnection().search(fields, vectors, filters, keywords, 
                                                 order_by, offset, limit, index_names, kb_ids, fusion)
        
        try:
            # ساخت query
            query = self._build_query(filters, keywords, vectors, fusion)
            
            # ساخت sort
            sort = self._build_sort(order_by)
            
            # اجرای جستجو
            response = self.es.search(
                index=",".join(index_names),
                body={
                    "query": query,
                    "sort": sort,
                    "from": offset,
                    "size": limit,
                    "_source": fields
                }
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Elasticsearch search error: {e}")
            return MockDocStoreConnection().search(fields, vectors, filters, keywords, 
                                                 order_by, offset, limit, index_names, kb_ids, fusion)
    
    def _build_query(self, filters: Dict, keywords: List, vectors: List, fusion=None):
        """ساخت query برای Elasticsearch"""
        must_clauses = []
        
        # فیلترها
        for field, value in filters.items():
            must_clauses.append({"term": {field: value}})
        
        # کلیدواژه‌ها
        if keywords:
            must_clauses.append({"multi_match": {
                "query": " ".join(keywords),
                "fields": ["content_ltks^2", "title_tks"],
                "type": "best_fields"
            }})
        
        # بردارها
        if vectors:
            for vector_expr in vectors:
                must_clauses.append({
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, '{}') + 1.0".format(
                                vector_expr.column_name
                            ),
                            "params": {"query_vector": vector_expr.vector}
                        }
                    }
                })
        
        if not must_clauses:
            return {"match_all": {}}
        
        return {"bool": {"must": must_clauses}}
    
    def _build_sort(self, order_by):
        """ساخت sort برای Elasticsearch"""
        if not order_by or not order_by.orders:
            return []
        
        sort_clauses = []
        for field, direction in order_by.orders:
            sort_clauses.append({field: {"order": direction}})
        
        return sort_clauses
    
    def getTotal(self, result) -> int:
        return result.get("hits", {}).get("total", {}).get("value", 0)
    
    def getIds(self, result) -> List[str]:
        hits = result.get("hits", {}).get("hits", [])
        return [hit["_id"] for hit in hits]
    
    def getFields(self, result, fields: List[str]) -> Dict[str, Dict]:
        hits = result.get("hits", {}).get("hits", [])
        fields_dict = {}
        
        for hit in hits:
            doc_id = hit["_id"]
            source = hit.get("_source", {})
            fields_dict[doc_id] = source
        
        return fields_dict
    
    def getHighlight(self, result) -> Dict[str, Dict]:
        hits = result.get("hits", {}).get("hits", [])
        highlights = {}
        
        for hit in hits:
            doc_id = hit["_id"]
            highlight = hit.get("highlight", {})
            if highlight:
                highlights[doc_id] = highlight
        
        return highlights
    
    def getAggregation(self, result) -> Dict[str, Any]:
        return result.get("aggregations", {})
    
    def getGroupDocs(self, result) -> List[List]:
        # پیاده‌سازی گروه‌بندی اسناد
        return []
    
    def sql_query(self, sql: str, fetch_size: int, format: str) -> Any:
        try:
            response = self.es.sql.query(body={"query": sql, "fetch_size": fetch_size})
            return response
        except Exception as e:
            self.logger.error(f"Elasticsearch SQL error: {e}")
            return {"results": [], "total": 0} 
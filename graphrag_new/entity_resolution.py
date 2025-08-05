# -*- coding: utf-8 -*-
"""
Entity Resolution - حل موجودیت‌های مشابه
"""
import logging
from typing import Dict, List, Any, Optional
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EntityResolution:
    """کلاس برای حل موجودیت‌های مشابه در گراف"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.resolved_entities = {}
    
    def calculate_similarity(self, entity1: str, entity2: str) -> float:
        """محاسبه شباهت بین دو موجودیت"""
        try:
            # تبدیل به حروف کوچک و حذف فاصله‌های اضافی
            e1 = entity1.lower().strip()
            e2 = entity2.lower().strip()
            
            # اگر دقیقاً یکسان باشند
            if e1 == e2:
                return 1.0
            
            # محاسبه شباهت بر اساس طول رشته مشترک
            common_length = 0
            for i in range(min(len(e1), len(e2))):
                if e1[i] == e2[i]:
                    common_length += 1
                else:
                    break
            
            # محاسبه شباهت
            max_length = max(len(e1), len(e2))
            if max_length == 0:
                return 0.0
            
            similarity = common_length / max_length
            
            # بررسی شباهت معنایی ساده
            if self._semantic_similarity(e1, e2):
                similarity += 0.2
            
            return min(similarity, 1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def _semantic_similarity(self, entity1: str, entity2: str) -> bool:
        """بررسی شباهت معنایی ساده"""
        # کلمات کلیدی مشترک
        keywords1 = set(entity1.split())
        keywords2 = set(entity2.split())
        
        if keywords1 & keywords2:  # اشتراک
            return True
        
        # بررسی مترادف‌های ساده
        synonyms = {
            "cancer": ["tumor", "neoplasm", "malignancy"],
            "gene": ["genetic", "dna", "chromosome"],
            "protein": ["enzyme", "peptide", "amino acid"],
            "disease": ["illness", "condition", "disorder"],
            "drug": ["medicine", "medication", "pharmaceutical"]
        }
        
        for word1 in keywords1:
            for word2 in keywords2:
                for synonym_list in synonyms.values():
                    if word1 in synonym_list and word2 in synonym_list:
                        return True
        
        return False
    
    def find_similar_entities(self, entities: List[str]) -> List[List[str]]:
        """یافتن گروه‌های موجودیت‌های مشابه"""
        groups = []
        processed = set()
        
        for i, entity1 in enumerate(entities):
            if entity1 in processed:
                continue
            
            group = [entity1]
            processed.add(entity1)
            
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if entity2 in processed:
                    continue
                
                similarity = self.calculate_similarity(entity1, entity2)
                if similarity >= self.similarity_threshold:
                    group.append(entity2)
                    processed.add(entity2)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def resolve_entities_in_graph(self, G: nx.Graph) -> nx.Graph:
        """حل موجودیت‌های مشابه در گراف"""
        try:
            # جمع‌آوری تمام نودها
            nodes = list(G.nodes())
            
            # یافتن گروه‌های مشابه
            similar_groups = self.find_similar_entities(nodes)
            
            # ادغام نودهای مشابه
            for group in similar_groups:
                if len(group) > 1:
                    # انتخاب نماینده (اولین نود)
                    representative = group[0]
                    
                    # ادغام ویژگی‌های نودها
                    merged_attrs = {}
                    for node in group:
                        if node in G.nodes():
                            node_attrs = G.nodes[node]
                            for key, value in node_attrs.items():
                                if key not in merged_attrs:
                                    merged_attrs[key] = value
                                elif isinstance(value, list) and isinstance(merged_attrs[key], list):
                                    merged_attrs[key].extend(value)
                                elif isinstance(value, dict) and isinstance(merged_attrs[key], dict):
                                    merged_attrs[key].update(value)
                    
                    # به‌روزرسانی نود نماینده
                    G.nodes[representative].update(merged_attrs)
                    
                    # انتقال یال‌ها به نود نماینده
                    for node in group[1:]:
                        if node in G:
                            # انتقال یال‌های ورودی
                            for pred in G.predecessors(node):
                                if pred != representative:
                                    edge_data = G.get_edge_data(pred, node)
                                    if not G.has_edge(pred, representative):
                                        G.add_edge(pred, representative, **edge_data)
                            
                            # انتقال یال‌های خروجی
                            for succ in G.successors(node):
                                if succ != representative:
                                    edge_data = G.get_edge_data(node, succ)
                                    if not G.has_edge(representative, succ):
                                        G.add_edge(representative, succ, **edge_data)
                            
                            # حذف نود قدیمی
                            G.remove_node(node)
                    
                    # ذخیره اطلاعات حل شده
                    self.resolved_entities[representative] = group
            
            return G
            
        except Exception as e:
            logging.error(f"Error in resolve_entities_in_graph: {e}")
            return G
    
    def get_resolution_summary(self) -> Dict[str, Any]:
        """دریافت خلاصه عملیات حل موجودیت"""
        return {
            "resolved_groups": len(self.resolved_entities),
            "total_resolved_entities": sum(len(group) for group in self.resolved_entities.values()),
            "representatives": list(self.resolved_entities.keys()),
            "resolution_mapping": self.resolved_entities
        }
    
    def clear_resolution_cache(self):
        """پاک کردن کش حل موجودیت"""
        self.resolved_entities.clear() 
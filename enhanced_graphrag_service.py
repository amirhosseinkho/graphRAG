# -*- coding: utf-8 -*-
"""
Enhanced GraphRAG Service - سرویس پیشرفته GraphRAG
"""

import json
import logging
import networkx as nx
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import time
import re
from dataclasses import dataclass

# Import new GraphRAG components
from graphrag_new.search import KGSearch
from graphrag_new.utils import get_entity_type2sampels, get_llm_cache, set_llm_cache
from graphrag_new.query_analyze_prompt import PROMPTS
from rag_new.nlp.search import Dealer, index_name
from rag_new.utils.doc_store_conn import OrderByExpr

class TokenExtractionMethod(Enum):
    """روش‌های استخراج توکن"""
    LLM_BASED = "llm_based"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"

class RetrievalAlgorithm(Enum):
    """الگوریتم‌های بازیابی"""
    BFS = "bfs"
    DFS = "dfs"
    DIJKSTRA = "dijkstra"
    PAGERANK = "pagerank"
    COMMUNITY_DETECTION = "community_detection"
    HYBRID = "hybrid"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    N_HOP = "n_hop"

class CommunityDetectionMethod(Enum):
    """روش‌های تشخیص جامعه"""
    LOUVAIN = "louvain"
    LABEL_PROPAGATION = "label_propagation"
    GIRVAN_NEWMAN = "girvan_newman"
    SPECTRAL = "spectral"

@dataclass
class RetrievalConfig:
    """تنظیمات بازیابی"""
    token_extraction_method: TokenExtractionMethod = TokenExtractionMethod.LLM_BASED
    retrieval_algorithm: RetrievalAlgorithm = RetrievalAlgorithm.HYBRID
    community_detection_method: CommunityDetectionMethod = CommunityDetectionMethod.LOUVAIN
    max_depth: int = 3
    max_nodes: int = 20
    max_edges: int = 40
    similarity_threshold: float = 0.3
    pagerank_alpha: float = 0.85
    community_resolution: float = 1.0
    enable_semantic_search: bool = True
    enable_community_detection: bool = True
    enable_n_hop_search: bool = True

class EnhancedGraphRAGService:
    """سرویس پیشرفته GraphRAG با قابلیت‌های جدید"""
    
    def __init__(self, graph_data_path: Optional[str] = None):
        """راه‌اندازی سرویس"""
        self.G = None
        self.kg_search = None
        self.config = RetrievalConfig()
        self.llm_cache = {}
        
        if graph_data_path:
            self.load_graph(graph_data_path)
    
    def load_graph(self, graph_path: str):
        """بارگذاری گراف"""
        try:
            if graph_path.endswith('.pkl'):
                import pickle
                with open(graph_path, 'rb') as f:
                    self.G = pickle.load(f)
            elif graph_path.endswith('.sif'):
                self.G = self._load_sif_graph(graph_path)
            else:
                raise ValueError(f"فرمت فایل {graph_path} پشتیبانی نمی‌شود")
            
            logging.info(f"گراف با {self.G.number_of_nodes()} نود و {self.G.number_of_edges()} یال بارگذاری شد")
            
        except Exception as e:
            logging.error(f"خطا در بارگذاری گراف: {e}")
            raise
    
    def _load_sif_graph(self, sif_path: str) -> nx.Graph:
        """بارگذاری گراف از فرمت SIF"""
        G = nx.Graph()
        
        with open(sif_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    source = parts[0]
                    relation = parts[1]
                    target = parts[2]
                    
                    # اضافه کردن نودها
                    if not G.has_node(source):
                        G.add_node(source, kind='entity', name=source)
                    if not G.has_node(target):
                        G.add_node(target, kind='entity', name=target)
                    
                    # اضافه کردن یال
                    G.add_edge(source, target, relation=relation, weight=1.0)
        
        return G
    
    def set_config(self, **kwargs):
        """تنظیم پیکربندی"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                if isinstance(value, str):
                    # تبدیل رشته به enum
                    if key == 'token_extraction_method':
                        setattr(self.config, key, TokenExtractionMethod(value))
                    elif key == 'retrieval_algorithm':
                        setattr(self.config, key, RetrievalAlgorithm(value))
                    elif key == 'community_detection_method':
                        setattr(self.config, key, CommunityDetectionMethod(value))
                    else:
                        setattr(self.config, key, value)
                else:
                    setattr(self.config, key, value)
    
    def get_config(self) -> Dict:
        """دریافت پیکربندی فعلی"""
        return {
            'token_extraction_method': self.config.token_extraction_method.value,
            'retrieval_algorithm': self.config.retrieval_algorithm.value,
            'community_detection_method': self.config.community_detection_method.value,
            'max_depth': self.config.max_depth,
            'max_nodes': self.config.max_nodes,
            'max_edges': self.config.max_edges,
            'similarity_threshold': self.config.similarity_threshold,
            'pagerank_alpha': self.config.pagerank_alpha,
            'community_resolution': self.config.community_resolution,
            'enable_semantic_search': self.config.enable_semantic_search,
            'enable_community_detection': self.config.enable_community_detection,
            'enable_n_hop_search': self.config.enable_n_hop_search
        }
    
    def extract_tokens_llm(self, query: str) -> Tuple[List[str], List[str]]:
        """استخراج توکن با استفاده از LLM"""
        try:
            # شبیه‌سازی LLM برای استخراج توکن
            # در واقعیت، اینجا باید از مدل LLM واقعی استفاده شود
            
            # استخراج موجودیت‌ها از سوال
            entities = self._extract_entities_from_query(query)
            
            # استخراج نوع پاسخ
            answer_types = self._extract_answer_types(query)
            
            return answer_types, entities
            
        except Exception as e:
            logging.error(f"خطا در استخراج توکن LLM: {e}")
            return [], []
    
    def extract_tokens_rule_based(self, query: str) -> Tuple[List[str], List[str]]:
        """استخراج توکن با استفاده از قوانین"""
        entities = []
        answer_types = []
        
        # قوانین ساده برای استخراج موجودیت‌ها
        words = query.lower().split()
        
        # تشخیص ژن‌ها (کلمات با حروف بزرگ)
        for word in words:
            if word.isupper() and len(word) > 2:
                entities.append(word)
        
        # تشخیص بیماری‌ها
        disease_keywords = ['cancer', 'diabetes', 'heart disease', 'alzheimer']
        for keyword in disease_keywords:
            if keyword in query.lower():
                entities.append(keyword)
        
        # تشخیص داروها
        drug_patterns = [r'\b[A-Z][a-z]+(?:ol|in|ine|ide)\b']
        for pattern in drug_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # تشخیص نوع پاسخ
        if any(word in query.lower() for word in ['what', 'which', 'how']):
            answer_types.append('DESCRIPTION')
        if any(word in query.lower() for word in ['when', 'time', 'date']):
            answer_types.append('TIME')
        if any(word in query.lower() for word in ['where', 'location']):
            answer_types.append('LOCATION')
        
        return answer_types, entities
    
    def extract_tokens_hybrid(self, query: str) -> Tuple[List[str], List[str]]:
        """استخراج توکن ترکیبی"""
        # ترکیب روش‌های LLM و قوانین
        llm_types, llm_entities = self.extract_tokens_llm(query)
        rule_types, rule_entities = self.extract_tokens_rule_based(query)
        
        # ترکیب نتایج
        combined_types = list(set(llm_types + rule_types))
        combined_entities = list(set(llm_entities + rule_entities))
        
        return combined_types, combined_entities
    
    def extract_tokens_semantic(self, query: str) -> Tuple[List[str], List[str]]:
        """استخراج توکن معنایی"""
        # پیاده‌سازی استخراج معنایی
        # اینجا می‌توان از مدل‌های embedding استفاده کرد
        
        entities = []
        answer_types = []
        
        # شبیه‌سازی استخراج معنایی
        if 'gene' in query.lower() or 'protein' in query.lower():
            answer_types.append('GENE')
        if 'disease' in query.lower() or 'cancer' in query.lower():
            answer_types.append('DISEASE')
        if 'drug' in query.lower() or 'compound' in query.lower():
            answer_types.append('DRUG')
        
        return answer_types, entities
    
    def extract_tokens(self, query: str) -> Tuple[List[str], List[str]]:
        """استخراج توکن بر اساس روش انتخاب شده"""
        if self.config.token_extraction_method == TokenExtractionMethod.LLM_BASED:
            return self.extract_tokens_llm(query)
        elif self.config.token_extraction_method == TokenExtractionMethod.RULE_BASED:
            return self.extract_tokens_rule_based(query)
        elif self.config.token_extraction_method == TokenExtractionMethod.HYBRID:
            return self.extract_tokens_hybrid(query)
        elif self.config.token_extraction_method == TokenExtractionMethod.SEMANTIC:
            return self.extract_tokens_semantic(query)
        else:
            return self.extract_tokens_llm(query)
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """استخراج موجودیت‌ها از سوال"""
        entities = []
        
        # الگوهای ساده برای تشخیص موجودیت‌ها
        patterns = [
            r'\b[A-Z]{2,}\b',  # کلمات با حروف بزرگ
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b',  # CamelCase
            r'\b\d+[A-Za-z]+\b',  # ترکیب اعداد و حروف
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        return list(set(entities))
    
    def _extract_answer_types(self, query: str) -> List[str]:
        """استخراج نوع پاسخ"""
        types = []
        
        # تشخیص نوع پاسخ بر اساس کلمات کلیدی
        if any(word in query.lower() for word in ['what', 'which', 'how']):
            types.append('DESCRIPTION')
        if any(word in query.lower() for word in ['when', 'time', 'date']):
            types.append('TIME')
        if any(word in query.lower() for word in ['where', 'location']):
            types.append('LOCATION')
        if any(word in query.lower() for word in ['why', 'cause', 'reason']):
            types.append('CAUSE')
        if any(word in query.lower() for word in ['effect', 'result', 'outcome']):
            types.append('EFFECT')
        
        return types
    
    def bfs_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """بازیابی با الگوریتم BFS"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'paths': []
        }
        
        visited = set()
        queue = [(node, 0) for node in start_nodes if self.G.has_node(node)]
        
        while queue and len(results['nodes']) < self.config.max_nodes:
            current_node, depth = queue.pop(0)
            
            if current_node in visited or depth > self.config.max_depth:
                continue
            
            visited.add(current_node)
            results['nodes'].append(current_node)
            
            # اضافه کردن یال‌های مرتبط
            for neighbor in self.G.neighbors(current_node):
                if neighbor not in visited:
                    edge_data = self.G.get_edge_data(current_node, neighbor)
                    results['edges'].append({
                        'source': current_node,
                        'target': neighbor,
                        'relation': edge_data.get('relation', ''),
                        'weight': edge_data.get('weight', 1.0)
                    })
                    queue.append((neighbor, depth + 1))
        
        return results
    
    def dfs_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """بازیابی با الگوریتم DFS"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'paths': []
        }
        
        visited = set()
        stack = [(node, 0) for node in start_nodes if self.G.has_node(node)]
        
        while stack and len(results['nodes']) < self.config.max_nodes:
            current_node, depth = stack.pop()
            
            if current_node in visited or depth > self.config.max_depth:
                continue
            
            visited.add(current_node)
            results['nodes'].append(current_node)
            
            # اضافه کردن یال‌های مرتبط
            for neighbor in self.G.neighbors(current_node):
                if neighbor not in visited:
                    edge_data = self.G.get_edge_data(current_node, neighbor)
                    results['edges'].append({
                        'source': current_node,
                        'target': neighbor,
                        'relation': edge_data.get('relation', ''),
                        'weight': edge_data.get('weight', 1.0)
                    })
                    stack.append((neighbor, depth + 1))
        
        return results
    
    def pagerank_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """بازیابی با الگوریتم PageRank"""
        if not self.G:
            return {}
        
        # محاسبه PageRank
        pagerank_scores = nx.pagerank(self.G, alpha=self.config.pagerank_alpha)
        
        # مرتب‌سازی نودها بر اساس PageRank
        sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = {
            'nodes': [],
            'edges': [],
            'paths': []
        }
        
        # انتخاب نودهای برتر
        for node, score in sorted_nodes[:self.config.max_nodes]:
            results['nodes'].append({
                'id': node,
                'pagerank': score,
                'attributes': dict(self.G.nodes[node])
            })
        
        # اضافه کردن یال‌های مرتبط
        for i, node_data in enumerate(results['nodes']):
            node = node_data['id']
            for neighbor in self.G.neighbors(node):
                if any(n['id'] == neighbor for n in results['nodes']):
                    edge_data = self.G.get_edge_data(node, neighbor)
                    results['edges'].append({
                        'source': node,
                        'target': neighbor,
                        'relation': edge_data.get('relation', ''),
                        'weight': edge_data.get('weight', 1.0)
                    })
        
        return results
    
    def community_detection_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """بازیابی با تشخیص جامعه"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'communities': []
        }
        
        # تشخیص جامعه بر اساس روش انتخاب شده
        if self.config.community_detection_method == CommunityDetectionMethod.LOUVAIN:
            communities = nx.community.louvain_communities(self.G, resolution=self.config.community_resolution)
        elif self.config.community_detection_method == CommunityDetectionMethod.LABEL_PROPAGATION:
            communities = nx.community.label_propagation_communities(self.G)
        else:
            communities = nx.community.louvain_communities(self.G)
        
        # یافتن جامعه‌های مرتبط با نودهای شروع
        relevant_communities = []
        for community in communities:
            if any(node in community for node in start_nodes):
                relevant_communities.append(community)
        
        # اضافه کردن نودها و یال‌های جامعه‌های مرتبط
        for community in relevant_communities:
            community_nodes = list(community)[:self.config.max_nodes // len(relevant_communities)]
            
            for node in community_nodes:
                results['nodes'].append({
                    'id': node,
                    'community': len(results['communities']),
                    'attributes': dict(self.G.nodes[node])
                })
            
            # اضافه کردن یال‌های درون جامعه
            for i, node1 in enumerate(community_nodes):
                for node2 in community_nodes[i+1:]:
                    if self.G.has_edge(node1, node2):
                        edge_data = self.G.get_edge_data(node1, node2)
                        results['edges'].append({
                            'source': node1,
                            'target': node2,
                            'relation': edge_data.get('relation', ''),
                            'weight': edge_data.get('weight', 1.0)
                        })
            
            results['communities'].append({
                'id': len(results['communities']),
                'nodes': community_nodes,
                'size': len(community_nodes)
            })
        
        return results
    
    def semantic_similarity_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """بازیابی بر اساس شباهت معنایی"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'similarities': []
        }
        
        # شبیه‌سازی محاسبه شباهت معنایی
        # در واقعیت، اینجا باید از مدل‌های embedding استفاده شود
        
        for node in self.G.nodes():
            if len(results['nodes']) >= self.config.max_nodes:
                break
            
            # محاسبه شباهت ساده (شبیه‌سازی)
            similarity = self._calculate_simple_similarity(query, node)
            
            if similarity > self.config.similarity_threshold:
                results['nodes'].append({
                    'id': node,
                    'similarity': similarity,
                    'attributes': dict(self.G.nodes[node])
                })
                results['similarities'].append({
                    'node': node,
                    'score': similarity
                })
        
        # مرتب‌سازی بر اساس شباهت
        results['nodes'].sort(key=lambda x: x['similarity'], reverse=True)
        results['similarities'].sort(key=lambda x: x['score'], reverse=True)
        
        # اضافه کردن یال‌های مرتبط
        for node_data in results['nodes'][:self.config.max_nodes]:
            node = node_data['id']
            for neighbor in self.G.neighbors(node):
                if any(n['id'] == neighbor for n in results['nodes']):
                    edge_data = self.G.get_edge_data(node, neighbor)
                    results['edges'].append({
                        'source': node,
                        'target': neighbor,
                        'relation': edge_data.get('relation', ''),
                        'weight': edge_data.get('weight', 1.0)
                    })
        
        return results
    
    def n_hop_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """بازیابی N-Hop"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'paths': []
        }
        
        # یافتن مسیرهای N-Hop
        for start_node in start_nodes:
            if not self.G.has_node(start_node):
                continue
            
            paths = self._find_n_hop_paths(start_node, self.config.max_depth)
            
            for path in paths:
                if len(results['nodes']) >= self.config.max_nodes:
                    break
                
                # اضافه کردن نودهای مسیر
                for node in path:
                    if not any(n['id'] == node for n in results['nodes']):
                        results['nodes'].append({
                            'id': node,
                            'attributes': dict(self.G.nodes[node])
                        })
                
                # اضافه کردن یال‌های مسیر
                for i in range(len(path) - 1):
                    edge_data = self.G.get_edge_data(path[i], path[i+1])
                    results['edges'].append({
                        'source': path[i],
                        'target': path[i+1],
                        'relation': edge_data.get('relation', ''),
                        'weight': edge_data.get('weight', 1.0)
                    })
                
                results['paths'].append(path)
        
        return results
    
    def _find_n_hop_paths(self, start_node: str, max_depth: int) -> List[List[str]]:
        """یافتن مسیرهای N-Hop"""
        paths = []
        visited = set()
        
        def dfs_paths(node: str, current_path: List[str], depth: int):
            if depth > max_depth or len(paths) >= 10:  # محدودیت تعداد مسیرها
                return
            
            current_path.append(node)
            visited.add(node)
            
            if depth > 0:  # مسیرهای غیر مستقیم
                paths.append(current_path.copy())
            
            for neighbor in self.G.neighbors(node):
                if neighbor not in visited:
                    dfs_paths(neighbor, current_path, depth + 1)
            
            current_path.pop()
            visited.remove(node)
        
        dfs_paths(start_node, [], 0)
        return paths
    
    def _calculate_simple_similarity(self, query: str, node: str) -> float:
        """محاسبه شباهت ساده"""
        query_words = set(query.lower().split())
        node_words = set(node.lower().split())
        
        if not query_words or not node_words:
            return 0.0
        
        intersection = query_words.intersection(node_words)
        union = query_words.union(node_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def hybrid_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """بازیابی ترکیبی"""
        results = {
            'nodes': [],
            'edges': [],
            'communities': [],
            'similarities': []
        }
        
        # ترکیب نتایج الگوریتم‌های مختلف
        bfs_results = self.bfs_retrieval(query, start_nodes)
        pagerank_results = self.pagerank_retrieval(query, start_nodes)
        semantic_results = self.semantic_similarity_retrieval(query, start_nodes)
        
        # ترکیب نودها
        all_nodes = {}
        for node_data in bfs_results.get('nodes', []):
            if isinstance(node_data, dict):
                node_id = node_data.get('id', node_data)
            else:
                node_id = node_data
            all_nodes[node_id] = {'source': 'bfs', 'data': node_data}
        
        for node_data in pagerank_results.get('nodes', []):
            node_id = node_data.get('id', node_data)
            if node_id not in all_nodes:
                all_nodes[node_id] = {'source': 'pagerank', 'data': node_data}
            else:
                all_nodes[node_id]['source'] = 'hybrid'
        
        for node_data in semantic_results.get('nodes', []):
            node_id = node_data.get('id', node_data)
            if node_id not in all_nodes:
                all_nodes[node_id] = {'source': 'semantic', 'data': node_data}
            else:
                all_nodes[node_id]['source'] = 'hybrid'
        
        # انتخاب نودهای برتر
        sorted_nodes = sorted(all_nodes.items(), key=lambda x: self._calculate_node_score(x[1]), reverse=True)
        
        for node_id, node_info in sorted_nodes[:self.config.max_nodes]:
            results['nodes'].append({
                'id': node_id,
                'source': node_info['source'],
                'data': node_info['data']
            })
        
        # ترکیب یال‌ها
        all_edges = set()
        for edge_data in bfs_results.get('edges', []):
            edge_key = (edge_data['source'], edge_data['target'])
            all_edges.add(edge_key)
            results['edges'].append(edge_data)
        
        for edge_data in pagerank_results.get('edges', []):
            edge_key = (edge_data['source'], edge_data['target'])
            if edge_key not in all_edges:
                all_edges.add(edge_key)
                results['edges'].append(edge_data)
        
        return results
    
    def _calculate_node_score(self, node_info: Dict) -> float:
        """محاسبه امتیاز نود"""
        score = 0.0
        
        if node_info['source'] == 'hybrid':
            score += 2.0
        elif node_info['source'] == 'pagerank':
            score += 1.5
        elif node_info['source'] == 'semantic':
            score += 1.0
        else:
            score += 0.5
        
        # اضافه کردن امتیاز بر اساس داده‌های نود
        if isinstance(node_info['data'], dict):
            if 'pagerank' in node_info['data']:
                score += node_info['data']['pagerank']
            if 'similarity' in node_info['data']:
                score += node_info['data']['similarity']
        
        return score
    
    def process_query(self, query: str, start_nodes: Optional[List[str]] = None) -> Dict:
        """پردازش سوال و بازیابی نتایج"""
        if not self.G:
            return {'error': 'گراف بارگذاری نشده است'}
        
        # استخراج توکن
        answer_types, entities = self.extract_tokens(query)
        
        # تعیین نودهای شروع
        if not start_nodes:
            start_nodes = entities if entities else list(self.G.nodes())[:5]
        
        # انتخاب الگوریتم بازیابی
        if self.config.retrieval_algorithm == RetrievalAlgorithm.BFS:
            results = self.bfs_retrieval(query, start_nodes)
        elif self.config.retrieval_algorithm == RetrievalAlgorithm.DFS:
            results = self.dfs_retrieval(query, start_nodes)
        elif self.config.retrieval_algorithm == RetrievalAlgorithm.PAGERANK:
            results = self.pagerank_retrieval(query, start_nodes)
        elif self.config.retrieval_algorithm == RetrievalAlgorithm.COMMUNITY_DETECTION:
            results = self.community_detection_retrieval(query, start_nodes)
        elif self.config.retrieval_algorithm == RetrievalAlgorithm.SEMANTIC_SIMILARITY:
            results = self.semantic_similarity_retrieval(query, start_nodes)
        elif self.config.retrieval_algorithm == RetrievalAlgorithm.N_HOP:
            results = self.n_hop_retrieval(query, start_nodes)
        elif self.config.retrieval_algorithm == RetrievalAlgorithm.HYBRID:
            results = self.hybrid_retrieval(query, start_nodes)
        else:
            results = self.hybrid_retrieval(query, start_nodes)
        
        # اضافه کردن اطلاعات تحلیل سوال
        results['query_analysis'] = {
            'query': query,
            'answer_types': answer_types,
            'entities': entities,
            'start_nodes': start_nodes,
            'algorithm': self.config.retrieval_algorithm.value,
            'token_extraction_method': self.config.token_extraction_method.value
        }
        
        return results
    
    def get_graph_statistics(self) -> Dict:
        """دریافت آمار گراف"""
        if not self.G:
            return {}
        
        return {
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'node_types': self._get_node_type_distribution(),
            'edge_types': self._get_edge_type_distribution(),
            'density': nx.density(self.G),
            'average_clustering': nx.average_clustering(self.G),
            'average_shortest_path': nx.average_shortest_path_length(self.G) if nx.is_connected(self.G) else None
        }
    
    def _get_node_type_distribution(self) -> Dict:
        """توزیع نوع نودها"""
        type_counts = defaultdict(int)
        for node, attrs in self.G.nodes(data=True):
            node_type = attrs.get('kind', 'unknown')
            type_counts[node_type] += 1
        return dict(type_counts)
    
    def _get_edge_type_distribution(self) -> Dict:
        """توزیع نوع یال‌ها"""
        type_counts = defaultdict(int)
        for source, target, attrs in self.G.edges(data=True):
            edge_type = attrs.get('relation', 'unknown')
            type_counts[edge_type] += 1
        return dict(type_counts) 
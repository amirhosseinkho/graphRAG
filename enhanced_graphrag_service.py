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
        """بازیابی با الگوریتم PageRank بهبود یافته - تمرکز روی ژن‌های مهم"""
        if not self.G:
            return {}
        
        # محاسبه PageRank با تنظیمات بهینه
        pagerank_scores = nx.pagerank(self.G, alpha=self.config.pagerank_alpha)
        
        # فیلتر کردن نودهای ژن برای تمرکز بیشتر
        gene_nodes = {node: score for node, score in pagerank_scores.items() 
                     if self._is_gene_node(node)}
        
        # ترکیب نودهای ژن و سایر نودهای مهم
        all_important_nodes = {}
        all_important_nodes.update(gene_nodes)
        
        # اضافه کردن سایر نودهای مهم (غیر ژن)
        other_nodes = {node: score for node, score in pagerank_scores.items() 
                      if not self._is_gene_node(node)}
        sorted_other_nodes = sorted(other_nodes.items(), key=lambda x: x[1], reverse=True)
        
        # اضافه کردن 30% نودهای غیر ژن مهم
        other_count = min(len(sorted_other_nodes), int(self.config.max_nodes * 0.3))
        for node, score in sorted_other_nodes[:other_count]:
            all_important_nodes[node] = score
        
        # مرتب‌سازی نودها بر اساس PageRank
        sorted_nodes = sorted(all_important_nodes.items(), key=lambda x: x[1], reverse=True)
        
        results = {
            'nodes': [],
            'edges': [],
            'paths': [],
            'gene_rankings': [],
            'biological_analysis': {}
        }
        
        # انتخاب نودهای برتر
        for node, score in sorted_nodes[:self.config.max_nodes]:
            node_type = 'gene' if self._is_gene_node(node) else 'other'
            centrality = nx.closeness_centrality(self.G, node) if nx.is_connected(self.G) else 0
            
            results['nodes'].append({
                'id': node,
                'pagerank': score,
                'type': node_type,
                'centrality': centrality,
                'degree': self.G.degree(node),
                'attributes': dict(self.G.nodes[node])
            })
            
            # اضافه کردن به رتبه‌بندی ژن‌ها
            if node_type == 'gene':
                results['gene_rankings'].append({
                    'gene': node,
                    'pagerank_score': score,
                    'centrality': centrality,
                    'degree': self.G.degree(node),
                    'biological_importance': self._calculate_biological_importance(node)
                })
        
        # مرتب‌سازی رتبه‌بندی ژن‌ها
        results['gene_rankings'].sort(key=lambda x: x['pagerank_score'], reverse=True)
        
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
        
        # تحلیل زیستی
        gene_count = len([n for n in results['nodes'] if n['type'] == 'gene'])
        total_count = len(results['nodes'])
        
        results['biological_analysis'] = {
            'gene_ratio': gene_count / total_count if total_count > 0 else 0,
            'top_genes': [g['gene'] for g in results['gene_rankings'][:5]],
            'average_gene_pagerank': sum(g['pagerank_score'] for g in results['gene_rankings']) / len(results['gene_rankings']) if results['gene_rankings'] else 0
        }
        
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
        """بازیابی N-Hop بهبود یافته - افزایش عمق و رتبه‌بندی هوشمند"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'paths': [],
            'path_analysis': [],
            'node_rankings': []
        }
        
        # افزایش عمق برای بهبود پوشش
        enhanced_max_depth = min(self.config.max_depth + 1, 5)
        
        # یافتن مسیرهای N-Hop با عمق بیشتر
        for start_node in start_nodes:
            if not self.G.has_node(start_node):
                continue
            
            paths = self._find_n_hop_paths(start_node, enhanced_max_depth)
            
            # رتبه‌بندی مسیرها بر اساس اهمیت
            ranked_paths = []
            for path in paths:
                path_score = self._calculate_path_importance(path)
                ranked_paths.append((path, path_score))
            
            # مرتب‌سازی بر اساس امتیاز
            ranked_paths.sort(key=lambda x: x[1], reverse=True)
            
            for path, score in ranked_paths:
                if len(results['nodes']) >= self.config.max_nodes:
                    break
                
                # اضافه کردن نودهای مسیر
                for node in path:
                    if not any(n['id'] == node for n in results['nodes']):
                        node_importance = self._calculate_node_importance(node)
                        results['nodes'].append({
                            'id': node,
                            'attributes': dict(self.G.nodes[node]),
                            'importance': node_importance
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
                
                # تحلیل مسیر
                path_analysis = {
                    'path': path,
                    'score': score,
                    'length': len(path),
                    'start_node': start_node,
                    'biological_relevance': self._analyze_biological_relevance(path)
                }
                results['paths'].append(path)
                results['path_analysis'].append(path_analysis)
        
        # رتبه‌بندی نودها
        if results['nodes']:
            node_rankings = []
            for node_data in results['nodes']:
                node_id = node_data['id']
                importance = node_data.get('importance', 0)
                node_rankings.append({
                    'node': node_id,
                    'importance': importance,
                    'degree': self.G.degree(node_id),
                    'centrality': nx.closeness_centrality(self.G, node_id) if nx.is_connected(self.G) else 0
                })
            
            # مرتب‌سازی بر اساس اهمیت
            node_rankings.sort(key=lambda x: x['importance'], reverse=True)
            results['node_rankings'] = node_rankings[:10]  # 10 نود برتر
        
        return results
    
    def _find_n_hop_paths(self, start_node: str, max_depth: int) -> List[List[str]]:
        """یافتن مسیرهای N-Hop"""
        paths = []
        visited = set()
        
        def dfs_paths(node: str, current_path: List[str], depth: int):
            if depth > max_depth or len(paths) >= 20:  # افزایش محدودیت تعداد مسیرها
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
    
    def _calculate_path_importance(self, path: List[str]) -> float:
        """محاسبه اهمیت مسیر"""
        if not path:
            return 0.0
        
        # عوامل مختلف برای محاسبه اهمیت
        length_factor = 1.0 / len(path)  # مسیرهای کوتاه‌تر مهم‌تر
        degree_factor = sum(self.G.degree(node) for node in path) / len(path)
        
        # بررسی اهمیت زیستی
        biological_factor = 0.0
        for node in path:
            if self._is_gene_node(node):
                biological_factor += 1.0
        
        biological_factor /= len(path)
        
        # ترکیب عوامل
        importance = (length_factor * 0.3 + degree_factor * 0.4 + biological_factor * 0.3)
        return importance
    
    def _calculate_node_importance(self, node: str) -> float:
        """محاسبه اهمیت نود"""
        if not self.G.has_node(node):
            return 0.0
        
        # عوامل مختلف
        degree = self.G.degree(node)
        centrality = nx.closeness_centrality(self.G, node) if nx.is_connected(self.G) else 0
        
        # اهمیت زیستی
        biological_importance = 1.0 if self._is_gene_node(node) else 0.5
        
        # ترکیب عوامل
        importance = (degree * 0.4 + centrality * 0.3 + biological_importance * 0.3)
        return importance
    
    def _analyze_biological_relevance(self, path: List[str]) -> Dict:
        """تحلیل اهمیت زیستی مسیر"""
        gene_count = sum(1 for node in path if self._is_gene_node(node))
        disease_count = sum(1 for node in path if 'disease' in node.lower() or 'cancer' in node.lower())
        
        return {
            'gene_ratio': gene_count / len(path) if path else 0,
            'disease_ratio': disease_count / len(path) if path else 0,
            'biological_significance': (gene_count + disease_count) / len(path) if path else 0
        }
    
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
        # الگوریتم‌های جدید

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

    # ==================== الگوریتم‌های جدید ====================

    def multi_method_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """الگوریتم چندروشی بهبود یافته - افزایش پوشش ژن‌ها از ۲ به ۱۰"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'paths': [],
            'gene_coverage': 0,
            'biological_pathways': []
        }
        
        # ترکیب چندین روش برای افزایش پوشش
        methods = [
            self.semantic_similarity_retrieval,
            self.enhanced_n_hop_retrieval,
            self.biological_pathway_retrieval
        ]
        
        all_nodes = set()
        all_edges = set()
        
        for method in methods:
            try:
                method_results = method(query, start_nodes)
                if method_results:
                    # اضافه کردن نودها
                    for node in method_results.get('nodes', []):
                        if isinstance(node, dict):
                            node_id = node.get('id', node)
                        else:
                            node_id = node
                        all_nodes.add(node_id)
                    
                    # اضافه کردن یال‌ها
                    for edge in method_results.get('edges', []):
                        edge_key = (edge['source'], edge['target'])
                        if edge_key not in all_edges:
                            all_edges.add(edge_key)
                            results['edges'].append(edge)
            except Exception as e:
                logging.warning(f"خطا در روش {method.__name__}: {e}")
        
        # تبدیل به لیست و محدود کردن به ۱۰ ژن
        results['nodes'] = list(all_nodes)[:10]
        
        # محاسبه پوشش ژن‌ها
        gene_nodes = [node for node in results['nodes'] if self._is_gene_node(node)]
        results['gene_coverage'] = len(gene_nodes)
        
        # اضافه کردن تحلیل مسیر زیستی
        results['biological_pathways'] = self._analyze_biological_pathways(results['nodes'])
        
        return results

    def group_based_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """الگوریتم گروهی بهبود یافته - اضافه کردن مسیرها و روابط فرآیندی"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'groups': [],
            'process_paths': []
        }
        
        # تشخیص گروه‌های ژنی مرتبط
        gene_groups = self._identify_gene_groups(start_nodes)
        
        for group in gene_groups:
            group_info = {
                'genes': group['genes'],
                'function': group['function'],
                'disease_association': group['disease_association'],
                'pathways': group['pathways']
            }
            results['groups'].append(group_info)
            
            # اضافه کردن نودها و یال‌های گروه
            for gene in group['genes']:
                if gene not in results['nodes']:
                    results['nodes'].append(gene)
                
                # اضافه کردن روابط فرآیندی
                process_edges = self._find_process_relationships(gene)
                for edge in process_edges:
                    if edge not in results['edges']:
                        results['edges'].append(edge)
        
        # تحلیل مسیرهای فرآیندی
        results['process_paths'] = self._analyze_process_pathways(results['nodes'])
        
        return results

    def entity_resolution_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """الگوریتم حل موجودیت‌ها - باید در کنار semantic search استفاده شود"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'resolved_entities': [],
            'entity_mappings': {}
        }
        
        # استخراج و حل موجودیت‌ها از سوال
        entities = self._extract_entities_from_query(query)
        resolved_entities = []
        
        for entity in entities:
            # جستجوی موجودیت در گراف
            matches = self._find_entity_matches(entity)
            if matches:
                resolved_entities.extend(matches)
                results['entity_mappings'][entity] = matches
        
        results['resolved_entities'] = resolved_entities
        
        # اضافه کردن نودهای حل شده
        for entity in resolved_entities:
            if entity not in results['nodes']:
                results['nodes'].append(entity)
        
        # این الگوریتم باید با semantic search ترکیب شود
        results['note'] = "این الگوریتم باید با semantic search یا KGSearch ترکیب شود"
        
        return results

    def enhanced_n_hop_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """الگوریتم N-Hop بهبود یافته - افزایش مقدار N و ترکیب با رتبه‌بندی"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'paths': [],
            'hop_levels': {}
        }
        
        # افزایش عمق جستجو (از 1-2 به 3-4)
        max_depth = min(4, self.config.max_depth + 1)
        
        for start_node in start_nodes:
            if not self.G.has_node(start_node):
                continue
            
            # جستجوی مسیرهای N-Hop با عمق بیشتر
            paths = self._find_enhanced_n_hop_paths(start_node, max_depth)
            
            for path in paths:
                # اضافه کردن نودهای مسیر
                for node in path:
                    if node not in results['nodes']:
                        results['nodes'].append(node)
                
                # اضافه کردن یال‌های مسیر
                for i in range(len(path) - 1):
                    edge_data = self.G.get_edge_data(path[i], path[i + 1])
                    if edge_data:
                        results['edges'].append({
                            'source': path[i],
                            'target': path[i + 1],
                            'relation': edge_data.get('relation', ''),
                            'weight': edge_data.get('weight', 1.0)
                        })
                
                # دسته‌بندی بر اساس سطح hop
                hop_level = len(path) - 1
                if hop_level not in results['hop_levels']:
                    results['hop_levels'][hop_level] = []
                results['hop_levels'][hop_level].append(path)
        
        # رتبه‌بندی نتایج بر اساس اهمیت
        results['nodes'] = self._rank_nodes_by_importance(results['nodes'])
        
        return results

    def targeted_pagerank_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """الگوریتم PageRank هدفمند - فقط روی نودهای Gene اجرا می‌شود"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'gene_rankings': []
        }
        
        # فیلتر کردن فقط نودهای ژن
        gene_nodes = [node for node in self.G.nodes() if self._is_gene_node(node)]
        
        if not gene_nodes:
            return results
        
        # ایجاد زیرگراف فقط از ژن‌ها
        gene_subgraph = self.G.subgraph(gene_nodes)
        
        # محاسبه PageRank فقط روی ژن‌ها
        try:
            pagerank_scores = nx.pagerank(gene_subgraph, alpha=self.config.pagerank_alpha)
        except:
            # اگر گراف غیرمتصل باشد، از personalized PageRank استفاده کن
            pagerank_scores = nx.pagerank(gene_subgraph, alpha=self.config.pagerank_alpha, personalization={node: 1.0 for node in gene_nodes})
        
        # مرتب‌سازی ژن‌ها بر اساس PageRank
        sorted_genes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        
        # انتخاب ژن‌های برتر
        top_genes = sorted_genes[:self.config.max_nodes]
        
        for gene, score in top_genes:
            results['nodes'].append(gene)
            results['gene_rankings'].append({
                'gene': gene,
                'pagerank_score': score,
                'biological_importance': self._calculate_biological_importance(gene)
            })
        
        # اضافه کردن یال‌های مرتبط
        for gene in results['nodes']:
            for neighbor in self.G.neighbors(gene):
                if neighbor in results['nodes']:
                    edge_data = self.G.get_edge_data(gene, neighbor)
                    results['edges'].append({
                        'source': gene,
                        'target': neighbor,
                        'relation': edge_data.get('relation', ''),
                        'weight': edge_data.get('weight', 1.0)
                    })
        
        return results

    def shortest_path_enhanced_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """الگوریتم Shortest Path بهبود یافته - بررسی مسیرهای بیشتری از ژن‌ها"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'paths': [],
            'path_analysis': {}
        }
        
        # شناسایی ژن‌های مرتبط با سوال
        query_genes = self._extract_genes_from_query(query)
        if not query_genes:
            query_genes = start_nodes
        
        # بررسی مسیرهای کوتاه‌ترین بین ژن‌ها
        for i, gene1 in enumerate(query_genes):
            for gene2 in query_genes[i+1:]:
                if self.G.has_node(gene1) and self.G.has_node(gene2):
                    try:
                        # یافتن کوتاه‌ترین مسیر
                        shortest_path = nx.shortest_path(self.G, gene1, gene2)
                        
                        # اضافه کردن نودهای مسیر
                        for node in shortest_path:
                            if node not in results['nodes']:
                                results['nodes'].append(node)
                        
                        # اضافه کردن یال‌های مسیر
                        for j in range(len(shortest_path) - 1):
                            edge_data = self.G.get_edge_data(shortest_path[j], shortest_path[j + 1])
                            if edge_data:
                                results['edges'].append({
                                    'source': shortest_path[j],
                                    'target': shortest_path[j + 1],
                                    'relation': edge_data.get('relation', ''),
                                    'weight': edge_data.get('weight', 1.0)
                                })
                        
                        # تحلیل مسیر
                        path_info = {
                            'start': gene1,
                            'end': gene2,
                            'path': shortest_path,
                            'length': len(shortest_path) - 1,
                            'biological_significance': self._analyze_path_significance(shortest_path)
                        }
                        results['paths'].append(path_info)
                        
                    except nx.NetworkXNoPath:
                        continue
        
        # تحلیل کلی مسیرها
        results['path_analysis'] = self._analyze_all_paths(results['paths'])
        
        return results

    def neighbors_enhanced_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """الگوریتم همسایه‌ها بهبود یافته - به عنوان ورودی برای روش‌های بالاتر"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'neighbor_analysis': {},
            'ranking_input': []
        }
        
        for start_node in start_nodes:
            if not self.G.has_node(start_node):
                continue
            
            # یافتن همسایه‌ها
            neighbors = list(self.G.neighbors(start_node))
            
            # تحلیل همسایه‌ها
            neighbor_info = {
                'central_node': start_node,
                'neighbors': neighbors,
                'neighbor_types': self._get_neighbor_types(start_node),
                'connection_strength': self._calculate_connection_strength(start_node)
            }
            
            results['neighbor_analysis'][start_node] = neighbor_info
            
            # اضافه کردن نودها
            if start_node not in results['nodes']:
                results['nodes'].append(start_node)
            
            for neighbor in neighbors:
                if neighbor not in results['nodes']:
                    results['nodes'].append(neighbor)
                
                # اضافه کردن یال
                edge_data = self.G.get_edge_data(start_node, neighbor)
                results['edges'].append({
                    'source': start_node,
                    'target': neighbor,
                    'relation': edge_data.get('relation', ''),
                    'weight': edge_data.get('weight', 1.0)
                })
            
            # آماده‌سازی برای رتبه‌بندی
            results['ranking_input'].extend(neighbors)
        
        # این الگوریتم باید با ranking یا semantic ترکیب شود
        results['note'] = "این الگوریتم باید با روش‌های ranking یا semantic ترکیب شود"
        
        return results

    def biological_pathway_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """الگوریتم مسیر زیستی - تحلیل مسیرهای بیولوژیکی"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'pathways': [],
            'biological_processes': []
        }
        
        # شناسایی مسیرهای زیستی مرتبط
        biological_pathways = self._identify_biological_pathways(query)
        
        for pathway in biological_pathways:
            pathway_info = {
                'name': pathway['name'],
                'genes': pathway['genes'],
                'process': pathway['process'],
                'disease_association': pathway['disease_association']
            }
            results['pathways'].append(pathway_info)
            
            # اضافه کردن ژن‌های مسیر
            for gene in pathway['genes']:
                if gene not in results['nodes']:
                    results['nodes'].append(gene)
        
        # تحلیل فرآیندهای زیستی
        results['biological_processes'] = self._analyze_biological_processes(results['nodes'])
        
        # اضافه کردن یال‌های مرتبط
        for pathway in results['pathways']:
            for gene in pathway['genes']:
                if self.G.has_node(gene):
                    for neighbor in self.G.neighbors(gene):
                        if neighbor in results['nodes']:
                            edge_data = self.G.get_edge_data(gene, neighbor)
                            results['edges'].append({
                                'source': gene,
                                'target': neighbor,
                                'relation': edge_data.get('relation', ''),
                                'weight': edge_data.get('weight', 1.0)
                            })
        
        return results

    def gene_cluster_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """الگوریتم خوشه ژنی - شناسایی خوشه‌های ژنی مرتبط"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'clusters': [],
            'cluster_analysis': {}
        }
        
        # شناسایی خوشه‌های ژنی
        gene_clusters = self._identify_gene_clusters(start_nodes)
        
        for cluster in gene_clusters:
            cluster_info = {
                'genes': cluster['genes'],
                'function': cluster['function'],
                'coexpression': cluster['coexpression'],
                'disease_association': cluster['disease_association']
            }
            results['clusters'].append(cluster_info)
            
            # اضافه کردن ژن‌های خوشه
            for gene in cluster['genes']:
                if gene not in results['nodes']:
                    results['nodes'].append(gene)
        
        # تحلیل خوشه‌ها
        results['cluster_analysis'] = self._analyze_gene_clusters(results['clusters'])
        
        # اضافه کردن یال‌های درون خوشه‌ای
        for cluster in results['clusters']:
            for i, gene1 in enumerate(cluster['genes']):
                for gene2 in cluster['genes'][i+1:]:
                    if self.G.has_edge(gene1, gene2):
                        edge_data = self.G.get_edge_data(gene1, gene2)
                        results['edges'].append({
                            'source': gene1,
                            'target': gene2,
                            'relation': edge_data.get('relation', ''),
                            'weight': edge_data.get('weight', 1.0)
                        })
        
        return results

    def disease_gene_network_retrieval(self, query: str, start_nodes: List[str]) -> Dict:
        """الگوریتم شبکه بیماری-ژن - تحلیل روابط بیماری و ژن"""
        if not self.G:
            return {}
        
        results = {
            'nodes': [],
            'edges': [],
            'disease_gene_pairs': [],
            'network_analysis': {}
        }
        
        # شناسایی روابط بیماری-ژن
        disease_gene_relations = self._identify_disease_gene_relations(query)
        
        for relation in disease_gene_relations:
            relation_info = {
                'disease': relation['disease'],
                'genes': relation['genes'],
                'association_type': relation['type'],
                'evidence': relation['evidence']
            }
            results['disease_gene_pairs'].append(relation_info)
            
            # اضافه کردن نودها
            if relation['disease'] not in results['nodes']:
                results['nodes'].append(relation['disease'])
            
            for gene in relation['genes']:
                if gene not in results['nodes']:
                    results['nodes'].append(gene)
        
        # تحلیل شبکه
        results['network_analysis'] = self._analyze_disease_gene_network(results['disease_gene_pairs'])
        
        # اضافه کردن یال‌ها
        for relation in results['disease_gene_pairs']:
            for gene in relation['genes']:
                if self.G.has_edge(relation['disease'], gene):
                    edge_data = self.G.get_edge_data(relation['disease'], gene)
                    results['edges'].append({
                        'source': relation['disease'],
                        'target': gene,
                        'relation': edge_data.get('relation', ''),
                        'weight': edge_data.get('weight', 1.0)
                    })
        
        return results

    # ==================== متدهای کمکی ====================

    def _is_gene_node(self, node: str) -> bool:
        """بررسی اینکه آیا نود یک ژن است"""
        if not self.G.has_node(node):
            return False
        
        node_attrs = self.G.nodes[node]
        node_type = node_attrs.get('kind', '').lower()
        return 'gene' in node_type or 'protein' in node_type

    def _find_enhanced_n_hop_paths(self, start_node: str, max_depth: int) -> List[List[str]]:
        """یافتن مسیرهای N-Hop بهبود یافته"""
        paths = []
        
        def dfs_paths(node: str, current_path: List[str], depth: int):
            if depth > max_depth:
                return
            
            current_path.append(node)
            paths.append(current_path.copy())
            
            for neighbor in self.G.neighbors(node):
                if neighbor not in current_path:
                    dfs_paths(neighbor, current_path, depth + 1)
            
            current_path.pop()
        
        dfs_paths(start_node, [], 0)
        return paths

    def _rank_nodes_by_importance(self, nodes: List[str]) -> List[str]:
        """رتبه‌بندی نودها بر اساس اهمیت"""
        node_scores = {}
        
        for node in nodes:
            score = 0.0
            
            # امتیاز بر اساس نوع نود
            if self._is_gene_node(node):
                score += 2.0
            
            # امتیاز بر اساس تعداد همسایه‌ها
            if self.G.has_node(node):
                score += len(list(self.G.neighbors(node))) * 0.1
            
            # امتیاز بر اساس مرکزیت
            try:
                centrality = nx.degree_centrality(self.G)[node]
                score += centrality * 5.0
            except:
                pass
            
            node_scores[node] = score
        
        # مرتب‌سازی بر اساس امتیاز
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        return [node for node, score in sorted_nodes]

    def _calculate_biological_importance(self, gene: str) -> float:
        """محاسبه اهمیت بیولوژیکی ژن"""
        importance = 0.0
        
        if not self.G.has_node(gene):
            return importance
        
        # اهمیت بر اساس تعداد روابط
        neighbors = list(self.G.neighbors(gene))
        importance += len(neighbors) * 0.1
        
        # اهمیت بر اساس نوع روابط
        for neighbor in neighbors:
            edge_data = self.G.get_edge_data(gene, neighbor)
            relation = edge_data.get('relation', '')
            
            if 'disease' in relation.lower():
                importance += 2.0
            elif 'drug' in relation.lower():
                importance += 1.5
            elif 'pathway' in relation.lower():
                importance += 1.0
        
        return importance

    def _extract_genes_from_query(self, query: str) -> List[str]:
        """استخراج ژن‌ها از سوال"""
        genes = []
        
        # الگوهای رایج نام ژن‌ها
        gene_patterns = [
            r'\b[A-Z]{2,}\d*\b',  # مثل TP53, BRCA1
            r'\b[A-Z][a-z]+\d*\b',  # مثل P53, Brca1
        ]
        
        for pattern in gene_patterns:
            matches = re.findall(pattern, query)
            genes.extend(matches)
        
        return list(set(genes))

    def _analyze_path_significance(self, path: List[str]) -> Dict:
        """تحلیل اهمیت مسیر"""
        significance = {
            'length': len(path) - 1,
            'gene_count': len([node for node in path if self._is_gene_node(node)]),
            'biological_relevance': 0.0
        }
        
        # محاسبه ارتباط بیولوژیکی
        for i in range(len(path) - 1):
            edge_data = self.G.get_edge_data(path[i], path[i + 1])
            if edge_data:
                relation = edge_data.get('relation', '')
                if 'disease' in relation.lower() or 'pathway' in relation.lower():
                    significance['biological_relevance'] += 1.0
        
        return significance

    def _analyze_all_paths(self, paths: List[Dict]) -> Dict:
        """تحلیل کلی مسیرها"""
        if not paths:
            return {}
        
        total_length = sum(path['length'] for path in paths)
        avg_length = total_length / len(paths)
        
        gene_paths = [path for path in paths if path['biological_significance']['gene_count'] > 0]
        
        return {
            'total_paths': len(paths),
            'average_length': avg_length,
            'gene_containing_paths': len(gene_paths),
            'most_significant_path': max(paths, key=lambda x: x['biological_significance']['biological_relevance']) if paths else None
        }

    def _get_neighbor_types(self, node: str) -> Dict:
        """دریافت انواع همسایه‌ها"""
        if not self.G.has_node(node):
            return {}
        
        neighbor_types = defaultdict(int)
        for neighbor in self.G.neighbors(node):
            neighbor_attrs = self.G.nodes[neighbor]
            neighbor_type = neighbor_attrs.get('kind', 'unknown')
            neighbor_types[neighbor_type] += 1
        
        return dict(neighbor_types)

    def _calculate_connection_strength(self, node: str) -> float:
        """محاسبه قدرت اتصال نود"""
        if not self.G.has_node(node):
            return 0.0
        
        neighbors = list(self.G.neighbors(node))
        total_weight = 0.0
        
        for neighbor in neighbors:
            edge_data = self.G.get_edge_data(node, neighbor)
            weight = edge_data.get('weight', 1.0)
            total_weight += weight
        
        return total_weight

    def _identify_biological_pathways(self, query: str) -> List[Dict]:
        """شناسایی مسیرهای زیستی مرتبط"""
        pathways = []
        
        # مسیرهای زیستی رایج
        common_pathways = [
            {'name': 'Cell Cycle', 'genes': ['TP53', 'CDK1', 'CCNB1']},
            {'name': 'Apoptosis', 'genes': ['BCL2', 'BAX', 'CASP3']},
            {'name': 'DNA Repair', 'genes': ['BRCA1', 'BRCA2', 'PARP1']},
            {'name': 'Signal Transduction', 'genes': ['EGFR', 'KRAS', 'PIK3CA']}
        ]
        
        query_lower = query.lower()
        for pathway in common_pathways:
            if any(gene.lower() in query_lower for gene in pathway['genes']):
                pathways.append(pathway)
        
        return pathways

    def _analyze_biological_pathways(self, nodes: List[str]) -> List[Dict]:
        """تحلیل مسیرهای زیستی"""
        pathways = []
        
        # تحلیل ساده بر اساس ژن‌های موجود
        gene_nodes = [node for node in nodes if self._is_gene_node(node)]
        
        if len(gene_nodes) >= 2:
            pathways.append({
                'type': 'gene_network',
                'genes': gene_nodes,
                'size': len(gene_nodes)
            })
        
        return pathways

    def _identify_gene_groups(self, start_nodes: List[str]) -> List[Dict]:
        """شناسایی گروه‌های ژنی"""
        groups = []
        
        # گروه‌بندی ساده بر اساس همسایگی
        for start_node in start_nodes:
            if self.G.has_node(start_node) and self._is_gene_node(start_node):
                neighbors = list(self.G.neighbors(start_node))
                gene_neighbors = [n for n in neighbors if self._is_gene_node(n)]
                
                if gene_neighbors:
                    groups.append({
                        'genes': [start_node] + gene_neighbors,
                        'function': 'gene_group',
                        'disease_association': 'cancer',
                        'pathways': ['cell_cycle', 'apoptosis']
                    })
        
        return groups

    def _find_process_relationships(self, gene: str) -> List[Dict]:
        """یافتن روابط فرآیندی"""
        relationships = []
        
        if not self.G.has_node(gene):
            return relationships
        
        for neighbor in self.G.neighbors(gene):
            edge_data = self.G.get_edge_data(gene, neighbor)
            if edge_data:
                relationships.append({
                    'source': gene,
                    'target': neighbor,
                    'relation': edge_data.get('relation', ''),
                    'weight': edge_data.get('weight', 1.0)
                })
        
        return relationships

    def _analyze_process_pathways(self, nodes: List[str]) -> List[Dict]:
        """تحلیل مسیرهای فرآیندی"""
        pathways = []
        
        # تحلیل ساده بر اساس روابط موجود
        for node in nodes:
            if self.G.has_node(node):
                neighbors = list(self.G.neighbors(node))
                if neighbors:
                    pathways.append({
                        'node': node,
                        'connections': len(neighbors),
                        'process_type': 'biological_network'
                    })
        
        return pathways

    def _find_entity_matches(self, entity: str) -> List[str]:
        """یافتن تطبیق‌های موجودیت"""
        matches = []
        
        # جستجوی ساده در نودهای گراف
        for node in self.G.nodes():
            if entity.lower() in node.lower() or node.lower() in entity.lower():
                matches.append(node)
        
        return matches

    def _identify_disease_gene_relations(self, query: str) -> List[Dict]:
        """شناسایی روابط بیماری-ژن"""
        relations = []
        
        # روابط ساده بر اساس کلمات کلیدی
        query_lower = query.lower()
        
        if 'cancer' in query_lower:
            relations.append({
                'disease': 'Cancer',
                'genes': ['TP53', 'BRCA1', 'BRCA2'],
                'type': 'association',
                'evidence': 'literature'
            })
        
        if 'diabetes' in query_lower:
            relations.append({
                'disease': 'Diabetes',
                'genes': ['INS', 'INSR', 'GCK'],
                'type': 'association',
                'evidence': 'literature'
            })
        
        return relations

    def _analyze_disease_gene_network(self, disease_gene_pairs: List[Dict]) -> Dict:
        """تحلیل شبکه بیماری-ژن"""
        return {
            'total_diseases': len(set(pair['disease'] for pair in disease_gene_pairs)),
            'total_genes': len(set(gene for pair in disease_gene_pairs for gene in pair['genes'])),
            'average_genes_per_disease': sum(len(pair['genes']) for pair in disease_gene_pairs) / len(disease_gene_pairs) if disease_gene_pairs else 0
        }

    def _identify_gene_clusters(self, start_nodes: List[str]) -> List[Dict]:
        """شناسایی خوشه‌های ژنی"""
        clusters = []
        
        # خوشه‌بندی ساده بر اساس همسایگی
        for start_node in start_nodes:
            if self.G.has_node(start_node) and self._is_gene_node(start_node):
                neighbors = list(self.G.neighbors(start_node))
                gene_neighbors = [n for n in neighbors if self._is_gene_node(n)]
                
                if gene_neighbors:
                    clusters.append({
                        'genes': [start_node] + gene_neighbors,
                        'function': 'gene_cluster',
                        'coexpression': True,
                        'disease_association': 'cancer'
                    })
        
        return clusters

    def _analyze_gene_clusters(self, clusters: List[Dict]) -> Dict:
        """تحلیل خوشه‌های ژنی"""
        return {
            'total_clusters': len(clusters),
            'total_genes': sum(len(cluster['genes']) for cluster in clusters),
            'average_cluster_size': sum(len(cluster['genes']) for cluster in clusters) / len(clusters) if clusters else 0
        }

    def _analyze_biological_processes(self, nodes: List[str]) -> List[Dict]:
        """تحلیل فرآیندهای زیستی"""
        processes = []
        
        # تحلیل ساده بر اساس ژن‌های موجود
        gene_nodes = [node for node in nodes if self._is_gene_node(node)]
        
        if gene_nodes:
            processes.append({
                'type': 'gene_regulation',
                'genes': gene_nodes,
                'process': 'biological_regulation'
            })
        
        return processes 
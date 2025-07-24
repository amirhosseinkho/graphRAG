# -*- coding: utf-8 -*-
"""
GraphRAG Service - سرویس اصلی GraphRAG
این سرویس تمام قابلیت‌های GraphRAG را فراهم می‌کند
"""

import pandas as pd
import networkx as nx
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import deque
import pickle
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class RetrievalMethod(Enum):
    """روش‌های بازیابی"""
    BFS = "BFS"
    DFS = "DFS"
    SHORTEST_PATH = "Shortest Path"
    NEIGHBORS = "Neighbors"
    HYBRID = "Hybrid"
    MULTI_METHOD = "Multi-Method"
    ENSEMBLE = "Ensemble"
    ADAPTIVE = "Adaptive"
    INTELLIGENT = "Intelligent Semantic Search"
    NO_RETRIEVAL = "بدون بازیابی (فقط مدل)"

class GenerationModel(Enum):
    """مدل‌های تولید متن"""
    SIMPLE = "Simple Template"
    GPT_SIMULATION = "GPT Simulation"
    CUSTOM = "Custom Model"
    HUGGINGFACE = "HuggingFace Models"
    OPENAI_GPT = "OpenAI GPT"
    ANTHROPIC_CLAUDE = "Anthropic Claude"
    GOOGLE_GEMINI = "Google Gemini"

@dataclass
class GraphNode:
    """نمایش یک نود گراف"""
    id: str
    name: str
    kind: str
    depth: int = 0
    score: float = 1.0

@dataclass
class GraphEdge:
    """نمایش یک یال گراف"""
    source: str
    target: str
    relation: str
    weight: float = 1.0

@dataclass
class RetrievalResult:
    """نتیجه بازیابی"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    paths: List[List[str]]
    context_text: str
    method: str
    query: str

@dataclass
class GenerationResult:
    """نتیجه تولید متن"""
    answer: str
    model: str
    context_used: str
    confidence: float

class GraphRAGService:
    """سرویس اصلی GraphRAG"""
    
    def __init__(self, graph_data_path: str = None):
        self.G = None
        self.nlp = None
        self.graph_data_path = graph_data_path
        self.initialize()
    
    def initialize(self):
        """راه‌اندازی سرویس"""
        print("🔧 راه‌اندازی GraphRAG Service...")
        
        # بارگذاری مدل spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ مدل spaCy بارگذاری شد")
        except:
            print("❌ خطا در بارگذاری مدل spaCy")
            return
        
        # بارگذاری یا ایجاد گراف
        if self.graph_data_path and os.path.exists(self.graph_data_path):
            self.load_graph_from_file()
        else:
            self.create_sample_graph()
    
    def create_sample_graph(self):
        """ایجاد گراف نمونه"""
        print("🔧 ایجاد گراف نمونه...")
        
        self.G = nx.Graph()
        
        # نودهای نمونه
        sample_nodes = [
            ('Gene::HMGB3', 'HMGB3', 'Gene'),
            ('Gene::PCNA', 'PCNA', 'Gene'),
            ('Gene::TP53', 'TP53', 'Gene'),
            ('Gene::BRCA1', 'BRCA1', 'Gene'),
            ('Disease::Diabetes', 'Type 2 Diabetes', 'Disease'),
            ('Disease::Cancer', 'Cancer', 'Disease'),
            ('Disease::HeartDisease', 'Heart Disease', 'Disease'),
            ('Drug::Metformin', 'Metformin', 'Drug'),
            ('Drug::Aspirin', 'Aspirin', 'Drug'),
            ('Drug::Insulin', 'Insulin', 'Drug'),
            ('Biological Process::GO:0008150', 'Metabolic Process', 'Biological Process'),
            ('Biological Process::GO:0006915', 'Apoptosis', 'Biological Process'),
            ('Biological Process::GO:0007049', 'Cell Cycle', 'Biological Process'),
            ('Anatomy::Heart', 'Heart', 'Anatomy'),
            ('Anatomy::Lung', 'Lung', 'Anatomy'),
            ('Anatomy::Brain', 'Brain', 'Anatomy'),
            ('Anatomy::Liver', 'Liver', 'Anatomy')
        ]
        
        for node_id, name, kind in sample_nodes:
            self.G.add_node(node_id, name=name, kind=kind)
        
        # یال‌های نمونه
        sample_edges = [
            ('Gene::HMGB3', 'Gene::PCNA', 'interacts_with'),
            ('Gene::PCNA', 'Disease::Diabetes', 'associates'),
            ('Gene::TP53', 'Disease::Cancer', 'causes'),
            ('Gene::BRCA1', 'Disease::Cancer', 'causes'),
            ('Drug::Metformin', 'Disease::Diabetes', 'treats'),
            ('Drug::Aspirin', 'Disease::HeartDisease', 'prevents'),
            ('Drug::Insulin', 'Disease::Diabetes', 'treats'),
            ('Gene::HMGB3', 'Biological Process::GO:0008150', 'participates_in'),
            ('Gene::TP53', 'Biological Process::GO:0006915', 'regulates'),
            ('Gene::BRCA1', 'Biological Process::GO:0007049', 'regulates'),
            ('Anatomy::Heart', 'Anatomy::Lung', 'adjacent_to'),
            ('Anatomy::Brain', 'Anatomy::Heart', 'controls'),
            ('Gene::HMGB3', 'Anatomy::Heart', 'expressed_in'),
            ('Gene::TP53', 'Anatomy::Brain', 'expressed_in'),
            ('Gene::BRCA1', 'Anatomy::Liver', 'expressed_in'),
            ('Disease::Diabetes', 'Anatomy::Heart', 'affects'),
            ('Disease::Cancer', 'Anatomy::Brain', 'affects')
        ]
        
        for source, target, relation in sample_edges:
            self.G.add_edge(source, target, metaedge=relation)
        
        print(f"✅ گراف نمونه ساخته شد: {self.G.number_of_nodes()} نود، {self.G.number_of_edges()} یال")
    
    def load_graph_from_file(self):
        """بارگذاری گراف از فایل"""
        try:
            with open(self.graph_data_path, 'rb') as f:
                self.G = pickle.load(f)
            print(f"✅ گراف از فایل بارگذاری شد: {self.G.number_of_nodes()} نود، {self.G.number_of_edges()} یال")
        except Exception as e:
            print(f"❌ خطا در بارگذاری گراف: {e}")
            self.create_sample_graph()
    
    def extract_keywords(self, text: str) -> List[str]:
        """استخراج کلمات کلیدی از متن"""
        doc = self.nlp(text)
        keywords = set()
        
        # موجودیت‌های نام‌دار
        for ent in doc.ents:
            if ent.label_ not in {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}:
                keywords.add(ent.text.lower())
        
        # اسم‌ها و اسم خاص‌ها
        for token in doc:
            if (token.pos_ in {"NOUN", "PROPN"} and 
                token.text.lower() not in STOP_WORDS and 
                token.is_alpha and len(token.text) > 2):
                keywords.add(token.text.lower())
        
        return sorted(keywords)
    
    def analyze_question_intent(self, query: str) -> Dict[str, Any]:
        """تحلیل عمیق سوال و استخراج قصد کاربر"""
        query_lower = query.lower()
        
        # تشخیص نوع سوال
        question_type = self._analyze_question_type(query_lower)
        
        # استخراج موجودیت‌های اصلی
        main_entities = []
        entity_types = []
        
        # کلمات کلیدی مرتبط با انواع موجودیت‌ها
        entity_keywords = {
            'Gene': ['ژن', 'gene', 'protein', 'پروتئین', 'dna', 'rna', 'mrna'],
            'Disease': ['بیماری', 'disease', 'disorder', 'syndrome', 'cancer', 'سرطان', 'diabetes', 'دیابت'],
            'Drug': ['دارو', 'drug', 'medicine', 'medication', 'treatment', 'درمان'],
            'Anatomy': ['قلب', 'heart', 'brain', 'مغز', 'liver', 'کبد', 'lung', 'ریه', 'kidney', 'کلیه'],
            'Biological_Process': ['process', 'فرآیند', 'pathway', 'مسیر', 'metabolism', 'متابولیسم'],
            'Compound': ['compound', 'ترکیب', 'chemical', 'شیمیایی', 'molecule', 'مولکول']
        }
        
        # تشخیص موجودیت‌های اصلی
        for entity_type, keywords in entity_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    main_entities.append(keyword)
                    entity_types.append(entity_type)
                    break
        
        # تشخیص روابط
        relationships = []
        relationship_keywords = {
            'interacts_with': ['تعامل', 'interact', 'interaction', 'تعامل می‌کند'],
            'associates': ['مرتبط', 'associate', 'association', 'ارتباط'],
            'treats': ['درمان', 'treat', 'treatment', 'cure', 'شفا'],
            'causes': ['سبب', 'cause', 'causation', 'علت'],
            'expressed_in': ['بیان', 'express', 'expression', 'بیان می‌شود'],
            'regulates': ['تنظیم', 'regulate', 'regulation', 'کنترل']
        }
        
        for rel_type, keywords in relationship_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    relationships.append(rel_type)
                    break
        
        # تشخیص سوال‌های خاص
        question_patterns = {
            'what_genes': ['چه ژن', 'what gene', 'which gene'],
            'what_diseases': ['چه بیماری', 'what disease', 'which disease'],
            'what_drugs': ['چه دارو', 'what drug', 'which drug'],
            'how_treat': ['چگونه درمان', 'how treat', 'how cure'],
            'what_causes': ['چه سبب', 'what cause', 'what causes'],
            'where_expressed': ['کجا بیان', 'where express', 'where expressed']
        }
        
        detected_patterns = []
        for pattern_name, patterns in question_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    detected_patterns.append(pattern_name)
                    break
        
        return {
            'question_type': question_type,
            'main_entities': main_entities,
            'entity_types': entity_types,
            'relationships': relationships,
            'patterns': detected_patterns,
            'keywords': self.extract_keywords(query)
        }
    
    def intelligent_semantic_search(self, query: str, max_depth: int = 3) -> List[Tuple[str, int, float, str]]:
        """جستجوی معنایی هوشمند بر اساس تحلیل سوال"""
        if not self.G:
            return []
        
        # تحلیل سوال
        intent = self.analyze_question_intent(query)
        print(f"تحلیل سوال: {intent}")
        
        # استخراج کلمات کلیدی
        keywords = intent['keywords']
        main_entities = intent['main_entities']
        
        # تطبیق با نودهای گراف
        matched_nodes = self.match_tokens_to_nodes(keywords + main_entities)
        
        if not matched_nodes:
            print("هیچ نودی تطبیق نیافت")
            return []
        
        print(f"نودهای تطبیق یافته: {matched_nodes}")
        
        # جستجوی هوشمند بر اساس نوع سوال
        results = []
        
        if intent['question_type'] == 'anatomy_expression':
            # سوالات مربوط به بیان ژن در آناتومی
            results = self._search_anatomy_expression(matched_nodes, intent, max_depth)
        elif intent['question_type'] == 'disease_info':
            # سوالات مربوط به بیماری‌ها
            results = self._search_disease_related(matched_nodes, intent, max_depth)
        elif intent['question_type'] == 'drug_treatment':
            # سوالات مربوط به داروها - بررسی اینکه آیا بیماری در matched_nodes وجود دارد
            if any(self.G.nodes[node_id]['kind'] == 'Disease' for node_id in matched_nodes.values()):
                # اگر بیماری وجود دارد، داروهای درمانی آن را جستجو کن
                results = self._search_disease_related(matched_nodes, intent, max_depth)
            else:
                # اگر دارو وجود دارد، بیماری‌های درمان شده توسط آن را جستجو کن
                results = self._search_drug_related(matched_nodes, intent, max_depth)
        elif intent['question_type'] == 'gene_function':
            # سوالات مربوط به عملکرد ژن‌ها
            results = self._search_gene_function(matched_nodes, intent, max_depth)
        else:
            # جستجوی عمومی
            results = self._search_general(matched_nodes, intent, max_depth)
        
        return results
    
    def _search_anatomy_expression(self, matched_nodes: Dict[str, str], intent: Dict, max_depth: int) -> List[Tuple[str, int, float, str]]:
        """جستجوی بیان ژن در آناتومی"""
        results = []
        
        for token, node_id in matched_nodes.items():
            if self.G.nodes[node_id]['kind'] == 'Anatomy':
                # یافتن ژن‌هایی که در این آناتومی بیان می‌شوند
                for neighbor in self.G.neighbors(node_id):
                    if self.G.nodes[neighbor]['kind'] == 'Gene':
                        # بررسی رابطه بیان
                        edge_data = self.G.get_edge_data(node_id, neighbor)
                        if edge_data and 'expressed_in' in edge_data.get('metaedge', ''):
                            results.append((neighbor, 1, 5.0, f"بیان در {self.G.nodes[node_id]['name']}"))
                
                # جستجوی عمیق‌تر
                for depth in range(2, max_depth + 1):
                    for path in nx.single_source_shortest_path(self.G, node_id, cutoff=depth).values():
                        if len(path) == depth + 1:
                            target_node = path[-1]
                            if self.G.nodes[target_node]['kind'] == 'Gene':
                                score = 5.0 / depth
                                results.append((target_node, depth, score, f"مسیر {depth} سطحی"))
        
        return results
    
    def _search_disease_related(self, matched_nodes: Dict[str, str], intent: Dict, max_depth: int) -> List[Tuple[str, int, float, str]]:
        """جستجوی مرتبط با بیماری‌ها"""
        results = []
        
        for token, node_id in matched_nodes.items():
            if self.G.nodes[node_id]['kind'] == 'Disease':
                # یافتن ژن‌های مرتبط با بیماری
                for neighbor in self.G.neighbors(node_id):
                    if self.G.nodes[neighbor]['kind'] == 'Gene':
                        edge_data = self.G.get_edge_data(node_id, neighbor)
                        if edge_data:
                            results.append((neighbor, 1, 5.0, f"مرتبط با {self.G.nodes[node_id]['name']}"))
                
                # یافتن داروهای درمانی
                for neighbor in self.G.neighbors(node_id):
                    if self.G.nodes[neighbor]['kind'] == 'Drug':
                        edge_data = self.G.get_edge_data(node_id, neighbor)
                        if edge_data and 'treats' in edge_data.get('metaedge', ''):
                            results.append((neighbor, 1, 4.5, f"درمان {self.G.nodes[node_id]['name']}"))
                
                # جستجوی عمیق‌تر
                for depth in range(2, max_depth + 1):
                    for path in nx.single_source_shortest_path(self.G, node_id, cutoff=depth).values():
                        if len(path) == depth + 1:
                            target_node = path[-1]
                            score = 4.0 / depth
                            results.append((target_node, depth, score, f"مسیر {depth} سطحی"))
        
        return results
    
    def _search_drug_related(self, matched_nodes: Dict[str, str], intent: Dict, max_depth: int) -> List[Tuple[str, int, float, str]]:
        """جستجوی مرتبط با داروها"""
        results = []
        
        for token, node_id in matched_nodes.items():
            if self.G.nodes[node_id]['kind'] == 'Drug':
                # یافتن بیماری‌هایی که این دارو درمان می‌کند
                for neighbor in self.G.neighbors(node_id):
                    if self.G.nodes[neighbor]['kind'] == 'Disease':
                        edge_data = self.G.get_edge_data(node_id, neighbor)
                        if edge_data and 'treats' in edge_data.get('metaedge', ''):
                            results.append((neighbor, 1, 5.0, f"درمان شده توسط {self.G.nodes[node_id]['name']}"))
                
                # یافتن ژن‌های هدف
                for neighbor in self.G.neighbors(node_id):
                    if self.G.nodes[neighbor]['kind'] == 'Gene':
                        edge_data = self.G.get_edge_data(node_id, neighbor)
                        if edge_data:
                            results.append((neighbor, 1, 4.5, f"هدف {self.G.nodes[node_id]['name']}"))
        
        return results
    
    def _search_gene_function(self, matched_nodes: Dict[str, str], intent: Dict, max_depth: int) -> List[Tuple[str, int, float, str]]:
        """جستجوی عملکرد ژن‌ها"""
        results = []
        
        for token, node_id in matched_nodes.items():
            if self.G.nodes[node_id]['kind'] == 'Gene':
                # یافتن فرآیندهای زیستی مرتبط
                for neighbor in self.G.neighbors(node_id):
                    if self.G.nodes[neighbor]['kind'] == 'Biological_Process':
                        edge_data = self.G.get_edge_data(node_id, neighbor)
                        if edge_data:
                            results.append((neighbor, 1, 4.5, f"فرآیند مرتبط با {self.G.nodes[node_id]['name']}"))
                
                # یافتن ژن‌های تعاملی
                for neighbor in self.G.neighbors(node_id):
                    if self.G.nodes[neighbor]['kind'] == 'Gene':
                        edge_data = self.G.get_edge_data(node_id, neighbor)
                        if edge_data and 'interacts_with' in edge_data.get('metaedge', ''):
                            results.append((neighbor, 1, 4.0, f"تعامل با {self.G.nodes[node_id]['name']}"))
        
        return results
    
    def _search_general(self, matched_nodes: Dict[str, str], intent: Dict, max_depth: int) -> List[Tuple[str, int, float, str]]:
        """جستجوی عمومی"""
        results = []
        
        for token, node_id in matched_nodes.items():
            # جستجوی همسایه‌ها
            for neighbor in self.G.neighbors(node_id):
                score = 4.0
                results.append((neighbor, 1, score, f"همسایه {self.G.nodes[node_id]['name']}"))
            
            # جستجوی عمیق‌تر
            for depth in range(2, max_depth + 1):
                for path in nx.single_source_shortest_path(self.G, node_id, cutoff=depth).values():
                    if len(path) == depth + 1:
                        target_node = path[-1]
                        score = 3.0 / depth
                        results.append((target_node, depth, score, f"مسیر {depth} سطحی"))
        
        return results
    
    def match_tokens_to_nodes(self, tokens: List[str]) -> Dict[str, str]:
        """تطبیق توکن‌ها با نودهای گراف"""
        matched = {}
        for token in tokens:
            token_lower = token.lower()
            for node_id, attrs in self.G.nodes(data=True):
                if token_lower in attrs['name'].lower():
                    matched[token] = node_id
                    break
        return matched
    
    def bfs_search(self, start_node: str, max_depth: int = 2) -> List[Tuple[str, int]]:
        """جستجوی سطح اول"""
        visited = set()
        queue = deque([(start_node, 0)])
        result = []
        
        while queue:
            node, depth = queue.popleft()
            if node in visited or depth > max_depth:
                continue
            visited.add(node)
            result.append((node, depth))
            for neighbor in self.G.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return result
    
    def dfs_search(self, start_node: str, max_depth: int = 2) -> List[Tuple[str, int]]:
        """جستجوی عمیق اول"""
        visited = set()
        result = []
        
        def dfs(node, depth):
            if depth > max_depth or node in visited:
                return
            visited.add(node)
            result.append((node, depth))
            for neighbor in self.G.neighbors(node):
                if neighbor not in visited:
                    dfs(neighbor, depth + 1)
        
        dfs(start_node, 0)
        return result
    
    def get_shortest_paths(self, source: str, target: str, max_paths: int = 3) -> List[List[str]]:
        """یافتن کوتاه‌ترین مسیرها"""
        try:
            paths = list(nx.all_shortest_paths(self.G, source=source, target=target))
            return paths[:max_paths]
        except nx.NetworkXNoPath:
            return []
    
    def get_neighbors_by_type(self, node_id: str, kind_filter: str = None) -> List[Tuple[str, str]]:
        """دریافت همسایه‌ها بر اساس نوع"""
        neighbors = []
        for neighbor in self.G.neighbors(node_id):
            kind = self.G.nodes[neighbor].get('kind')
            if kind_filter is None or kind == kind_filter:
                neighbors.append((neighbor, self.G.nodes[neighbor]['name']))
        return neighbors
    
    def hybrid_search(self, nodes: List[str], max_depth: int = 2) -> List[Tuple[str, int]]:
        """جستجوی ترکیبی"""
        all_results = []
        for node in nodes:
            bfs_result = self.bfs_search(node, max_depth)
            all_results.extend(bfs_result)
        
        # حذف تکراری‌ها و مرتب‌سازی بر اساس عمق
        unique_results = {}
        for node, depth in all_results:
            if node not in unique_results or depth < unique_results[node]:
                unique_results[node] = depth
        
        return sorted(unique_results.items(), key=lambda x: x[1])
    
    def multi_method_search(self, nodes: List[str], max_depth: int = 2) -> List[Tuple[str, int, str]]:
        """جستجوی چند روشی - ترکیب BFS، DFS، و همسایه‌ها"""
        all_results = []
        
        for node in nodes:
            # BFS
            bfs_result = self.bfs_search(node, max_depth)
            for n, depth in bfs_result:
                all_results.append((n, depth, 'BFS'))
            
            # DFS
            dfs_result = self.dfs_search(node, max_depth)
            for n, depth in dfs_result:
                all_results.append((n, depth, 'DFS'))
            
            # همسایه‌های مستقیم
            neighbors = self.get_neighbors_by_type(node)
            for nid, name in neighbors:
                all_results.append((nid, 1, 'Neighbors'))
        
        # حذف تکراری‌ها و امتیازدهی
        unique_results = {}
        for node, depth, method in all_results:
            if node not in unique_results:
                unique_results[node] = {'depth': depth, 'methods': [method], 'score': 1.0}
            else:
                unique_results[node]['methods'].append(method)
                unique_results[node]['score'] += 0.5  # امتیاز بیشتر برای روش‌های مختلف
                if depth < unique_results[node]['depth']:
                    unique_results[node]['depth'] = depth
        
        # مرتب‌سازی بر اساس امتیاز و عمق
        sorted_results = []
        for node, info in unique_results.items():
            sorted_results.append((node, info['depth'], ', '.join(info['methods'])))
        
        return sorted(sorted_results, key=lambda x: (x[1], -len(x[2].split(', '))))
    
    def ensemble_search(self, nodes: List[str], max_depth: int = 2) -> List[Tuple[str, int, float]]:
        """جستجوی گروهی - ترکیب همه روش‌ها با وزن‌دهی"""
        method_weights = {
            'BFS': 1.0,
            'DFS': 0.8,
            'SHORTEST_PATH': 1.2,
            'NEIGHBORS': 0.9
        }
        
        all_results = {}
        
        for node in nodes:
            # BFS
            bfs_result = self.bfs_search(node, max_depth)
            for n, depth in bfs_result:
                if n not in all_results:
                    all_results[n] = {'score': 0, 'depth': depth, 'count': 0}
                all_results[n]['score'] += method_weights['BFS'] / (depth + 1)
                all_results[n]['count'] += 1
            
            # DFS
            dfs_result = self.dfs_search(node, max_depth)
            for n, depth in dfs_result:
                if n not in all_results:
                    all_results[n] = {'score': 0, 'depth': depth, 'count': 0}
                all_results[n]['score'] += method_weights['DFS'] / (depth + 1)
                all_results[n]['count'] += 1
            
            # همسایه‌ها
            neighbors = self.get_neighbors_by_type(node)
            for nid, name in neighbors:
                if nid not in all_results:
                    all_results[nid] = {'score': 0, 'depth': 1, 'count': 0}
                all_results[nid]['score'] += method_weights['NEIGHBORS']
                all_results[nid]['count'] += 1
        
        # مرتب‌سازی بر اساس امتیاز
        sorted_results = []
        for node, info in all_results.items():
            final_score = info['score'] * (1 + 0.1 * info['count'])  # پاداش برای تکرار
            sorted_results.append((node, info['depth'], final_score))
        
        return sorted(sorted_results, key=lambda x: x[2], reverse=True)
    
    def adaptive_search(self, nodes: List[str], max_depth: int = 2) -> List[Tuple[str, int, str]]:
        """جستجوی تطبیقی - انتخاب روش بر اساس نوع نود"""
        all_results = []
        
        for node in nodes:
            node_kind = self.G.nodes[node]['kind']
            
            # انتخاب روش بر اساس نوع نود
            if node_kind in ['Gene', 'Disease']:
                # برای ژن‌ها و بیماری‌ها از BFS و همسایه‌ها
                bfs_result = self.bfs_search(node, max_depth)
                for n, depth in bfs_result:
                    all_results.append((n, depth, 'BFS'))
                
                neighbors = self.get_neighbors_by_type(node)
                for nid, name in neighbors:
                    all_results.append((nid, 1, 'Neighbors'))
            
            elif node_kind in ['Drug', 'Compound']:
                # برای داروها از DFS و کوتاه‌ترین مسیر
                dfs_result = self.dfs_search(node, max_depth)
                for n, depth in dfs_result:
                    all_results.append((n, depth, 'DFS'))
            
            elif node_kind in ['Anatomy', 'Biological Process']:
                # برای آناتومی و فرآیندهای زیستی از همه روش‌ها
                bfs_result = self.bfs_search(node, max_depth)
                for n, depth in bfs_result:
                    all_results.append((n, depth, 'BFS'))
                
                dfs_result = self.dfs_search(node, max_depth)
                for n, depth in dfs_result:
                    all_results.append((n, depth, 'DFS'))
            
            else:
                # برای بقیه از روش ترکیبی
                hybrid_result = self.hybrid_search([node], max_depth)
                for n, depth in hybrid_result:
                    all_results.append((n, depth, 'Hybrid'))
        
        # حذف تکراری‌ها
        unique_results = {}
        for node, depth, method in all_results:
            if node not in unique_results:
                unique_results[node] = (depth, method)
            elif depth < unique_results[node][0]:
                unique_results[node] = (depth, method)
        
        return [(node, depth, method) for node, (depth, method) in unique_results.items()]
    
    def retrieve_information(self, query: str, method: RetrievalMethod, 
                           max_depth: int = 2, max_nodes: int = 10) -> RetrievalResult:
        """بازیابی اطلاعات از گراف"""
        print(f"🔍 بازیابی اطلاعات با روش {method.value if hasattr(method, 'value') else method}...")
        
        # استخراج کلمات کلیدی
        keywords = self.extract_keywords(query)
        print(f"کلمات کلیدی: {keywords}")
        
        # تطبیق با نودهای گراف
        matches = self.match_tokens_to_nodes(keywords)
        print(f"تطبیق‌های یافت شده: {matches}")
        
        nodes = []
        edges = []
        paths = []
        
        if method == RetrievalMethod.BFS:
            # BFS برای هر نود تطبیق یافته
            for token, node_id in matches.items():
                bfs_result = self.bfs_search(node_id, max_depth)
                for node, depth in bfs_result[:max_nodes]:
                    nodes.append(GraphNode(
                        id=node,
                        name=self.G.nodes[node]['name'],
                        kind=self.G.nodes[node]['kind'],
                        depth=depth
                    ))
        
        elif method == RetrievalMethod.DFS:
            # DFS برای هر نود تطبیق یافته
            for token, node_id in matches.items():
                dfs_result = self.dfs_search(node_id, max_depth)
                for node, depth in dfs_result[:max_nodes]:
                    nodes.append(GraphNode(
                        id=node,
                        name=self.G.nodes[node]['name'],
                        kind=self.G.nodes[node]['kind'],
                        depth=depth
                    ))
        
        elif method == RetrievalMethod.SHORTEST_PATH:
            # کوتاه‌ترین مسیر بین نودها
            if len(matches) >= 2:
                node_ids = list(matches.values())
                for i in range(len(node_ids)):
                    for j in range(i+1, len(node_ids)):
                        paths.extend(self.get_shortest_paths(node_ids[i], node_ids[j]))
                        
                        # اضافه کردن نودهای مسیر
                        for path in paths:
                            for k, node in enumerate(path):
                                nodes.append(GraphNode(
                                    id=node,
                                    name=self.G.nodes[node]['name'],
                                    kind=self.G.nodes[node]['kind'],
                                    depth=k
                                ))
        
        elif method == RetrievalMethod.NEIGHBORS:
            # همسایه‌های مستقیم
            for token, node_id in matches.items():
                neighbors = self.get_neighbors_by_type(node_id)
                for nid, name in neighbors[:max_nodes]:
                    nodes.append(GraphNode(
                        id=nid,
                        name=name,
                        kind=self.G.nodes[nid]['kind'],
                        depth=1
                    ))
        
        elif method == RetrievalMethod.HYBRID:
            # ترکیبی از روش‌ها
            if len(matches) >= 2:
                node_ids = list(matches.values())
                hybrid_result = self.hybrid_search(node_ids, max_depth)
                for node, depth in hybrid_result[:max_nodes]:
                    nodes.append(GraphNode(
                        id=node,
                        name=self.G.nodes[node]['name'],
                        kind=self.G.nodes[node]['kind'],
                        depth=depth
                    ))
        
        elif method == RetrievalMethod.MULTI_METHOD:
            # جستجوی چند روشی
            node_ids = list(matches.values())
            multi_result = self.multi_method_search(node_ids, max_depth)
            for node, depth, methods in multi_result[:max_nodes]:
                nodes.append(GraphNode(
                    id=node,
                    name=self.G.nodes[node]['name'],
                    kind=self.G.nodes[node]['kind'],
                    depth=depth,
                    score=len(methods.split(', '))  # امتیاز بر اساس تعداد روش‌ها
                ))
        
        elif method == RetrievalMethod.ENSEMBLE:
            # جستجوی گروهی
            node_ids = list(matches.values())
            ensemble_result = self.ensemble_search(node_ids, max_depth)
            for node, depth, score in ensemble_result[:max_nodes]:
                nodes.append(GraphNode(
                    id=node,
                    name=self.G.nodes[node]['name'],
                    kind=self.G.nodes[node]['kind'],
                    depth=depth,
                    score=score
                ))
        
        elif method == RetrievalMethod.ADAPTIVE:
            # جستجوی تطبیقی
            node_ids = list(matches.values())
            adaptive_result = self.adaptive_search(node_ids, max_depth)
            for node, depth, method in adaptive_result[:max_nodes]:
                nodes.append(GraphNode(
                    id=node,
                    name=self.G.nodes[node]['name'],
                    kind=self.G.nodes[node]['kind'],
                    depth=depth
                ))
        
        elif method == RetrievalMethod.INTELLIGENT:
            # جستجوی معنایی هوشمند
            intelligent_result = self.intelligent_semantic_search(query, max_depth)
            
            # تبدیل نتایج به GraphNode
            for node_id, depth, score, reason in intelligent_result[:max_nodes]:
                nodes.append(GraphNode(
                    id=node_id,
                    name=self.G.nodes[node_id]['name'],
                    kind=self.G.nodes[node_id]['kind'],
                    depth=depth,
                    score=score
                ))
            
            # یافتن مسیرهای ارتباطی بین نودها
            if len(nodes) >= 2:
                node_ids = [node.id for node in nodes]
                for i in range(len(node_ids)):
                    for j in range(i+1, len(node_ids)):
                        paths.extend(self.get_shortest_paths(node_ids[i], node_ids[j]))
            
            # یافتن یال‌های مرتبط
            for node in nodes:
                for neighbor in self.G.neighbors(node.id):
                    if any(n.id == neighbor for n in nodes):
                        edge_data = self.G.get_edge_data(node.id, neighbor)
                        if edge_data:
                            edges.append(GraphEdge(
                                source=node.id,
                                target=neighbor,
                                relation=edge_data.get('metaedge', 'related'),
                                weight=edge_data.get('weight', 1.0)
                            ))
        
        elif method == RetrievalMethod.NO_RETRIEVAL:
            # بدون بازیابی - فقط مدل
            print("🔍 بدون بازیابی از گراف - فقط استفاده از مدل")
            # ایجاد یک نود خالی برای حفظ ساختار
            nodes.append(GraphNode(
                id="no_retrieval",
                name="بدون بازیابی",
                kind="System",
                depth=0
            ))
        
        # حذف تکراری‌ها
        unique_nodes = {}
        for node in nodes:
            if node.id not in unique_nodes:
                unique_nodes[node.id] = node
        
        nodes = list(unique_nodes.values())
        
        # ایجاد یال‌ها بین نودهای مرتبط
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                if self.G.has_edge(node1.id, node2.id):
                    edge_data = self.G.get_edge_data(node1.id, node2.id)
                    edges.append(GraphEdge(
                        source=node1.id,
                        target=node2.id,
                        relation=edge_data['metaedge']
                    ))
        
        # ایجاد متن زمینه
        context_text = self.create_context_text(nodes, edges, paths)
        
        return RetrievalResult(
            nodes=nodes,
            edges=edges,
            paths=paths,
            context_text=context_text,
            method=method.value if hasattr(method, 'value') else str(method),
            query=query
        )
    
    def create_context_text(self, nodes: List[GraphNode], edges: List[GraphEdge], 
                           paths: List[List[str]]) -> str:
        """ایجاد متن زمینه بهبود یافته از نتایج بازیابی"""
        context_parts = []
        
        # دسته‌بندی نودها بر اساس نوع
        nodes_by_type = {}
        for node in nodes:
            if node.kind not in nodes_by_type:
                nodes_by_type[node.kind] = []
            nodes_by_type[node.kind].append(node)
        
        # اطلاعات نودها به صورت دسته‌بندی شده
        context_parts.append("📊 ENTITIES FOUND IN KNOWLEDGE GRAPH:")
        context_parts.append("=" * 50)
        
        for kind, kind_nodes in nodes_by_type.items():
            context_parts.append(f"\n🔹 {kind.upper()} ({len(kind_nodes)} entities):")
            for node in kind_nodes:
                score_info = f" [Score: {node.score:.2f}]" if hasattr(node, 'score') and node.score != 1.0 else ""
                depth_info = f" [Depth: {node.depth}]" if node.depth > 0 else ""
                context_parts.append(f"  • {node.name}{score_info}{depth_info}")
        
        # روابط با جزئیات بیشتر
        if edges:
            context_parts.append(f"\n🔗 RELATIONSHIPS ({len(edges)} connections):")
            context_parts.append("=" * 50)
            
            # دسته‌بندی روابط بر اساس نوع
            relations_by_type = {}
            for edge in edges:
                if edge.relation not in relations_by_type:
                    relations_by_type[edge.relation] = []
                relations_by_type[edge.relation].append(edge)
            
            for relation, relation_edges in relations_by_type.items():
                context_parts.append(f"\n📌 {relation.upper()} ({len(relation_edges)} connections):")
                for edge in relation_edges[:10]:  # حداکثر 10 رابطه از هر نوع
                    source_name = next(n.name for n in nodes if n.id == edge.source)
                    target_name = next(n.name for n in nodes if n.id == edge.target)
                    source_kind = next(n.kind for n in nodes if n.id == edge.source)
                    target_kind = next(n.kind for n in nodes if n.id == edge.target)
                    context_parts.append(f"  • {source_name} ({source_kind}) → {target_name} ({target_kind})")
        
        # مسیرهای مهم
        if paths:
            context_parts.append(f"\n🛤️ IMPORTANT PATHS ({len(paths)} paths):")
            context_parts.append("=" * 50)
            for i, path in enumerate(paths[:5], 1):  # حداکثر 5 مسیر
                path_names = [self.G.nodes[node]['name'] for node in path]
                path_kinds = [self.G.nodes[node]['kind'] for node in path]
                context_parts.append(f"\nPath {i} ({len(path)} steps):")
                for j, (name, kind) in enumerate(zip(path_names, path_kinds)):
                    context_parts.append(f"  {j+1}. {name} ({kind})")
        
        # آمار کلی
        context_parts.append(f"\n📈 SUMMARY:")
        context_parts.append("=" * 50)
        context_parts.append(f"• Total entities: {len(nodes)}")
        context_parts.append(f"• Total relationships: {len(edges)}")
        context_parts.append(f"• Entity types: {len(nodes_by_type)}")
        context_parts.append(f"• Relationship types: {len(set(e.relation for e in edges))}")
        if paths:
            context_parts.append(f"• Important paths: {len(paths)}")
        
        # راهنمای تفسیر
        context_parts.append(f"\n💡 INTERPRETATION GUIDE:")
        context_parts.append("=" * 50)
        context_parts.append("• Genes often participate in biological processes")
        context_parts.append("• Drugs can treat diseases and interact with genes")
        context_parts.append("• Diseases are associated with specific genes and symptoms")
        context_parts.append("• Anatomy expresses genes and can be affected by diseases")
        context_parts.append("• Compounds can interact with genes and biological processes")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, retrieval_result: RetrievalResult, 
                       model: GenerationModel) -> GenerationResult:
        """تولید پاسخ بر اساس نتایج بازیابی"""
        print(f"🤖 تولید پاسخ با مدل {model.value}...")
        
        if model == GenerationModel.SIMPLE:
            answer = self.simple_template_generation(retrieval_result)
            confidence = 0.7
        elif model == GenerationModel.GPT_SIMULATION:
            answer = self.gpt_simulation_generation(retrieval_result)
            confidence = 0.85
        elif model == GenerationModel.CUSTOM:
            answer = self.custom_generation(retrieval_result)
            confidence = 0.9
        elif model == GenerationModel.HUGGINGFACE:
            answer = self.huggingface_generation(retrieval_result)
            confidence = 0.92
        elif model == GenerationModel.OPENAI_GPT:
            answer = self.openai_gpt_generation(retrieval_result)
            confidence = 0.95
        elif model == GenerationModel.ANTHROPIC_CLAUDE:
            answer = self.anthropic_claude_generation(retrieval_result)
            confidence = 0.94
        elif model == GenerationModel.GOOGLE_GEMINI:
            answer = self.google_gemini_generation(retrieval_result)
            confidence = 0.93
        else:
            answer = "متأسفانه مدل انتخاب شده در دسترس نیست."
            confidence = 0.0
        
        return GenerationResult(
            answer=answer,
            model=model.value,
            context_used=retrieval_result.context_text,
            confidence=confidence
        )
    
    def simple_template_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ ساده با قالب بهبود یافته"""
        query_lower = retrieval_result.query.lower()
        
        # تحلیل نوع سوال
        question_type = self._analyze_question_type(query_lower)
        
        if question_type == "relationship":
            return self._generate_relationship_answer(retrieval_result)
        elif question_type == "drug_treatment":
            return self._generate_drug_treatment_answer(retrieval_result)
        elif question_type == "gene_function":
            return self._generate_gene_function_answer(retrieval_result)
        elif question_type == "disease_info":
            return self._generate_disease_info_answer(retrieval_result)
        elif question_type == "anatomy_expression":
            return self._generate_anatomy_expression_answer(retrieval_result)
        else:
            return self._generate_general_answer(retrieval_result)
    
    def _analyze_question_type(self, query_lower: str) -> str:
        """تحلیل نوع سوال"""
        if any(word in query_lower for word in ["relationship", "relation", "connect", "link"]):
            return "relationship"
        elif any(word in query_lower for word in ["drug", "treat", "medicine", "therapy"]):
            return "drug_treatment"
        elif any(word in query_lower for word in ["gene", "function", "regulate", "express"]):
            return "gene_function"
        elif any(word in query_lower for word in ["disease", "disorder", "condition", "symptom"]):
            return "disease_info"
        elif any(word in query_lower for word in ["anatomy", "organ", "tissue", "heart", "brain", "liver"]):
            return "anatomy_expression"
        else:
            return "general"
    
    def _generate_relationship_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات رابطه"""
        if not retrieval_result.edges:
            return "❌ No direct relationships found between the specified entities in the knowledge graph."
        
        answer_parts = ["🔗 RELATIONSHIPS FOUND:"]
        
        # دسته‌بندی روابط
        relations_by_type = {}
        for edge in retrieval_result.edges:
            if edge.relation not in relations_by_type:
                relations_by_type[edge.relation] = []
            relations_by_type[edge.relation].append(edge)
        
        for relation, edges in relations_by_type.items():
            answer_parts.append(f"\n📌 {relation.upper()} ({len(edges)} connections):")
            for edge in edges[:5]:  # حداکثر 5 رابطه از هر نوع
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"  • {source_name} → {target_name}")
        
        return "\n".join(answer_parts)
    
    def _generate_drug_treatment_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات درمان دارویی"""
        drug_nodes = [n for n in retrieval_result.nodes if n.kind in ['Drug', 'Compound']]
        disease_nodes = [n for n in retrieval_result.nodes if n.kind == 'Disease']
        
        if not drug_nodes and not disease_nodes:
            return "❌ No drug or disease information found in the retrieved context."
        
        answer_parts = ["💊 DRUG-DISEASE RELATIONSHIPS:"]
        
        # روابط درمان
        treatment_edges = [e for e in retrieval_result.edges if 'treat' in e.relation.lower() or 'therapy' in e.relation.lower()]
        if treatment_edges:
            answer_parts.append(f"\n✅ TREATMENT RELATIONSHIPS ({len(treatment_edges)} found):")
            for edge in treatment_edges[:10]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"  • {source_name} treats {target_name}")
        
        # داروهای یافت شده
        if drug_nodes:
            answer_parts.append(f"\n💊 DRUGS FOUND ({len(drug_nodes)}):")
            for drug in drug_nodes[:10]:
                answer_parts.append(f"  • {drug.name}")
        
        # بیماری‌های یافت شده
        if disease_nodes:
            answer_parts.append(f"\n🏥 DISEASES FOUND ({len(disease_nodes)}):")
            for disease in disease_nodes[:10]:
                answer_parts.append(f"  • {disease.name}")
        
        return "\n".join(answer_parts)
    
    def _generate_gene_function_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات عملکرد ژن"""
        gene_nodes = [n for n in retrieval_result.nodes if n.kind == 'Gene']
        process_nodes = [n for n in retrieval_result.nodes if n.kind == 'Biological Process']
        
        if not gene_nodes:
            return "❌ No gene information found in the retrieved context."
        
        answer_parts = ["🧬 GENE FUNCTION ANALYSIS:"]
        
        # ژن‌های یافت شده
        answer_parts.append(f"\n🧬 GENES FOUND ({len(gene_nodes)}):")
        for gene in gene_nodes[:10]:
            answer_parts.append(f"  • {gene.name}")
        
        # فرآیندهای زیستی مرتبط
        if process_nodes:
            answer_parts.append(f"\n⚙️ BIOLOGICAL PROCESSES ({len(process_nodes)}):")
            for process in process_nodes[:10]:
                answer_parts.append(f"  • {process.name}")
        
        # روابط ژن-فرآیند
        gene_process_edges = [e for e in retrieval_result.edges if 'participate' in e.relation.lower() or 'regulate' in e.relation.lower()]
        if gene_process_edges:
            answer_parts.append(f"\n🔗 GENE-PROCESS RELATIONSHIPS ({len(gene_process_edges)}):")
            for edge in gene_process_edges[:10]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"  • {source_name} → {target_name}")
        
        return "\n".join(answer_parts)
    
    def _generate_disease_info_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات اطلاعات بیماری"""
        disease_nodes = [n for n in retrieval_result.nodes if n.kind == 'Disease']
        gene_nodes = [n for n in retrieval_result.nodes if n.kind == 'Gene']
        symptom_nodes = [n for n in retrieval_result.nodes if n.kind == 'Symptom']
        
        if not disease_nodes:
            return "❌ No disease information found in the retrieved context."
        
        answer_parts = ["🏥 DISEASE ANALYSIS:"]
        
        # بیماری‌های یافت شده
        answer_parts.append(f"\n🏥 DISEASES FOUND ({len(disease_nodes)}):")
        for disease in disease_nodes[:10]:
            answer_parts.append(f"  • {disease.name}")
        
        # ژن‌های مرتبط
        if gene_nodes:
            answer_parts.append(f"\n🧬 ASSOCIATED GENES ({len(gene_nodes)}):")
            for gene in gene_nodes[:10]:
                answer_parts.append(f"  • {gene.name}")
        
        # علائم مرتبط
        if symptom_nodes:
            answer_parts.append(f"\n🤒 ASSOCIATED SYMPTOMS ({len(symptom_nodes)}):")
            for symptom in symptom_nodes[:10]:
                answer_parts.append(f"  • {symptom.name}")
        
        return "\n".join(answer_parts)
    
    def _generate_anatomy_expression_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات بیان ژن در آناتومی"""
        anatomy_nodes = [n for n in retrieval_result.nodes if n.kind == 'Anatomy']
        gene_nodes = [n for n in retrieval_result.nodes if n.kind == 'Gene']
        
        if not anatomy_nodes:
            return "❌ No anatomy information found in the retrieved context."
        
        answer_parts = ["🫀 ANATOMY-GENE EXPRESSION:"]
        
        # اندام‌های یافت شده
        answer_parts.append(f"\n🫀 ANATOMICAL STRUCTURES ({len(anatomy_nodes)}):")
        for anatomy in anatomy_nodes[:10]:
            answer_parts.append(f"  • {anatomy.name}")
        
        # ژن‌های بیان شده
        if gene_nodes:
            answer_parts.append(f"\n🧬 EXPRESSED GENES ({len(gene_nodes)}):")
            for gene in gene_nodes[:10]:
                answer_parts.append(f"  • {gene.name}")
        
        # روابط بیان
        expression_edges = [e for e in retrieval_result.edges if 'express' in e.relation.lower()]
        if expression_edges:
            answer_parts.append(f"\n🔗 EXPRESSION RELATIONSHIPS ({len(expression_edges)}):")
            for edge in expression_edges[:10]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"  • {source_name} expresses {target_name}")
        
        return "\n".join(answer_parts)
    
    def _generate_general_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ عمومی"""
        if not retrieval_result.nodes:
            return "❌ No relevant information found in the knowledge graph for your query."
        
        answer_parts = ["📊 GENERAL INFORMATION FOUND:"]
        
        # دسته‌بندی نودها
        nodes_by_type = {}
        for node in retrieval_result.nodes:
            if node.kind not in nodes_by_type:
                nodes_by_type[node.kind] = []
            nodes_by_type[node.kind].append(node)
        
        for kind, nodes in nodes_by_type.items():
            answer_parts.append(f"\n🔹 {kind.upper()} ({len(nodes)} entities):")
            for node in nodes[:5]:
                answer_parts.append(f"  • {node.name}")
        
        # روابط مهم
        if retrieval_result.edges:
            answer_parts.append(f"\n🔗 KEY RELATIONSHIPS ({len(retrieval_result.edges)}):")
            for edge in retrieval_result.edges[:10]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"  • {source_name} → {target_name} ({edge.relation})")
        
        return "\n".join(answer_parts)
    
    def gpt_simulation_generation(self, retrieval_result: RetrievalResult) -> str:
        """شبیه‌سازی تولید پاسخ GPT بهبود یافته"""
        query = retrieval_result.query
        query_lower = query.lower()
        
        # تحلیل نوع سوال
        question_type = self._analyze_question_type(query_lower)
        
        # استخراج اطلاعات مهم از نتایج بازیابی
        gene_nodes = [n for n in retrieval_result.nodes if n.kind == 'Gene']
        disease_nodes = [n for n in retrieval_result.nodes if n.kind == 'Disease']
        drug_nodes = [n for n in retrieval_result.nodes if n.kind in ['Drug', 'Compound']]
        anatomy_nodes = [n for n in retrieval_result.nodes if n.kind == 'Anatomy']
        process_nodes = [n for n in retrieval_result.nodes if n.kind == 'Biological Process']
        
        # تولید پاسخ بر اساس نوع سوال
        if question_type == "relationship":
            return self._generate_intelligent_relationship_answer(retrieval_result, gene_nodes, disease_nodes, drug_nodes)
        elif question_type == "drug_treatment":
            return self._generate_intelligent_drug_answer(retrieval_result, drug_nodes, disease_nodes)
        elif question_type == "gene_function":
            return self._generate_intelligent_gene_answer(retrieval_result, gene_nodes, process_nodes)
        elif question_type == "disease_info":
            return self._generate_intelligent_disease_answer(retrieval_result, disease_nodes, gene_nodes)
        elif question_type == "anatomy_expression":
            return self._generate_intelligent_anatomy_answer(retrieval_result, anatomy_nodes, gene_nodes)
        else:
            return self._generate_intelligent_general_answer(retrieval_result, gene_nodes, disease_nodes, drug_nodes, anatomy_nodes, process_nodes)
    
    def _generate_intelligent_relationship_answer(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes, drug_nodes) -> str:
        """تولید پاسخ هوشمند برای سوالات رابطه"""
        if not retrieval_result.edges:
            return "🔍 **تحلیل رابطه:**\n\nمتأسفانه رابطه مستقیمی بین موجودیت‌های مورد نظر در گراف دانش یافت نشد. این ممکن است به دلیل:\n• فاصله زیاد بین نودها در گراف\n• نیاز به جستجوی عمیق‌تر\n• عدم وجود رابطه مستقیم در داده‌های موجود"
        
        answer_parts = ["🔍 **تحلیل رابطه:**\n"]
        
        # تحلیل روابط مهم
        important_relations = {}
        for edge in retrieval_result.edges:
            if edge.relation not in important_relations:
                important_relations[edge.relation] = []
            important_relations[edge.relation].append(edge)
        
        # نمایش مهم‌ترین روابط
        answer_parts.append("**روابط مهم یافت شده:**\n")
        for relation, edges in sorted(important_relations.items(), key=lambda x: len(x[1]), reverse=True)[:3]:
            answer_parts.append(f"• **{relation}** ({len(edges)} رابطه):")
            for edge in edges[:3]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"  - {source_name} → {target_name}")
            answer_parts.append("")
        
        # تحلیل آماری
        answer_parts.append("**آمار کلی:**")
        answer_parts.append(f"• تعداد کل روابط: {len(retrieval_result.edges)}")
        answer_parts.append(f"• انواع روابط: {len(important_relations)}")
        answer_parts.append(f"• نودهای مرتبط: {len(retrieval_result.nodes)}")
        
        return "\n".join(answer_parts)
    
    def _generate_intelligent_drug_answer(self, retrieval_result: RetrievalResult, drug_nodes, disease_nodes) -> str:
        """تولید پاسخ هوشمند برای سوالات دارویی"""
        answer_parts = ["💊 **تحلیل دارویی:**\n"]
        
        if drug_nodes:
            answer_parts.append("**داروهای یافت شده:**")
            for drug in drug_nodes[:5]:
                score_info = f" (امتیاز: {drug.score:.2f})" if hasattr(drug, 'score') and drug.score != 1.0 else ""
                answer_parts.append(f"• {drug.name}{score_info}")
            answer_parts.append("")
        
        if disease_nodes:
            answer_parts.append("**بیماری‌های مرتبط:**")
            for disease in disease_nodes[:5]:
                answer_parts.append(f"• {disease.name}")
            answer_parts.append("")
        
        # روابط درمان
        treatment_edges = [e for e in retrieval_result.edges if 'treat' in e.relation.lower() or 'therapy' in e.relation.lower()]
        if treatment_edges:
            answer_parts.append("**روابط درمانی:**")
            for edge in treatment_edges[:5]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"• {source_name} درمان می‌کند: {target_name}")
        
        if not drug_nodes and not disease_nodes:
            answer_parts.append("❌ اطلاعات دارویی یا بیماری در نتایج یافت نشد.")
        
        return "\n".join(answer_parts)
    
    def _generate_intelligent_gene_answer(self, retrieval_result: RetrievalResult, gene_nodes, process_nodes) -> str:
        """تولید پاسخ هوشمند برای سوالات ژن"""
        answer_parts = ["🧬 **تحلیل ژنتیکی:**\n"]
        
        if gene_nodes:
            answer_parts.append("**ژن‌های مهم یافت شده:**")
            # مرتب‌سازی بر اساس امتیاز
            sorted_genes = sorted(gene_nodes, key=lambda x: getattr(x, 'score', 1.0), reverse=True)
            for gene in sorted_genes[:5]:
                score_info = f" (امتیاز: {gene.score:.2f})" if hasattr(gene, 'score') and gene.score != 1.0 else ""
                answer_parts.append(f"• {gene.name}{score_info}")
            answer_parts.append("")
        
        if process_nodes:
            answer_parts.append("**فرآیندهای زیستی مرتبط:**")
            for process in process_nodes[:5]:
                answer_parts.append(f"• {process.name}")
            answer_parts.append("")
        
        # روابط ژن-فرآیند
        gene_process_edges = [e for e in retrieval_result.edges if 'participate' in e.relation.lower() or 'regulate' in e.relation.lower()]
        if gene_process_edges:
            answer_parts.append("**روابط ژن-فرآیند:**")
            for edge in gene_process_edges[:5]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"• {source_name} → {target_name}")
        
        if not gene_nodes:
            answer_parts.append("❌ اطلاعات ژنتیکی در نتایج یافت نشد.")
        
        return "\n".join(answer_parts)
    
    def _generate_intelligent_disease_answer(self, retrieval_result: RetrievalResult, disease_nodes, gene_nodes) -> str:
        """تولید پاسخ هوشمند برای سوالات بیماری"""
        answer_parts = ["🏥 **تحلیل بیماری:**\n"]
        
        if disease_nodes:
            answer_parts.append("**بیماری‌های یافت شده:**")
            for disease in disease_nodes[:5]:
                answer_parts.append(f"• {disease.name}")
            answer_parts.append("")
        
        if gene_nodes:
            answer_parts.append("**ژن‌های مرتبط:**")
            for gene in gene_nodes[:5]:
                answer_parts.append(f"• {gene.name}")
            answer_parts.append("")
        
        # روابط بیماری-ژن
        disease_gene_edges = [e for e in retrieval_result.edges if 'cause' in e.relation.lower() or 'associate' in e.relation.lower()]
        if disease_gene_edges:
            answer_parts.append("**روابط بیماری-ژن:**")
            for edge in disease_gene_edges[:5]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"• {source_name} → {target_name}")
        
        if not disease_nodes:
            answer_parts.append("❌ اطلاعات بیماری در نتایج یافت نشد.")
        
        return "\n".join(answer_parts)
    
    def _generate_intelligent_anatomy_answer(self, retrieval_result: RetrievalResult, anatomy_nodes, gene_nodes) -> str:
        """تولید پاسخ هوشمند برای سوالات آناتومی"""
        answer_parts = ["🫀 **تحلیل آناتومیک:**\n"]
        
        if anatomy_nodes:
            answer_parts.append("**ساختارهای آناتومیک:**")
            for anatomy in anatomy_nodes[:5]:
                answer_parts.append(f"• {anatomy.name}")
            answer_parts.append("")
        
        if gene_nodes:
            answer_parts.append("**ژن‌های بیان شده:**")
            for gene in gene_nodes[:5]:
                answer_parts.append(f"• {gene.name}")
            answer_parts.append("")
        
        # روابط بیان
        expression_edges = [e for e in retrieval_result.edges if 'express' in e.relation.lower()]
        if expression_edges:
            answer_parts.append("**روابط بیان ژن:**")
            for edge in expression_edges[:5]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"• {source_name} بیان می‌کند: {target_name}")
        
        if not anatomy_nodes:
            answer_parts.append("❌ اطلاعات آناتومیک در نتایج یافت نشد.")
        
        return "\n".join(answer_parts)
    
    def _generate_intelligent_general_answer(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes, drug_nodes, anatomy_nodes, process_nodes) -> str:
        """تولید پاسخ هوشمند عمومی"""
        answer_parts = ["📊 **تحلیل جامع:**\n"]
        
        # خلاصه آماری
        total_entities = len(retrieval_result.nodes)
        total_relationships = len(retrieval_result.edges)
        
        answer_parts.append(f"**خلاصه آماری:**")
        answer_parts.append(f"• کل موجودیت‌ها: {total_entities}")
        answer_parts.append(f"• کل روابط: {total_relationships}")
        answer_parts.append("")
        
        # دسته‌بندی موجودیت‌ها
        if gene_nodes:
            answer_parts.append(f"**ژن‌ها ({len(gene_nodes)}):**")
            for gene in gene_nodes[:3]:
                score_info = f" (امتیاز: {gene.score:.2f})" if hasattr(gene, 'score') and gene.score != 1.0 else ""
                answer_parts.append(f"• {gene.name}{score_info}")
            answer_parts.append("")
        
        if disease_nodes:
            answer_parts.append(f"**بیماری‌ها ({len(disease_nodes)}):**")
            for disease in disease_nodes[:3]:
                answer_parts.append(f"• {disease.name}")
            answer_parts.append("")
        
        if drug_nodes:
            answer_parts.append(f"**داروها ({len(drug_nodes)}):**")
            for drug in drug_nodes[:3]:
                answer_parts.append(f"• {drug.name}")
            answer_parts.append("")
        
        if anatomy_nodes:
            answer_parts.append(f"**ساختارهای آناتومیک ({len(anatomy_nodes)}):**")
            for anatomy in anatomy_nodes[:3]:
                answer_parts.append(f"• {anatomy.name}")
            answer_parts.append("")
        
        if process_nodes:
            answer_parts.append(f"**فرآیندهای زیستی ({len(process_nodes)}):**")
            for process in process_nodes[:3]:
                answer_parts.append(f"• {process.name}")
            answer_parts.append("")
        
        # مهم‌ترین روابط
        if retrieval_result.edges:
            answer_parts.append("**مهم‌ترین روابط:**")
            # مرتب‌سازی بر اساس نوع رابطه
            relations_count = {}
            for edge in retrieval_result.edges:
                relations_count[edge.relation] = relations_count.get(edge.relation, 0) + 1
            
            for relation, count in sorted(relations_count.items(), key=lambda x: x[1], reverse=True)[:3]:
                answer_parts.append(f"• {relation}: {count} رابطه")
        
        if not retrieval_result.nodes:
            answer_parts.append("❌ اطلاعات مرتبطی در گراف دانش یافت نشد.")
        
        return "\n".join(answer_parts)
    
    def custom_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ سفارشی پیشرفته"""
        query = retrieval_result.query
        query_lower = query.lower()
        
        # تحلیل عمیق‌تر سوال
        question_type = self._analyze_question_type(query_lower)
        
        # استخراج اطلاعات با جزئیات بیشتر
        gene_nodes = [n for n in retrieval_result.nodes if n.kind == 'Gene']
        disease_nodes = [n for n in retrieval_result.nodes if n.kind == 'Disease']
        drug_nodes = [n for n in retrieval_result.nodes if n.kind in ['Drug', 'Compound']]
        anatomy_nodes = [n for n in retrieval_result.nodes if n.kind == 'Anatomy']
        process_nodes = [n for n in retrieval_result.nodes if n.kind == 'Biological Process']
        symptom_nodes = [n for n in retrieval_result.nodes if n.kind == 'Symptom']
        
        # تولید پاسخ سفارشی با تحلیل عمیق
        answer_parts = ["🎯 **تحلیل سفارشی پیشرفته:**\n"]
        
        # تحلیل کیفیت نتایج
        total_score = sum(getattr(n, 'score', 1.0) for n in retrieval_result.nodes)
        avg_score = total_score / len(retrieval_result.nodes) if retrieval_result.nodes else 0
        
        answer_parts.append(f"**کیفیت نتایج:**")
        answer_parts.append(f"• میانگین امتیاز: {avg_score:.2f}")
        answer_parts.append(f"• تعداد نودهای با امتیاز بالا: {len([n for n in retrieval_result.nodes if getattr(n, 'score', 1.0) > 2.0])}")
        answer_parts.append("")
        
        # تحلیل بر اساس نوع سوال
        if question_type == "relationship":
            answer_parts.extend(self._custom_relationship_analysis(retrieval_result, gene_nodes, disease_nodes, drug_nodes))
        elif question_type == "drug_treatment":
            answer_parts.extend(self._custom_drug_analysis(retrieval_result, drug_nodes, disease_nodes))
        elif question_type == "gene_function":
            answer_parts.extend(self._custom_gene_analysis(retrieval_result, gene_nodes, process_nodes))
        elif question_type == "disease_info":
            answer_parts.extend(self._custom_disease_analysis(retrieval_result, disease_nodes, gene_nodes, symptom_nodes))
        elif question_type == "anatomy_expression":
            answer_parts.extend(self._custom_anatomy_analysis(retrieval_result, anatomy_nodes, gene_nodes))
        else:
            answer_parts.extend(self._custom_general_analysis(retrieval_result, gene_nodes, disease_nodes, drug_nodes, anatomy_nodes, process_nodes))
        
        # توصیه‌های کاربردی
        answer_parts.append("**💡 توصیه‌های کاربردی:**")
        if len(retrieval_result.nodes) < 5:
            answer_parts.append("• افزایش عمق جستجو برای یافتن نتایج بیشتر")
        if len(retrieval_result.edges) < 3:
            answer_parts.append("• استفاده از روش‌های ترکیبی برای یافتن روابط بیشتر")
        if avg_score < 2.0:
            answer_parts.append("• استفاده از روش Ensemble برای بهبود کیفیت نتایج")
        
        return "\n".join(answer_parts)
    
    def _custom_relationship_analysis(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes, drug_nodes) -> List[str]:
        """تحلیل سفارشی روابط"""
        parts = []
        parts.append("**🔍 تحلیل روابط:**")
        
        if retrieval_result.edges:
            # تحلیل پیچیدگی روابط
            edge_types = {}
            for edge in retrieval_result.edges:
                edge_types[edge.relation] = edge_types.get(edge.relation, 0) + 1
            
            parts.append(f"• انواع روابط یافت شده: {len(edge_types)}")
            parts.append(f"• پیچیدگی شبکه: {len(retrieval_result.edges)} / {len(retrieval_result.nodes)} = {len(retrieval_result.edges)/len(retrieval_result.nodes):.2f}")
            
            # مهم‌ترین روابط
            most_common = max(edge_types.items(), key=lambda x: x[1])
            parts.append(f"• رایج‌ترین رابطه: {most_common[0]} ({most_common[1]} بار)")
        else:
            parts.append("• هیچ رابطه مستقیمی یافت نشد")
        
        return parts
    
    def _custom_drug_analysis(self, retrieval_result: RetrievalResult, drug_nodes, disease_nodes) -> List[str]:
        """تحلیل سفارشی دارویی"""
        parts = []
        parts.append("**💊 تحلیل دارویی:**")
        
        if drug_nodes:
            # تحلیل داروهای مهم
            high_score_drugs = [d for d in drug_nodes if getattr(d, 'score', 1.0) > 2.0]
            parts.append(f"• داروهای با امتیاز بالا: {len(high_score_drugs)}")
            
            if high_score_drugs:
                parts.append("• مهم‌ترین داروها:")
                for drug in high_score_drugs[:3]:
                    parts.append(f"  - {drug.name} (امتیاز: {drug.score:.2f})")
        
        if disease_nodes:
            parts.append(f"• بیماری‌های مرتبط: {len(disease_nodes)}")
        
        return parts
    
    def _custom_gene_analysis(self, retrieval_result: RetrievalResult, gene_nodes, process_nodes) -> List[str]:
        """تحلیل سفارشی ژنتیکی"""
        parts = []
        parts.append("**🧬 تحلیل ژنتیکی:**")
        
        if gene_nodes:
            # تحلیل ژن‌های مهم
            sorted_genes = sorted(gene_nodes, key=lambda x: getattr(x, 'score', 1.0), reverse=True)
            parts.append(f"• ژن‌های یافت شده: {len(gene_nodes)}")
            parts.append("• مهم‌ترین ژن‌ها:")
            for gene in sorted_genes[:3]:
                parts.append(f"  - {gene.name} (امتیاز: {gene.score:.2f})")
        
        if process_nodes:
            parts.append(f"• فرآیندهای زیستی: {len(process_nodes)}")
        
        return parts
    
    def _custom_disease_analysis(self, retrieval_result: RetrievalResult, disease_nodes, gene_nodes, symptom_nodes) -> List[str]:
        """تحلیل سفارشی بیماری"""
        parts = []
        parts.append("**🏥 تحلیل بیماری:**")
        
        if disease_nodes:
            parts.append(f"• بیماری‌های یافت شده: {len(disease_nodes)}")
        
        if gene_nodes:
            parts.append(f"• ژن‌های مرتبط: {len(gene_nodes)}")
        
        if symptom_nodes:
            parts.append(f"• علائم مرتبط: {len(symptom_nodes)}")
        
        return parts
    
    def _custom_anatomy_analysis(self, retrieval_result: RetrievalResult, anatomy_nodes, gene_nodes) -> List[str]:
        """تحلیل سفارشی آناتومیک"""
        parts = []
        parts.append("**🫀 تحلیل آناتومیک:**")
        
        if anatomy_nodes:
            parts.append(f"• ساختارهای آناتومیک: {len(anatomy_nodes)}")
        
        if gene_nodes:
            parts.append(f"• ژن‌های بیان شده: {len(gene_nodes)}")
        
        return parts
    
    def _custom_general_analysis(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes, drug_nodes, anatomy_nodes, process_nodes) -> List[str]:
        """تحلیل سفارشی عمومی"""
        parts = []
        parts.append("**📊 تحلیل جامع:**")
        
        # آمار کلی
        parts.append(f"• کل موجودیت‌ها: {len(retrieval_result.nodes)}")
        parts.append(f"• کل روابط: {len(retrieval_result.edges)}")
        parts.append(f"• تراکم شبکه: {len(retrieval_result.edges)/max(len(retrieval_result.nodes), 1):.2f}")
        
        # توزیع انواع
        type_distribution = {}
        for node in retrieval_result.nodes:
            type_distribution[node.kind] = type_distribution.get(node.kind, 0) + 1
        
        parts.append("• توزیع انواع:")
        for kind, count in sorted(type_distribution.items(), key=lambda x: x[1], reverse=True):
            parts.append(f"  - {kind}: {count}")
        
        return parts
    
    def process_query(self, query: str, retrieval_method: RetrievalMethod, 
                     generation_model: GenerationModel, max_depth: int = 2) -> Dict[str, Any]:
        """پردازش کامل یک سوال"""
        print(f"🚀 پردازش سوال: {query}")
        
        # مرحله 1: بازیابی
        retrieval_result = self.retrieve_information(query, retrieval_method, max_depth)
        
        # مرحله 2: تولید پاسخ
        generation_result = self.generate_answer(retrieval_result, generation_model)
        
        # آماده‌سازی نتیجه
        result = {
            "query": query,
            "retrieval_method": retrieval_method.value,
            "generation_model": generation_model.value,
            "keywords": self.extract_keywords(query),
            "matched_nodes": {k: self.G.nodes[v]['name'] for k, v in self.match_tokens_to_nodes(self.extract_keywords(query)).items()},
            "retrieved_nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "kind": node.kind,
                    "depth": node.depth,
                    "score": node.score
                } for node in retrieval_result.nodes
            ],
            "retrieved_edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "weight": edge.weight
                } for edge in retrieval_result.edges
            ],
            "paths": retrieval_result.paths,
            "context_text": retrieval_result.context_text,
            "answer": generation_result.answer,
            "confidence": generation_result.confidence,
            "process_steps": [
                "1. استخراج کلمات کلیدی از سوال",
                "2. تطبیق کلمات کلیدی با نودهای گراف",
                f"3. بازیابی اطلاعات با روش {retrieval_method.value}",
                "4. ایجاد متن زمینه از نتایج",
                f"5. تولید پاسخ با مدل {generation_model.value}"
            ]
        }
        
        return result
    
    def huggingface_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ با مدل‌های HuggingFace (رایگان)"""
        try:
            # استفاده از مدل‌های رایگان HuggingFace
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # مدل‌های رایگان قدرتمند
            models = [
                "microsoft/DialoGPT-medium",  # چت‌بات
                "gpt2",  # GPT-2
                "distilgpt2",  # GPT-2 سبک
                "EleutherAI/gpt-neo-125M",  # GPT-Neo کوچک
                "microsoft/DialoGPT-small"  # چت‌بات کوچک
            ]
            
            # انتخاب بهترین مدل موجود
            selected_model = None
            for model_name in models:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    selected_model = model_name
                    break
                except:
                    continue
            
            if selected_model is None:
                return self._fallback_generation(retrieval_result, "HuggingFace")
            
            # آماده‌سازی متن ورودی
            prompt = self._create_advanced_prompt(retrieval_result)
            
            # تولید پاسخ
            inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=300,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # استخراج پاسخ از خروجی
            if len(response) > len(prompt):
                answer = response[len(prompt):].strip()
            else:
                answer = response.strip()
            
            return answer if answer else self._fallback_generation(retrieval_result, "HuggingFace")
            
        except Exception as e:
            print(f"خطا در HuggingFace: {e}")
            return self._fallback_generation(retrieval_result, "HuggingFace")
    
    def openai_gpt_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ با OpenAI GPT (نیاز به API Key)"""
        try:
            from openai import OpenAI
            
            # بررسی وجود API Key
            if not hasattr(self, 'openai_api_key') or not self.openai_api_key:
                return "🔑 برای استفاده از OpenAI GPT، لطفاً API Key را تنظیم کنید.\n\n" + self._fallback_generation(retrieval_result, "OpenAI")
            
            # ایجاد کلاینت OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            # آماده‌سازی متن ورودی
            prompt = self._create_advanced_prompt(retrieval_result)
            
            # انتخاب مدل بر اساس هزینه و کیفیت
            # gpt-3.5-turbo: ارزان و سریع
            # gpt-4: گران‌تر اما کیفیت بهتر
            # gpt-4-turbo-preview: جدیدترین و بهترین
            model_choice = "gpt-3.5-turbo"  # می‌توانید به gpt-4 تغییر دهید
            
            # درخواست به OpenAI
            response = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": "You are a biomedical expert analyzing knowledge graph data. Provide detailed, accurate, and well-structured answers in Persian with proper formatting and emojis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,  # افزایش تعداد توکن‌ها
                temperature=0.7,
                presence_penalty=0.1,  # تشویق به تنوع
                frequency_penalty=0.1   # کاهش تکرار
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"خطا در OpenAI: {e}")
            return self._fallback_generation(retrieval_result, "OpenAI")
    
    def anthropic_claude_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ با Anthropic Claude (نیاز به API Key)"""
        try:
            import anthropic
            
            # بررسی وجود API Key
            if not hasattr(self, 'anthropic_api_key') or not self.anthropic_api_key:
                return "🔑 برای استفاده از Claude، لطفاً API Key را تنظیم کنید.\n\n" + self._fallback_generation(retrieval_result, "Claude")
            
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            # آماده‌سازی متن ورودی
            prompt = self._create_advanced_prompt(retrieval_result)
            
            # درخواست به Claude
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            print(f"خطا در Claude: {e}")
            return self._fallback_generation(retrieval_result, "Claude")
    
    def google_gemini_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ با Google Gemini (نیاز به API Key)"""
        try:
            import google.generativeai as genai
            
            # بررسی وجود API Key
            if not hasattr(self, 'gemini_api_key') or not self.gemini_api_key:
                return "🔑 برای استفاده از Gemini، لطفاً API Key را تنظیم کنید.\n\n" + self._fallback_generation(retrieval_result, "Gemini")
            
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # آماده‌سازی متن ورودی
            prompt = self._create_advanced_prompt(retrieval_result)
            
            # درخواست به Gemini
            response = model.generate_content(prompt)
            
            return response.text.strip()
            
        except Exception as e:
            print(f"خطا در Gemini: {e}")
            return self._fallback_generation(retrieval_result, "Gemini")
    
    def _create_advanced_prompt(self, retrieval_result: RetrievalResult) -> str:
        """ایجاد متن ورودی پیشرفته برای مدل‌های AI"""
        query = retrieval_result.query
        context = retrieval_result.context_text
        method = retrieval_result.method
        
        # تحلیل نوع سوال
        question_type = self._analyze_question_type(query.lower())
        
        # بررسی اینکه آیا از گراف استفاده شده یا نه
        if method == "بدون بازیابی (فقط مدل)":
            # فقط مدل - بدون اطلاعات گراف
            system_prompt = """You are an expert biomedical AI assistant with comprehensive knowledge of:
            - Molecular biology and genetics
            - Drug discovery and pharmacology
            - Disease mechanisms and pathology
            - Biological pathways and networks
            - Clinical medicine and therapeutics
            
            Your task is to provide detailed, accurate, and well-structured answers to biomedical questions
            based on your training knowledge. Focus on:
            - Scientific accuracy and current understanding
            - Comprehensive analysis and insights
            - Practical implications and applications
            - Research directions and future possibilities
            
            Always answer in Persian with proper formatting and structure your response with clear sections.
            Do not use emojis in your response."""
            
            user_prompt = f"""
            **سوال پزشکی-زیستی:**
            {query}
            
            **دستورالعمل پاسخ‌دهی:**
            لطفاً بر اساس دانش تخصصی خود در زمینه علوم زیستی و پزشکی، پاسخ جامع و دقیقی ارائه دهید که شامل:
            
            1. **تحلیل موضوع:** بررسی دقیق سوال و جنبه‌های مختلف آن
            2. **مبانی علمی:** توضیح مکانیزم‌ها و فرآیندهای زیستی مرتبط
            3. **جنبه‌های درمانی:** در صورت مرتبط بودن، روش‌های درمانی و دارویی
            4. **تحقیقات:** وضعیت فعلی تحقیقات و مطالعات مرتبط
            5. **چشم‌انداز آینده:** مسیرهای تحقیقاتی و پیشرفت‌های آینده
            6. **توصیه‌های کاربردی:** نکات مهم برای پژوهشگران و پزشکان
            
            پاسخ را به صورت ساختاریافته و با فرمت‌بندی مناسب ارائه دهید.
            """
            
        else:
            # با اطلاعات گراف
            system_prompt = """You are a biomedical knowledge graph expert analyzing data from a comprehensive
            biological knowledge graph containing information about:
            - Genes, proteins, and their functions
            - Diseases and their molecular mechanisms
            - Drugs, compounds, and their therapeutic effects
            - Biological processes and pathways
            - Anatomical structures and gene expression
            - Clinical relationships and treatment outcomes
            
            Your task is to analyze the retrieved information from the knowledge graph and provide:
            - Comprehensive interpretation of the data
            - Biological significance and implications
            - Clinical relevance and applications
            - Research insights and recommendations
            - Quality assessment of the retrieved information
            
            IMPORTANT: If the retrieved information is insufficient or limited, supplement your analysis
            with your general biomedical knowledge to provide a comprehensive and useful answer.
            Focus on providing valuable insights even when graph data is limited.
            
            Always answer in Persian with proper formatting and structure your response with clear sections.
            Do not use emojis in your response."""
            
            user_prompt = f"""
            **سوال پزشکی-زیستی:**
            {query}
            
            **اطلاعات بازیابی شده از گراف دانش زیستی:**
            روش بازیابی: {method}
            
            {context}
            
            **دستورالعمل تحلیل:**
            بر اساس اطلاعات بازیابی شده از گراف دانش، تحلیل جامع و دقیقی ارائه دهید که شامل:
            
            1. **خلاصه آماری:** آمار کلی اطلاعات بازیابی شده
            2. **تحلیل روابط:** بررسی روابط و ارتباطات یافت شده
            3. **اهمیت زیستی:** تفسیر اهمیت زیستی و پزشکی یافته‌ها
            4. **جنبه‌های درمانی:** کاربردهای درمانی و دارویی
            5. **کیفیت داده‌ها:** ارزیابی کیفیت و اعتبار اطلاعات
            6. **توصیه‌های کاربردی:** پیشنهادات برای پژوهش و کاربرد
            7. **مسیرهای آینده:** جهت‌گیری‌های تحقیقاتی پیشنهادی
            
            **نکته مهم:** اگر اطلاعات بازیابی شده محدود یا ناکافی باشد، از دانش عمومی خود در زمینه علوم زیستی
            برای تکمیل تحلیل استفاده کنید و پاسخ جامع و مفیدی ارائه دهید. هدف ارائه اطلاعات ارزشمند
            به کاربر است، حتی اگر داده‌های گراف محدود باشند.
            
            پاسخ را به صورت ساختاریافته و با فرمت‌بندی مناسب ارائه دهید.
            """
        
        return f"{system_prompt}\n\n{user_prompt}"
    
    def _fallback_generation(self, retrieval_result: RetrievalResult, model_name: str) -> str:
        """تولید پاسخ پشتیبان در صورت خطا"""
        return f"""🤖 **تحلیل با {model_name} (پاسخ پشتیبان):**

{self.gpt_simulation_generation(retrieval_result)}

---
💡 **نکته:** برای استفاده از مدل‌های پیشرفته‌تر، لطفاً API Key مربوطه را تنظیم کنید.
"""
    
    def set_openai_api_key(self, api_key: str):
        """تنظیم API Key برای OpenAI"""
        self.openai_api_key = api_key
        print("✅ OpenAI API Key تنظیم شد")
    
    def set_anthropic_api_key(self, api_key: str):
        """تنظیم API Key برای Anthropic"""
        self.anthropic_api_key = api_key
        print("✅ Anthropic API Key تنظیم شد")
    
    def set_gemini_api_key(self, api_key: str):
        """تنظیم API Key برای Google Gemini"""
        self.gemini_api_key = api_key
        print("✅ Gemini API Key تنظیم شد")

# نمونه استفاده
if __name__ == "__main__":
    service = GraphRAGService()
    
    # تست سرویس
    result = service.process_query(
        query="What is the relationship between HMGB3 and diabetes?",
        retrieval_method=RetrievalMethod.BFS,
        generation_model=GenerationModel.GPT_SIMULATION
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False)) 
# -*- coding: utf-8 -*-
"""
GraphRAG Utils - توابع کمکی GraphRAG
"""
import json
import logging
import xxhash
from typing import Dict, Any, List, Optional
import networkx as nx

# Import from rag_new
from rag_new.utils import REDIS_CONN, num_tokens_from_string, get_float

def get_llm_cache(key: str) -> Optional[str]:
    """دریافت کش LLM از Redis"""
    try:
        return REDIS_CONN.get(f"llm_cache:{key}")
    except Exception as e:
        logging.warning(f"Failed to get LLM cache: {e}")
        return None

def set_llm_cache(key: str, value: str, expire: int = 3600) -> bool:
    """ذخیره کش LLM در Redis"""
    try:
        return REDIS_CONN.set(f"llm_cache:{key}", value, expire)
    except Exception as e:
        logging.warning(f"Failed to set LLM cache: {e}")
        return False

def get_entity_type2sampels() -> Dict[str, List[str]]:
    """دریافت نمونه‌های انواع موجودیت‌ها"""
    return {
        "Gene": ["TP53", "BRCA1", "EGFR", "KRAS", "PIK3CA"],
        "Disease": ["Cancer", "Diabetes", "Heart Disease", "Alzheimer", "Parkinson"],
        "Drug": ["Aspirin", "Ibuprofen", "Morphine", "Insulin", "Chemotherapy"],
        "Protein": ["Insulin", "Hemoglobin", "Collagen", "Myosin", "Actin"],
        "Pathway": ["Glycolysis", "Krebs Cycle", "DNA Replication", "Protein Synthesis"],
        "Cell": ["Neuron", "Muscle Cell", "Blood Cell", "Stem Cell", "Cancer Cell"],
        "Tissue": ["Brain", "Heart", "Liver", "Muscle", "Skin"],
        "Organ": ["Brain", "Heart", "Liver", "Lung", "Kidney"],
        "Process": ["Metabolism", "Cell Division", "Apoptosis", "Differentiation"],
        "Function": ["Transport", "Catalysis", "Binding", "Regulation", "Signaling"]
    }

def get_relation(relation_name: str) -> Dict[str, Any]:
    """دریافت اطلاعات رابطه"""
    # این تابع می‌تواند از دیتابیس یا فایل‌های محلی خوانده شود
    return {
        "name": relation_name,
        "type": "biological",
        "description": f"Relation: {relation_name}"
    }

def clean_graph_data(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """پاکسازی داده‌های گراف"""
    cleaned_data = {}
    
    # پاکسازی نودها
    if "nodes" in graph_data:
        cleaned_nodes = []
        for node in graph_data["nodes"]:
            if isinstance(node, dict) and "id" in node:
                cleaned_node = {
                    "id": str(node["id"]).strip(),
                    "label": str(node.get("label", "")).strip(),
                    "type": str(node.get("type", "")).strip(),
                    "properties": node.get("properties", {})
                }
                cleaned_nodes.append(cleaned_node)
        cleaned_data["nodes"] = cleaned_nodes
    
    # پاکسازی یال‌ها
    if "edges" in graph_data:
        cleaned_edges = []
        for edge in graph_data["edges"]:
            if isinstance(edge, dict) and "source" in edge and "target" in edge:
                cleaned_edge = {
                    "source": str(edge["source"]).strip(),
                    "target": str(edge["target"]).strip(),
                    "label": str(edge.get("label", "")).strip(),
                    "properties": edge.get("properties", {})
                }
                cleaned_edges.append(cleaned_edge)
        cleaned_data["edges"] = cleaned_edges
    
    return cleaned_data

def create_networkx_graph(graph_data: Dict[str, Any]) -> nx.Graph:
    """ایجاد گراف NetworkX از داده‌های گراف"""
    G = nx.Graph()
    
    # اضافه کردن نودها
    if "nodes" in graph_data:
        for node in graph_data["nodes"]:
            G.add_node(node["id"], **node.get("properties", {}))
    
    # اضافه کردن یال‌ها
    if "edges" in graph_data:
        for edge in graph_data["edges"]:
            G.add_edge(edge["source"], edge["target"], 
                      label=edge.get("label", ""), 
                      **edge.get("properties", {}))
    
    return G

def calculate_node_importance(G: nx.Graph, node: str) -> float:
    """محاسبه اهمیت نود بر اساس PageRank"""
    try:
        pagerank = nx.pagerank(G)
        return pagerank.get(node, 0.0)
    except Exception as e:
        logging.warning(f"Failed to calculate PageRank: {e}")
        return 0.0

def find_shortest_paths(G: nx.Graph, source: str, target: str, max_paths: int = 5) -> List[List[str]]:
    """یافتن کوتاه‌ترین مسیرها بین دو نود"""
    try:
        paths = list(nx.all_simple_paths(G, source, target, cutoff=10))
        # مرتب‌سازی بر اساس طول مسیر
        paths.sort(key=len)
        return paths[:max_paths]
    except Exception as e:
        logging.warning(f"Failed to find shortest paths: {e}")
        return []

def extract_subgraph(G: nx.Graph, nodes: List[str], max_depth: int = 2) -> nx.Graph:
    """استخراج زیرگراف حول نودهای مشخص شده"""
    try:
        # یافتن نودهای همسایه تا عمق مشخص شده
        subgraph_nodes = set(nodes)
        current_nodes = set(nodes)
        
        for depth in range(max_depth):
            neighbors = set()
            for node in current_nodes:
                if node in G:
                    neighbors.update(G.neighbors(node))
            subgraph_nodes.update(neighbors)
            current_nodes = neighbors
        
        return G.subgraph(subgraph_nodes)
    except Exception as e:
        logging.warning(f"Failed to extract subgraph: {e}")
        return nx.Graph()

def generate_cache_key(*args, **kwargs) -> str:
    """تولید کلید کش از آرگومان‌ها"""
    # ترکیب آرگومان‌ها
    key_parts = []
    
    # اضافه کردن آرگومان‌های موقعیتی
    for arg in args:
        key_parts.append(str(arg))
    
    # اضافه کردن آرگومان‌های کلیدی
    for key, value in sorted(kwargs.items()):
        key_parts.append(f"{key}:{value}")
    
    # ترکیب و هش کردن
    key_string = "|".join(key_parts)
    return xxhash.xxh64(key_string.encode()).hexdigest()

class GraphChange:
    """کلاس برای ردیابی تغییرات گراف"""
    
    def __init__(self):
        self.added_nodes = []
        self.removed_nodes = []
        self.added_edges = []
        self.removed_edges = []
        self.modified_nodes = []
    
    def add_node(self, node_id: str, properties: Dict[str, Any] = None):
        """اضافه کردن نود جدید"""
        self.added_nodes.append({
            "id": node_id,
            "properties": properties or {}
        })
    
    def remove_node(self, node_id: str):
        """حذف نود"""
        self.removed_nodes.append(node_id)
    
    def add_edge(self, source: str, target: str, properties: Dict[str, Any] = None):
        """اضافه کردن یال جدید"""
        self.added_edges.append({
            "source": source,
            "target": target,
            "properties": properties or {}
        })
    
    def remove_edge(self, source: str, target: str):
        """حذف یال"""
        self.removed_edges.append({
            "source": source,
            "target": target
        })
    
    def modify_node(self, node_id: str, properties: Dict[str, Any]):
        """تغییر نود"""
        self.modified_nodes.append({
            "id": node_id,
            "properties": properties
        })
    
    def get_summary(self) -> Dict[str, int]:
        """دریافت خلاصه تغییرات"""
        return {
            "added_nodes": len(self.added_nodes),
            "removed_nodes": len(self.removed_nodes),
            "added_edges": len(self.added_edges),
            "removed_edges": len(self.removed_edges),
            "modified_nodes": len(self.modified_nodes)
        }
    
    def clear(self):
        """پاک کردن تمام تغییرات"""
        self.added_nodes.clear()
        self.removed_nodes.clear()
        self.added_edges.clear()
        self.removed_edges.clear()
        self.modified_nodes.clear() 
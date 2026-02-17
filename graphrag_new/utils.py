# -*- coding: utf-8 -*-
"""
GraphRAG Utils - توابع کمکی GraphRAG
"""
import json
import logging
import xxhash
from typing import Dict, Any, List, Optional
from itertools import islice
import networkx as nx

# Import from rag_new
from rag_new.utils import REDIS_CONN, num_tokens_from_string, get_float

def make_llm_cache_key(llm_name: str, system: Any, history: Any, gen_conf: Any) -> str:
    """تبدیل پارامترهای LLM به کلید یکتا برای کش."""
    try:
        payload = json.dumps({
            "llm": llm_name,
            "system": system,
            "history": history,
            "gen": gen_conf
        }, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        payload = f"{llm_name}|{str(system)}|{str(history)}|{str(gen_conf)}"
    return f"llm_cache:{xxhash.xxh64(payload.encode()).hexdigest()}"


def get_llm_cache(llm_name_or_key: str, system: Any = None, history: Any = None, gen_conf: Any = None) -> Optional[str]:
    """دریافت کش LLM از Redis (سازگار با دو امضا)."""
    try:
        if system is None and history is None and gen_conf is None:
            key = llm_name_or_key if str(llm_name_or_key).startswith("llm_cache:") else f"llm_cache:{llm_name_or_key}"
        else:
            key = make_llm_cache_key(llm_name_or_key, system, history, gen_conf)
        return REDIS_CONN.get(key)
    except Exception as e:
        logging.warning(f"Failed to get LLM cache: {e}")
        return None

def set_llm_cache(*args, **kwargs) -> bool:
    """ذخیره کش LLM در Redis (سازگار با دو امضا)."""
    try:
        expire = kwargs.get("expire", 3600)
        if len(args) == 2 or (len(args) == 3 and isinstance(args[2], int)):
            key = args[0]
            value = args[1]
            if len(args) == 3 and isinstance(args[2], int):
                expire = args[2]
            full_key = key if str(key).startswith("llm_cache:") else f"llm_cache:{key}"
            return REDIS_CONN.set(full_key, value, expire)
        elif len(args) >= 5:
            llm_name, system, value, history, gen_conf = args[:5]
            if len(args) >= 6 and isinstance(args[5], int):
                expire = args[5]
            cache_key = make_llm_cache_key(llm_name, system, history, gen_conf)
            return REDIS_CONN.set(cache_key, value, expire)
        else:
            logging.warning("set_llm_cache called with unsupported signature")
            return False
    except Exception as e:
        logging.warning(f"Failed to set LLM cache: {e}")
        return False

def get_entity_type2samples(idxnms: Any = None, kb_ids: Any = None) -> Dict[str, List[str]]:
    """نمونه‌های انواع موجودیت‌ها بر اساس متانودهای Hetionet."""
    return {
        "Gene": ["TP53", "BRCA1", "EGFR", "KRAS", "PIK3CA"],
        "Disease": ["breast cancer", "glioblastoma", "diabetes", "Alzheimer's disease"],
        "Compound": ["trastuzumab", "cisplatin", "imatinib", "aspirin"],
        "Pathway": ["PI3K-Akt signaling", "Wnt signaling", "DNA repair"],
        "Biological Process": ["apoptosis", "cell cycle", "DNA repair"],
        "Molecular Function": ["kinase activity", "DNA binding"],
        "Cellular Component": ["nucleus", "mitochondrion"],
        "Pharmacologic Class": ["antineoplastic agents", "tyrosine kinase inhibitors"],
        "Anatomy": ["brain", "liver", "breast"],
        "Symptom": ["fever", "pain", "fatigue"]
    }

def get_entity_type2sampels(idxnms: Any = None, kb_ids: Any = None) -> Dict[str, List[str]]:
    return get_entity_type2samples(idxnms, kb_ids)

def get_relation(*args) -> Any:
    if len(args) == 1:
        relation_name = args[0]
        return {
            "name": relation_name,
            "type": "biological",
            "description": f"Relation: {relation_name}"
        }
    elif len(args) == 2:
        return "related"
    else:
        return "related"

def clean_graph_data(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """پاکسازی داده‌های گراف"""
    cleaned_data = {}
    
    # پاکسازی نودها
    if "nodes" in graph_data:
        cleaned_nodes = []
        for node in graph_data["nodes"]:
            if isinstance(node, dict) and "id" in node:
                props = node.get("properties", {}) or {}
                label = str(node.get("label", props.get("label", ""))).strip()
                ntype = str(node.get("type", props.get("type", ""))).strip()
                cleaned_node = {
                    "id": str(node["id"]).strip(),
                    "label": label,
                    "type": ntype,
                    "properties": props
                }
                cleaned_nodes.append(cleaned_node)
            else:
                logging.debug(f"Dropped invalid node: {node}")
        cleaned_data["nodes"] = cleaned_nodes
    
    # پاکسازی یال‌ها
    if "edges" in graph_data:
        cleaned_edges = []
        for edge in graph_data["edges"]:
            if isinstance(edge, dict) and "source" in edge and "target" in edge:
                props = edge.get("properties", {}) or {}
                cleaned_edge = {
                    "source": str(edge["source"]).strip(),
                    "target": str(edge["target"]).strip(),
                    "label": str(edge.get("label", props.get("label", ""))).strip(),
                    "properties": props
                }
                cleaned_edges.append(cleaned_edge)
            else:
                logging.debug(f"Dropped invalid edge: {edge}")
        cleaned_data["edges"] = cleaned_edges
    
    return cleaned_data

def create_networkx_graph(graph_data: Dict[str, Any]) -> nx.MultiDiGraph:
    """ایجاد گراف NetworkX (MultiDiGraph) از داده‌های گراف"""
    G = nx.MultiDiGraph()
    for node in graph_data.get("nodes", []):
        G.add_node(node["id"], label=node.get("label", ""), type=node.get("type", ""), **node.get("properties", {}))
    for edge in graph_data.get("edges", []):
        G.add_edge(edge["source"], edge["target"], label=edge.get("label", ""), **edge.get("properties", {}))
    return G

def calculate_node_importance(G: nx.Graph, node: str, pagerank_map: Optional[Dict[str, float]] = None, alpha: float = 0.85) -> float:
    """محاسبه اهمیت نود بر اساس PageRank (با کش خارجی اختیاری)."""
    try:
        if pagerank_map is None:
            pagerank_map = nx.pagerank(G, alpha=alpha)
        return pagerank_map.get(node, 0.0)
    except Exception as e:
        logging.warning(f"Failed to calculate PageRank: {e}")
        return 0.0

def find_shortest_paths(G: nx.Graph, source: str, target: str, max_paths: int = 5) -> List[List[str]]:
    """یافتن کوتاه‌ترین مسیرها بین دو نود (shortest_simple_paths)."""
    try:
        gen = nx.shortest_simple_paths(G, source, target)
        return list(islice(gen, max_paths))
    except Exception as e:
        logging.warning(f"Failed to find shortest paths: {e}")
        return []

def extract_subgraph(G: nx.Graph, nodes: List[str], max_depth: int = 2, direction: str = "both", max_neighbors_per_hop: Optional[int] = None) -> nx.Graph:
    """استخراج زیرگراف حول نودهای مشخص شده با درنظرگرفتن جهت و کنترل هاب."""
    try:
        subgraph_nodes = set(nodes)
        frontier = list(nodes)
        visited = set(nodes)
        for _ in range(1, max_depth + 1):
            next_frontier: List[str] = []
            for node in frontier:
                if node not in G:
                    continue
                if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
                    if direction == "out":
                        neighbors_iter = G.successors(node)
                    elif direction == "in":
                        neighbors_iter = G.predecessors(node)
                    else:
                        neighbors_iter = list(G.successors(node)) + list(G.predecessors(node))
                else:
                    neighbors_iter = G.neighbors(node)
                count = 0
                for nbr in neighbors_iter:
                    if nbr in visited:
                        continue
                    subgraph_nodes.add(nbr)
                    visited.add(nbr)
                    next_frontier.append(nbr)
                    count += 1
                    if max_neighbors_per_hop is not None and count >= max_neighbors_per_hop:
                        break
            frontier = next_frontier
        return G.subgraph(subgraph_nodes).copy()
    except Exception as e:
        logging.warning(f"Failed to extract subgraph: {e}")
        return nx.MultiDiGraph() if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)) else nx.Graph()

def generate_cache_key(*args, **kwargs) -> str:
    """تولید کلید کش پایدار از آرگومان‌ها (JSON مرتب برای ساختارهای پیچیده)."""
    parts: List[str] = []
    for a in args:
        if isinstance(a, (dict, list, tuple)):
            parts.append(json.dumps(a, ensure_ascii=False, sort_keys=True))
        else:
            parts.append(str(a))
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (dict, list, tuple)):
            v_str = json.dumps(v, ensure_ascii=False, sort_keys=True)
        else:
            v_str = str(v)
        parts.append(f"{k}:{v_str}")
    return xxhash.xxh64("|".join(parts).encode()).hexdigest()

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
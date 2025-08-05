# -*- coding: utf-8 -*-
"""
Base Extractor - کلاس پایه استخراج
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import networkx as nx

class BaseExtractor(ABC):
    """کلاس پایه برای استخراج اطلاعات از گراف"""
    
    def __init__(self, graph: nx.Graph = None):
        self.graph = graph
        self.extracted_data = {}
    
    @abstractmethod
    def extract(self, **kwargs) -> Dict[str, Any]:
        """استخراج اطلاعات از گراف - باید در کلاس‌های فرزند پیاده‌سازی شود"""
        pass
    
    def set_graph(self, graph: nx.Graph):
        """تنظیم گراف برای استخراج"""
        self.graph = graph
    
    def get_graph_info(self) -> Dict[str, Any]:
        """دریافت اطلاعات پایه گراف"""
        if not self.graph:
            return {}
        
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "is_directed": self.graph.is_directed(),
            "is_weighted": any(self.graph.get_edge_data(u, v).get('weight') 
                              for u, v in self.graph.edges()),
            "node_types": self._get_node_types(),
            "edge_types": self._get_edge_types()
        }
    
    def _get_node_types(self) -> Dict[str, int]:
        """دریافت انواع نودها و تعداد آنها"""
        node_types = {}
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        return node_types
    
    def _get_edge_types(self) -> Dict[str, int]:
        """دریافت انواع یال‌ها و تعداد آنها"""
        edge_types = {}
        for u, v, attrs in self.graph.edges(data=True):
            edge_type = attrs.get('label', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        return edge_types
    
    def get_node_attributes(self, node: str) -> Dict[str, Any]:
        """دریافت ویژگی‌های نود"""
        if not self.graph or node not in self.graph:
            return {}
        return dict(self.graph.nodes[node])
    
    def get_edge_attributes(self, source: str, target: str) -> Dict[str, Any]:
        """دریافت ویژگی‌های یال"""
        if not self.graph or not self.graph.has_edge(source, target):
            return {}
        return dict(self.graph.get_edge_data(source, target))
    
    def find_nodes_by_type(self, node_type: str) -> List[str]:
        """یافتن نودها بر اساس نوع"""
        if not self.graph:
            return []
        
        nodes = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == node_type:
                nodes.append(node)
        return nodes
    
    def find_edges_by_type(self, edge_type: str) -> List[tuple]:
        """یافتن یال‌ها بر اساس نوع"""
        if not self.graph:
            return []
        
        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            if attrs.get('label') == edge_type:
                edges.append((u, v))
        return edges
    
    def get_neighbors(self, node: str) -> List[str]:
        """دریافت همسایه‌های نود"""
        if not self.graph or node not in self.graph:
            return []
        return list(self.graph.neighbors(node))
    
    def get_degree(self, node: str) -> int:
        """دریافت درجه نود"""
        if not self.graph or node not in self.graph:
            return 0
        return self.graph.degree(node)
    
    def calculate_centrality(self, node: str, centrality_type: str = "degree") -> float:
        """محاسبه مرکزیت نود"""
        if not self.graph or node not in self.graph:
            return 0.0
        
        try:
            if centrality_type == "degree":
                return self.graph.degree(node)
            elif centrality_type == "betweenness":
                return nx.betweenness_centrality(self.graph).get(node, 0.0)
            elif centrality_type == "closeness":
                return nx.closeness_centrality(self.graph).get(node, 0.0)
            elif centrality_type == "eigenvector":
                return nx.eigenvector_centrality(self.graph, max_iter=1000).get(node, 0.0)
            else:
                logging.warning(f"Unknown centrality type: {centrality_type}")
                return 0.0
        except Exception as e:
            logging.warning(f"Error calculating centrality: {e}")
            return 0.0
    
    def get_subgraph(self, nodes: List[str]) -> nx.Graph:
        """دریافت زیرگراف شامل نودهای مشخص شده"""
        if not self.graph:
            return nx.Graph()
        
        # اضافه کردن نودهای همسایه تا عمق 1
        all_nodes = set(nodes)
        for node in nodes:
            if node in self.graph:
                all_nodes.update(self.graph.neighbors(node))
        
        return self.graph.subgraph(all_nodes)
    
    def validate_extraction(self, extracted_data: Dict[str, Any]) -> bool:
        """اعتبارسنجی داده‌های استخراج شده"""
        if not extracted_data:
            return False
        
        # بررسی وجود کلیدهای ضروری
        required_keys = ['nodes', 'edges']
        for key in required_keys:
            if key not in extracted_data:
                logging.warning(f"Missing required key: {key}")
                return False
        
        # بررسی نوع داده‌ها
        if not isinstance(extracted_data['nodes'], list):
            logging.warning("Nodes must be a list")
            return False
        
        if not isinstance(extracted_data['edges'], list):
            logging.warning("Edges must be a list")
            return False
        
        return True
    
    def save_extracted_data(self, filepath: str):
        """ذخیره داده‌های استخراج شده"""
        try:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.extracted_data, f, ensure_ascii=False, indent=2)
            logging.info(f"Extracted data saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving extracted data: {e}")
    
    def load_extracted_data(self, filepath: str):
        """بارگذاری داده‌های استخراج شده"""
        try:
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                self.extracted_data = json.load(f)
            logging.info(f"Extracted data loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading extracted data: {e}")
    
    def clear_extracted_data(self):
        """پاک کردن داده‌های استخراج شده"""
        self.extracted_data.clear() 
# -*- coding: utf-8 -*-
"""
Graph Index - ایندکس گراف
"""
import logging
from typing import Dict, List, Any, Optional
import networkx as nx
import json
import pickle
from pathlib import Path

class GraphIndex:
    """کلاس برای ایندکس کردن و مدیریت گراف"""
    
    def __init__(self, graph: nx.Graph = None, index_path: str = None):
        self.graph = graph
        self.index_path = index_path
        self.node_index = {}
        self.edge_index = {}
        self.type_index = {}
        self.attribute_index = {}
        
    def build_index(self, graph: nx.Graph = None):
        """ساخت ایندکس برای گراف"""
        if graph:
            self.graph = graph
        
        if not self.graph:
            logging.warning("No graph provided for indexing")
            return
        
        try:
            # ایندکس نودها
            self._build_node_index()
            
            # ایندکس یال‌ها
            self._build_edge_index()
            
            # ایندکس انواع
            self._build_type_index()
            
            # ایندکس ویژگی‌ها
            self._build_attribute_index()
            
            logging.info(f"Index built successfully: {len(self.node_index)} nodes, {len(self.edge_index)} edges")
            
        except Exception as e:
            logging.error(f"Error building index: {e}")
    
    def _build_node_index(self):
        """ساخت ایندکس نودها"""
        self.node_index = {}
        
        for node, attrs in self.graph.nodes(data=True):
            # ایندکس بر اساس نام نود
            self.node_index[node] = {
                "id": node,
                "type": attrs.get('type', 'unknown'),
                "attributes": dict(attrs),
                "degree": self.graph.degree(node)
            }
    
    def _build_edge_index(self):
        """ساخت ایندکس یال‌ها"""
        self.edge_index = {}
        
        for u, v, attrs in self.graph.edges(data=True):
            edge_id = f"{u}_{v}"
            self.edge_index[edge_id] = {
                "source": u,
                "target": v,
                "label": attrs.get('label', 'unknown'),
                "attributes": dict(attrs)
            }
    
    def _build_type_index(self):
        """ساخت ایندکس انواع"""
        self.type_index = {
            "nodes": {},
            "edges": {}
        }
        
        # ایندکس انواع نودها
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            if node_type not in self.type_index["nodes"]:
                self.type_index["nodes"][node_type] = []
            self.type_index["nodes"][node_type].append(node)
        
        # ایندکس انواع یال‌ها
        for u, v, attrs in self.graph.edges(data=True):
            edge_type = attrs.get('label', 'unknown')
            if edge_type not in self.type_index["edges"]:
                self.type_index["edges"][edge_type] = []
            self.type_index["edges"][edge_type].append((u, v))
    
    def _build_attribute_index(self):
        """ساخت ایندکس ویژگی‌ها"""
        self.attribute_index = {
            "node_attributes": {},
            "edge_attributes": {}
        }
        
        # ایندکس ویژگی‌های نودها
        for node, attrs in self.graph.nodes(data=True):
            for key, value in attrs.items():
                if key not in self.attribute_index["node_attributes"]:
                    self.attribute_index["node_attributes"][key] = {}
                if value not in self.attribute_index["node_attributes"][key]:
                    self.attribute_index["node_attributes"][key][value] = []
                self.attribute_index["node_attributes"][key][value].append(node)
        
        # ایندکس ویژگی‌های یال‌ها
        for u, v, attrs in self.graph.edges(data=True):
            for key, value in attrs.items():
                if key not in self.attribute_index["edge_attributes"]:
                    self.attribute_index["edge_attributes"][key] = {}
                if value not in self.attribute_index["edge_attributes"][key]:
                    self.attribute_index["edge_attributes"][key][value] = []
                self.attribute_index["edge_attributes"][key][value].append((u, v))
    
    def search_nodes(self, query: str, search_type: str = "name") -> List[str]:
        """جستجوی نودها"""
        results = []
        query_lower = query.lower()
        
        if search_type == "name":
            # جستجو بر اساس نام
            for node in self.node_index:
                if query_lower in node.lower():
                    results.append(node)
        
        elif search_type == "type":
            # جستجو بر اساس نوع
            if query_lower in self.type_index["nodes"]:
                results = self.type_index["nodes"][query_lower]
        
        elif search_type == "attribute":
            # جستجو بر اساس ویژگی
            for attr_name, attr_values in self.attribute_index["node_attributes"].items():
                for value, nodes in attr_values.items():
                    if query_lower in str(value).lower():
                        results.extend(nodes)
        
        return list(set(results))  # حذف تکرار
    
    def search_edges(self, query: str, search_type: str = "label") -> List[tuple]:
        """جستجوی یال‌ها"""
        results = []
        query_lower = query.lower()
        
        if search_type == "label":
            # جستجو بر اساس برچسب
            if query_lower in self.type_index["edges"]:
                results = self.type_index["edges"][query_lower]
        
        elif search_type == "attribute":
            # جستجو بر اساس ویژگی
            for attr_name, attr_values in self.attribute_index["edge_attributes"].items():
                for value, edges in attr_values.items():
                    if query_lower in str(value).lower():
                        results.extend(edges)
        
        return list(set(results))  # حذف تکرار
    
    def get_nodes_by_type(self, node_type: str) -> List[str]:
        """دریافت نودها بر اساس نوع"""
        return self.type_index["nodes"].get(node_type, [])
    
    def get_edges_by_type(self, edge_type: str) -> List[tuple]:
        """دریافت یال‌ها بر اساس نوع"""
        return self.type_index["edges"].get(edge_type, [])
    
    def get_node_attributes(self, node: str) -> Dict[str, Any]:
        """دریافت ویژگی‌های نود"""
        return self.node_index.get(node, {}).get("attributes", {})
    
    def get_edge_attributes(self, source: str, target: str) -> Dict[str, Any]:
        """دریافت ویژگی‌های یال"""
        edge_id = f"{source}_{target}"
        return self.edge_index.get(edge_id, {}).get("attributes", {})
    
    def get_neighbors(self, node: str) -> List[str]:
        """دریافت همسایه‌های نود"""
        if not self.graph or node not in self.graph:
            return []
        return list(self.graph.neighbors(node))
    
    def get_connected_components(self) -> List[List[str]]:
        """دریافت اجزای متصل"""
        if not self.graph:
            return []
        return list(nx.connected_components(self.graph))
    
    def get_shortest_paths(self, source: str, target: str, max_paths: int = 5) -> List[List[str]]:
        """دریافت کوتاه‌ترین مسیرها"""
        if not self.graph or source not in self.graph or target not in self.graph:
            return []
        
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=10))
            paths.sort(key=len)  # مرتب‌سازی بر اساس طول
            return paths[:max_paths]
        except Exception as e:
            logging.warning(f"Error finding shortest paths: {e}")
            return []
    
    def get_subgraph(self, nodes: List[str]) -> nx.Graph:
        """دریافت زیرگراف"""
        if not self.graph:
            return nx.Graph()
        
        # اضافه کردن نودهای همسایه تا عمق 1
        all_nodes = set(nodes)
        for node in nodes:
            if node in self.graph:
                all_nodes.update(self.graph.neighbors(node))
        
        return self.graph.subgraph(all_nodes)
    
    def save_index(self, filepath: str = None):
        """ذخیره ایندکس"""
        if filepath is None:
            filepath = self.index_path
        
        if filepath is None:
            logging.warning("No filepath provided for saving index")
            return
        
        try:
            index_data = {
                "node_index": self.node_index,
                "edge_index": self.edge_index,
                "type_index": self.type_index,
                "attribute_index": self.attribute_index
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(index_data, f)
            
            logging.info(f"Index saved to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving index: {e}")
    
    def load_index(self, filepath: str = None):
        """بارگذاری ایندکس"""
        if filepath is None:
            filepath = self.index_path
        
        if filepath is None:
            logging.warning("No filepath provided for loading index")
            return
        
        try:
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            
            self.node_index = index_data.get("node_index", {})
            self.edge_index = index_data.get("edge_index", {})
            self.type_index = index_data.get("type_index", {})
            self.attribute_index = index_data.get("attribute_index", {})
            
            logging.info(f"Index loaded from {filepath}")
            
        except Exception as e:
            logging.error(f"Error loading index: {e}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """دریافت آمار ایندکس"""
        return {
            "num_indexed_nodes": len(self.node_index),
            "num_indexed_edges": len(self.edge_index),
            "node_types": len(self.type_index.get("nodes", {})),
            "edge_types": len(self.type_index.get("edges", {})),
            "node_attributes": len(self.attribute_index.get("node_attributes", {})),
            "edge_attributes": len(self.attribute_index.get("edge_attributes", {}))
        }
    
    def clear_index(self):
        """پاک کردن ایندکس"""
        self.node_index.clear()
        self.edge_index.clear()
        self.type_index.clear()
        self.attribute_index.clear() 
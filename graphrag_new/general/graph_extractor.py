# -*- coding: utf-8 -*-
"""
Graph Extractor - استخراج گراف
"""
import logging
from typing import Dict, List, Any, Optional
import networkx as nx
from .extractor import BaseExtractor

class GraphExtractor(BaseExtractor):
    """کلاس برای استخراج اطلاعات از گراف"""
    
    def __init__(self, graph: nx.Graph = None):
        super().__init__(graph)
        self.extraction_methods = {
            'basic': self._extract_basic_info,
            'centrality': self._extract_centrality_info,
            'communities': self._extract_community_info,
            'paths': self._extract_path_info,
            'subgraphs': self._extract_subgraph_info
        }
    
    def extract(self, methods: List[str] = None, **kwargs) -> Dict[str, Any]:
        """استخراج اطلاعات از گراف"""
        if not self.graph:
            logging.warning("No graph provided for extraction")
            return {}
        
        if methods is None:
            methods = ['basic']
        
        extracted_data = {}
        
        for method in methods:
            if method in self.extraction_methods:
                try:
                    method_data = self.extraction_methods[method](**kwargs)
                    extracted_data[method] = method_data
                except Exception as e:
                    logging.error(f"Error in extraction method {method}: {e}")
                    extracted_data[method] = {}
            else:
                logging.warning(f"Unknown extraction method: {method}")
        
        self.extracted_data = extracted_data
        return extracted_data
    
    def _extract_basic_info(self, **kwargs) -> Dict[str, Any]:
        """استخراج اطلاعات پایه گراف"""
        return {
            "graph_info": self.get_graph_info(),
            "nodes": self._extract_nodes_info(),
            "edges": self._extract_edges_info(),
            "statistics": self._calculate_statistics()
        }
    
    def _extract_nodes_info(self) -> List[Dict[str, Any]]:
        """استخراج اطلاعات نودها"""
        nodes_info = []
        for node, attrs in self.graph.nodes(data=True):
            node_info = {
                "id": node,
                "type": attrs.get('type', 'unknown'),
                "degree": self.graph.degree(node),
                "attributes": dict(attrs)
            }
            nodes_info.append(node_info)
        return nodes_info
    
    def _extract_edges_info(self) -> List[Dict[str, Any]]:
        """استخراج اطلاعات یال‌ها"""
        edges_info = []
        for u, v, attrs in self.graph.edges(data=True):
            edge_info = {
                "source": u,
                "target": v,
                "label": attrs.get('label', 'unknown'),
                "attributes": dict(attrs)
            }
            edges_info.append(edge_info)
        return edges_info
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """محاسبه آمار گراف"""
        if not self.graph:
            return {}
        
        try:
            stats = {
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "is_connected": nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
                "num_components": nx.number_connected_components(self.graph),
                "average_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
            }
            
            # محاسبه قطر گراف (اگر متصل باشد)
            if stats["is_connected"]:
                try:
                    stats["diameter"] = nx.diameter(self.graph)
                except:
                    stats["diameter"] = None
            
            return stats
        except Exception as e:
            logging.warning(f"Error calculating statistics: {e}")
            return {}
    
    def _extract_centrality_info(self, **kwargs) -> Dict[str, Any]:
        """استخراج اطلاعات مرکزیت"""
        centrality_types = kwargs.get('centrality_types', ['degree', 'betweenness', 'closeness'])
        centrality_data = {}
        
        for centrality_type in centrality_types:
            try:
                if centrality_type == "degree":
                    centrality_data[centrality_type] = dict(self.graph.degree())
                elif centrality_type == "betweenness":
                    centrality_data[centrality_type] = nx.betweenness_centrality(self.graph)
                elif centrality_type == "closeness":
                    centrality_data[centrality_type] = nx.closeness_centrality(self.graph)
                elif centrality_type == "eigenvector":
                    centrality_data[centrality_type] = nx.eigenvector_centrality(self.graph, max_iter=1000)
                elif centrality_type == "pagerank":
                    centrality_data[centrality_type] = nx.pagerank(self.graph)
            except Exception as e:
                logging.warning(f"Error calculating {centrality_type} centrality: {e}")
                centrality_data[centrality_type] = {}
        
        return centrality_data
    
    def _extract_community_info(self, **kwargs) -> Dict[str, Any]:
        """استخراج اطلاعات جامعه‌ها"""
        try:
            # استفاده از الگوریتم Louvain برای تشخیص جامعه‌ها
            from community import best_partition
            partition = best_partition(self.graph)
            
            # گروه‌بندی نودها بر اساس جامعه
            communities = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)
            
            # محاسبه آمار جامعه‌ها
            community_stats = []
            for community_id, nodes in communities.items():
                subgraph = self.graph.subgraph(nodes)
                stats = {
                    "community_id": community_id,
                    "num_nodes": len(nodes),
                    "num_edges": subgraph.number_of_edges(),
                    "density": nx.density(subgraph),
                    "nodes": nodes
                }
                community_stats.append(stats)
            
            return {
                "communities": communities,
                "community_stats": community_stats,
                "num_communities": len(communities)
            }
        except ImportError:
            logging.warning("python-louvain not installed, skipping community detection")
            return {}
        except Exception as e:
            logging.warning(f"Error in community detection: {e}")
            return {}
    
    def _extract_path_info(self, **kwargs) -> Dict[str, Any]:
        """استخراج اطلاعات مسیرها"""
        max_paths = kwargs.get('max_paths', 10)
        max_length = kwargs.get('max_length', 5)
        
        path_info = {
            "shortest_paths": {},
            "all_paths": {},
            "path_statistics": {}
        }
        
        # محاسبه کوتاه‌ترین مسیرها
        try:
            shortest_paths = dict(nx.all_pairs_shortest_path(self.graph, cutoff=max_length))
            path_info["shortest_paths"] = shortest_paths
        except Exception as e:
            logging.warning(f"Error calculating shortest paths: {e}")
        
        # محاسبه تمام مسیرهای ساده
        try:
            all_paths = {}
            nodes = list(self.graph.nodes())
            for i, source in enumerate(nodes[:max_paths]):
                for target in nodes[i+1:max_paths]:
                    paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
                    if paths:
                        all_paths[(source, target)] = paths[:5]  # حداکثر 5 مسیر
        except Exception as e:
            logging.warning(f"Error calculating all paths: {e}")
        
        path_info["all_paths"] = all_paths
        
        return path_info
    
    def _extract_subgraph_info(self, **kwargs) -> Dict[str, Any]:
        """استخراج اطلاعات زیرگراف‌ها"""
        subgraph_info = {
            "node_type_subgraphs": {},
            "edge_type_subgraphs": {},
            "neighborhood_subgraphs": {}
        }
        
        # زیرگراف‌ها بر اساس نوع نود
        node_types = set()
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            node_types.add(node_type)
        
        for node_type in node_types:
            nodes = self.find_nodes_by_type(node_type)
            if nodes:
                subgraph = self.graph.subgraph(nodes)
                subgraph_info["node_type_subgraphs"][node_type] = {
                    "num_nodes": subgraph.number_of_nodes(),
                    "num_edges": subgraph.number_of_edges(),
                    "nodes": list(subgraph.nodes())
                }
        
        # زیرگراف‌ها بر اساس نوع یال
        edge_types = set()
        for u, v, attrs in self.graph.edges(data=True):
            edge_type = attrs.get('label', 'unknown')
            edge_types.add(edge_type)
        
        for edge_type in edge_types:
            edges = self.find_edges_by_type(edge_type)
            if edges:
                nodes = set()
                for u, v in edges:
                    nodes.add(u)
                    nodes.add(v)
                subgraph = self.graph.subgraph(nodes)
                subgraph_info["edge_type_subgraphs"][edge_type] = {
                    "num_nodes": subgraph.number_of_nodes(),
                    "num_edges": subgraph.number_of_edges(),
                    "edges": edges
                }
        
        return subgraph_info
    
    def extract_for_query(self, query: str, max_nodes: int = 50) -> Dict[str, Any]:
        """استخراج اطلاعات مرتبط با کوئری"""
        # این متد می‌تواند برای استخراج اطلاعات مرتبط با کوئری خاص استفاده شود
        # فعلاً یک پیاده‌سازی ساده ارائه می‌دهیم
        
        extracted_data = {
            "query": query,
            "relevant_nodes": [],
            "relevant_edges": [],
            "subgraph": None
        }
        
        # یافتن نودهای مرتبط با کوئری (بر اساس نام)
        query_lower = query.lower()
        relevant_nodes = []
        
        for node in self.graph.nodes():
            if query_lower in node.lower():
                relevant_nodes.append(node)
        
        if relevant_nodes:
            # محدود کردن تعداد نودها
            relevant_nodes = relevant_nodes[:max_nodes]
            
            # ایجاد زیرگراف از نودهای مرتبط
            subgraph = self.get_subgraph(relevant_nodes)
            
            extracted_data["relevant_nodes"] = relevant_nodes
            extracted_data["subgraph"] = {
                "num_nodes": subgraph.number_of_nodes(),
                "num_edges": subgraph.number_of_edges(),
                "nodes": list(subgraph.nodes()),
                "edges": list(subgraph.edges())
            }
        
        return extracted_data 
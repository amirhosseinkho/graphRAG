#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
دیباگ گراف
"""

import sys
import os

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def debug_graph():
    """دیباگ گراف"""
    
    print("🔍 دیباگ گراف")
    print("=" * 30)
    
    # ایجاد سرویس
    service = GraphRAGService()
    service.initialize()
    
    print(f"تعداد نودها: {service.G.number_of_nodes()}")
    print(f"تعداد یال‌ها: {service.G.number_of_edges()}")
    
    print("\nنودهای گراف:")
    for node_id, attrs in service.G.nodes(data=True):
        print(f"  {node_id}: {attrs['name']} ({attrs['kind']})")
    
    print("\nیال‌های گراف:")
    for source, target, attrs in service.G.edges(data=True):
        print(f"  {source} -> {target}: {attrs}")
    
    # تست جستجوی داروهای دیابت
    print("\n🔍 تست جستجوی داروهای دیابت:")
    
    # یافتن نود دیابت
    diabetes_node = None
    for node_id, attrs in service.G.nodes(data=True):
        if 'diabetes' in attrs['name'].lower():
            diabetes_node = node_id
            break
    
    if diabetes_node:
        print(f"نود دیابت یافت شد: {diabetes_node}")
        
        # یافتن همسایه‌های دیابت
        neighbors = list(service.G.neighbors(diabetes_node))
        print(f"همسایه‌های دیابت: {neighbors}")
        
        for neighbor in neighbors:
            neighbor_attrs = service.G.nodes[neighbor]
            edge_data = service.G.get_edge_data(diabetes_node, neighbor)
            print(f"  همسایه: {neighbor} ({neighbor_attrs['kind']})")
            print(f"  یال: {edge_data}")
            
            # بررسی معکوس
            reverse_edge = service.G.get_edge_data(neighbor, diabetes_node)
            print(f"  یال معکوس: {reverse_edge}")
    else:
        print("نود دیابت یافت نشد!")

if __name__ == "__main__":
    debug_graph() 
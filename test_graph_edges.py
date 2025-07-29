#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تست بررسی یال‌های گراف
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def test_graph_edges():
    """تست بررسی یال‌های گراف"""
    print("🔍 تست بررسی یال‌های گراف")
    
    # ایجاد سرویس
    service = GraphRAGService()
    service.initialize()
    
    # بررسی نودهای آناتومی
    print("\n📋 نودهای آناتومی:")
    anatomy_nodes = [(nid, attrs) for nid, attrs in service.G.nodes(data=True) 
                    if attrs.get('kind') == 'Anatomy' or attrs.get('metanode') == 'Anatomy']
    for nid, attrs in anatomy_nodes:
        print(f"  • {attrs['name']} (ID: {nid})")
    
    # بررسی نودهای ژن
    print("\n📋 نودهای ژن:")
    gene_nodes = [(nid, attrs) for nid, attrs in service.G.nodes(data=True) 
                 if attrs.get('kind') == 'Gene' or attrs.get('metanode') == 'Gene']
    for nid, attrs in gene_nodes[:10]:  # فقط 10 تا اول
        print(f"  • {attrs['name']} (ID: {nid})")
    
    # بررسی یال‌های AeG
    print("\n🔗 یال‌های AeG (Anatomy → expresses → Gene):")
    aeG_edges = [(u, v, data) for u, v, data in service.G.edges(data=True) 
                if data.get('relation') == 'AeG']
    
    if aeG_edges:
        print(f"  ✅ {len(aeG_edges)} یال AeG یافت شد:")
        for u, v, data in aeG_edges[:5]:  # فقط 5 تا اول
            u_name = service.G.nodes[u]['name']
            v_name = service.G.nodes[v]['name']
            print(f"    • {u_name} → {v_name} (AeG)")
    else:
        print("  ❌ هیچ یال AeG یافت نشد!")
    
    # بررسی یال‌های AuG
    print("\n🔗 یال‌های AuG (Anatomy → upregulates → Gene):")
    auG_edges = [(u, v, data) for u, v, data in service.G.edges(data=True) 
                if data.get('relation') == 'AuG']
    
    if auG_edges:
        print(f"  ✅ {len(auG_edges)} یال AuG یافت شد:")
        for u, v, data in auG_edges[:3]:
            u_name = service.G.nodes[u]['name']
            v_name = service.G.nodes[v]['name']
            print(f"    • {u_name} → {v_name} (AuG)")
    else:
        print("  ❌ هیچ یال AuG یافت نشد!")
    
    # بررسی یال‌های AdG
    print("\n🔗 یال‌های AdG (Anatomy → downregulates → Gene):")
    adG_edges = [(u, v, data) for u, v, data in service.G.edges(data=True) 
                if data.get('relation') == 'AdG']
    
    if adG_edges:
        print(f"  ✅ {len(adG_edges)} یال AdG یافت شد:")
        for u, v, data in adG_edges[:3]:
            u_name = service.G.nodes[u]['name']
            v_name = service.G.nodes[v]['name']
            print(f"    • {u_name} → {v_name} (AdG)")
    else:
        print("  ❌ هیچ یال AdG یافت نشد!")
    
    # بررسی یال‌های مربوط به قلب
    print("\n💓 یال‌های مربوط به قلب:")
    heart_edges = []
    for u, v, data in service.G.edges(data=True):
        u_name = service.G.nodes[u]['name'].lower()
        v_name = service.G.nodes[v]['name'].lower()
        if 'heart' in u_name or 'heart' in v_name:
            heart_edges.append((u, v, data))
    
    if heart_edges:
        print(f"  ✅ {len(heart_edges)} یال مربوط به قلب یافت شد:")
        for u, v, data in heart_edges:
            u_name = service.G.nodes[u]['name']
            v_name = service.G.nodes[v]['name']
            relation = data.get('relation', 'Unknown')
            print(f"    • {u_name} → {v_name} ({relation})")
    else:
        print("  ❌ هیچ یال مربوط به قلب یافت نشد!")
    
    # بررسی همسایه‌های قلب
    print("\n🔍 همسایه‌های نود قلب:")
    heart_node = None
    for nid, attrs in service.G.nodes(data=True):
        if 'heart' in attrs['name'].lower():
            heart_node = nid
            break
    
    if heart_node:
        print(f"  ✅ نود قلب یافت شد: {service.G.nodes[heart_node]['name']}")
        neighbors = list(service.G.neighbors(heart_node))
        print(f"  📊 {len(neighbors)} همسایه:")
        for neighbor in neighbors[:5]:
            neighbor_name = service.G.nodes[neighbor]['name']
            edge_data = service.G.get_edge_data(heart_node, neighbor)
            relation = edge_data.get('relation', 'Unknown') if edge_data else 'Unknown'
            print(f"    • {neighbor_name} (رابطه: {relation})")
    else:
        print("  ❌ نود قلب یافت نشد!")
    
    print("\n✅ تست بررسی یال‌های گراف تکمیل شد!")

if __name__ == "__main__":
    test_graph_edges() 
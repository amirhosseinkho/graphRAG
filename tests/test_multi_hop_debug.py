#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تست دیباگ برای جستجوی چندمرحله‌ای
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def test_multi_hop_debug():
    """تست دیباگ برای جستجوی چندمرحله‌ای"""
    print("🔍 تست دیباگ جستجوی چندمرحله‌ای")
    
    # ایجاد سرویس
    service = GraphRAGService()
    service.initialize()
    
    # بررسی یال‌های CuG
    print("\n📋 بررسی یال‌های CuG:")
    cuG_edges = [(u, v, data) for u, v, data in service.G.edges(data=True) 
                if data.get('relation') == 'CuG']
    
    if cuG_edges:
        print(f"  ✅ {len(cuG_edges)} یال CuG یافت شد:")
        for u, v, data in cuG_edges[:5]:
            u_name = service.G.nodes[u]['name']
            v_name = service.G.nodes[v]['name']
            print(f"    • {u_name} → {v_name} (CuG)")
    else:
        print("  ❌ هیچ یال CuG یافت نشد!")
    
    # بررسی یال‌های CdG
    print("\n📋 بررسی یال‌های CdG:")
    cdG_edges = [(u, v, data) for u, v, data in service.G.edges(data=True) 
                if data.get('relation') == 'CdG']
    
    if cdG_edges:
        print(f"  ✅ {len(cdG_edges)} یال CdG یافت شد:")
        for u, v, data in cdG_edges[:5]:
            u_name = service.G.nodes[u]['name']
            v_name = service.G.nodes[v]['name']
            print(f"    • {u_name} → {v_name} (CdG)")
    else:
        print("  ❌ هیچ یال CdG یافت نشد!")
    
    # تست مسیرهای چندمرحله‌ای
    print("\n📋 تست مسیرهای چندمرحله‌ای:")
    
    # از نود Heart شروع کن
    heart_node = 'Anatomy::Heart'
    if service.G.has_node(heart_node):
        print(f"  شروع از نود: {service.G.nodes[heart_node]['name']}")
        
        # جستجوی مسیرهای AeG → CuG
        pattern = ['AeG', 'CuG']
        paths = service._find_paths_with_pattern(heart_node, pattern, max_depth=3)
        
        print(f"  الگو: {' → '.join(pattern)}")
        print(f"  تعداد مسیرها: {len(paths)}")
        
        for i, (path, metaedges) in enumerate(paths[:3]):
            path_names = [service.G.nodes[node]['name'] for node in path]
            print(f"    مسیر {i+1}: {' → '.join(path_names)}")
            print(f"    Metaedges: {' → '.join(metaedges)}")
    
    # تست جستجوی چندمرحله‌ای کامل
    print("\n📋 تست جستجوی چندمرحله‌ای کامل:")
    query = "What compounds upregulate genes expressed in the heart?"
    results = service.multi_hop_search(query, max_depth=3)
    
    print(f"  سوال: {query}")
    print(f"  تعداد نتایج: {len(results)}")
    
    for i, (node_id, depth, score, explanation, path) in enumerate(results[:5]):
        node_name = service.G.nodes[node_id]['name'] if service.G.has_node(node_id) else node_id
        print(f"    {i+1}. {node_name} (عمق: {depth}, امتیاز: {score:.2f})")
        print(f"       توضیح: {explanation}")
        print(f"       مسیر: {path}")
    
    print("\n✅ تست دیباگ جستجوی چندمرحله‌ای تکمیل شد!")

if __name__ == "__main__":
    test_multi_hop_debug() 
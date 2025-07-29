#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تست برای بررسی الگوی CdG → AeG
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def test_cdg_aeg_pattern():
    """تست برای بررسی الگوی CdG → AeG"""
    print("🔍 تست برای بررسی الگوی CdG → AeG")
    
    # ایجاد سرویس
    service = GraphRAGService()
    service.initialize()
    
    # بررسی یال‌های CdG
    print("\n📋 تمام یال‌های CdG:")
    cdG_edges = [(u, v, data) for u, v, data in service.G.edges(data=True) 
                if data.get('relation') == 'CdG']
    
    for u, v, data in cdG_edges:
        u_name = service.G.nodes[u]['name']
        v_name = service.G.nodes[v]['name']
        print(f"  • {u_name} → {v_name} (CdG)")
    
    # بررسی یال‌های AeG
    print("\n📋 تمام یال‌های AeG:")
    aeG_edges = [(u, v, data) for u, v, data in service.G.edges(data=True) 
                if data.get('relation') == 'AeG']
    
    for u, v, data in aeG_edges:
        u_name = service.G.nodes[u]['name']
        v_name = service.G.nodes[v]['name']
        print(f"  • {u_name} → {v_name} (AeG)")
    
    # بررسی مسیرهای CdG → AeG
    print("\n📋 بررسی مسیرهای CdG → AeG:")
    
    for compound_node, gene_node, _ in cdG_edges:
        compound_name = service.G.nodes[compound_node]['name']
        gene_name = service.G.nodes[gene_node]['name']
        print(f"  بررسی: {compound_name} → {gene_name} (CdG)")
        
        # بررسی یال‌های AeG از این ژن
        gene_aeG_edges = [(u, v, data) for u, v, data in service.G.edges(data=True) 
                          if data.get('relation') == 'AeG' and u == gene_node]
        
        if gene_aeG_edges:
            for u, v, data in gene_aeG_edges:
                anatomy_name = service.G.nodes[v]['name']
                print(f"    → {gene_name} → {anatomy_name} (AeG)")
                print(f"      مسیر کامل: {compound_name} → {gene_name} (CdG) → {anatomy_name} (AeG)")
        else:
            print(f"    → {gene_name} → هیچ AeG یافت نشد")
    
    # تست الگوی CdG → AeG
    print("\n📋 تست الگوی CdG → AeG:")
    pattern = ['CdG', 'AeG']
    
    # شروع از Caffeine
    caffeine_node = 'Compound::Caffeine'
    if service.G.has_node(caffeine_node):
        print(f"  شروع از نود: {service.G.nodes[caffeine_node]['name']}")
        
        # استفاده از تابع _find_paths_with_pattern
        paths = service._find_paths_with_pattern(caffeine_node, pattern, max_depth=3)
        
        print(f"  تعداد مسیرها: {len(paths)}")
        for i, (path, metaedges) in enumerate(paths):
            path_names = [service.G.nodes[node]['name'] for node in path]
            print(f"    مسیر {i+1}: {' → '.join(path_names)}")
            print(f"    Metaedges: {' → '.join(metaedges)}")
    
    print("\n✅ تست برای بررسی الگوی CdG → AeG تکمیل شد!")

if __name__ == "__main__":
    test_cdg_aeg_pattern() 
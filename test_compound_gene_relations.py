#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تست بررسی روابط Compound-Gene
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def test_compound_gene_relations():
    """تست بررسی روابط Compound-Gene"""
    print("🔍 تست بررسی روابط Compound-Gene")
    
    # ایجاد سرویس
    service = GraphRAGService()
    service.initialize()
    
    # بررسی تمام یال‌های CuG
    print("\n📋 تمام یال‌های CuG:")
    cuG_edges = [(u, v, data) for u, v, data in service.G.edges(data=True) 
                if data.get('relation') == 'CuG']
    
    for u, v, data in cuG_edges:
        u_name = service.G.nodes[u]['name']
        v_name = service.G.nodes[v]['name']
        print(f"  • {u_name} → {v_name} (CuG)")
    
    # بررسی تمام یال‌های CdG
    print("\n📋 تمام یال‌های CdG:")
    cdG_edges = [(u, v, data) for u, v, data in service.G.edges(data=True) 
                if data.get('relation') == 'CdG']
    
    for u, v, data in cdG_edges:
        u_name = service.G.nodes[u]['name']
        v_name = service.G.nodes[v]['name']
        print(f"  • {u_name} → {v_name} (CdG)")
    
    # بررسی ژن‌هایی که در قلب بیان می‌شوند
    print("\n📋 ژن‌هایی که در قلب بیان می‌شوند:")
    heart_genes = []
    for u, v, data in service.G.edges(data=True):
        if data.get('relation') == 'AeG' and 'Heart' in u:
            gene_name = service.G.nodes[v]['name']
            heart_genes.append(v)
            print(f"  • {gene_name}")
    
    # بررسی روابط Compound-Gene برای ژن‌های قلب
    print("\n📋 روابط Compound-Gene برای ژن‌های قلب:")
    for gene_node in heart_genes:
        gene_name = service.G.nodes[gene_node]['name']
        print(f"  ژن: {gene_name}")
        
        # بررسی یال‌های CuG
        cuG_compounds = []
        for u, v, data in service.G.edges(data=True):
            if data.get('relation') == 'CuG' and v == gene_node:
                compound_name = service.G.nodes[u]['name']
                cuG_compounds.append(compound_name)
        
        if cuG_compounds:
            print(f"    CuG: {', '.join(cuG_compounds)}")
        else:
            print(f"    CuG: هیچ")
        
        # بررسی یال‌های CdG
        cdG_compounds = []
        for u, v, data in service.G.edges(data=True):
            if data.get('relation') == 'CdG' and v == gene_node:
                compound_name = service.G.nodes[u]['name']
                cdG_compounds.append(compound_name)
        
        if cdG_compounds:
            print(f"    CdG: {', '.join(cdG_compounds)}")
        else:
            print(f"    CdG: هیچ")
    
    # بررسی مسیرهای واقعی Heart → Gene → Compound
    print("\n📋 مسیرهای واقعی Heart → Gene → Compound:")
    for gene_node in heart_genes:
        gene_name = service.G.nodes[gene_node]['name']
        
        # بررسی یال‌های CuG
        for u, v, data in service.G.edges(data=True):
            if data.get('relation') == 'CuG' and v == gene_node:
                compound_name = service.G.nodes[u]['name']
                print(f"  Heart → {gene_name} (AeG) → {compound_name} (CuG)")
        
        # بررسی یال‌های CdG
        for u, v, data in service.G.edges(data=True):
            if data.get('relation') == 'CdG' and v == gene_node:
                compound_name = service.G.nodes[u]['name']
                print(f"  Heart → {gene_name} (AeG) → {compound_name} (CdG)")
    
    print("\n✅ تست بررسی روابط Compound-Gene تکمیل شد!")

if __name__ == "__main__":
    test_compound_gene_relations() 
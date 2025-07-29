#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست ساده بدون spaCy برای بررسی تغییرات
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_basic_functionality():
    """تست عملکرد پایه"""
    print("🧪 تست عملکرد پایه")
    print("=" * 50)
    
    # ایجاد سرویس
    service = GraphRAGService()
    
    # تست 1: بررسی گراف
    print(f"\n📊 اطلاعات گراف:")
    print(f"  تعداد نودها: {service.G.number_of_nodes()}")
    print(f"  تعداد یال‌ها: {service.G.number_of_edges()}")
    
    # نمایش چند نود نمونه
    print(f"\n🔍 نمونه نودها:")
    node_count = 0
    for node_id, attrs in service.G.nodes(data=True):
        if node_count < 5:
            print(f"  {node_id}: {attrs.get('name', 'Unknown')} ({attrs.get('kind', 'Unknown')})")
            node_count += 1
    
    # نمایش چند یال نمونه
    print(f"\n🔗 نمونه یال‌ها:")
    edge_count = 0
    for source, target, attrs in service.G.edges(data=True):
        if edge_count < 5:
            print(f"  {source} → {target} ({attrs.get('metaedge', 'Unknown')})")
            edge_count += 1
    
    # تست 2: تطبیق توکن‌ها
    print(f"\n🔍 تست تطبیق توکن‌ها:")
    test_tokens = ['heart', 'genes', 'brain', 'disease']
    for token in test_tokens:
        matched = service.match_tokens_to_nodes([token])
        if matched:
            for token_name, node_id in matched.items():
                node_name = service.G.nodes[node_id]['name']
                node_kind = service.G.nodes[node_id]['kind']
                print(f"  '{token}' → {node_name} ({node_kind})")
        else:
            print(f"  '{token}' → تطبیق یافت نشد")
    
    # تست 3: جستجوی مستقیم AeG
    print(f"\n🔍 تست جستجوی مستقیم AeG:")
    heart_nodes = []
    for node_id, attrs in service.G.nodes(data=True):
        if attrs.get('kind') == 'Anatomy' and 'heart' in attrs.get('name', '').lower():
            heart_nodes.append(node_id)
    
    if heart_nodes:
        print(f"  نودهای قلب یافت شد: {heart_nodes}")
        for heart_node in heart_nodes:
            print(f"  بررسی نود: {service.G.nodes[heart_node]['name']}")
            aeG_genes = []
            for neighbor in service.G.neighbors(heart_node):
                if service.G.nodes[neighbor]['kind'] == 'Gene':
                    edge_data = service.G.get_edge_data(heart_node, neighbor)
                    if edge_data and edge_data.get('metaedge') == 'AeG':
                        aeG_genes.append(neighbor)
            
            if aeG_genes:
                print(f"    ژن‌های AeG یافت شد: {len(aeG_genes)}")
                for gene_id in aeG_genes:
                    gene_name = service.G.nodes[gene_id]['name']
                    print(f"      🧬 {gene_name}")
            else:
                print(f"    هیچ ژن AeG یافت نشد")
    else:
        print(f"  هیچ نود قلب یافت نشد")
    
    print(f"\n" + "=" * 50)
    print("✅ تست پایه کامل شد")

if __name__ == "__main__":
    print("🚀 شروع تست ساده")
    test_basic_functionality()
    print("\n🎉 تست کامل شد!") 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
دیباگ بازیابی TP53
"""

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def debug_tp53_retrieval():
    """دیباگ بازیابی TP53"""
    print("🔍 دیباگ بازیابی TP53...")
    
    # ایجاد سرویس
    service = GraphRAGService()
    
    # سوال تست
    query = "How does TP53 relate to cancer?"
    print(f"🔍 سوال: {query}")
    
    # بررسی استخراج کلمات کلیدی
    keywords = service.extract_keywords(query)
    print(f"🔑 کلمات کلیدی استخراج شده: {keywords}")
    
    # بررسی تطبیق توکن‌ها
    matched_nodes = service.match_tokens_to_nodes(keywords)
    print(f"🎯 نودهای تطبیق یافته: {matched_nodes}")
    
    # بررسی همه نودهای ژن در گراف
    print("\n🔍 بررسی همه ژن‌های موجود در گراف:")
    gene_nodes = []
    for node_id, attrs in service.G.nodes(data=True):
        if attrs.get('kind') == 'Gene':
            gene_nodes.append((node_id, attrs['name']))
    
    print(f"📊 تعداد کل ژن‌ها: {len(gene_nodes)}")
    
    # جستجوی TP53 در گراف
    tp53_found = False
    for node_id, name in gene_nodes:
        if 'TP53' in name.upper() or 'P53' in name.upper():
            print(f"✅ TP53 یافت شد: {name} (ID: {node_id})")
            tp53_found = True
    
    if not tp53_found:
        print("❌ TP53 در گراف یافت نشد!")
        print("🔍 جستجوی مشابه:")
        for node_id, name in gene_nodes:
            if any(keyword in name.upper() for keyword in ['TUMOR', 'P53', 'SUPPRESSOR']):
                print(f"  • {name} (ID: {node_id})")
    
    # بررسی روابط سرطان
    print("\n🔍 بررسی روابط سرطان:")
    cancer_nodes = []
    for node_id, attrs in service.G.nodes(data=True):
        if attrs.get('kind') == 'Disease':
            name_lower = attrs['name'].lower()
            if any(keyword in name_lower for keyword in ['cancer', 'tumor', 'malignancy']):
                cancer_nodes.append((node_id, attrs['name']))
    
    print(f"📊 سرطان‌های یافت شده: {len(cancer_nodes)}")
    for node_id, name in cancer_nodes:
        print(f"  • {name} (ID: {node_id})")
    
    # تست جستجوی مستقیم TP53
    print("\n🔍 تست جستجوی مستقیم TP53:")
    if tp53_found:
        for node_id, name in gene_nodes:
            if 'TP53' in name.upper():
                print(f"🔍 بررسی همسایه‌های {name}:")
                for neighbor in service.G.neighbors(node_id):
                    neighbor_attrs = service.G.nodes[neighbor]
                    edge_data = service.G.get_edge_data(node_id, neighbor)
                    print(f"  • {neighbor_attrs['name']} ({neighbor_attrs.get('kind', 'Unknown')}) - {edge_data.get('metaedge', 'Unknown')}")
    
    return {
        'keywords': keywords,
        'matched_nodes': matched_nodes,
        'gene_nodes': gene_nodes,
        'cancer_nodes': cancer_nodes,
        'tp53_found': tp53_found
    }

if __name__ == "__main__":
    debug_tp53_retrieval() 
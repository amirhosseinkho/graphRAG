#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Runner for GraphRAG Tests
"""

import sys
import os
from pathlib import Path

# اضافه کردن مسیر اصلی پروژه به sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# حالا می‌توانیم import کنیم
from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def run_tp53_test():
    """اجرای تست TP53"""
    print("🧪 شروع تست TP53...")
    
    # ایجاد سرویس
    service = GraphRAGService()
    
    # سوال تست
    query = "How does TP53 relate to cancer?"
    
    # پردازش سوال
    result = service.process_query(
        query=query,
        retrieval_method=RetrievalMethod.INTELLIGENT,
        generation_model=GenerationModel.GPT_SIMULATION,
        max_depth=3
    )
    
    print("\n📊 نتایج:")
    print(f"• روش بازیابی: {result.get('retrieval_method', 'N/A')}")
    print(f"• مدل تولید: {result.get('generation_model', 'N/A')}")
    
    # بررسی نودهای بازیابی شده
    retrieved_nodes = result.get('retrieved_nodes', [])
    print(f"• تعداد نودها: {len(retrieved_nodes)}")
    
    print("\n🎯 نودهای یافت شده:")
    for node in retrieved_nodes:
        print(f"  • {node['name']} ({node['kind']}) - امتیاز: {node.get('score', 'N/A')}")
    
    # بررسی یال‌های بازیابی شده
    retrieved_edges = result.get('retrieved_edges', [])
    print(f"• تعداد یال‌ها: {len(retrieved_edges)}")
    
    if retrieved_edges:
        print("\n🔗 یال‌های یافت شده:")
        for edge in retrieved_edges:
            print(f"  • {edge['source']} → {edge['target']} ({edge['relation']})")
    
    # بررسی مسیرها
    paths = result.get('paths', [])
    if paths:
        print(f"\n🛤️ مسیرهای یافت شده: {len(paths)}")
        for i, path in enumerate(paths[:3]):  # فقط 3 مسیر اول
            print(f"  {i+1}. {' → '.join(path)}")
    
    # بررسی متن زمینه
    context_text = result.get('context_text', '')
    if context_text:
        print(f"\n📝 متن زمینه (اول 200 کاراکتر):")
        print(f"  {context_text[:200]}...")
    
    # بررسی پاسخ
    answer = result.get('answer', '')
    if answer:
        print(f"\n🤖 پاسخ تولید شده:")
        print(answer)
    
    # بررسی اطمینان
    confidence = result.get('confidence', 0)
    print(f"\n🎯 سطح اطمینان: {confidence}")
    
    # بررسی مراحل پردازش
    process_steps = result.get('process_steps', [])
    if process_steps:
        print(f"\n📋 مراحل پردازش:")
        for step in process_steps:
            print(f"  • {step}")
    
    # بررسی کلمات کلیدی
    keywords = result.get('keywords', [])
    print(f"\n🔑 کلمات کلیدی: {keywords}")
    
    # بررسی نودهای تطبیق یافته
    matched_nodes = result.get('matched_nodes', {})
    print(f"\n🎯 نودهای تطبیق یافته: {matched_nodes}")
    
    return result

def run_debug_tp53():
    """اجرای دیباگ TP53"""
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
    
    return {
        'keywords': keywords,
        'matched_nodes': matched_nodes,
        'gene_nodes': gene_nodes,
        'tp53_found': tp53_found
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "tp53":
            run_tp53_test()
        elif test_type == "debug":
            run_debug_tp53()
        else:
            print("استفاده: python test_runner.py [tp53|debug]")
    else:
        # اجرای تست TP53 به صورت پیش‌فرض
        run_tp53_test() 
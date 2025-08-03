#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست اصلاحات TP53
"""

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_tp53_query():
    """تست سوال TP53"""
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

if __name__ == "__main__":
    test_tp53_query()
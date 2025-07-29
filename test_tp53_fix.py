#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست بهبودهای TP53
"""

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_tp53_query():
    """تست سوال TP53"""
    print("🧪 شروع تست TP53...")
    
    # ایجاد سرویس
    service = GraphRAGService()
    
    # سوال تست
    query = "How does TP53 relate to cancer?"
    print(f"🔍 سوال: {query}")
    
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
    
    # بررسی ساختار نتیجه
    if 'retrieval_result' in result:
        retrieval_result = result['retrieval_result']
        print(f"• تعداد نودها: {len(retrieval_result.nodes)}")
        print(f"• تعداد یال‌ها: {len(retrieval_result.edges)}")
        
        print("\n🎯 نودهای یافت شده:")
        for node in retrieval_result.nodes:
            print(f"  • {node.name} ({node.kind})")
        
        print("\n🔗 یال‌های یافت شده:")
        for edge in retrieval_result.edges:
            source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
            target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
            print(f"  • {source_name} → {target_name} ({edge.relation})")
    else:
        print("❌ نتیجه بازیابی در دسترس نیست")
    
    if 'generation_result' in result:
        print("\n🤖 پاسخ تولید شده:")
        print(result['generation_result'].answer)
    else:
        print("❌ نتیجه تولید در دسترس نیست")
    
    # نمایش کل نتیجه برای دیباگ
    print("\n🔍 کل نتیجه:")
    for key, value in result.items():
        if key not in ['retrieval_result', 'generation_result']:
            print(f"  • {key}: {value}")
    
    return result

if __name__ == "__main__":
    test_tp53_query()
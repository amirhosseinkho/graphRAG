#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست ساده متن زمینه
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService, RetrievalMethod

def test_simple_context():
    """تست ساده متن زمینه"""
    print("🧪 تست ساده متن زمینه")
    print("=" * 50)
    
    # ایجاد سرویس
    service = GraphRAGService()
    
    # تست سوال ساده
    query = "What genes are expressed in heart?"
    
    print(f"🔍 سوال: {query}")
    
    # بازیابی اطلاعات
    retrieval_result = service.retrieve_information(
        query=query,
        method=RetrievalMethod.INTELLIGENT,
        max_depth=2
    )
    
    print(f"📊 نتایج بازیابی:")
    print(f"  • نودها: {len(retrieval_result.nodes)}")
    print(f"  • یال‌ها: {len(retrieval_result.edges)}")
    print(f"  • مسیرها: {len(retrieval_result.paths)}")
    
    # تست متن زمینه ساده
    print(f"\n📝 متن زمینه ساده:")
    print("-" * 30)
    
    simple_context = service._create_simple_context_text(retrieval_result)
    
    # نمایش خلاصه متن زمینه
    lines = simple_context.split('\n')
    print(f"  طول متن: {len(lines)} خط")
    print(f"  کاراکترها: {len(simple_context)}")
    
    # نمایش چند خط اول
    for j, line in enumerate(lines[:8]):
        if line.strip():
            print(f"  {j+1}: {line[:70]}{'...' if len(line) > 70 else ''}")
    
    if len(lines) > 8:
        print(f"  ... و {len(lines) - 8} خط دیگر")
    
    print("✅ تست ساده متن زمینه کامل شد")

if __name__ == "__main__":
    test_simple_context()
    print("\n🎉 تست کامل شد!") 
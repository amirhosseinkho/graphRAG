#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تست ساده برای بررسی تطبیق نودها
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def test_node_matching():
    """تست تطبیق نودها"""
    print("🔍 تست تطبیق نودها")
    
    # ایجاد سرویس
    service = GraphRAGService()
    service.initialize()
    
    # تست 1: تطبیق مستقیم
    print("\n📋 تست 1: تطبیق مستقیم")
    tokens = ["heart", "genes"]
    matched = service.match_tokens_to_nodes(tokens)
    print(f"توکن‌ها: {tokens}")
    print(f"نتیجه: {matched}")
    
    # تست 2: استخراج کلمات کلیدی
    print("\n📋 تست 2: استخراج کلمات کلیدی")
    query = "What genes are expressed in the heart?"
    keywords = service.extract_keywords(query)
    print(f"سوال: {query}")
    print(f"کلمات کلیدی: {keywords}")
    
    # تست 3: تحلیل قصد سوال
    print("\n📋 تست 3: تحلیل قصد سوال")
    intent = service.analyze_question_intent(query)
    print(f"نوع سوال: {intent['question_type']}")
    print(f"metaedges: {intent['metaedges']}")
    print(f"موجودیت‌ها: {intent['entities']}")
    print(f"کلمات کلیدی: {intent['keywords']}")
    
    # تست 4: تطبیق با کلمات کلیدی استخراج شده
    print("\n📋 تست 4: تطبیق با کلمات کلیدی استخراج شده")
    matched_from_keywords = service.match_tokens_to_nodes(intent['keywords'])
    print(f"نتیجه تطبیق: {matched_from_keywords}")
    
    # تست 5: جستجوی معنایی هوشمند
    print("\n📋 تست 5: جستجوی معنایی هوشمند")
    results = service.intelligent_semantic_search(query, max_depth=2)
    print(f"تعداد نتایج: {len(results)}")
    for i, (node_id, depth, score, explanation) in enumerate(results[:5]):
        node_name = service.G.nodes[node_id]['name'] if service.G.has_node(node_id) else node_id
        print(f"  {i+1}. {node_name} (عمق: {depth}, امتیاز: {score:.2f})")
        print(f"     توضیح: {explanation}")
    
    print("\n✅ تست تطبیق نودها تکمیل شد!")

if __name__ == "__main__":
    test_node_matching() 
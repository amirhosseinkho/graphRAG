#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست سوالات فارسی - بررسی مشکل بازیابی اطلاعات از گراف انگلیسی
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_persian_queries():
    """تست سوالات فارسی مختلف"""
    
    # راه‌اندازی سرویس
    service = GraphRAGService()
    service.initialize()
    
    # سوالات فارسی مختلف برای تست
    persian_queries = [
        "ژن TP53 چه کاری انجام می‌دهد؟",
        "کدام ژن‌ها در کبد بیان می‌شوند؟",
        "سرطان سینه با کدام ژن‌ها مرتبط است؟",
        "آسپرین چه بیماری‌هایی را درمان می‌کند؟",
        "ژن BRCA1 در کجا بیان می‌شود؟",
        "کدام داروها برای درمان دیابت استفاده می‌شوند؟",
        "ژن‌های مرتبط با سرطان ریه کدامند؟",
        "مغز چه ژن‌هایی را بیان می‌کند؟",
        "کدام ژن‌ها در فرآیند آپوپتوز شرکت دارند؟",
        "ژن APOE چه نقشی در بیماری‌ها دارد؟"
    ]
    
    print("🔍 تست سوالات فارسی")
    print("=" * 50)
    
    for i, query in enumerate(persian_queries, 1):
        print(f"\n📝 سوال {i}: {query}")
        print("-" * 40)
        
        try:
            # استخراج کلمات کلیدی
            keywords = service.extract_keywords(query)
            print(f"🔑 کلمات کلیدی استخراج شده: {keywords}")
            
            # تطبیق با نودهای گراف
            matched_nodes = service.match_tokens_to_nodes(keywords)
            print(f"🎯 نودهای تطبیق یافته: {len(matched_nodes)}")
            for token, node_id in matched_nodes.items():
                node_name = service.G.nodes[node_id]['name']
                node_kind = service.G.nodes[node_id].get('kind', 'Unknown')
                print(f"   '{token}' -> {node_name} ({node_kind})")
            
            # تحلیل قصد سوال
            intent = service.analyze_question_intent(query)
            print(f"🧠 قصد سوال: {intent.get('question_type', 'نامشخص')}")
            
            # بازیابی اطلاعات
            result = service.retrieve_information(
                query=query,
                method=RetrievalMethod.INTELLIGENT,
                max_depth=2
            )
            
            print(f"📊 تعداد نودهای بازیابی شده: {len(result.nodes)}")
            print(f"🔗 تعداد یال‌های بازیابی شده: {len(result.edges)}")
            
            if result.nodes:
                print("📋 نودهای بازیابی شده:")
                for node in result.nodes[:5]:  # فقط 5 نود اول
                    print(f"   - {node.name} ({node.kind})")
                if len(result.nodes) > 5:
                    print(f"   ... و {len(result.nodes) - 5} نود دیگر")
            else:
                print("❌ هیچ نودی بازیابی نشد!")
            
            print(f"📝 متن زمینه: {len(result.context_text)} کاراکتر")
            if result.context_text:
                print(f"   نمونه: {result.context_text[:200]}...")
            
        except Exception as e:
            print(f"❌ خطا در پردازش سوال: {e}")
        
        print()

def test_english_queries_for_comparison():
    """تست سوالات انگلیسی برای مقایسه"""
    
    service = GraphRAGService()
    service.initialize()
    
    english_queries = [
        "What does TP53 gene do?",
        "Which genes are expressed in liver?",
        "What genes are associated with breast cancer?",
        "What diseases does aspirin treat?",
        "Where is BRCA1 gene expressed?",
        "What drugs are used for diabetes treatment?",
        "What genes are related to lung cancer?",
        "What genes are expressed in brain?",
        "Which genes participate in apoptosis?",
        "What is the role of APOE gene in diseases?"
    ]
    
    print("\n🔍 تست سوالات انگلیسی (برای مقایسه)")
    print("=" * 50)
    
    for i, query in enumerate(english_queries, 1):
        print(f"\n📝 سوال {i}: {query}")
        print("-" * 40)
        
        try:
            keywords = service.extract_keywords(query)
            print(f"🔑 کلمات کلیدی: {keywords}")
            
            matched_nodes = service.match_tokens_to_nodes(keywords)
            print(f"🎯 نودهای تطبیق یافته: {len(matched_nodes)}")
            
            result = service.retrieve_information(
                query=query,
                method=RetrievalMethod.INTELLIGENT,
                max_depth=2
            )
            
            print(f"📊 نودهای بازیابی شده: {len(result.nodes)}")
            print(f"🔗 یال‌های بازیابی شده: {len(result.edges)}")
            
        except Exception as e:
            print(f"❌ خطا: {e}")
        
        print()

if __name__ == "__main__":
    print("🚀 شروع تست سوالات فارسی")
    test_persian_queries()
    test_english_queries_for_comparison()
    print("\n✅ تست کامل شد!") 
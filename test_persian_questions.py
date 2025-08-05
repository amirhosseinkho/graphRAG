#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست کامل سوالات فارسی - بررسی عملکرد سیستم با سوالات واقعی
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_persian_questions():
    """تست سوالات فارسی واقعی"""
    
    service = GraphRAGService()
    service.initialize()
    
    # سوالات فارسی مختلف
    persian_questions = [
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
    
    for i, question in enumerate(persian_questions, 1):
        print(f"\n📝 سوال {i}: {question}")
        print("-" * 40)
        
        try:
            # استخراج کلمات کلیدی
            keywords = service.extract_keywords(question)
            print(f"🔑 کلمات کلیدی: {keywords}")
            
            # تطبیق با نودهای گراف
            matched_nodes = service.match_tokens_to_nodes(keywords)
            print(f"🎯 نودهای تطبیق یافته: {len(matched_nodes)}")
            
            if matched_nodes:
                print("📋 نودهای تطبیق یافته:")
                for token, node_id in matched_nodes.items():
                    node_name = service.G.nodes[node_id]['name']
                    node_kind = service.G.nodes[node_id].get('kind', 'Unknown')
                    print(f"   '{token}' -> {node_name} ({node_kind})")
            
            # تحلیل قصد سوال
            intent = service.analyze_question_intent(question)
            print(f"🧠 قصد سوال: {intent.get('question_type', 'نامشخص')}")
            
            # بازیابی اطلاعات
            result = service.retrieve_information(
                query=question,
                method=RetrievalMethod.INTELLIGENT,
                max_depth=2
            )
            
            print(f"📊 نودهای بازیابی شده: {len(result.nodes)}")
            print(f"🔗 یال‌های بازیابی شده: {len(result.edges)}")
            
            if result.nodes:
                print("📋 نمونه نودهای بازیابی شده:")
                for node in result.nodes[:3]:  # فقط 3 نود اول
                    print(f"   - {node.name} ({node.kind})")
                if len(result.nodes) > 3:
                    print(f"   ... و {len(result.nodes) - 3} نود دیگر")
            else:
                print("❌ هیچ نودی بازیابی نشد!")
            
            print(f"📝 متن زمینه: {len(result.context_text)} کاراکتر")
            if result.context_text:
                print(f"   نمونه: {result.context_text[:150]}...")
            
        except Exception as e:
            print(f"❌ خطا در پردازش سوال: {e}")
        
        print()

def test_specific_persian_queries():
    """تست سوالات فارسی خاص"""
    
    service = GraphRAGService()
    service.initialize()
    
    # سوالات خاص برای تست
    specific_queries = [
        "سرطان",
        "کبد",
        "مغز", 
        "ژن",
        "دارو",
        "بیماری",
        "آسپرین",
        "TP53",
        "BRCA1",
        "کبد و مغز"
    ]
    
    print("\n🔍 تست کلمات کلیدی فارسی")
    print("=" * 40)
    
    for query in specific_queries:
        print(f"\n📝 کلمه/عبارت: {query}")
        print("-" * 30)
        
        try:
            keywords = service.extract_keywords(query)
            print(f"🔑 کلمات کلیدی: {keywords}")
            
            matched_nodes = service.match_tokens_to_nodes(keywords)
            print(f"🎯 نودهای تطبیق یافته: {len(matched_nodes)}")
            
            if matched_nodes:
                for token, node_id in matched_nodes.items():
                    node_name = service.G.nodes[node_id]['name']
                    node_kind = service.G.nodes[node_id].get('kind', 'Unknown')
                    print(f"   '{token}' -> {node_name} ({node_kind})")
            else:
                print("❌ هیچ نودی تطبیق نیافت!")
                
        except Exception as e:
            print(f"❌ خطا: {e}")
        
        print()

if __name__ == "__main__":
    print("🚀 شروع تست سوالات فارسی")
    test_specific_persian_queries()
    test_persian_questions()
    print("\n✅ تست کامل شد!") 
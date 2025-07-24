#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تست روش جستجوی معنایی هوشمند
"""

import json
import sys
import os

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_intelligent_search():
    """تست روش جستجوی معنایی هوشمند"""
    
    print("🧠 تست روش جستجوی معنایی هوشمند")
    print("=" * 50)
    
    # ایجاد سرویس
    service = GraphRAGService()
    service.initialize()
    
    # تنظیم API Key برای OpenAI
    service.set_openai_api_key("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")
    
    # سوالات تست مختلف
    test_questions = [
        {
            "question": "چه ژن‌هایی در قلب بیان می‌شوند؟",
            "description": "سوال آناتومی - بیان ژن"
        },
        {
            "question": "چه داروهایی برای درمان دیابت استفاده می‌شوند؟",
            "description": "سوال دارو - درمان بیماری"
        },
        {
            "question": "ژن HMGB3 چه عملکردی دارد؟",
            "description": "سوال ژن - عملکرد"
        },
        {
            "question": "چه بیماری‌هایی با ژن‌های قلبی مرتبط هستند؟",
            "description": "سوال بیماری - ارتباط با ژن"
        },
        {
            "question": "چه ترکیباتی با پروتئین‌های قلب تعامل دارند؟",
            "description": "سوال ترکیب - تعامل"
        }
    ]
    
    # مدل‌های مختلف برای تست
    models = [
        GenerationModel.OPENAI_GPT,
        GenerationModel.GPT_SIMULATION,
        GenerationModel.CUSTOM
    ]
    
    for i, test_case in enumerate(test_questions, 1):
        print(f"\n📝 تست {i}: {test_case['description']}")
        print(f"سوال: {test_case['question']}")
        print("-" * 40)
        
        for model in models:
            print(f"\n🤖 مدل: {model.value}")
            
            try:
                # پردازش با روش هوشمند
                result = service.process_query(
                    query=test_case['question'],
                    retrieval_method=RetrievalMethod.INTELLIGENT,
                    generation_model=model,
                    max_depth=3
                )
                
                # نمایش نتایج
                print(f"✅ موفقیت")
                print(f"تعداد نودهای بازیابی شده: {len(result['retrieval_result']['nodes'])}")
                print(f"تعداد یال‌های بازیابی شده: {len(result['retrieval_result']['edges'])}")
                print(f"تعداد مسیرهای یافت شده: {len(result['retrieval_result']['paths'])}")
                
                # نمایش نودهای مهم
                if result['retrieval_result']['nodes']:
                    print("\nنودهای مهم:")
                    for node in result['retrieval_result']['nodes'][:5]:  # فقط 5 نود اول
                        print(f"  - {node['name']} ({node['kind']}) [امتیاز: {node['score']:.2f}]")
                
                # نمایش بخشی از پاسخ
                answer = result['generation_result']['answer']
                if len(answer) > 200:
                    answer = answer[:200] + "..."
                print(f"\nپاسخ: {answer}")
                
            except Exception as e:
                print(f"❌ خطا: {e}")
            
            print("-" * 30)
    
    print("\n🎉 تست روش جستجوی معنایی هوشمند تکمیل شد!")

def test_question_analysis():
    """تست تحلیل سوال"""
    
    print("\n🔍 تست تحلیل سوال")
    print("=" * 30)
    
    service = GraphRAGService()
    service.initialize()
    
    test_questions = [
        "چه ژن‌هایی در قلب بیان می‌شوند؟",
        "چه داروهایی برای درمان دیابت استفاده می‌شوند؟",
        "ژن HMGB3 چه عملکردی دارد؟",
        "چه بیماری‌هایی با ژن‌های قلبی مرتبط هستند؟",
        "چه ترکیباتی با پروتئین‌های قلب تعامل دارند؟"
    ]
    
    for question in test_questions:
        print(f"\nسوال: {question}")
        
        # تحلیل سوال
        intent = service.analyze_question_intent(question)
        
        print(f"نوع سوال: {intent['question_type']}")
        print(f"موجودیت‌های اصلی: {intent['main_entities']}")
        print(f"انواع موجودیت: {intent['entity_types']}")
        print(f"روابط: {intent['relationships']}")
        print(f"الگوهای تشخیص داده شده: {intent['patterns']}")
        print(f"کلمات کلیدی: {intent['keywords'][:5]}...")  # فقط 5 کلمه اول

if __name__ == "__main__":
    print("🚀 شروع تست‌های جستجوی معنایی هوشمند")
    
    # تست تحلیل سوال
    test_question_analysis()
    
    # تست جستجوی هوشمند
    test_intelligent_search()
    
    print("\n✅ تمام تست‌ها تکمیل شد!") 
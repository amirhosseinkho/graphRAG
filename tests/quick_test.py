#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تست سریع روش جستجوی معنایی هوشمند
"""

import sys
import os

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def quick_test():
    """تست سریع روش جدید"""
    
    print("🚀 تست سریع روش جستجوی معنایی هوشمند")
    print("=" * 50)
    
    # ایجاد سرویس
    service = GraphRAGService()
    service.initialize()
    
    # تنظیم API Key برای OpenAI
    service.set_openai_api_key("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")
    
    # تست سوال داروهای دیابت
    test_question = "What drugs treat diabetes?"
    
    print(f"📝 سوال تست: {test_question}")
    print("-" * 40)
    
    try:
        # تحلیل سوال
        intent = service.analyze_question_intent(test_question)
        print(f"✅ تحلیل سوال موفق:")
        print(f"   نوع سوال: {intent['question_type']}")
        print(f"   موجودیت‌های اصلی: {intent['main_entities']}")
        print(f"   انواع موجودیت: {intent['entity_types']}")
        print(f"   الگوهای تشخیص داده شده: {intent['patterns']}")
        
        # پردازش با روش هوشمند
        print(f"\n🔍 پردازش با روش جستجوی معنایی هوشمند...")
        result = service.process_query(
            query=test_question,
            retrieval_method=RetrievalMethod.INTELLIGENT,
            generation_model=GenerationModel.OPENAI_GPT,
            max_depth=3
        )
        
        # نمایش نتایج
        print(f"✅ پردازش موفق")
        print(f"تعداد نودهای بازیابی شده: {len(result['retrieved_nodes'])}")
        print(f"تعداد یال‌های بازیابی شده: {len(result['retrieved_edges'])}")
        print(f"تعداد مسیرهای یافت شده: {len(result['paths'])}")
        
        # نمایش نودهای مهم
        if result['retrieved_nodes']:
            print(f"\nنودهای مهم:")
            for node in result['retrieved_nodes'][:5]:  # فقط 5 نود اول
                print(f"  - {node['name']} ({node['kind']}) [امتیاز: {node['score']:.2f}]")
        
        # نمایش بخشی از پاسخ
        answer = result['answer']
        print(f"\nپاسخ تولید شده:")
        print("-" * 30)
        print(answer)
        
        print(f"\n🎉 تست موفق!")
        
    except Exception as e:
        print(f"❌ خطا: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test() 
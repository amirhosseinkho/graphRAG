#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست مدل‌های تولید متن بدون استفاده از گراف
"""

import json
from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_model_only():
    """تست مدل‌های مختلف بدون بازیابی از گراف"""
    
    print("🧪 تست مدل‌های تولید متن بدون گراف")
    print("=" * 50)
    
    # ایجاد سرویس
    service = GraphRAGService()
    
    # سوالات تست
    test_questions = [
        "رابطه بین ژن HMGB3 و دیابت چیست؟",
        "آیا داروی متفورمین برای درمان دیابت موثر است؟",
        "ژن BRCA1 چه نقشی در سرطان سینه دارد؟",
        "مکانیزم عمل داروی آسپرین چیست؟",
        "بیماری آلزایمر چه علل ژنتیکی دارد؟"
    ]
    
    # مدل‌های مختلف برای تست
    models_to_test = [
        GenerationModel.HUGGINGFACE,
        GenerationModel.GPT_SIMULATION,
        GenerationModel.CUSTOM
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 سوال {i}: {question}")
        print("-" * 40)
        
        for model in models_to_test:
            print(f"\n🤖 مدل: {model.value}")
            print("-" * 20)
            
            try:
                # پردازش سوال بدون بازیابی از گراف
                result = service.process_query(
                    query=question,
                    retrieval_method=RetrievalMethod.NO_RETRIEVAL,
                    generation_model=model
                )
                
                # نمایش نتیجه
                print("📝 پاسخ:")
                print(result['answer'])
                print(f"\n🎯 اطمینان: {result['confidence']:.2f}")
                
            except Exception as e:
                print(f"❌ خطا: {e}")
            
            print("\n" + "="*30)

def test_specific_question():
    """تست یک سوال خاص با همه مدل‌ها"""
    
    print("\n🎯 تست سوال خاص با همه مدل‌ها")
    print("=" * 50)
    
    service = GraphRAGService()
    question = "ژن TP53 چه نقشی در سرطان دارد و چگونه می‌توان از آن برای درمان استفاده کرد؟"
    
    print(f"سوال: {question}")
    print("-" * 40)
    
    models = [
        GenerationModel.HUGGINGFACE,
        GenerationModel.GPT_SIMULATION,
        GenerationModel.CUSTOM
    ]
    
    results = {}
    
    for model in models:
        print(f"\n🤖 تست مدل: {model.value}")
        print("-" * 30)
        
        try:
            result = service.process_query(
                query=question,
                retrieval_method=RetrievalMethod.NO_RETRIEVAL,
                generation_model=model
            )
            
            results[model.value] = {
                'answer': result['answer'],
                'confidence': result['confidence'],
                'processing_time': result.get('processing_time', 'N/A')
            }
            
            print("✅ موفق")
            
        except Exception as e:
            print(f"❌ خطا: {e}")
            results[model.value] = {'error': str(e)}
    
    # نمایش مقایسه
    print("\n📊 مقایسه نتایج:")
    print("=" * 50)
    
    for model_name, result in results.items():
        print(f"\n🤖 {model_name}:")
        if 'error' in result:
            print(f"❌ خطا: {result['error']}")
        else:
            print(f"🎯 اطمینان: {result['confidence']:.2f}")
            print(f"⏱️ زمان: {result['processing_time']}")
            print("📝 پاسخ:")
            print(result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer'])

if __name__ == "__main__":
    print("🚀 شروع تست مدل‌های تولید متن")
    
    # تست کلی
    test_model_only()
    
    # تست سوال خاص
    test_specific_question()
    
    print("\n✅ تست‌ها کامل شد!") 
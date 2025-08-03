#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست OpenAI GPT با API Key
"""

import json
from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_openai_gpt():
    """تست OpenAI GPT"""
    
    print("🚀 تست OpenAI GPT")
    print("=" * 50)
    
    # ایجاد سرویس
    service = GraphRAGService()
    
    # تنظیم API Key
    OPENAI_API_KEY = "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"
    service.set_openai_api_key(OPENAI_API_KEY)
    
    # سوالات تست
    test_questions = [
        "ژن TP53 چه نقشی در سرطان دارد؟",
        "مکانیزم عمل داروی متفورمین چیست؟",
        "بیماری آلزایمر چه علل ژنتیکی دارد؟",
        "ژن BRCA1 چگونه با سرطان سینه مرتبط است؟",
        "آسپرین چگونه از لخته شدن خون جلوگیری می‌کند؟"
    ]
    
    print("🔍 شروع تست OpenAI GPT...")
    print("-" * 40)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 سوال {i}: {question}")
        print("-" * 30)
        
        try:
            # تست بدون گراف
            result_no_graph = service.process_query(
                query=question,
                retrieval_method=RetrievalMethod.NO_RETRIEVAL,
                generation_model=GenerationModel.OPENAI_GPT
            )
            
            print("🤖 پاسخ OpenAI GPT (بدون گراف):")
            print(result_no_graph['answer'])
            print(f"🎯 اطمینان: {result_no_graph['confidence']:.2f}")
            
            # تست با گراف (اگر موجود باشد)
            if service.G and service.G.number_of_nodes() > 0:
                print("\n" + "="*40)
                print("🔗 تست با گراف:")
                
                result_with_graph = service.process_query(
                    query=question,
                    retrieval_method=RetrievalMethod.ENSEMBLE,
                    generation_model=GenerationModel.OPENAI_GPT
                )
                
                print("🤖 پاسخ OpenAI GPT (با گراف):")
                print(result_with_graph['answer'])
                print(f"🎯 اطمینان: {result_with_graph['confidence']:.2f}")
            
        except Exception as e:
            print(f"❌ خطا: {e}")
        
        print("\n" + "="*50)

def test_openai_models():
    """تست مدل‌های مختلف OpenAI"""
    
    print("\n🧪 تست مدل‌های مختلف OpenAI")
    print("=" * 50)
    
    service = GraphRAGService()
    OPENAI_API_KEY = "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"
    service.set_openai_api_key(OPENAI_API_KEY)
    
    question = "ژن TP53 چه نقشی در سرطان دارد و چگونه می‌توان از آن برای درمان استفاده کرد؟"
    
    print(f"سوال: {question}")
    print("-" * 40)
    
    # تست مدل‌های مختلف
    models = [
        ("gpt-3.5-turbo", "GPT-3.5 Turbo (ارزان و سریع)"),
        ("gpt-4", "GPT-4 (کیفیت بهتر)"),
        ("gpt-4-turbo-preview", "GPT-4 Turbo (جدیدترین)")
    ]
    
    for model_name, description in models:
        print(f"\n🤖 تست {description}")
        print("-" * 30)
        
        try:
            # تغییر مدل در کد
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # ایجاد prompt
            prompt = f"""
            🧬 سوال پزشکی-زیستی:
            {question}
            
            📋 دستورالعمل پاسخ‌دهی:
            لطفاً بر اساس دانش تخصصی خود در زمینه علوم زیستی و پزشکی، پاسخ جامع و دقیقی ارائه دهید که شامل:
            
            1. 🎯 تحلیل موضوع: بررسی دقیق سوال و جنبه‌های مختلف آن
            2. 🔬 مبانی علمی: توضیح مکانیزم‌ها و فرآیندهای زیستی مرتبط
            3. 💊 جنبه‌های درمانی: در صورت مرتبط بودن، روش‌های درمانی و دارویی
            4. 🧪 تحقیقات: وضعیت فعلی تحقیقات و مطالعات مرتبط
            5. 🔮 چشم‌انداز آینده: مسیرهای تحقیقاتی و پیشرفت‌های آینده
            6. 💡 توصیه‌های کاربردی: نکات مهم برای پژوهشگران و پزشکان
            
            پاسخ را به صورت ساختاریافته، با استفاده از ایموجی‌ها و فرمت‌بندی مناسب ارائه دهید.
            """
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a biomedical expert. Provide detailed, accurate, and well-structured answers in Persian with proper formatting and emojis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            usage = response.usage
            
            print("✅ موفق")
            print(f"📊 استفاده توکن: {usage.total_tokens}")
            print(f"💰 هزینه تقریبی: ${usage.total_tokens * 0.000002:.6f}")
            print("📝 پاسخ:")
            print(answer[:300] + "..." if len(answer) > 300 else answer)
            
        except Exception as e:
            print(f"❌ خطا: {e}")

if __name__ == "__main__":
    print("🚀 شروع تست OpenAI GPT")
    
    # تست اصلی
    test_openai_gpt()
    
    # تست مدل‌های مختلف
    test_openai_models()
    
    print("\n✅ تست‌های OpenAI کامل شد!") 
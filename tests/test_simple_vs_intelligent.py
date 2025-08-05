#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تست مقایسه نوع تولید متن ساده و هوشمند
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_text_generation_types():
    """تست مقایسه نوع تولید متن ساده و هوشمند"""
    
    print("🧪 شروع تست مقایسه نوع تولید متن...")
    print("=" * 60)
    
    # راه‌اندازی سرویس
    service = GraphRAGService()
    service.initialize()
    
    # سوال تست
    test_query = "What genes are expressed in heart?"
    
    print(f"📝 سوال تست: {test_query}")
    print()
    
    # تست با نوع تولید متن ساده
    print("🔹 تست نوع تولید متن ساده:")
    print("-" * 40)
    
    try:
        result_simple = service.process_query(
            query=test_query,
            retrieval_method=RetrievalMethod.BFS,
            generation_model=GenerationModel.GENERAL_SIMPLE,
            text_generation_type='SIMPLE',
            max_depth=2
        )
        
        print("✅ نتیجه نوع ساده:")
        print(f"• پاسخ: {result_simple['answer'][:200]}...")
        print(f"• اطمینان: {result_simple['confidence']}")
        print(f"• نوع متن زمینه: ساده")
        
    except Exception as e:
        print(f"❌ خطا در نوع ساده: {e}")
    
    print()
    print("=" * 60)
    print()
    
    # تست با نوع تولید متن هوشمند
    print("🔹 تست نوع تولید متن هوشمند:")
    print("-" * 40)
    
    try:
        result_intelligent = service.process_query(
            query=test_query,
            retrieval_method=RetrievalMethod.BFS,
            generation_model=GenerationModel.GENERAL_SIMPLE,
            text_generation_type='INTELLIGENT',
            max_depth=2
        )
        
        print("✅ نتیجه نوع هوشمند:")
        print(f"• پاسخ: {result_intelligent['answer'][:200]}...")
        print(f"• اطمینان: {result_intelligent['confidence']}")
        print(f"• نوع متن زمینه: هوشمند")
        
    except Exception as e:
        print(f"❌ خطا در نوع هوشمند: {e}")
    
    print()
    print("=" * 60)
    print()
    
    # مقایسه نتایج
    print("📊 مقایسه نتایج:")
    print("-" * 40)
    
    if 'result_simple' in locals() and 'result_intelligent' in locals():
        print("✅ هر دو نوع تولید متن با موفقیت اجرا شدند")
        print(f"• تفاوت اطمینان: {result_intelligent['confidence'] - result_simple['confidence']:.2f}")
        
        # بررسی تفاوت در متن زمینه
        if 'context_text' in result_simple and 'context_text' in result_intelligent:
            simple_context_length = len(result_simple['context_text'])
            intelligent_context_length = len(result_intelligent['context_text'])
            print(f"• طول متن زمینه ساده: {simple_context_length} کاراکتر")
            print(f"• طول متن زمینه هوشمند: {intelligent_context_length} کاراکتر")
            print(f"• تفاوت طول: {intelligent_context_length - simple_context_length} کاراکتر")
    else:
        print("❌ امکان مقایسه کامل وجود ندارد")
    
    print()
    print("🎉 تست کامل شد!")

if __name__ == "__main__":
    test_text_generation_types() 
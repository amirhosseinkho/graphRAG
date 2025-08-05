#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تست مدل‌های جدید اضافه شده
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_new_models():
    """تست مدل‌های جدید"""
    
    print("🧪 شروع تست مدل‌های جدید...")
    print("=" * 60)
    
    # راه‌اندازی سرویس
    service = GraphRAGService()
    service.initialize()
    
    # سوال تست
    test_query = "What genes are expressed in heart?"
    
    print(f"📝 سوال تست: {test_query}")
    print()
    
    # تست مدل‌های مختلف
    models_to_test = [
        GenerationModel.OPENAI_GPT_4O,
        GenerationModel.OPENAI_GPT_4O_MINI,
        GenerationModel.ANTHROPIC_CLAUDE_3_5_SONNET,
        GenerationModel.GOOGLE_GEMINI_1_5_PRO,
        GenerationModel.META_LLAMA_3_1,
        GenerationModel.MISTRAL_AI
    ]
    
    for model in models_to_test:
        print(f"🔹 تست مدل: {model.value}")
        print("-" * 40)
        
        try:
            result = service.process_query(
                query=test_query,
                retrieval_method=RetrievalMethod.BFS,
                generation_model=model,
                text_generation_type='SIMPLE',
                max_depth=2
            )
            
            print("✅ نتیجه:")
            print(f"• مدل: {result['model']}")
            print(f"• اطمینان: {result['confidence']}")
            print(f"• پاسخ: {result['answer'][:100]}...")
            
        except Exception as e:
            print(f"❌ خطا: {e}")
        
        print()
        print("=" * 60)
        print()
    
    print("🎉 تست مدل‌های جدید کامل شد!")

if __name__ == "__main__":
    test_new_models() 
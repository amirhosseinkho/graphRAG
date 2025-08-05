#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست متن زمینه پیشرفته - بررسی انواع مختلف متن زمینه پیشرفته
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_advanced_context():
    """تست متن زمینه پیشرفته"""
    print("🧪 تست متن زمینه پیشرفته")
    print("=" * 60)
    
    # ایجاد سرویس
    service = GraphRAGService()
    
    # تست سوالات مختلف
    test_queries = [
        "What genes are expressed in heart?",
        "How does TP53 relate to cancer?",
        "What drugs treat breast cancer?",
        "What biological processes involve insulin?",
        "Which tissues express EGFR?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 تست {i}: {query}")
        print("-" * 40)
        
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
        
        # تست متن زمینه پیشرفته
        print(f"\n📝 متن زمینه پیشرفته:")
        print("-" * 30)
        
        advanced_context = service._create_advanced_context_text(retrieval_result)
        
        # نمایش خلاصه متن زمینه
        lines = advanced_context.split('\n')
        print(f"  طول متن: {len(lines)} خط")
        print(f"  کاراکترها: {len(advanced_context)}")
        
        # نمایش چند خط اول
        for j, line in enumerate(lines[:10]):
            if line.strip():
                print(f"  {j+1}: {line[:80]}{'...' if len(line) > 80 else ''}")
        
        if len(lines) > 10:
            print(f"  ... و {len(lines) - 10} خط دیگر")
        
        print(f"\n{'='*60}")
    
    print("✅ تست متن زمینه پیشرفته کامل شد")

def test_context_comparison():
    """مقایسه انواع مختلف متن زمینه"""
    print("\n🧪 مقایسه انواع مختلف متن زمینه")
    print("=" * 60)
    
    service = GraphRAGService()
    
    # تست سوال پیچیده
    complex_query = "What is the relationship between TP53, breast cancer, and drug treatments?"
    
    print(f"🔍 سوال پیچیده: {complex_query}")
    
    # بازیابی اطلاعات
    retrieval_result = service.retrieve_information(
        query=complex_query,
        method=RetrievalMethod.INTELLIGENT,
        max_depth=2
    )
    
    # تست انواع مختلف متن زمینه
    context_types = [
        ('SIMPLE', 'متن ساده', service._create_simple_context_text),
        ('INTELLIGENT', 'متن هوشمند', service._create_intelligent_context_text),
        ('ADVANCED', 'متن پیشرفته', service._create_advanced_context_text),
        ('ENHANCED', 'متن بهبود یافته', service._create_enhanced_context_text)
    ]
    
    for context_type, description, context_func in context_types:
        print(f"\n📝 {description}:")
        print("-" * 30)
        
        context_text = context_func(retrieval_result)
        
        # نمایش خلاصه متن زمینه
        lines = context_text.split('\n')
        print(f"  طول متن: {len(lines)} خط")
        print(f"  کاراکترها: {len(context_text)}")
        
        # نمایش چند خط اول
        for j, line in enumerate(lines[:5]):
            if line.strip():
                print(f"  {j+1}: {line[:60]}{'...' if len(line) > 60 else ''}")
        
        if len(lines) > 5:
            print(f"  ... و {len(lines) - 5} خط دیگر")
    
    print("✅ مقایسه انواع مختلف متن زمینه کامل شد")

def test_context_quality():
    """تست کیفیت متن زمینه"""
    print("\n🧪 تست کیفیت متن زمینه")
    print("=" * 60)
    
    service = GraphRAGService()
    
    # تست سوالات مختلف
    test_queries = [
        "What genes are expressed in heart?",
        "How does TP53 relate to cancer?",
        "What drugs treat breast cancer?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 تست {i}: {query}")
        print("-" * 40)
        
        # بازیابی اطلاعات
        retrieval_result = service.retrieve_information(
            query=query,
            method=RetrievalMethod.INTELLIGENT,
            max_depth=2
        )
        
        # تست متن زمینه پیشرفته
        advanced_context = service._create_advanced_context_text(retrieval_result)
        
        # تحلیل کیفیت
        lines = advanced_context.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        print(f"📊 تحلیل کیفیت:")
        print(f"  • کل خطوط: {len(lines)}")
        print(f"  • خطوط غیر خالی: {len(non_empty_lines)}")
        print(f"  • تراکم محتوا: {len(non_empty_lines)/len(lines)*100:.1f}%")
        print(f"  • طول متن: {len(advanced_context)} کاراکتر")
        
        # بررسی وجود بخش‌های مهم
        important_sections = [
            "تحلیل آماری پیشرفته",
            "تحلیل نوع‌شناسی نودها",
            "تحلیل روابط معنادار",
            "تحلیل مسیرهای زیستی",
            "استنتاجات زیستی",
            "دستورالعمل هوشمند"
        ]
        
        print(f"  • بخش‌های مهم موجود:")
        for section in important_sections:
            if section in advanced_context:
                print(f"    ✅ {section}")
            else:
                print(f"    ❌ {section}")
        
        print(f"\n{'='*60}")
    
    print("✅ تست کیفیت متن زمینه کامل شد")

if __name__ == "__main__":
    print("🚀 شروع تست متن زمینه پیشرفته")
    test_advanced_context()
    test_context_comparison()
    test_context_quality()
    print("\n🎉 تمام تست‌ها کامل شد!") 
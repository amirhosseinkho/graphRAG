#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست تولید متن زمینه - بررسی انواع مختلف متن زمینه
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel, ContextTextType

def test_context_generation():
    """تست تولید انواع مختلف متن زمینه"""
    print("🧪 تست تولید متن زمینه")
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
        
        # تست انواع مختلف متن زمینه
        context_types = [
            ('SIMPLE', 'متن ساده'),
            ('INTELLIGENT', 'متن هوشمند'),
            ('SCIENTIFIC_ANALYTICAL', 'متن علمی-تحلیلی'),
            ('NARRATIVE', 'متن روایی'),
            ('DATA_DRIVEN', 'متن داده‌محور'),
            ('STEP_BY_STEP', 'متن گام به گام'),
            ('COMPACT_DIRECT', 'متن فشرده'),
            ('BIOLOGICAL_PATHWAY', 'متن مسیر زیستی'),
            ('CLINICAL_RELEVANCE', 'متن بالینی'),
            ('MECHANISTIC_DETAILED', 'متن مکانیسمی')
        ]
        
        for context_type, description in context_types:
            print(f"\n📝 {description}:")
            print("-" * 30)
            
            # ایجاد متن زمینه بر اساس نوع
            if context_type == 'SIMPLE':
                context_text = service._create_simple_context_text(retrieval_result)
            elif context_type == 'INTELLIGENT':
                context_text = service._create_intelligent_context_text(retrieval_result)
            elif context_type == 'SCIENTIFIC_ANALYTICAL':
                context_text = service._create_scientific_analytical_context(retrieval_result)
            elif context_type == 'NARRATIVE':
                context_text = service._create_narrative_context(retrieval_result)
            elif context_type == 'DATA_DRIVEN':
                context_text = service._create_data_driven_context(retrieval_result)
            elif context_type == 'STEP_BY_STEP':
                context_text = service._create_step_by_step_context(retrieval_result)
            elif context_type == 'COMPACT_DIRECT':
                context_text = service._create_compact_direct_context(retrieval_result)
            elif context_type == 'BIOLOGICAL_PATHWAY':
                context_text = service._create_biological_pathway_context(retrieval_result)
            elif context_type == 'CLINICAL_RELEVANCE':
                context_text = service._create_clinical_relevance_context(retrieval_result)
            elif context_type == 'MECHANISTIC_DETAILED':
                context_text = service._create_mechanistic_detailed_context(retrieval_result)
            else:
                context_text = "نوع متن زمینه نامعتبر"
            
            # نمایش خلاصه متن زمینه
            lines = context_text.split('\n')
            print(f"  طول متن: {len(lines)} خط")
            print(f"  کاراکترها: {len(context_text)}")
            
            # نمایش چند خط اول
            for j, line in enumerate(lines[:5]):
                if line.strip():
                    print(f"  {j+1}: {line[:80]}{'...' if len(line) > 80 else ''}")
            
            if len(lines) > 5:
                print(f"  ... و {len(lines) - 5} خط دیگر")
        
        print(f"\n{'='*60}")
    
    print("✅ تست تولید متن زمینه کامل شد")

def test_enhanced_context():
    """تست متن زمینه بهبود یافته"""
    print("\n🧪 تست متن زمینه بهبود یافته")
    print("=" * 60)
    
    service = GraphRAGService()
    
    # تست سوال پیچیده
    complex_query = "What is the relationship between TP53, breast cancer, and drug treatments?"
    
    print(f"🔍 سوال پیچیده: {complex_query}")
    
    # بازیابی هدفمند
    retrieval_result = service.retrieve_information(
        query=complex_query,
        method=RetrievalMethod.INTELLIGENT,
        max_depth=3
    )
    
    # تست متن زمینه بهبود یافته
    enhanced_context = service._create_enhanced_context_text(retrieval_result)
    
    print(f"\n📊 نتایج بازیابی:")
    print(f"  • نودها: {len(retrieval_result.nodes)}")
    print(f"  • یال‌ها: {len(retrieval_result.edges)}")
    print(f"  • مسیرها: {len(retrieval_result.paths)}")
    
    print(f"\n📝 متن زمینه بهبود یافته:")
    print("-" * 40)
    lines = enhanced_context.split('\n')
    for i, line in enumerate(lines[:10]):
        print(f"  {i+1}: {line}")
    
    if len(lines) > 10:
        print(f"  ... و {len(lines) - 10} خط دیگر")
    
    print("✅ تست متن زمینه بهبود یافته کامل شد")

if __name__ == "__main__":
    print("🚀 شروع تست تولید متن زمینه")
    test_context_generation()
    test_enhanced_context()
    print("\n🎉 تمام تست‌ها کامل شد!") 
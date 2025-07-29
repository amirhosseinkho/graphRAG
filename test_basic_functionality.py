#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست قابلیت‌های اصلی سیستم GraphRAG
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def test_basic_functionality():
    """تست قابلیت‌های اصلی"""
    print("🚀 شروع تست قابلیت‌های اصلی")
    print("=" * 60)
    
    # ایجاد سرویس
    service = GraphRAGService()
    
    # تست 1: تشخیص نوع سوال
    print("\n🔍 تست 1: تشخیص نوع سوال")
    test_questions = [
        "What genes are expressed in the heart?",
        "Which genes interact with TP53?",
        "What compounds treat heart disease?"
    ]
    
    for question in test_questions:
        print(f"\nسوال: {question}")
        intent = service.analyze_question_intent(question)
        print(f"  نوع: {intent['question_type']}")
        print(f"  Metaedges: {intent['metaedges']}")
        print(f"  توضیح: {intent['description']}")
    
    # تست 2: جستجوی هوشمند
    print("\n🔍 تست 2: جستجوی هوشمند")
    question = "What genes are expressed in the heart?"
    print(f"سوال: {question}")
    
    try:
        results = service.intelligent_semantic_search(question, max_depth=2)
        print(f"  تعداد نتایج: {len(results)}")
        
        if results:
            print("  نتایج:")
            for i, (node_id, depth, score, explanation) in enumerate(results[:3], 1):
                node_name = service.G.nodes[node_id]['name']
                node_kind = service.G.nodes[node_id]['kind']
                print(f"    {i}. {node_name} ({node_kind}) - امتیاز: {score:.2f}")
                print(f"       توضیح: {explanation}")
        else:
            print("  ❌ هیچ نتیجه‌ای یافت نشد")
            
    except Exception as e:
        print(f"  ❌ خطا: {e}")
    
    # تست 3: جستجوی چندمرحله‌ای
    print("\n🔄 تست 3: جستجوی چندمرحله‌ای")
    complex_question = "What compounds upregulate genes expressed in the heart?"
    print(f"سوال پیچیده: {complex_question}")
    
    try:
        results = service.multi_hop_search(complex_question, max_depth=3)
        print(f"  تعداد نتایج: {len(results)}")
        
        if results:
            print("  نتایج:")
            for i, (node_id, depth, score, explanation, path) in enumerate(results[:3], 1):
                node_name = service.G.nodes[node_id]['name']
                node_kind = service.G.nodes[node_id]['kind']
                print(f"    {i}. {node_name} ({node_kind}) - عمق: {depth}, امتیاز: {score:.2f}")
                print(f"       مسیر: {' → '.join([service.G.nodes[p]['name'] for p in path])}")
        else:
            print("  ❌ هیچ نتیجه‌ای یافت نشد")
            
    except Exception as e:
        print(f"  ❌ خطا: {e}")
    
    print("\n✅ تست قابلیت‌های اصلی کامل شد")

if __name__ == "__main__":
    test_basic_functionality() 
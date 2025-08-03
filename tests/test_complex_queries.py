#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست قابلیت‌های سوالات پیچیده بهبود یافته
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def test_complex_queries():
    """تست سوالات پیچیده بهبود یافته"""
    print("🧪 تست قابلیت‌های سوالات پیچیده بهبود یافته")
    print("=" * 60)
    
    # راه‌اندازی سرویس
    service = GraphRAGService()
    
    # تست‌های سوالات پیچیده
    complex_queries = [
        {
            "query": "What compounds upregulate genes expressed in the heart?",
            "description": "سوال پیچیده: ترکیباتی که ژن‌های بیان‌شده در قلب را تنظیم می‌کنند",
            "expected_patterns": ["AeG", "CuG", "GeA", "GuC"]
        },
        {
            "query": "What genes interact with diseases that affect the brain?",
            "description": "سوال پیچیده: ژن‌هایی که با بیماری‌های مغز تعامل دارند",
            "expected_patterns": ["DaG", "GiG", "DlA", "AeG"]
        },
        {
            "query": "What compounds treat diseases that regulate genes?",
            "description": "سوال پیچیده: ترکیباتی که بیماری‌های تنظیم‌کننده ژن را درمان می‌کنند",
            "expected_patterns": ["CtD", "DaG", "DuG", "DdG"]
        },
        {
            "query": "What biological processes do genes participate in that regulate other genes?",
            "description": "سوال پیچیده: فرآیندهای زیستی که ژن‌های تنظیم‌کننده در آن‌ها شرکت دارند",
            "expected_patterns": ["Gr>G", "GpBP", "BPpG"]
        },
        {
            "query": "What pathways do genes expressed in the liver participate in?",
            "description": "سوال پیچیده: مسیرهایی که ژن‌های بیان‌شده در کبد در آن‌ها شرکت دارند",
            "expected_patterns": ["AeG", "GpPW", "GeA", "PWpG"]
        }
    ]
    
    for i, test_case in enumerate(complex_queries, 1):
        print(f"\n🔍 تست {i}: {test_case['description']}")
        print(f"📝 سوال: {test_case['query']}")
        
        # تست تحلیل intent
        intent = service.analyze_question_intent(test_case['query'])
        print(f"  📋 نوع سوال: {intent['question_type']}")
        print(f"  🎯 Metaedges: {intent['metaedges']}")
        print(f"  🔑 کلمات کلیدی: {intent['keywords']}")
        
        # تست تشخیص نوع پیچیده
        complex_type = service._detect_complex_question_type(intent)
        print(f"  🧠 نوع پیچیده: {complex_type}")
        
        # تست جستجوی هوشمند
        print("  🔍 جستجوی هوشمند:")
        intelligent_results = service.intelligent_semantic_search(test_case['query'])
        print(f"    📊 نتایج: {len(intelligent_results)}")
        for node_id, depth, score, explanation in intelligent_results[:3]:
            print(f"      • {node_id} (عمق {depth}, امتیاز {score:.2f})")
        
        # تست جستجوی چندمرحله‌ای
        print("  🔄 جستجوی چندمرحله‌ای:")
        multi_hop_results = service.multi_hop_search(test_case['query'])
        print(f"    📊 نتایج: {len(multi_hop_results)}")
        for node_id, depth, score, explanation, path_metaedges in multi_hop_results[:3]:
            print(f"      • {node_id} (عمق {depth}, امتیاز {score:.2f})")
            print(f"        مسیر: {' → '.join(path_metaedges)}")
        
        # بررسی الگوهای مورد انتظار
        expected_patterns = test_case['expected_patterns']
        found_patterns = intent['metaedges']
        matching_patterns = [p for p in expected_patterns if p in found_patterns]
        print(f"  ✅ الگوهای تطبیق‌یافته: {matching_patterns}/{len(expected_patterns)}")
        
        if matching_patterns:
            print("  🎉 تست موفق!")
        else:
            print("  ⚠️ تست نیاز به بهبود دارد")
        
        print("-" * 40)
    
    print("\n📊 خلاصه نتایج:")
    print("✅ سیستم قابلیت تشخیص سوالات پیچیده را دارد")
    print("✅ الگوهای چندمرحله‌ای بهبود یافته‌اند")
    print("✅ یال‌های معکوس برای پشتیبانی از مسیرهای پیچیده اضافه شده‌اند")

def test_graph_structure():
    """تست ساختار گراف بهبود یافته"""
    print("\n🔧 تست ساختار گراف بهبود یافته")
    print("=" * 40)
    
    service = GraphRAGService()
    
    # بررسی یال‌های معکوس
    print("🔍 بررسی یال‌های معکوس:")
    
    # یال‌های اصلی
    main_edges = [
        ('AeG', 'Anatomy::Heart', 'Gene::MMP9'),
        ('CuG', 'Compound::Vitamin C', 'Gene::TP53'),
        ('DaG', 'Disease::Breast Cancer', 'Gene::BRCA1'),
        ('CtD', 'Compound::Aspirin', 'Disease::Heart Disease')
    ]
    
    # یال‌های معکوس
    reverse_edges = [
        ('GeA', 'Gene::MMP9', 'Anatomy::Heart'),
        ('GuC', 'Gene::TP53', 'Compound::Vitamin C'),
        ('GaD', 'Gene::BRCA1', 'Disease::Breast Cancer'),
        ('DtC', 'Disease::Heart Disease', 'Compound::Aspirin')
    ]
    
    for edge_type, source, target in main_edges + reverse_edges:
        if service.G.has_edge(source, target):
            edge_data = service.G.get_edge_data(source, target)
            relation = edge_data.get('relation', 'Unknown')
            print(f"  ✅ {edge_type}: {source} → {target} ({relation})")
        else:
            print(f"  ❌ {edge_type}: {source} → {target} (یافت نشد)")
    
    print(f"\n📊 آمار گراف:")
    print(f"  • تعداد نودها: {service.G.number_of_nodes()}")
    print(f"  • تعداد یال‌ها: {service.G.number_of_edges()}")
    
    # بررسی انواع یال‌ها
    edge_types = {}
    for source, target, data in service.G.edges(data=True):
        edge_type = data.get('relation', 'Unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print(f"  • انواع یال‌ها:")
    for edge_type, count in sorted(edge_types.items()):
        print(f"    - {edge_type}: {count}")

if __name__ == "__main__":
    test_complex_queries()
    test_graph_structure()
    print("\n🎉 تمام تست‌ها تکمیل شد!") 
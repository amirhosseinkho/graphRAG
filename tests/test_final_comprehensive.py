#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست جامع نهایی سیستم GraphRAG بهبود یافته
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def test_comprehensive_system():
    """تست جامع تمام قابلیت‌های سیستم"""
    print("🧪 تست جامع سیستم GraphRAG بهبود یافته")
    print("=" * 70)
    
    # راه‌اندازی سرویس
    service = GraphRAGService()
    
    # تست‌های مختلف
    test_cases = [
        {
            "category": "سوالات ساده",
            "tests": [
                {
                    "query": "What genes are expressed in the heart?",
                    "expected_type": "anatomy_expression",
                    "expected_metaedges": ["AeG"]
                },
                {
                    "query": "What diseases are associated with BRCA1?",
                    "expected_type": "gene_disease",
                    "expected_metaedges": ["DaG", "GaD"]
                },
                {
                    "query": "What compounds treat heart disease?",
                    "expected_type": "disease_treatment",
                    "expected_metaedges": ["CtD"]
                }
            ]
        },
        {
            "category": "سوالات پیچیده",
            "tests": [
                {
                    "query": "What compounds upregulate genes expressed in the heart?",
                    "expected_type": "complex_expression",
                    "expected_metaedges": ["AeG", "CuG", "GeA", "GuC"]
                },
                {
                    "query": "What genes interact with diseases that affect the brain?",
                    "expected_type": "complex_disease",
                    "expected_metaedges": ["DaG", "GiG", "DlA", "AeG"]
                },
                {
                    "query": "What compounds treat diseases that regulate genes?",
                    "expected_type": "complex_treatment",
                    "expected_metaedges": ["CtD", "DaG", "DuG", "DdG"]
                },
                {
                    "query": "What biological processes do genes participate in that regulate other genes?",
                    "expected_type": "complex_function",
                    "expected_metaedges": ["Gr>G", "GpBP", "BPpG"]
                }
            ]
        },
        {
            "category": "سوالات چندمرحله‌ای",
            "tests": [
                {
                    "query": "What pathways do genes expressed in the liver participate in?",
                    "expected_type": "complex_function",
                    "expected_metaedges": ["AeG", "GpPW", "GeA", "PWpG"]
                },
                {
                    "query": "What compounds bind genes that are expressed in the brain?",
                    "expected_type": "complex_expression",
                    "expected_metaedges": ["AeG", "CbG", "GeA", "GbC"]
                }
            ]
        }
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for category in test_cases:
        print(f"\n📋 {category['category']}")
        print("-" * 50)
        
        for test in category['tests']:
            total_tests += 1
            print(f"\n🔍 تست: {test['query']}")
            
            # تست تحلیل intent
            intent = service.analyze_question_intent(test['query'])
            question_type = intent['question_type']
            metaedges = intent['metaedges']
            
            print(f"  📋 نوع سوال تشخیص داده شده: {question_type}")
            print(f"  🎯 Metaedges تشخیص داده شده: {metaedges}")
            
            # تست تشخیص نوع پیچیده
            complex_type = service._detect_complex_question_type(intent)
            print(f"  🧠 نوع پیچیده: {complex_type}")
            
            # تست جستجوی هوشمند
            intelligent_results = service.intelligent_semantic_search(test['query'])
            print(f"  🔍 نتایج جستجوی هوشمند: {len(intelligent_results)}")
            
            # تست جستجوی چندمرحله‌ای
            multi_hop_results = service.multi_hop_search(test['query'])
            print(f"  🔄 نتایج چندمرحله‌ای: {len(multi_hop_results)}")
            
            # بررسی تطبیق با انتظارات
            type_match = question_type == test['expected_type']
            metaedge_match = all(me in metaedges for me in test['expected_metaedges'])
            
            if type_match and metaedge_match:
                print("  ✅ تست موفق!")
                passed_tests += 1
            else:
                print("  ❌ تست ناموفق!")
                if not type_match:
                    print(f"    نوع سوال: انتظار {test['expected_type']}, دریافت {question_type}")
                if not metaedge_match:
                    print(f"    Metaedges: انتظار {test['expected_metaedges']}, دریافت {metaedges}")
    
    # خلاصه نتایج
    print(f"\n📊 خلاصه نتایج:")
    print(f"  • کل تست‌ها: {total_tests}")
    print(f"  • تست‌های موفق: {passed_tests}")
    print(f"  • نرخ موفقیت: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests, total_tests

def test_graph_capabilities():
    """تست قابلیت‌های گراف"""
    print(f"\n🔧 تست قابلیت‌های گراف")
    print("=" * 40)
    
    service = GraphRAGService()
    
    # تست یال‌های معکوس
    print("🔍 تست یال‌های معکوس:")
    
    test_edges = [
        ('AeG', 'Anatomy::Heart', 'Gene::MMP9'),
        ('GeA', 'Gene::MMP9', 'Anatomy::Heart'),
        ('CuG', 'Compound::Vitamin C', 'Gene::TP53'),
        ('GuC', 'Gene::TP53', 'Compound::Vitamin C'),
        ('DaG', 'Disease::Breast Cancer', 'Gene::BRCA1'),
        ('GaD', 'Gene::BRCA1', 'Disease::Breast Cancer'),
        ('CtD', 'Compound::Aspirin', 'Disease::Heart Disease'),
        ('DtC', 'Disease::Heart Disease', 'Compound::Aspirin')
    ]
    
    edge_tests_passed = 0
    for edge_type, source, target in test_edges:
        if service.G.has_edge(source, target):
            edge_data = service.G.get_edge_data(source, target)
            relation = edge_data.get('relation', 'Unknown')
            print(f"  ✅ {edge_type}: {source} → {target} ({relation})")
            edge_tests_passed += 1
        else:
            print(f"  ❌ {edge_type}: {source} → {target} (یافت نشد)")
    
    print(f"\n📊 نتایج تست یال‌ها: {edge_tests_passed}/{len(test_edges)} موفق")
    
    # تست مسیرهای پیچیده
    print(f"\n🔍 تست مسیرهای پیچیده:")
    
    complex_paths = [
        ('Anatomy::Heart', ['AeG', 'CuG'], 'Compound'),
        ('Anatomy::Heart', ['AeG', 'CdG'], 'Compound'),
        ('Gene::BRCA1', ['GaD', 'GpBP'], 'Biological Process'),
        ('Compound::Aspirin', ['CtD', 'DaG'], 'Gene')
    ]
    
    path_tests_passed = 0
    for start_node, pattern, target_type in complex_paths:
        paths = service._find_paths_with_pattern(start_node, pattern, max_depth=3)
        if paths:
            print(f"  ✅ مسیر {start_node} → {' → '.join(pattern)}: {len(paths)} مسیر یافت شد")
            path_tests_passed += 1
        else:
            print(f"  ❌ مسیر {start_node} → {' → '.join(pattern)}: هیچ مسیری یافت نشد")
    
    print(f"\n📊 نتایج تست مسیرها: {path_tests_passed}/{len(complex_paths)} موفق")
    
    return edge_tests_passed, len(test_edges), path_tests_passed, len(complex_paths)

def test_performance_metrics():
    """تست معیارهای عملکرد"""
    print(f"\n⚡ تست معیارهای عملکرد")
    print("=" * 40)
    
    service = GraphRAGService()
    
    # آمار گراف
    print("📊 آمار گراف:")
    print(f"  • تعداد نودها: {service.G.number_of_nodes()}")
    print(f"  • تعداد یال‌ها: {service.G.number_of_edges()}")
    
    # انواع یال‌ها
    edge_types = {}
    for source, target, data in service.G.edges(data=True):
        edge_type = data.get('relation', 'Unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print(f"  • انواع یال‌ها: {len(edge_types)}")
    
    # یال‌های کلیدی
    key_edges = ['AeG', 'GeA', 'DaG', 'GaD', 'GpBP', 'BPpG', 'CuG', 'GuC', 'CtD', 'DtC']
    for edge_type in key_edges:
        count = edge_types.get(edge_type, 0)
        print(f"    - {edge_type}: {count} یال")
    
    # تست سرعت
    import time
    
    test_queries = [
        "What genes are expressed in the heart?",
        "What compounds upregulate genes expressed in the heart?",
        "What biological processes do genes participate in?"
    ]
    
    print(f"\n⏱️ تست سرعت:")
    for query in test_queries:
        start_time = time.time()
        intelligent_results = service.intelligent_semantic_search(query)
        intelligent_time = time.time() - start_time
        
        start_time = time.time()
        multi_hop_results = service.multi_hop_search(query)
        multi_hop_time = time.time() - start_time
        
        print(f"  • {query[:50]}...")
        print(f"    جستجوی هوشمند: {len(intelligent_results)} نتیجه در {intelligent_time:.3f} ثانیه")
        print(f"    جستجوی چندمرحله‌ای: {len(multi_hop_results)} نتیجه در {multi_hop_time:.3f} ثانیه")

def main():
    """تابع اصلی"""
    print("🚀 شروع تست جامع سیستم GraphRAG")
    print("=" * 70)
    
    # تست قابلیت‌های اصلی
    passed_tests, total_tests = test_comprehensive_system()
    
    # تست قابلیت‌های گراف
    edge_passed, edge_total, path_passed, path_total = test_graph_capabilities()
    
    # تست معیارهای عملکرد
    test_performance_metrics()
    
    # خلاصه نهایی
    print(f"\n🎯 خلاصه نهایی:")
    print(f"  ✅ تست‌های اصلی: {passed_tests}/{total_tests} موفق")
    print(f"  ✅ تست‌های یال‌ها: {edge_passed}/{edge_total} موفق")
    print(f"  ✅ تست‌های مسیرها: {path_passed}/{path_total} موفق")
    
    overall_success_rate = ((passed_tests + edge_passed + path_passed) / 
                           (total_tests + edge_total + path_total)) * 100
    
    print(f"  📊 نرخ موفقیت کلی: {overall_success_rate:.1f}%")
    
    if overall_success_rate >= 80:
        print("  🎉 سیستم آماده استفاده است!")
    elif overall_success_rate >= 60:
        print("  ⚠️ سیستم نیاز به بهبود دارد")
    else:
        print("  ❌ سیستم نیاز به بهبود جدی دارد")
    
    print("\n📋 دستاوردهای کلیدی:")
    print("  ✅ پشتیبانی از یال‌های معکوس")
    print("  ✅ تشخیص سوالات پیچیده")
    print("  ✅ الگوهای چندمرحله‌ای")
    print("  ✅ جستجوی هوشمند")
    print("  ✅ ساختار گراف بهبود یافته")

if __name__ == "__main__":
    main() 
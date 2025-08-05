#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست سرویس پیشرفته GraphRAG
"""

import sys
import os
import json
import time
from datetime import datetime

# اضافه کردن مسیر پروژه به sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_graphrag_service import EnhancedGraphRAGService, TokenExtractionMethod, RetrievalAlgorithm, CommunityDetectionMethod

def test_enhanced_service():
    """تست سرویس پیشرفته GraphRAG"""
    
    print("🧪 شروع تست سرویس پیشرفته GraphRAG")
    print("=" * 50)
    
    # راه‌اندازی سرویس
    print("📦 راه‌اندازی سرویس...")
    try:
        # بررسی وجود فایل گراف
        graph_files = [f for f in os.listdir('.') if f.startswith('hetionet_graph_') and f.endswith('.pkl')]
        if graph_files:
            latest_graph_file = max(graph_files)
            print(f"📊 استفاده از گراف: {latest_graph_file}")
            service = EnhancedGraphRAGService(graph_data_path=latest_graph_file)
        else:
            print("⚠️ فایل گراف یافت نشد، استفاده از سرویس بدون گراف")
            service = EnhancedGraphRAGService()
        
        print("✅ سرویس با موفقیت راه‌اندازی شد")
        
    except Exception as e:
        print(f"❌ خطا در راه‌اندازی سرویس: {e}")
        return
    
    # تست تنظیمات
    print("\n🔧 تست تنظیمات...")
    test_config(service)
    
    # تست استخراج توکن
    print("\n🔍 تست استخراج توکن...")
    test_token_extraction(service)
    
    # تست الگوریتم‌های بازیابی
    print("\n🎯 تست الگوریتم‌های بازیابی...")
    test_retrieval_algorithms(service)
    
    # تست سوالات مختلف
    print("\n❓ تست سوالات مختلف...")
    test_queries(service)
    
    print("\n🎉 تست‌ها با موفقیت انجام شد!")

def test_config(service):
    """تست تنظیمات سرویس"""
    
    # دریافت تنظیمات فعلی
    config = service.get_config()
    print(f"📋 تنظیمات فعلی: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    # تغییر تنظیمات
    new_config = {
        'token_extraction_method': 'hybrid',
        'retrieval_algorithm': 'pagerank',
        'max_depth': 4,
        'max_nodes': 25,
        'similarity_threshold': 0.4
    }
    
    service.set_config(**new_config)
    updated_config = service.get_config()
    print(f"📋 تنظیمات به‌روز شده: {json.dumps(updated_config, indent=2, ensure_ascii=False)}")
    
    # تست آمار گراف
    if service.G:
        stats = service.get_graph_statistics()
        print(f"📊 آمار گراف: {json.dumps(stats, indent=2, ensure_ascii=False)}")

def test_token_extraction(service):
    """تست روش‌های استخراج توکن"""
    
    test_queries = [
        "What genes are associated with diabetes?",
        "How does TP53 relate to cancer?",
        "What drugs treat heart disease?",
        "Which pathways are involved in metabolism?"
    ]
    
    methods = [
        TokenExtractionMethod.LLM_BASED,
        TokenExtractionMethod.RULE_BASED,
        TokenExtractionMethod.HYBRID,
        TokenExtractionMethod.SEMANTIC
    ]
    
    for query in test_queries:
        print(f"\n🔍 سوال: {query}")
        
        for method in methods:
            service.config.token_extraction_method = method
            try:
                answer_types, entities = service.extract_tokens(query)
                print(f"  📝 {method.value}:")
                print(f"    نوع پاسخ: {answer_types}")
                print(f"    موجودیت‌ها: {entities}")
            except Exception as e:
                print(f"    ❌ خطا: {e}")

def test_retrieval_algorithms(service):
    """تست الگوریتم‌های بازیابی"""
    
    if not service.G:
        print("⚠️ گراف بارگذاری نشده، تست الگوریتم‌ها انجام نمی‌شود")
        return
    
    test_query = "What genes are associated with diabetes?"
    start_nodes = ["DIABETES", "GENE", "TP53"]  # نودهای نمونه
    
    algorithms = [
        RetrievalAlgorithm.BFS,
        RetrievalAlgorithm.DFS,
        RetrievalAlgorithm.PAGERANK,
        RetrievalAlgorithm.COMMUNITY_DETECTION,
        RetrievalAlgorithm.SEMANTIC_SIMILARITY,
        RetrievalAlgorithm.N_HOP,
        RetrievalAlgorithm.HYBRID
    ]
    
    for algorithm in algorithms:
        print(f"\n🎯 الگوریتم: {algorithm.value}")
        
        service.config.retrieval_algorithm = algorithm
        service.config.max_nodes = 10
        service.config.max_depth = 3
        
        try:
            start_time = time.time()
            result = service.process_query(test_query, start_nodes)
            end_time = time.time()
            
            print(f"  ⏱️ زمان پردازش: {end_time - start_time:.2f} ثانیه")
            print(f"  📊 تعداد نودها: {len(result.get('nodes', []))}")
            print(f"  📊 تعداد یال‌ها: {len(result.get('edges', []))}")
            
            if 'query_analysis' in result:
                analysis = result['query_analysis']
                print(f"  🔍 تحلیل سوال:")
                print(f"    نوع پاسخ: {analysis.get('answer_types', [])}")
                print(f"    موجودیت‌ها: {analysis.get('entities', [])}")
                print(f"    نودهای شروع: {analysis.get('start_nodes', [])}")
            
        except Exception as e:
            print(f"  ❌ خطا: {e}")

def test_queries(service):
    """تست سوالات مختلف"""
    
    if not service.G:
        print("⚠️ گراف بارگذاری نشده، تست سوالات انجام نمی‌شود")
        return
    
    test_cases = [
        {
            "query": "What genes are associated with diabetes?",
            "description": "جستجوی ژن‌های مرتبط با دیابت",
            "expected_entities": ["DIABETES", "GENE"]
        },
        {
            "query": "How does TP53 relate to cancer?",
            "description": "تحلیل رابطه TP53 با سرطان",
            "expected_entities": ["TP53", "CANCER"]
        },
        {
            "query": "What drugs treat heart disease?",
            "description": "جستجوی داروهای درمان بیماری قلبی",
            "expected_entities": ["DRUG", "HEART", "DISEASE"]
        },
        {
            "query": "Which pathways are involved in metabolism?",
            "description": "تحلیل مسیرهای متابولیسم",
            "expected_entities": ["PATHWAY", "METABOLISM"]
        },
        {
            "query": "What are the side effects of aspirin?",
            "description": "جستجوی عوارض جانبی آسپرین",
            "expected_entities": ["ASPIRIN", "SIDE_EFFECT"]
        }
    ]
    
    # تنظیم الگوریتم ترکیبی
    service.config.retrieval_algorithm = RetrievalAlgorithm.HYBRID
    service.config.token_extraction_method = TokenExtractionMethod.HYBRID
    service.config.max_nodes = 15
    service.config.max_depth = 3
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 تست {i}: {test_case['description']}")
        print(f"❓ سوال: {test_case['query']}")
        
        try:
            start_time = time.time()
            result = service.process_query(test_case['query'])
            end_time = time.time()
            
            print(f"  ⏱️ زمان پردازش: {end_time - start_time:.2f} ثانیه")
            
            if 'query_analysis' in result:
                analysis = result['query_analysis']
                print(f"  🔍 تحلیل سوال:")
                print(f"    نوع پاسخ: {analysis.get('answer_types', [])}")
                print(f"    موجودیت‌ها: {analysis.get('entities', [])}")
                print(f"    نودهای شروع: {analysis.get('start_nodes', [])}")
            
            print(f"  📊 نتایج:")
            print(f"    نودها: {len(result.get('nodes', []))}")
            print(f"    یال‌ها: {len(result.get('edges', []))}")
            
            if 'communities' in result and result['communities']:
                print(f"    جامعه‌ها: {len(result['communities'])}")
            
            if 'similarities' in result and result['similarities']:
                print(f"    شباهت‌ها: {len(result['similarities'])}")
            
            if 'paths' in result and result['paths']:
                print(f"    مسیرها: {len(result['paths'])}")
            
        except Exception as e:
            print(f"  ❌ خطا: {e}")

def test_performance():
    """تست عملکرد"""
    
    print("\n⚡ تست عملکرد...")
    
    # تست سرعت استخراج توکن
    service = EnhancedGraphRAGService()
    
    test_queries = [
        "What genes are associated with diabetes?",
        "How does TP53 relate to cancer?",
        "What drugs treat heart disease?",
        "Which pathways are involved in metabolism?",
        "What are the side effects of aspirin?"
    ]
    
    methods = [
        TokenExtractionMethod.LLM_BASED,
        TokenExtractionMethod.RULE_BASED,
        TokenExtractionMethod.HYBRID,
        TokenExtractionMethod.SEMANTIC
    ]
    
    performance_results = {}
    
    for method in methods:
        service.config.token_extraction_method = method
        method_times = []
        
        for query in test_queries:
            try:
                start_time = time.time()
                service.extract_tokens(query)
                end_time = time.time()
                method_times.append(end_time - start_time)
            except Exception as e:
                print(f"❌ خطا در {method.value}: {e}")
        
        if method_times:
            avg_time = sum(method_times) / len(method_times)
            performance_results[method.value] = {
                'average_time': avg_time,
                'total_time': sum(method_times),
                'count': len(method_times)
            }
    
    print("📊 نتایج عملکرد:")
    for method, results in performance_results.items():
        print(f"  {method}:")
        print(f"    میانگین زمان: {results['average_time']:.4f} ثانیه")
        print(f"    کل زمان: {results['total_time']:.4f} ثانیه")
        print(f"    تعداد تست: {results['count']}")

def main():
    """تابع اصلی"""
    
    print("🚀 شروع تست‌های سرویس پیشرفته GraphRAG")
    print(f"📅 تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # تست اصلی
        test_enhanced_service()
        
        # تست عملکرد
        test_performance()
        
        print("\n🎉 تمام تست‌ها با موفقیت انجام شد!")
        
    except KeyboardInterrupt:
        print("\n⏹️ تست توسط کاربر متوقف شد")
    except Exception as e:
        print(f"\n❌ خطای کلی: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
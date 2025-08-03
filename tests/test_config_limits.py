#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست تنظیمات و محدودیت‌های سیستم GraphRAG
"""

import sys
import os
from pathlib import Path

# اضافه کردن مسیر اصلی پروژه به sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_default_limits():
    """تست محدودیت‌های پیش‌فرض"""
    print("🧪 تست محدودیت‌های پیش‌فرض")
    print("=" * 50)
    
    service = GraphRAGService()
    
    # نمایش تنظیمات پیش‌فرض
    config = service.get_config()
    print("📋 تنظیمات پیش‌فرض:")
    for key, value in config.items():
        print(f"  • {key}: {value}")
    
    # تست با تنظیمات پیش‌فرض
    query = "How does TP53 relate to cancer?"
    result = service.process_query(
        query=query,
        retrieval_method=RetrievalMethod.INTELLIGENT,
        generation_model=GenerationModel.GPT_SIMULATION,
        max_depth=config['max_depth']
    )
    
    print(f"\n📊 نتایج با تنظیمات پیش‌فرض:")
    print(f"  • تعداد نودها: {len(result.get('retrieved_nodes', []))}")
    print(f"  • تعداد یال‌ها: {len(result.get('retrieved_edges', []))}")
    print(f"  • طول پاسخ: {len(result.get('answer', ''))} کاراکتر")
    
    return result

def test_increased_limits():
    """تست با افزایش محدودیت‌ها"""
    print("\n🧪 تست با افزایش محدودیت‌ها")
    print("=" * 50)
    
    service = GraphRAGService()
    
    # افزایش محدودیت‌ها
    service.set_config(
        max_nodes=20,           # افزایش از 10 به 20
        max_edges=40,           # افزایش از 20 به 40
        max_depth=4,            # افزایش از 3 به 4
        max_paths=10,           # افزایش از 5 به 10
        max_context_length=3000, # افزایش از 2000 به 3000
        max_answer_tokens=1500,  # افزایش از 1000 به 1500
        max_prompt_tokens=6000   # افزایش از 4000 به 6000
    )
    
    # نمایش تنظیمات جدید
    config = service.get_config()
    print("📋 تنظیمات جدید:")
    for key, value in config.items():
        print(f"  • {key}: {value}")
    
    # تست با تنظیمات جدید
    query = "How does TP53 relate to cancer?"
    result = service.process_query(
        query=query,
        retrieval_method=RetrievalMethod.INTELLIGENT,
        generation_model=GenerationModel.GPT_SIMULATION,
        max_depth=config['max_depth']
    )
    
    print(f"\n📊 نتایج با تنظیمات جدید:")
    print(f"  • تعداد نودها: {len(result.get('retrieved_nodes', []))}")
    print(f"  • تعداد یال‌ها: {len(result.get('retrieved_edges', []))}")
    print(f"  • طول پاسخ: {len(result.get('answer', ''))} کاراکتر")
    
    return result

def test_decreased_limits():
    """تست با کاهش محدودیت‌ها"""
    print("\n🧪 تست با کاهش محدودیت‌ها")
    print("=" * 50)
    
    service = GraphRAGService()
    
    # کاهش محدودیت‌ها
    service.set_config(
        max_nodes=5,            # کاهش از 10 به 5
        max_edges=10,           # کاهش از 20 به 10
        max_depth=2,            # کاهش از 3 به 2
        max_paths=3,            # کاهش از 5 به 3
        max_context_length=1000, # کاهش از 2000 به 1000
        max_answer_tokens=500,   # کاهش از 1000 به 500
        max_prompt_tokens=2000   # کاهش از 4000 به 2000
    )
    
    # نمایش تنظیمات جدید
    config = service.get_config()
    print("📋 تنظیمات جدید:")
    for key, value in config.items():
        print(f"  • {key}: {value}")
    
    # تست با تنظیمات جدید
    query = "How does TP53 relate to cancer?"
    result = service.process_query(
        query=query,
        retrieval_method=RetrievalMethod.INTELLIGENT,
        generation_model=GenerationModel.GPT_SIMULATION,
        max_depth=config['max_depth']
    )
    
    print(f"\n📊 نتایج با تنظیمات جدید:")
    print(f"  • تعداد نودها: {len(result.get('retrieved_nodes', []))}")
    print(f"  • تعداد یال‌ها: {len(result.get('retrieved_edges', []))}")
    print(f"  • طول پاسخ: {len(result.get('answer', ''))} کاراکتر")
    
    return result

def test_performance_comparison():
    """مقایسه عملکرد با تنظیمات مختلف"""
    print("\n🧪 مقایسه عملکرد")
    print("=" * 50)
    
    queries = [
        "How does TP53 relate to cancer?",
        "What genes are expressed in heart?",
        "Which drugs treat diabetes?",
        "What is the role of BRCA1 in breast cancer?"
    ]
    
    configs = {
        'کم': {'max_nodes': 5, 'max_depth': 2, 'max_answer_tokens': 500},
        'متوسط': {'max_nodes': 10, 'max_depth': 3, 'max_answer_tokens': 1000},
        'زیاد': {'max_nodes': 20, 'max_depth': 4, 'max_answer_tokens': 1500}
    }
    
    for config_name, config_values in configs.items():
        print(f"\n🔧 تنظیمات {config_name}:")
        service = GraphRAGService()
        service.set_config(**config_values)
        
        total_nodes = 0
        total_edges = 0
        total_answer_length = 0
        
        for query in queries:
            result = service.process_query(
                query=query,
                retrieval_method=RetrievalMethod.INTELLIGENT,
                generation_model=GenerationModel.GPT_SIMULATION
            )
            
            total_nodes += len(result.get('retrieved_nodes', []))
            total_edges += len(result.get('retrieved_edges', []))
            total_answer_length += len(result.get('answer', ''))
        
        avg_nodes = total_nodes / len(queries)
        avg_edges = total_edges / len(queries)
        avg_answer_length = total_answer_length / len(queries)
        
        print(f"  • میانگین نودها: {avg_nodes:.1f}")
        print(f"  • میانگین یال‌ها: {avg_edges:.1f}")
        print(f"  • میانگین طول پاسخ: {avg_answer_length:.0f} کاراکتر")

def test_web_app_config():
    """تست تنظیمات رابط وب"""
    print("\n🌐 تست تنظیمات رابط وب")
    print("=" * 50)
    
    service = GraphRAGService()
    
    # تست تغییر تنظیمات
    print("📝 تغییر تنظیمات...")
    service.set_config(
        max_nodes=15,
        max_edges=30,
        max_depth=3,
        max_answer_tokens=1200
    )
    
    # بررسی تنظیمات جدید
    config = service.get_config()
    print("✅ تنظیمات جدید:")
    for key, value in config.items():
        print(f"  • {key}: {value}")
    
    # تست بازنشانی تنظیمات
    print("\n🔄 بازنشانی تنظیمات...")
    service.set_config(
        max_nodes=10,
        max_edges=20,
        max_depth=3,
        max_paths=5,
        max_context_length=2000,
        max_answer_tokens=1000,
        max_prompt_tokens=4000,
        enable_verbose_logging=True,
        enable_biological_enrichment=True,
        enable_smart_filtering=True
    )
    
    config = service.get_config()
    print("✅ تنظیمات بازنشانی شده:")
    for key, value in config.items():
        print(f"  • {key}: {value}")

def main():
    """تابع اصلی"""
    print("🧪 تست تنظیمات و محدودیت‌های GraphRAG")
    print("=" * 60)
    
    # تست محدودیت‌های پیش‌فرض
    test_default_limits()
    
    # تست با افزایش محدودیت‌ها
    test_increased_limits()
    
    # تست با کاهش محدودیت‌ها
    test_decreased_limits()
    
    # مقایسه عملکرد
    test_performance_comparison()
    
    # تست تنظیمات رابط وب
    test_web_app_config()
    
    print("\n✅ تمام تست‌ها با موفقیت انجام شد!")
    print("\n🌐 برای تست رابط وب:")
    print("   1. سرور را اجرا کنید: python web_app.py")
    print("   2. به آدرس http://localhost:5000 بروید")
    print("   3. روی دکمه 'تنظیمات' کلیک کنید")
    print("   4. تنظیمات را تغییر دهید و ذخیره کنید")

if __name__ == "__main__":
    main() 
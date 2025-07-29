#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست سیستم بازیابی هدفمند جدید
"""

from graphrag_service import GraphRAGService
import os

def main():
    """تست اصلی سیستم"""
    print("🧪 تست سیستم بازیابی هدفمند جدید")
    print("=" * 50)
    
    # راه‌اندازی سرویس
    graph_files = [f for f in os.listdir('.') if f.startswith('hetionet_graph_') and f.endswith('.pkl')]
    if graph_files:
        latest_graph_file = max(graph_files)
        print(f"🔧 استفاده از گراف Hetionet: {latest_graph_file}")
        graphrag_service = GraphRAGService(graph_data_path=latest_graph_file)
    else:
        print("⚠️ فایل گراف Hetionet یافت نشد، استفاده از گراف نمونه")
        graphrag_service = GraphRAGService()
    
    # تست سوالات مختلف
    test_queries = [
        "What genes are expressed in the heart?",
        "What genes are associated with diabetes?",
        "What drugs treat cancer?",
        "What genes participate in cell cycle regulation?",
        "How do genes interact with TP53?",
        "What pathways are involved in cancer progression?",
        "What symptoms are associated with diabetes?",
        "What side effects does aspirin cause?"
    ]
    
    print(f"\n📋 تست {len(test_queries)} سوال مختلف:")
    print("-" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 تست {i}/{len(test_queries)}: {query}")
        print("-" * 30)
        
        try:
            # تست بازیابی خلاصه
            result = graphrag_service.test_compact_retrieval(query)
            
            # نمایش خلاصه نتایج
            retrieval_data = result['retrieval_data']
            print(f"✅ نتایج:")
            print(f"   • ژن‌های اصلی: {len(retrieval_data['primary_genes'])}")
            print(f"   • فرآیندهای زیستی: {len(retrieval_data['biological_processes'])}")
            print(f"   • مسیرهای زیستی: {len(retrieval_data['pathways'])}")
            print(f"   • بیماری‌ها: {len(retrieval_data['diseases'])}")
            print(f"   • داروها: {len(retrieval_data['drugs'])}")
            print(f"   • طول متن: {result['text_length']} کاراکتر")
            
            if retrieval_data['primary_genes']:
                print(f"   • بهترین ژن: {retrieval_data['primary_genes'][0]['name']} (امتیاز: {retrieval_data['primary_genes'][0]['score']:.2f})")
            
        except Exception as e:
            print(f"❌ خطا در پردازش سوال: {e}")
        
        print()
    
    print("🎉 تست کامل شد!")

if __name__ == "__main__":
    main() 
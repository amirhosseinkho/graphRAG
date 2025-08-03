#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست متن کوتاه‌تر برای مدل
"""

from graphrag_service import GraphRAGService
import os

def main():
    """تست متن کوتاه‌تر"""
    print("🧪 تست متن کوتاه‌تر برای مدل")
    print("=" * 40)
    
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
        "How do genes interact with TP53?"
    ]
    
    print(f"\n📋 تست {len(test_queries)} سوال:")
    print("-" * 40)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 تست {i}/{len(test_queries)}: {query}")
        print("-" * 30)
        
        try:
            # تست بازیابی خلاصه
            result = graphrag_service.test_compact_retrieval(query)
            
            # نمایش خلاصه
            print(f"✅ طول متن: {result['text_length']} کاراکتر")
            
            if result['text_length'] > 1000:
                print("❌ متن خیلی طولانی است!")
            elif result['text_length'] > 500:
                print("⚠️ متن متوسط است")
            else:
                print("✅ متن کوتاه و مناسب است")
            
        except Exception as e:
            print(f"❌ خطا: {e}")
        
        print()
    
    print("🎉 تست کامل شد!")

if __name__ == "__main__":
    main() 
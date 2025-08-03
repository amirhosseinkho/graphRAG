#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست سیستم بهبود یافته
"""

from graphrag_service import GraphRAGService
import os

def main():
    """تست سیستم بهبود یافته"""
    print("🧪 تست سیستم بهبود یافته")
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
            # تست بازیابی بهبود یافته
            result = graphrag_service.test_compact_retrieval(query)
            
            # نمایش خلاصه
            print(f"✅ طول متن: {result['text_length']} کاراکتر")
            
            if result['text_length'] > 1500:
                print("❌ متن خیلی طولانی است!")
            elif result['text_length'] > 800:
                print("⚠️ متن متوسط است")
            elif result['text_length'] > 400:
                print("✅ متن مناسب است")
            else:
                print("✅ متن کوتاه و عالی است")
            
            # بررسی کیفیت محتوا
            retrieval_data = result['retrieval_data']
            if retrieval_data['primary_genes']:
                genes_with_info = sum(1 for gene in retrieval_data['primary_genes'] 
                                    if gene.get('biological_processes') or gene.get('pathways') or gene.get('diseases'))
                print(f"📊 کیفیت محتوا: {genes_with_info}/{len(retrieval_data['primary_genes'])} ژن با اطلاعات زیستی")
            
            if retrieval_data['metaedges_used']:
                print(f"🔗 روابط استفاده شده: {len(retrieval_data['metaedges_used'])} نوع")
            
        except Exception as e:
            print(f"❌ خطا: {e}")
        
        print()
    
    print("🎉 تست کامل شد!")

if __name__ == "__main__":
    main() 
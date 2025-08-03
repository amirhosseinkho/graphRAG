#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست سوال مربوط به اثر بیماری بر بافت‌ها
"""

from graphrag_service import GraphRAGService
import os

def main():
    """تست سوال مربوط به اثر بیماری بر بافت‌ها"""
    print("🧪 تست سوال: How does cancer affect different tissues?")
    print("=" * 60)
    
    # راه‌اندازی سرویس
    graph_files = [f for f in os.listdir('.') if f.startswith('hetionet_graph_') and f.endswith('.pkl')]
    if graph_files:
        latest_graph_file = max(graph_files)
        print(f"🔧 استفاده از گراف Hetionet: {latest_graph_file}")
        graphrag_service = GraphRAGService(graph_data_path=latest_graph_file)
    else:
        print("⚠️ فایل گراف Hetionet یافت نشد، استفاده از گراف نمونه")
        graphrag_service = GraphRAGService()
    
    # تست سوال اصلی
    query = "How does cancer affect different tissues?"
    
    print(f"\n🔍 تست سوال: {query}")
    print("-" * 50)
    
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
            print(f"📋 Metaedges: {', '.join(retrieval_data['metaedges_used'])}")
        
        # بررسی مسیرهای ترکیبی
        if retrieval_data.get('tissue_disease_paths'):
            print(f"🔄 مسیرهای ترکیبی یافت شده: {len(retrieval_data['tissue_disease_paths'])}")
            for i, path in enumerate(retrieval_data['tissue_disease_paths'][:3], 1):
                print(f"  {i}. {path['disease']} → {path['tissue']} ({len(path['genes'])} ژن)")
        else:
            print("⚠️ هیچ مسیر ترکیبی یافت نشد")
        
        # نمایش متن کامل
        print(f"\n📝 متن کامل برای مدل:")
        print("-" * 60)
        print(result['structured_text'])
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ خطا: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 تست کامل شد!")

if __name__ == "__main__":
    main() 
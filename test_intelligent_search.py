#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست بهبودهای intelligent_semantic_search
این فایل تغییرات جدید در جستجوی ژن‌های بیان شده را آزمایش می‌کند
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_improved_intelligent_search():
    """تست بهبودهای intelligent_semantic_search"""
    print("🧪 تست بهبودهای intelligent_semantic_search")
    print("=" * 60)
    
    # ایجاد سرویس
    service = GraphRAGService()
    
    # تست سوال اصلی
    test_query = "What genes are expressed in the heart?"
    print(f"\n🔍 تست سوال: {test_query}")
    
    # تست 1: جستجوی مستقیم intelligent_semantic_search
    print("\n📋 تست 1: جستجوی مستقیم intelligent_semantic_search")
    try:
        results = service.intelligent_semantic_search(test_query, max_depth=3)
        print(f"✅ تعداد نتایج: {len(results)}")
        
        if results:
            print("📊 نتایج یافت شده:")
            for i, (node_id, depth, score, explanation) in enumerate(results[:5], 1):
                node_name = service.G.nodes[node_id]['name']
                node_kind = service.G.nodes[node_id]['kind']
                print(f"  {i}. {node_name} ({node_kind}) - عمق: {depth}, امتیاز: {score:.2f}")
                print(f"     توضیح: {explanation}")
        else:
            print("❌ هیچ نتیجه‌ای یافت نشد")
    except Exception as e:
        print(f"❌ خطا در تست 1: {e}")
    
    # تست 2: پردازش کامل سوال
    print("\n📋 تست 2: پردازش کامل سوال")
    try:
        result = service.process_query(
            query=test_query,
            retrieval_method=RetrievalMethod.INTELLIGENT,
            generation_model=GenerationModel.GPT_SIMULATION,
            max_depth=3
        )
        
        print("✅ پردازش کامل انجام شد")
        print(f"📊 تعداد نودهای بازیابی شده: {len(result['retrieval_result'].nodes)}")
        print(f"📊 تعداد یال‌های بازیابی شده: {len(result['retrieval_result'].edges)}")
        
        # نمایش پاسخ
        print("\n💬 پاسخ تولید شده:")
        print(result['generation_result'].answer)
        
    except Exception as e:
        print(f"❌ خطا در تست 2: {e}")
    
    # تست 3: مقایسه روش‌های مختلف
    print("\n📋 تست 3: مقایسه روش‌های مختلف")
    methods = [
        RetrievalMethod.INTELLIGENT,
        RetrievalMethod.ADAPTIVE,
        RetrievalMethod.DFS
    ]
    
    for method in methods:
        print(f"\n🔍 تست روش: {method.value}")
        try:
            result = service.process_query(
                query=test_query,
                retrieval_method=method,
                generation_model=GenerationModel.SIMPLE,
                max_depth=2
            )
            
            node_count = len(result['retrieval_result'].nodes)
            edge_count = len(result['retrieval_result'].edges)
            print(f"  📊 نودها: {node_count}, یال‌ها: {edge_count}")
            
            # بررسی وجود ژن‌های بیان شده
            gene_nodes = [n for n in result['retrieval_result'].nodes if n.kind == 'Gene']
            anatomy_nodes = [n for n in result['retrieval_result'].nodes if n.kind == 'Anatomy']
            aeG_edges = [e for e in result['retrieval_result'].edges if e.relation == 'AeG']
            
            print(f"  🧬 ژن‌ها: {len(gene_nodes)}, 🫀 آناتومی: {len(anatomy_nodes)}, 🔗 یال‌های AeG: {len(aeG_edges)}")
            
        except Exception as e:
            print(f"  ❌ خطا: {e}")
    
    # تست 4: سوالات مشابه
    print("\n📋 تست 4: سوالات مشابه")
    similar_queries = [
        "What genes are expressed in the brain?",
        "Which genes are expressed in the liver?",
        "Genes expressed in the heart",
        "What genes are active in the heart?",
        "Heart gene expression"
    ]
    
    for query in similar_queries:
        print(f"\n🔍 تست: {query}")
        try:
            results = service.intelligent_semantic_search(query, max_depth=2)
            gene_count = len([r for r in results if service.G.nodes[r[0]]['kind'] == 'Gene'])
            print(f"  ✅ ژن‌های یافت شده: {gene_count}")
        except Exception as e:
            print(f"  ❌ خطا: {e}")
    
    print("\n" + "=" * 60)
    print("✅ تست‌ها کامل شد")

def test_anatomy_expression_specific():
    """تست خاص برای بیان ژن در آناتومی"""
    print("\n🧬 تست خاص برای بیان ژن در آناتومی")
    print("=" * 60)
    
    service = GraphRAGService()
    
    # تست تابع جدید _search_genes_expressed_in_anatomy
    print("\n📋 تست تابع _search_genes_expressed_in_anatomy")
    
    # ایجاد داده‌های تست
    test_matched_nodes = {}
    test_intent = {'question_type': 'anatomy_expression'}
    
    # یافتن نود آناتومی heart
    for node_id, attrs in service.G.nodes(data=True):
        if attrs.get('kind') == 'Anatomy' and 'heart' in attrs.get('name', '').lower():
            test_matched_nodes['heart'] = node_id
            break
    
    if test_matched_nodes:
        print(f"✅ نود آناتومی یافت شد: {test_matched_nodes}")
        
        try:
            results = service._search_genes_expressed_in_anatomy(test_matched_nodes, test_intent, max_depth=2)
            print(f"✅ تعداد ژن‌های بیان شده یافت شد: {len(results)}")
            
            for gene_id, depth, score, explanation in results:
                gene_name = service.G.nodes[gene_id]['name']
                print(f"  🧬 {gene_name} - امتیاز: {score:.2f}")
                print(f"     توضیح: {explanation}")
                
        except Exception as e:
            print(f"❌ خطا در تست تابع: {e}")
    else:
        print("❌ نود آناتومی heart یافت نشد")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    print("🚀 شروع تست‌های بهبود intelligent_semantic_search")
    
    # تست اصلی
    test_improved_intelligent_search()
    
    # تست خاص
    test_anatomy_expression_specific()
    
    print("\n🎉 تمام تست‌ها کامل شد!") 
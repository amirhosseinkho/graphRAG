# -*- coding: utf-8 -*-
"""
تست الگوریتم‌های جدید و بهبود یافته
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_graphrag_service import EnhancedGraphRAGService, RetrievalAlgorithm
import networkx as nx
import json

def create_test_graph():
    """ایجاد گراف تست"""
    G = nx.Graph()
    
    # اضافه کردن نودهای ژن
    genes = ['TP53', 'BRCA1', 'BRCA2', 'EGFR', 'KRAS', 'PIK3CA', 'CDK1', 'CCNB1', 'BCL2', 'BAX']
    for gene in genes:
        G.add_node(gene, kind='Gene')
    
    # اضافه کردن نودهای بیماری
    diseases = ['Cancer', 'Breast Cancer', 'Lung Cancer', 'Diabetes']
    for disease in diseases:
        G.add_node(disease, kind='Disease')
    
    # اضافه کردن نودهای دارو
    drugs = ['Tamoxifen', 'Cisplatin', 'Doxorubicin']
    for drug in drugs:
        G.add_node(drug, kind='Drug')
    
    # اضافه کردن یال‌ها
    edges = [
        # روابط ژن-بیماری
        ('TP53', 'Cancer', {'relation': 'DaG', 'weight': 2.0}),
        ('BRCA1', 'Breast Cancer', {'relation': 'DaG', 'weight': 2.0}),
        ('BRCA2', 'Breast Cancer', {'relation': 'DaG', 'weight': 2.0}),
        ('EGFR', 'Lung Cancer', {'relation': 'DaG', 'weight': 1.5}),
        ('KRAS', 'Lung Cancer', {'relation': 'DaG', 'weight': 1.5}),
        
        # روابط ژن-ژن
        ('TP53', 'CDK1', {'relation': 'GiG', 'weight': 1.0}),
        ('TP53', 'CCNB1', {'relation': 'GiG', 'weight': 1.0}),
        ('BRCA1', 'BRCA2', {'relation': 'GiG', 'weight': 1.5}),
        ('EGFR', 'KRAS', {'relation': 'GiG', 'weight': 1.0}),
        ('BCL2', 'BAX', {'relation': 'GiG', 'weight': 1.0}),
        
        # روابط دارو-ژن
        ('Tamoxifen', 'BRCA1', {'relation': 'CtD', 'weight': 1.0}),
        ('Cisplatin', 'TP53', {'relation': 'CtD', 'weight': 1.0}),
        ('Doxorubicin', 'BCL2', {'relation': 'CtD', 'weight': 1.0}),
        
        # روابط بیماری-دارو
        ('Breast Cancer', 'Tamoxifen', {'relation': 'CtD', 'weight': 1.5}),
        ('Cancer', 'Cisplatin', {'relation': 'CtD', 'weight': 1.5}),
    ]
    
    for source, target, attrs in edges:
        G.add_edge(source, target, **attrs)
    
    return G

def test_enhanced_algorithms():
    """تست الگوریتم‌های جدید"""
    print("🧪 شروع تست الگوریتم‌های جدید...")
    
    # ایجاد سرویس
    service = EnhancedGraphRAGService()
    
    # ایجاد گراف تست
    test_graph = create_test_graph()
    service.G = test_graph
    
    # سوالات تست
    test_queries = [
        "Tell me 10 genes that are about cancer",
        "What genes are associated with breast cancer?",
        "How do TP53 and BRCA1 relate to cancer?",
        "What drugs treat cancer?",
        "Show me gene clusters related to cancer"
    ]
    
    # الگوریتم‌های جدید
    new_algorithms = [
        RetrievalAlgorithm.MULTI_METHOD,
        RetrievalAlgorithm.GROUP_BASED,
        RetrievalAlgorithm.ENHANCED_N_HOP,
        RetrievalAlgorithm.TARGETED_PAGERANK,
        RetrievalAlgorithm.SHORTEST_PATH_ENHANCED,
        RetrievalAlgorithm.BIOLOGICAL_PATHWAY,
        RetrievalAlgorithm.GENE_CLUSTER,
        RetrievalAlgorithm.DISEASE_GENE_NETWORK
    ]
    
    results = {}
    
    for algorithm in new_algorithms:
        print(f"\n🔍 تست الگوریتم: {algorithm.value}")
        service.set_config(retrieval_algorithm=algorithm)
        
        algorithm_results = {}
        
        for i, query in enumerate(test_queries):
            try:
                result = service.process_query(query)
                algorithm_results[f"query_{i+1}"] = {
                    'query': query,
                    'nodes_found': len(result.get('nodes', [])),
                    'edges_found': len(result.get('edges', [])),
                    'success': True
                }
                print(f"  ✅ سوال {i+1}: {len(result.get('nodes', []))} نود یافت شد")
            except Exception as e:
                algorithm_results[f"query_{i+1}"] = {
                    'query': query,
                    'error': str(e),
                    'success': False
                }
                print(f"  ❌ سوال {i+1}: خطا - {e}")
        
        results[algorithm.value] = algorithm_results
    
    # ذخیره نتایج
    with open('enhanced_algorithms_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 نتایج تست در فایل 'enhanced_algorithms_test_results.json' ذخیره شد")
    
    return results

def test_algorithm_comparison():
    """مقایسه الگوریتم‌های قدیمی و جدید"""
    print("\n📈 مقایسه الگوریتم‌های قدیمی و جدید...")
    
    service = EnhancedGraphRAGService()
    test_graph = create_test_graph()
    service.G = test_graph
    
    query = "Tell me genes related to cancer"
    
    # الگوریتم‌های قدیمی
    old_algorithms = [
        RetrievalAlgorithm.BFS,
        RetrievalAlgorithm.DFS,
        RetrievalAlgorithm.PAGERANK,
        RetrievalAlgorithm.N_HOP,
        RetrievalAlgorithm.HYBRID
    ]
    
    # الگوریتم‌های جدید
    new_algorithms = [
        RetrievalAlgorithm.MULTI_METHOD,
        RetrievalAlgorithm.TARGETED_PAGERANK,
        RetrievalAlgorithm.ENHANCED_N_HOP,
        RetrievalAlgorithm.BIOLOGICAL_PATHWAY
    ]
    
    comparison_results = {
        'old_algorithms': {},
        'new_algorithms': {}
    }
    
    print("\n🔍 الگوریتم‌های قدیمی:")
    for algorithm in old_algorithms:
        service.set_config(retrieval_algorithm=algorithm)
        try:
            result = service.process_query(query)
            comparison_results['old_algorithms'][algorithm.value] = {
                'nodes_found': len(result.get('nodes', [])),
                'edges_found': len(result.get('edges', [])),
                'success': True
            }
            print(f"  {algorithm.value}: {len(result.get('nodes', []))} نود")
        except Exception as e:
            comparison_results['old_algorithms'][algorithm.value] = {
                'error': str(e),
                'success': False
            }
            print(f"  {algorithm.value}: خطا")
    
    print("\n🔍 الگوریتم‌های جدید:")
    for algorithm in new_algorithms:
        service.set_config(retrieval_algorithm=algorithm)
        try:
            result = service.process_query(query)
            comparison_results['new_algorithms'][algorithm.value] = {
                'nodes_found': len(result.get('nodes', [])),
                'edges_found': len(result.get('edges', [])),
                'success': True
            }
            print(f"  {algorithm.value}: {len(result.get('nodes', []))} نود")
        except Exception as e:
            comparison_results['new_algorithms'][algorithm.value] = {
                'error': str(e),
                'success': False
            }
            print(f"  {algorithm.value}: خطا")
    
    # ذخیره نتایج مقایسه
    with open('algorithm_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 نتایج مقایسه در فایل 'algorithm_comparison_results.json' ذخیره شد")
    
    return comparison_results

def test_specific_features():
    """تست ویژگی‌های خاص الگوریتم‌های جدید"""
    print("\n🎯 تست ویژگی‌های خاص...")
    
    service = EnhancedGraphRAGService()
    test_graph = create_test_graph()
    service.G = test_graph
    
    # تست PageRank هدفمند
    print("\n🎯 تست PageRank هدفمند:")
    service.set_config(retrieval_algorithm=RetrievalAlgorithm.TARGETED_PAGERANK)
    result = service.process_query("Find important genes")
    gene_rankings = result.get('gene_rankings', [])
    print(f"  ژن‌های رتبه‌بندی شده: {len(gene_rankings)}")
    for ranking in gene_rankings[:3]:
        print(f"    {ranking['gene']}: {ranking['pagerank_score']:.4f}")
    
    # تست چندروشی
    print("\n🎯 تست چندروشی:")
    service.set_config(retrieval_algorithm=RetrievalAlgorithm.MULTI_METHOD)
    result = service.process_query("Find cancer genes")
    gene_coverage = result.get('gene_coverage', 0)
    print(f"  پوشش ژن‌ها: {gene_coverage}")
    
    # تست مسیر زیستی
    print("\n🎯 تست مسیر زیستی:")
    service.set_config(retrieval_algorithm=RetrievalAlgorithm.BIOLOGICAL_PATHWAY)
    result = service.process_query("Show biological pathways")
    pathways = result.get('pathways', [])
    print(f"  مسیرهای زیستی یافت شده: {len(pathways)}")
    
    # تست خوشه ژنی
    print("\n🎯 تست خوشه ژنی:")
    service.set_config(retrieval_algorithm=RetrievalAlgorithm.GENE_CLUSTER)
    result = service.process_query("Find gene clusters")
    clusters = result.get('clusters', [])
    print(f"  خوشه‌های ژنی یافت شده: {len(clusters)}")
    
    print("\n✅ تست ویژگی‌های خاص تکمیل شد")

if __name__ == "__main__":
    print("🚀 شروع تست الگوریتم‌های جدید و بهبود یافته")
    
    # تست اصلی
    test_enhanced_algorithms()
    
    # تست مقایسه
    test_algorithm_comparison()
    
    # تست ویژگی‌های خاص
    test_specific_features()
    
    print("\n🎉 تمام تست‌ها تکمیل شد!") 
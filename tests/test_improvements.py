# -*- coding: utf-8 -*-
"""
تست بهبودهای GraphRAG
"""

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_improvements():
    """تست بهبودهای اعمال شده"""
    print("🧪 شروع تست بهبودهای GraphRAG...")
    
    # راه‌اندازی سرویس
    service = GraphRAGService()
    
    # تست 1: تطبیق نودها با نوع موجودیت
    print("\n1️⃣ تست تطبیق نودها با نوع موجودیت:")
    test_query = "What genes are expressed in the heart?"
    keywords = service.extract_keywords(test_query)
    print(f"کلمات کلیدی: {keywords}")
    
    matches = service.match_tokens_to_nodes(keywords)
    print(f"تطبیق‌های یافت شده: {matches}")
    
    for token, node_id in matches.items():
        node_name = service.G.nodes[node_id]['name']
        node_kind = service.G.nodes[node_id]['kind']
        print(f"  '{token}' -> {node_name} ({node_kind})")
    
    # تست 2: جستجوی هوشمند آناتومی
    print("\n2️⃣ تست جستجوی هوشمند آناتومی:")
    result = service.process_query(
        query="What genes are expressed in the heart?",
        retrieval_method=RetrievalMethod.INTELLIGENT,
        generation_model=GenerationModel.CUSTOM,
        max_depth=3
    )
    
    print(f"تعداد نودهای یافت شده: {len(result['retrieval_result']['nodes'])}")
    print(f"تعداد یال‌های یافت شده: {len(result['retrieval_result']['edges'])}")
    
    # نمایش نودهای ژن
    gene_nodes = [n for n in result['retrieval_result']['nodes'] if n['kind'] == 'Gene']
    print(f"ژن‌های یافت شده: {[n['name'] for n in gene_nodes]}")
    
    # تست 3: جستجوی تطبیقی
    print("\n3️⃣ تست جستجوی تطبیقی:")
    result_adaptive = service.process_query(
        query="What genes are expressed in the heart?",
        retrieval_method=RetrievalMethod.ADAPTIVE,
        generation_model=GenerationModel.CUSTOM,
        max_depth=3
    )
    
    print(f"تعداد نودهای یافت شده (تطبیقی): {len(result_adaptive['retrieval_result']['nodes'])}")
    
    # تست 4: مقایسه روش‌ها
    print("\n4️⃣ مقایسه روش‌های مختلف:")
    methods = [
        RetrievalMethod.BFS,
        RetrievalMethod.DFS,
        RetrievalMethod.INTELLIGENT,
        RetrievalMethod.ADAPTIVE
    ]
    
    for method in methods:
        result = service.process_query(
            query="What genes are expressed in the heart?",
            retrieval_method=method,
            generation_model=GenerationModel.CUSTOM,
            max_depth=2
        )
        
        gene_count = len([n for n in result['retrieval_result']['nodes'] if n['kind'] == 'Gene'])
        print(f"  {method.value}: {gene_count} ژن یافت شد")
    
    # تست 5: پاسخ‌دهی بهبود یافته
    print("\n5️⃣ تست پاسخ‌دهی بهبود یافته:")
    result = service.process_query(
        query="What genes are expressed in the heart?",
        retrieval_method=RetrievalMethod.INTELLIGENT,
        generation_model=GenerationModel.CUSTOM,
        max_depth=3
    )
    
    print("پاسخ تولید شده:")
    print(result['answer'])
    
    print("\n✅ تست‌ها تکمیل شد!")

if __name__ == "__main__":
    test_improvements() 
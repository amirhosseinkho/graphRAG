#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test to debug shortest path issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag_service import GraphRAGService, RetrievalMethod

def test_shortest_path_issue():
    """Test the shortest path retrieval issue"""
    
    # Initialize service
    service = GraphRAGService()
    service.initialize()
    
    # Test queries
    test_queries = [
        "ژن TP53 چه نقشی در سرطان دارد؟",
        "TP53",
        "BRCA1",
        "سرطان سینه",
        "ژن TP53 و BRCA1",
        "ژن TP53 و سرطان سینه"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"تست سوال: {query}")
        print(f"{'='*50}")
        
        # Extract keywords
        keywords = service.extract_keywords(query)
        print(f"کلمات کلیدی استخراج شده: {keywords}")
        
        # Match tokens to nodes
        matches = service.match_tokens_to_nodes(keywords)
        print(f"نودهای تطبیق یافته: {matches}")
        
        if len(matches) < 2:
            print("⚠️ مشکل: کمتر از 2 نود پیدا شد - SHORTEST_PATH نیاز به حداقل 2 نود دارد")
            print("پیشنهاد: از روش‌های دیگر مانند BFS یا DFS استفاده کنید")
        else:
            print("✅ تعداد نودها کافی است")
            
            # Test shortest path
            try:
                node_ids = list(matches.values())
                paths = service.get_shortest_paths(node_ids[0], node_ids[1])
                print(f"مسیرهای پیدا شده: {paths}")
                
                if not paths:
                    print("⚠️ هیچ مسیری بین نودها پیدا نشد")
                else:
                    print("✅ مسیر پیدا شد")
                    
            except Exception as e:
                print(f"❌ خطا در یافتن مسیر: {e}")
        
        # Test other methods
        print("\nتست روش‌های دیگر:")
        
        # BFS
        try:
            result = service.retrieve_information(query, RetrievalMethod.BFS, max_depth=2)
            print(f"BFS - تعداد نودها: {len(result.nodes)}")
        except Exception as e:
            print(f"❌ خطا در BFS: {e}")
        
        # DFS
        try:
            result = service.retrieve_information(query, RetrievalMethod.DFS, max_depth=2)
            print(f"DFS - تعداد نودها: {len(result.nodes)}")
        except Exception as e:
            print(f"❌ خطا در DFS: {e}")
        
        # NEIGHBORS
        try:
            result = service.retrieve_information(query, RetrievalMethod.NEIGHBORS, max_depth=2)
            print(f"NEIGHBORS - تعداد نودها: {len(result.nodes)}")
        except Exception as e:
            print(f"❌ خطا در NEIGHBORS: {e}")

if __name__ == "__main__":
    test_shortest_path_issue() 
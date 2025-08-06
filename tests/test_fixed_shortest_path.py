#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test to verify that the shortest path issue is fixed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag_service import GraphRAGService, RetrievalMethod

def test_fixed_shortest_path():
    """Test that shortest path now works with fallback to BFS"""
    
    # Initialize service
    service = GraphRAGService()
    service.initialize()
    
    # Test queries that might have only one node
    test_queries = [
        "ژن TP53",
        "TP53",
        "سرطان سینه",
        "BRCA1",
        "ژن TP53 چه نقشی دارد؟"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"تست سوال: {query}")
        print(f"{'='*60}")
        
        try:
            # Test SHORTEST_PATH method
            result = service.retrieve_information(query, RetrievalMethod.SHORTEST_PATH, max_depth=2)
            print(f"✅ SHORTEST_PATH - تعداد نودها: {len(result.nodes)}")
            print(f"   نودهای پیدا شده: {[node.name for node in result.nodes[:5]]}")
            
            if result.nodes:
                print("✅ مشکل برطرف شد - SHORTEST_PATH با موفقیت کار کرد")
            else:
                print("⚠️ هنوز هیچ نودی پیدا نشد")
                
        except Exception as e:
            print(f"❌ خطا در SHORTEST_PATH: {e}")
        
        # Also test BFS for comparison
        try:
            result = service.retrieve_information(query, RetrievalMethod.BFS, max_depth=2)
            print(f"✅ BFS - تعداد نودها: {len(result.nodes)}")
        except Exception as e:
            print(f"❌ خطا در BFS: {e}")
        
        # Test DFS for comparison
        try:
            result = service.retrieve_information(query, RetrievalMethod.DFS, max_depth=2)
            print(f"✅ DFS - تعداد نودها: {len(result.nodes)}")
        except Exception as e:
            print(f"❌ خطا در DFS: {e}")

if __name__ == "__main__":
    test_fixed_shortest_path() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست ساده برای بررسی مشکل کلمات فارسی
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def test_simple_persian():
    """تست ساده کلمات فارسی"""
    
    service = GraphRAGService()
    service.initialize()
    
    # تست کلمات فارسی مختلف
    persian_words = [
        "سرطان",
        "کبد", 
        "مغز",
        "ژن",
        "دارو",
        "آسپرین",
        "بیماری",
        "بافت",
        "فرآیند",
        "آپوپتوز"
    ]
    
    print("🔍 تست کلمات فارسی")
    print("=" * 30)
    
    for word in persian_words:
        print(f"\n📝 کلمه: {word}")
        print("-" * 20)
        
        try:
            # استخراج کلمات کلیدی
            keywords = service.extract_keywords(word)
            print(f"🔑 کلمات کلیدی: {keywords}")
            
            # تطبیق با نودهای گراف
            matched_nodes = service.match_tokens_to_nodes(keywords)
            print(f"🎯 نودهای تطبیق یافته: {len(matched_nodes)}")
            
            for token, node_id in matched_nodes.items():
                node_name = service.G.nodes[node_id]['name']
                node_kind = service.G.nodes[node_id].get('kind', 'Unknown')
                print(f"   '{token}' -> {node_name} ({node_kind})")
            
        except Exception as e:
            print(f"❌ خطا: {e}")
        
        print()

if __name__ == "__main__":
    test_simple_persian() 
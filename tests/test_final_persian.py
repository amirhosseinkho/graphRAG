#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست نهایی سوالات فارسی - بررسی بهبودهای اعمال شده
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def test_final_persian():
    """تست نهایی کلمات فارسی"""
    
    service = GraphRAGService()
    service.initialize()
    
    # تست کلمات فارسی مختلف
    test_words = [
        "سرطان",
        "کبد", 
        "مغز",
        "ژن",
        "دارو",
        "آسپرین",
        "بیماری",
        "بافت",
        "فرآیند",
        "آپوپتوز",
        "TP53",
        "BRCA1"
    ]
    
    print("🔍 تست نهایی کلمات فارسی")
    print("=" * 40)
    
    for word in test_words:
        print(f"\n📝 کلمه: {word}")
        print("-" * 25)
        
        try:
            # استخراج کلمات کلیدی
            keywords = service.extract_keywords(word)
            print(f"🔑 کلمات کلیدی: {keywords}")
            
            # تطبیق با نودهای گراف
            matched_nodes = service.match_tokens_to_nodes(keywords)
            print(f"🎯 نودهای تطبیق یافته: {len(matched_nodes)}")
            
            if matched_nodes:
                for token, node_id in matched_nodes.items():
                    node_name = service.G.nodes[node_id]['name']
                    node_kind = service.G.nodes[node_id].get('kind', 'Unknown')
                    print(f"   '{token}' -> {node_name} ({node_kind})")
            else:
                print("❌ هیچ نودی تطبیق نیافت!")
                
        except Exception as e:
            print(f"❌ خطا: {e}")
        
        print()

def test_persian_sentences():
    """تست جملات فارسی"""
    
    service = GraphRAGService()
    service.initialize()
    
    # جملات فارسی برای تست
    sentences = [
        "ژن TP53 چه کاری انجام می‌دهد؟",
        "کدام ژن‌ها در کبد بیان می‌شوند؟",
        "سرطان سینه با کدام ژن‌ها مرتبط است؟",
        "آسپرین چه بیماری‌هایی را درمان می‌کند؟"
    ]
    
    print("\n🔍 تست جملات فارسی")
    print("=" * 40)
    
    for sentence in sentences:
        print(f"\n📝 جمله: {sentence}")
        print("-" * 30)
        
        try:
            keywords = service.extract_keywords(sentence)
            print(f"🔑 کلمات کلیدی: {keywords}")
            
            matched_nodes = service.match_tokens_to_nodes(keywords)
            print(f"🎯 نودهای تطبیق یافته: {len(matched_nodes)}")
            
            if matched_nodes:
                for token, node_id in matched_nodes.items():
                    node_name = service.G.nodes[node_id]['name']
                    node_kind = service.G.nodes[node_id].get('kind', 'Unknown')
                    print(f"   '{token}' -> {node_name} ({node_kind})")
            else:
                print("❌ هیچ نودی تطبیق نیافت!")
                
        except Exception as e:
            print(f"❌ خطا: {e}")
        
        print()

if __name__ == "__main__":
    print("🚀 شروع تست نهایی فارسی")
    test_final_persian()
    test_persian_sentences()
    print("\n✅ تست کامل شد!") 
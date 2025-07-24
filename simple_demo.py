# -*- coding: utf-8 -*-
"""
GraphRAG - Simple Demo (بدون دانلود)
این اسکریپت عملکرد پایه GraphRAG را نشان می‌دهد
"""

import pandas as pd
import networkx as nx
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import os

def create_sample_graph():
    """ایجاد یک گراف نمونه کوچک برای نمایش"""
    print("🔧 ایجاد گراف نمونه...")
    
    # ایجاد گراف نمونه
    G = nx.Graph()
    
    # افزودن نودهای نمونه
    sample_nodes = [
        ('Gene::HMGB3', 'HMGB3', 'Gene'),
        ('Gene::PCNA', 'PCNA', 'Gene'),
        ('Disease::Diabetes', 'Type 2 Diabetes', 'Disease'),
        ('Drug::Metformin', 'Metformin', 'Drug'),
        ('Biological Process::GO:0008150', 'Metabolic Process', 'Biological Process'),
        ('Anatomy::Heart', 'Heart', 'Anatomy'),
        ('Anatomy::Lung', 'Lung', 'Anatomy')
    ]
    
    for node_id, name, kind in sample_nodes:
        G.add_node(node_id, name=name, kind=kind)
    
    # افزودن یال‌های نمونه
    sample_edges = [
        ('Gene::HMGB3', 'Gene::PCNA', 'interacts_with'),
        ('Gene::PCNA', 'Disease::Diabetes', 'associates'),
        ('Drug::Metformin', 'Disease::Diabetes', 'treats'),
        ('Gene::HMGB3', 'Biological Process::GO:0008150', 'participates_in'),
        ('Anatomy::Heart', 'Anatomy::Lung', 'adjacent_to'),
        ('Gene::HMGB3', 'Anatomy::Heart', 'expressed_in')
    ]
    
    for source, target, relation in sample_edges:
        G.add_edge(source, target, metaedge=relation)
    
    print(f"گراف نمونه ساخته شد: {G.number_of_nodes()} نود، {G.number_of_edges()} یال")
    return G

def extract_keywords(text):
    """استخراج کلمات کلیدی از متن"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    keywords = set()
    
    # استخراج موجودیت‌های نام‌دار
    for ent in doc.ents:
        if ent.label_ not in {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}:
            keywords.add(ent.text.lower())
    
    # استخراج اسم‌ها و اسم خاص‌ها
    for token in doc:
        if (token.pos_ in {"NOUN", "PROPN"} and 
            token.text.lower() not in STOP_WORDS and 
            token.is_alpha and len(token.text) > 2):
            keywords.add(token.text.lower())
    
    return sorted(keywords)

def match_tokens_to_nodes(graph, tokens):
    """تطبیق توکن‌ها با نودهای گراف"""
    matched = {}
    for token in tokens:
        token_lower = token.lower()
        for node_id, attrs in graph.nodes(data=True):
            if token_lower in attrs['name'].lower():
                matched[token] = node_id
                break
    return matched

def get_shortest_path(graph, source, target):
    """یافتن کوتاه‌ترین مسیر بین دو نود"""
    try:
        path = nx.shortest_path(graph, source=source, target=target)
        return path
    except nx.NetworkXNoPath:
        return None

def get_neighbors_by_type(graph, node_id, kind_filter=None):
    """دریافت همسایه‌ها بر اساس نوع"""
    neighbors = []
    for neighbor in graph.neighbors(node_id):
        kind = graph.nodes[neighbor].get('kind')
        if kind_filter is None or kind == kind_filter:
            neighbors.append((neighbor, graph.nodes[neighbor]['name']))
    return neighbors

def demo():
    """نمایش عملکرد سیستم"""
    print("🚀 شروع نمایش GraphRAG (نسخه نمونه)...")
    
    # ایجاد گراف نمونه
    G = create_sample_graph()
    
    # نمایش نودها
    print("\n📋 نودهای موجود در گراف:")
    for node_id, attrs in G.nodes(data=True):
        print(f"  {node_id}: {attrs['name']} ({attrs['kind']})")
    
    # مثال سوال 1
    question1 = "What is the relationship between HMGB3 and diabetes?"
    print(f"\n❓ سوال 1: {question1}")
    
    tokens1 = extract_keywords(question1)
    print(f"🔍 کلمات کلیدی: {tokens1}")
    
    matches1 = match_tokens_to_nodes(G, tokens1)
    print(f"✅ تطبیق‌های یافت شده:")
    for token, node_id in matches1.items():
        print(f"  {token} → {G.nodes[node_id]['name']} ({G.nodes[node_id]['kind']})")
    
    # یافتن مسیر
    if len(matches1) >= 2:
        node_ids = list(matches1.values())
        path1 = get_shortest_path(G, node_ids[0], node_ids[1])
        
        if path1:
            print(f"\n🛤️ مسیر یافت شده:")
            for i, node in enumerate(path1):
                print(f"  {i+1}. {G.nodes[node]['name']} ({G.nodes[node]['kind']})")
                if i < len(path1) - 1:
                    edge_data = G.get_edge_data(node, path1[i+1])
                    print(f"     ↓ [{edge_data['metaedge']}]")
        else:
            print("❌ مسیری بین نودهای تطبیق یافته یافت نشد.")
    
    # مثال سوال 2
    question2 = "What drugs treat diabetes?"
    print(f"\n❓ سوال 2: {question2}")
    
    tokens2 = extract_keywords(question2)
    print(f"🔍 کلمات کلیدی: {tokens2}")
    
    matches2 = match_tokens_to_nodes(G, tokens2)
    print(f"✅ تطبیق‌های یافت شده:")
    for token, node_id in matches2.items():
        print(f"  {token} → {G.nodes[node_id]['name']} ({G.nodes[node_id]['kind']})")
    
    # جستجوی همسایه‌ها
    if 'diabetes' in matches2:
        diabetes_node = matches2['diabetes']
        neighbors = get_neighbors_by_type(G, diabetes_node, kind_filter='Drug')
        print(f"\n💊 داروهای مرتبط با دیابت:")
        for nid, name in neighbors:
            print(f"  - {name}")
    
    # مثال سوال 3
    question3 = "What genes are expressed in the heart?"
    print(f"\n❓ سوال 3: {question3}")
    
    tokens3 = extract_keywords(question3)
    print(f"🔍 کلمات کلیدی: {tokens3}")
    
    matches3 = match_tokens_to_nodes(G, tokens3)
    print(f"✅ تطبیق‌های یافت شده:")
    for token, node_id in matches3.items():
        print(f"  {token} → {G.nodes[node_id]['name']} ({G.nodes[node_id]['kind']})")
    
    # جستجوی همسایه‌ها
    if 'heart' in matches3:
        heart_node = matches3['heart']
        neighbors = get_neighbors_by_type(G, heart_node, kind_filter='Gene')
        print(f"\n🧬 ژن‌های بیان شده در قلب:")
        for nid, name in neighbors:
            print(f"  - {name}")

if __name__ == "__main__":
    demo() 
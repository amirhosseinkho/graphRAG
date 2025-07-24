# -*- coding: utf-8 -*-
"""
GraphRAG - Simple Demo Script
این اسکریپت یک نسخه ساده از پروژه GraphRAG را اجرا می‌کند
"""

import pandas as pd
import networkx as nx
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import deque
import pickle
import os

def download_data():
    """دانلود داده‌های Hetionet"""
    import urllib.request
    import gzip
    import subprocess
    import sys
    
    print("📥 دانلود داده‌های Hetionet...")
    
    # دانلود نودها
    if not os.path.exists('hetionet-v1.0-nodes.tsv'):
        print("دانلود فایل نودها...")
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/hetio/hetionet/master/hetnet/tsv/hetionet-v1.0-nodes.tsv',
            'hetionet-v1.0-nodes.tsv'
        )
    
    # دانلود یال‌ها - استفاده از wget یا curl
    if not os.path.exists('hetionet-v1.0-edges.sif') or os.path.getsize('hetionet-v1.0-edges.sif') == 0:
        print("دانلود فایل یال‌ها...")
        
        # حذف فایل‌های قدیمی
        if os.path.exists('hetionet-v1.0-edges.sif.gz'):
            os.remove('hetionet-v1.0-edges.sif.gz')
        if os.path.exists('hetionet-v1.0-edges.sif'):
            os.remove('hetionet-v1.0-edges.sif')
        
        try:
            # استفاده از PowerShell برای دانلود
            cmd = [
                'powershell', '-Command',
                'Invoke-WebRequest -Uri "https://raw.githubusercontent.com/hetio/hetionet/master/hetnet/tsv/hetionet-v1.0-edges.sif.gz" -OutFile "hetionet-v1.0-edges.sif.gz"'
            ]
            subprocess.run(cmd, check=True)
            
            # استخراج فایل فشرده
            with gzip.open('hetionet-v1.0-edges.sif.gz', 'rb') as f_in:
                with open('hetionet-v1.0-edges.sif', 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # حذف فایل فشرده
            os.remove('hetionet-v1.0-edges.sif.gz')
            print("فایل یال‌ها با موفقیت دانلود و استخراج شد.")
            
        except Exception as e:
            print(f"خطا در دانلود: {e}")
            print("تلاش برای دانلود مستقیم...")
            try:
                cmd = [
                    'powershell', '-Command',
                    'Invoke-WebRequest -Uri "https://raw.githubusercontent.com/hetio/hetionet/master/hetnet/tsv/hetionet-v1.0-edges.sif" -OutFile "hetionet-v1.0-edges.sif"'
                ]
                subprocess.run(cmd, check=True)
                print("فایل یال‌ها با موفقیت دانلود شد.")
            except Exception as e2:
                print(f"خطا در دانلود مستقیم: {e2}")
                print("لطفاً فایل‌ها را به صورت دستی دانلود کنید.")
                return False
    
    return True

def load_graph():
    """بارگذاری گراف"""
    print("📊 بارگذاری گراف...")
    
    # خواندن نودها
    nodes = pd.read_csv('hetionet-v1.0-nodes.tsv', sep='\t', encoding='utf-8-sig')
    print(f"تعداد نودها: {len(nodes)}")
    
    # خواندن یال‌ها
    edges = pd.read_csv('hetionet-v1.0-edges.sif', sep='\t')
    print(f"تعداد یال‌ها: {len(edges)}")
    
    # ساخت گراف
    G = nx.Graph()
    
    # افزودن نودها
    for _, row in nodes.iterrows():
        G.add_node(row['id'], name=row['name'], kind=row['kind'])
    
    # افزودن یال‌ها
    for _, row in edges.iterrows():
        G.add_edge(row['source'], row['target'], metaedge=row['metaedge'])
    
    print(f"گراف ساخته شد: {G.number_of_nodes()} نود، {G.number_of_edges()} یال")
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

def demo():
    """نمایش عملکرد سیستم"""
    print("🚀 شروع نمایش GraphRAG...")
    
    # دانلود داده‌ها
    if not download_data():
        print("❌ خطا در دانلود داده‌ها. برنامه متوقف می‌شود.")
        return
    
    # بارگذاری گراف
    G = load_graph()
    
    # مثال سوال
    question = "What is the relationship between HMGB3 and pulmonary valve formation?"
    print(f"\n❓ سوال: {question}")
    
    # استخراج کلمات کلیدی
    tokens = extract_keywords(question)
    print(f"🔍 کلمات کلیدی استخراج شده: {tokens}")
    
    # تطبیق با نودهای گراف
    matches = match_tokens_to_nodes(G, tokens)
    print(f"✅ تطبیق‌های یافت شده:")
    for token, node_id in matches.items():
        print(f"  {token} → {G.nodes[node_id]['name']} ({G.nodes[node_id]['kind']})")
    
    # یافتن مسیر بین دو نود
    if len(matches) >= 2:
        node_ids = list(matches.values())
        path = get_shortest_path(G, node_ids[0], node_ids[1])
        
        if path:
            print(f"\n🛤️ مسیر یافت شده بین '{G.nodes[node_ids[0]]['name']}' و '{G.nodes[node_ids[1]]['name']}':")
            for i, node in enumerate(path):
                print(f"  {i+1}. {G.nodes[node]['name']} ({G.nodes[node]['kind']})")
                if i < len(path) - 1:
                    edge_data = G.get_edge_data(node, path[i+1])
                    print(f"     ↓ [{edge_data['metaedge']}]")
        else:
            print("❌ مسیری بین نودهای تطبیق یافته یافت نشد.")
    else:
        print("❌ کمتر از 2 توکن در گراف تطبیق یافت.")

if __name__ == "__main__":
    demo() 
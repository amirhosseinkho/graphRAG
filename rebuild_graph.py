# -*- coding: utf-8 -*-
"""
Rebuild Graph Script - بازسازی گراف با داده‌های جدید
"""

import pandas as pd
import networkx as nx
import pickle
import os
from datetime import datetime

def rebuild_graph():
    """بازسازی گراف با داده‌های جدید"""
    print("🔧 شروع بازسازی گراف...")
    
    # بررسی وجود فایل‌ها
    nodes_file = 'hetionet-v1.0-nodes.tsv'
    edges_file = 'edges.sif'  # فایل جدید
    
    if not os.path.exists(nodes_file):
        print(f"❌ فایل نودها یافت نشد: {nodes_file}")
        return False
    
    if not os.path.exists(edges_file):
        print(f"❌ فایل یال‌ها یافت نشد: {edges_file}")
        return False
    
    try:
        # خواندن نودها
        print("📖 خواندن فایل نودها...")
        nodes = pd.read_csv(nodes_file, sep='\t', encoding='utf-8-sig')
        print(f"✅ {len(nodes)} نود خوانده شد")
        print("نمونه نودها:")
        print(nodes.head())
        
        # خواندن یال‌ها
        print("\n📖 خواندن فایل یال‌ها...")
        edges = pd.read_csv(edges_file, sep='\t')
        print(f"✅ {len(edges)} یال خوانده شد")
        print("نمونه یال‌ها:")
        print(edges.head())
        
        # بررسی ستون‌ها
        print(f"\nستون‌های نودها: {list(nodes.columns)}")
        print(f"ستون‌های یال‌ها: {list(edges.columns)}")
        
        # ساخت گراف
        print("\n🔧 ساخت گراف...")
        G = nx.Graph()
        
        # افزودن نودها
        print("➕ افزودن نودها...")
        for _, row in nodes.iterrows():
            node_id = row['id']
            node_name = row['name']
            node_kind = row['kind']
            G.add_node(node_id, name=node_name, kind=node_kind)
        
        print(f"✅ {G.number_of_nodes()} نود به گراف اضافه شد")
        
        # افزودن یال‌ها
        print("➕ افزودن یال‌ها...")
        edge_count = 0
        for _, row in edges.iterrows():
            try:
                source = row['source']
                target = row['target']
                metaedge = row['metaedge']
                
                # بررسی وجود نودها
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, metaedge=metaedge)
                    edge_count += 1
                else:
                    print(f"⚠️ نود یافت نشد: {source} یا {target}")
            except Exception as e:
                print(f"⚠️ خطا در افزودن یال: {e}")
                continue
        
        print(f"✅ {edge_count} یال به گراف اضافه شد")
        
        # آمار گراف
        print(f"\n📊 آمار گراف:")
        print(f"تعداد نودها: {G.number_of_nodes()}")
        print(f"تعداد یال‌ها: {G.number_of_edges()}")
        
        # آمار انواع نودها
        node_types = {}
        for node, attrs in G.nodes(data=True):
            kind = attrs.get('kind', 'Unknown')
            node_types[kind] = node_types.get(kind, 0) + 1
        
        print(f"\nانواع نودها:")
        for kind, count in sorted(node_types.items()):
            print(f"  {kind}: {count}")
        
        # آمار انواع یال‌ها
        edge_types = {}
        for _, _, attrs in G.edges(data=True):
            metaedge = attrs.get('metaedge', 'Unknown')
            edge_types[metaedge] = edge_types.get(metaedge, 0) + 1
        
        print(f"\nانواع یال‌ها:")
        for metaedge, count in sorted(edge_types.items()):
            print(f"  {metaedge}: {count}")
        
        # ذخیره گراف
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_filename = f"hetionet_graph_{timestamp}.pkl"
        
        print(f"\n💾 ذخیره گراف در فایل: {graph_filename}")
        with open(graph_filename, "wb") as f:
            pickle.dump(G, f)
        
        # ایجاد فایل آمار
        stats_filename = f"graph_stats_{timestamp}.txt"
        with open(stats_filename, "w", encoding="utf-8") as f:
            f.write("آمار گراف Hetionet\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"تاریخ ایجاد: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"تعداد کل نودها: {G.number_of_nodes()}\n")
            f.write(f"تعداد کل یال‌ها: {G.number_of_edges()}\n\n")
            
            f.write("انواع نودها:\n")
            f.write("-" * 20 + "\n")
            for kind, count in sorted(node_types.items()):
                f.write(f"{kind}: {count}\n")
            
            f.write("\nانواع یال‌ها:\n")
            f.write("-" * 20 + "\n")
            for metaedge, count in sorted(edge_types.items()):
                f.write(f"{metaedge}: {count}\n")
        
        print(f" آمار گراف در فایل: {stats_filename}")
        
        # تست عملکرد
        print(f"\n تست عملکرد گراف...")
        
        # پیدا کردن چند نود نمونه
        sample_nodes = list(G.nodes())[:5]
        print(f"نودهای نمونه: {sample_nodes}")
        
        for node in sample_nodes:
            neighbors = list(G.neighbors(node))
            print(f"نود {G.nodes[node]['name']} ({G.nodes[node]['kind']}): {len(neighbors)} همسایه")
        
        print(f"\n گراف با موفقیت بازسازی شد!")
        print(f"فایل گراف: {graph_filename}")
        print(f"فایل آمار: {stats_filename}")
        
        return True
        
    except Exception as e:
        print(f" خطا در بازسازی گراف: {e}")
        import traceback
        traceback.print_exc()
        return False

def update_service_graph():
    """به‌روزرسانی گراف در سرویس"""
    print("\n🔄 به‌روزرسانی گراف در سرویس...")
    
    # پیدا کردن جدیدترین فایل گراف
    graph_files = [f for f in os.listdir('.') if f.startswith('hetionet_graph_') and f.endswith('.pkl')]
    
    if not graph_files:
        print("❌ هیچ فایل گرافی یافت نشد")
        return False
    
    # انتخاب جدیدترین فایل
    latest_graph_file = max(graph_files)
    print(f"استفاده از فایل: {latest_graph_file}")
    
    # کپی به نام استاندارد
    import shutil
    shutil.copy(latest_graph_file, 'hetionet_graph.pkl')
    print(" فایل گراف به‌روزرسانی شد")
    
    return True

if __name__ == "__main__":
    print("🚀 شروع بازسازی گراف Hetionet")
    print("=" * 50)
    
    # بازسازی گراف
    if rebuild_graph():
        # به‌روزرسانی در سرویس
        update_service_graph()
        
        print("\n✅ عملیات کامل شد!")
        print("\nبرای استفاده از گراف جدید:")
        print("1. برنامه وب را متوقف کنید (Ctrl+C)")
        print("2. دوباره اجرا کنید: python web_app.py")
    else:
        print("\n❌ خطا در بازسازی گراف") 
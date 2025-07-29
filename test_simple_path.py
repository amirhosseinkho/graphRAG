#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تست ساده برای بررسی مسیر Heart → MMP9 → Caffeine
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def test_simple_path():
    """تست ساده برای بررسی مسیر Heart → MMP9 → Caffeine"""
    print("🔍 تست ساده برای بررسی مسیر Heart → MMP9 → Caffeine")
    
    # ایجاد سرویس
    service = GraphRAGService()
    service.initialize()
    
    # بررسی نودهای مورد نظر
    heart_node = 'Anatomy::Heart'
    mmp9_node = 'Gene::MMP9'
    caffeine_node = 'Compound::Caffeine'
    
    print(f"\n📋 بررسی نودها:")
    print(f"  Heart: {service.G.nodes[heart_node]['name']}")
    print(f"  MMP9: {service.G.nodes[mmp9_node]['name']}")
    print(f"  Caffeine: {service.G.nodes[caffeine_node]['name']}")
    
    # بررسی یال Heart → MMP9
    heart_mmp9_edge = service.G.get_edge_data(heart_node, mmp9_node)
    print(f"\n📋 یال Heart → MMP9:")
    if heart_mmp9_edge:
        print(f"  ✅ وجود دارد: {heart_mmp9_edge.get('relation', 'Unknown')}")
    else:
        print(f"  ❌ وجود ندارد")
    
    # بررسی یال MMP9 → Caffeine
    mmp9_caffeine_edge = service.G.get_edge_data(mmp9_node, caffeine_node)
    print(f"\n📋 یال MMP9 → Caffeine:")
    if mmp9_caffeine_edge:
        print(f"  ✅ وجود دارد: {mmp9_caffeine_edge.get('relation', 'Unknown')}")
    else:
        print(f"  ❌ وجود ندارد")
    
    # بررسی یال Caffeine → MMP9 (جهت معکوس)
    caffeine_mmp9_edge = service.G.get_edge_data(caffeine_node, mmp9_node)
    print(f"\n📋 یال Caffeine → MMP9:")
    if caffeine_mmp9_edge:
        print(f"  ✅ وجود دارد: {caffeine_mmp9_edge.get('relation', 'Unknown')}")
    else:
        print(f"  ❌ وجود ندارد")
    
    # تست DFS ساده
    print(f"\n📋 تست DFS ساده:")
    
    def simple_dfs(node, target, path, max_depth=2):
        if len(path) >= max_depth:
            return []
        
        path.append(node)
        results = []
        
        if node == target and len(path) > 1:
            results.append(path.copy())
        
        for neighbor in service.G.neighbors(node):
            if neighbor not in path:
                edge_data = service.G.get_edge_data(node, neighbor)
                if edge_data:
                    relation = edge_data.get('relation', 'Unknown')
                    print(f"    {service.G.nodes[node]['name']} → {service.G.nodes[neighbor]['name']} ({relation})")
                
                sub_results = simple_dfs(neighbor, target, path, max_depth)
                results.extend(sub_results)
        
        path.pop()
        return results
    
    # جستجوی مسیر از Heart به Caffeine
    print(f"\n📋 جستجوی مسیر از Heart به Caffeine:")
    paths = simple_dfs(heart_node, caffeine_node, [], max_depth=3)
    
    print(f"  تعداد مسیرها: {len(paths)}")
    for i, path in enumerate(paths):
        path_names = [service.G.nodes[node]['name'] for node in path]
        print(f"    مسیر {i+1}: {' → '.join(path_names)}")
    
    # تست الگوی AeG → CdG
    print(f"\n📋 تست الگوی AeG → CdG:")
    pattern = ['AeG', 'CdG']
    
    def find_pattern_paths(start_node, pattern, current_path=[], current_metaedges=[], depth=0):
        if depth >= 3:
            return []
        
        current_path.append(start_node)
        results = []
        
        # بررسی اینکه آیا مسیر فعلی با الگو مطابقت دارد
        if len(current_metaedges) == len(pattern):
            results.append((current_path.copy(), current_metaedges.copy()))
        
        # جستجوی همسایه‌ها
        for neighbor in service.G.neighbors(start_node):
            if neighbor not in current_path:
                edge_data = service.G.get_edge_data(start_node, neighbor)
                if edge_data and edge_data.get('relation'):
                    metaedge = edge_data.get('relation')
                    print(f"      بررسی: {service.G.nodes[start_node]['name']} → {service.G.nodes[neighbor]['name']} ({metaedge})")
                    
                    # بررسی اینکه آیا این metaedge در الگو است
                    if len(current_metaedges) < len(pattern) and metaedge == pattern[len(current_metaedges)]:
                        new_metaedges = current_metaedges + [metaedge]
                        print(f"        ✅ تطبیق: {metaedge} == {pattern[len(current_metaedges)]}")
                        sub_results = find_pattern_paths(neighbor, pattern, current_path, new_metaedges, depth + 1)
                        results.extend(sub_results)
                    elif metaedge in pattern:
                        new_metaedges = current_metaedges + [metaedge]
                        print(f"        ✅ در الگو: {metaedge} در {pattern}")
                        sub_results = find_pattern_paths(neighbor, pattern, current_path, new_metaedges, depth + 1)
                        results.extend(sub_results)
                    else:
                        print(f"        ❌ تطبیق نکرد: {metaedge} در {pattern}")
        
        current_path.pop()
        return results
    
    pattern_paths = find_pattern_paths(heart_node, pattern)
    print(f"  تعداد مسیرهای الگو: {len(pattern_paths)}")
    for i, (path, metaedges) in enumerate(pattern_paths):
        path_names = [service.G.nodes[node]['name'] for node in path]
        print(f"    مسیر {i+1}: {' → '.join(path_names)}")
        print(f"    Metaedges: {' → '.join(metaedges)}")
    
    print("\n✅ تست ساده برای بررسی مسیر تکمیل شد!")

if __name__ == "__main__":
    test_simple_path() 
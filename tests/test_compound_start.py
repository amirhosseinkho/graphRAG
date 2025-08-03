#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تست برای بررسی شروع از نودهای Compound
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def test_compound_start():
    """تست برای بررسی شروع از نودهای Compound"""
    print("🔍 تست برای بررسی شروع از نودهای Compound")
    
    # ایجاد سرویس
    service = GraphRAGService()
    service.initialize()
    
    # بررسی نودهای Compound
    print("\n📋 نودهای Compound:")
    compound_nodes = [nid for nid, attrs in service.G.nodes(data=True) 
                     if attrs.get('kind') == 'Compound' or attrs.get('metanode') == 'Compound']
    
    for compound_node in compound_nodes:
        compound_name = service.G.nodes[compound_node]['name']
        print(f"  • {compound_name}")
    
    # بررسی یال‌های CdG از Caffeine
    print("\n📋 یال‌های CdG از Caffeine:")
    caffeine_node = 'Compound::Caffeine'
    if service.G.has_node(caffeine_node):
        caffeine_neighbors = list(service.G.neighbors(caffeine_node))
        print(f"  همسایه‌های Caffeine: {len(caffeine_neighbors)}")
        
        for neighbor in caffeine_neighbors:
            neighbor_name = service.G.nodes[neighbor]['name']
            edge_data = service.G.get_edge_data(caffeine_node, neighbor)
            relation = edge_data.get('relation', 'Unknown') if edge_data else 'Unknown'
            print(f"    • Caffeine → {neighbor_name} ({relation})")
    
    # تست الگوی CdG → AeG (معکوس)
    print("\n📋 تست الگوی CdG → AeG (معکوس):")
    pattern = ['CdG', 'AeG']
    
    def find_reverse_pattern_paths(start_node, pattern, current_path=[], current_metaedges=[], depth=0):
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
                        sub_results = find_reverse_pattern_paths(neighbor, pattern, current_path, new_metaedges, depth + 1)
                        results.extend(sub_results)
                    elif metaedge in pattern:
                        new_metaedges = current_metaedges + [metaedge]
                        print(f"        ✅ در الگو: {metaedge} در {pattern}")
                        sub_results = find_reverse_pattern_paths(neighbor, pattern, current_path, new_metaedges, depth + 1)
                        results.extend(sub_results)
                    else:
                        print(f"        ❌ تطبیق نکرد: {metaedge} در {pattern}")
        
        current_path.pop()
        return results
    
    # شروع از Caffeine
    reverse_pattern_paths = find_reverse_pattern_paths(caffeine_node, pattern)
    print(f"  تعداد مسیرهای الگوی معکوس: {len(reverse_pattern_paths)}")
    for i, (path, metaedges) in enumerate(reverse_pattern_paths):
        path_names = [service.G.nodes[node]['name'] for node in path]
        print(f"    مسیر {i+1}: {' → '.join(path_names)}")
        print(f"    Metaedges: {' → '.join(metaedges)}")
    
    print("\n✅ تست برای بررسی شروع از نودهای Compound تکمیل شد!")

if __name__ == "__main__":
    test_compound_start() 
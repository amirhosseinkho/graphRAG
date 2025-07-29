#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تست بررسی مسیرهای واقعی در گراف
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def test_path_finding():
    """تست بررسی مسیرهای واقعی در گراف"""
    print("🔍 تست بررسی مسیرهای واقعی در گراف")
    
    # ایجاد سرویس
    service = GraphRAGService()
    service.initialize()
    
    # بررسی مسیرهای واقعی
    print("\n📋 بررسی مسیرهای واقعی:")
    
    # از Heart به Compound
    heart_node = 'Anatomy::Heart'
    if service.G.has_node(heart_node):
        print(f"  شروع از نود: {service.G.nodes[heart_node]['name']}")
        
        # بررسی همسایه‌های Heart
        heart_neighbors = list(service.G.neighbors(heart_node))
        print(f"  همسایه‌های Heart: {len(heart_neighbors)}")
        
        for neighbor in heart_neighbors:
            neighbor_name = service.G.nodes[neighbor]['name']
            edge_data = service.G.get_edge_data(heart_node, neighbor)
            relation = edge_data.get('relation', 'Unknown') if edge_data else 'Unknown'
            print(f"    • Heart → {neighbor_name} ({relation})")
            
            # بررسی همسایه‌های همسایه
            second_neighbors = list(service.G.neighbors(neighbor))
            print(f"      همسایه‌های {neighbor_name}: {len(second_neighbors)}")
            
            for second_neighbor in second_neighbors:
                second_name = service.G.nodes[second_neighbor]['name']
                second_edge_data = service.G.get_edge_data(neighbor, second_neighbor)
                second_relation = second_edge_data.get('relation', 'Unknown') if second_edge_data else 'Unknown'
                print(f"        • {neighbor_name} → {second_name} ({second_relation})")
    
    # بررسی مسیرهای AeG → CuG
    print("\n📋 بررسی مسیرهای AeG → CuG:")
    
    # پیدا کردن تمام یال‌های AeG
    aeG_edges = [(u, v) for u, v, data in service.G.edges(data=True) 
                if data.get('relation') == 'AeG']
    
    print(f"  تعداد یال‌های AeG: {len(aeG_edges)}")
    
    for source, target in aeG_edges[:3]:  # فقط 3 تا اول
        source_name = service.G.nodes[source]['name']
        target_name = service.G.nodes[target]['name']
        print(f"    {source_name} → {target_name} (AeG)")
        
        # بررسی همسایه‌های target
        target_neighbors = list(service.G.neighbors(target))
        cuG_neighbors = []
        
        for neighbor in target_neighbors:
            edge_data = service.G.get_edge_data(target, neighbor)
            if edge_data and edge_data.get('relation') == 'CuG':
                neighbor_name = service.G.nodes[neighbor]['name']
                cuG_neighbors.append(neighbor_name)
        
        if cuG_neighbors:
            print(f"      → {target_name} → {', '.join(cuG_neighbors)} (CuG)")
        else:
            print(f"      → {target_name} → هیچ CuG یافت نشد")
    
    # بررسی مسیرهای AeG → CdG
    print("\n📋 بررسی مسیرهای AeG → CdG:")
    
    for source, target in aeG_edges[:3]:  # فقط 3 تا اول
        source_name = service.G.nodes[source]['name']
        target_name = service.G.nodes[target]['name']
        print(f"    {source_name} → {target_name} (AeG)")
        
        # بررسی همسایه‌های target
        target_neighbors = list(service.G.neighbors(target))
        cdG_neighbors = []
        
        for neighbor in target_neighbors:
            edge_data = service.G.get_edge_data(target, neighbor)
            if edge_data and edge_data.get('relation') == 'CdG':
                neighbor_name = service.G.nodes[neighbor]['name']
                cdG_neighbors.append(neighbor_name)
        
        if cdG_neighbors:
            print(f"      → {target_name} → {', '.join(cdG_neighbors)} (CdG)")
        else:
            print(f"      → {target_name} → هیچ CdG یافت نشد")
    
    print("\n✅ تست بررسی مسیرهای واقعی تکمیل شد!")

if __name__ == "__main__":
    test_path_finding() 
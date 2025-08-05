#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست سیستم بهبود یافته NodeLookupSystem
و ادغام آن با GraphRAGService
"""

import sys
import os
from typing import Dict, List, Tuple, Optional
import json

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from node_lookup_system import NodeLookupSystem
from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_node_lookup_integration():
    """تست ادغام NodeLookupSystem با GraphRAGService"""
    print("🧬 تست ادغام NodeLookupSystem با GraphRAGService")
    print("=" * 60)
    
    # 1. راه‌اندازی سیستم‌ها
    print("📋 راه‌اندازی سیستم‌ها...")
    lookup_system = NodeLookupSystem()
    graphrag_service = GraphRAGService()
    
    print("✅ سیستم‌ها راه‌اندازی شدند")
    
    # 2. تست تبدیل شناسه‌ها به نام‌های معنادار
    print("\n🔍 تست تبدیل شناسه‌ها:")
    test_node_ids = [
        "Gene::7157",  # TP53
        "Gene::4087",  # SMAD2
        "Compound::DB00262",  # Carmustine
        "Disease::DOID:162",  # malignant glioma
        "Anatomy::UBERON:0000955",  # brain
        "Biological Process::GO:0006915"  # apoptosis
    ]
    
    for node_id in test_node_ids:
        node_info = lookup_system.get_node_info(node_id)
        if node_info:
            print(f"• {node_id} → {node_info.name} ({node_info.kind})")
            if node_info.description:
                print(f"  توضیح: {node_info.description}")
        else:
            print(f"• {node_id} → یافت نشد")
    
    # 3. تست نمایش یال‌ها
    print("\n🔗 تست نمایش یال‌ها:")
    test_edges = [
        ("Gene::7157", "Gene::4087", "GiG"),
        ("Compound::DB00262", "Disease::DOID:162", "CtD"),
        ("Gene::7157", "Biological Process::GO:0006915", "GpBP")
    ]
    
    for source, target, relation in test_edges:
        edge_display = lookup_system.format_edge_for_display(source, target, relation)
        print(f"• {edge_display}")
    
    # 4. تست نمایش مسیرها
    print("\n🛤️ تست نمایش مسیرها:")
    test_path = ["Gene::7157", "Gene::4087", "Disease::DOID:162"]
    test_edges_for_path = [
        ("Gene::7157", "Gene::4087", "GiG"),
        ("Gene::4087", "Disease::DOID:162", "DaG")
    ]
    
    path_display = lookup_system.format_path_for_display(test_path, test_edges_for_path)
    print(f"• مسیر: {path_display}")
    
    # 5. تست با GraphRAGService
    print("\n🔧 تست با GraphRAGService:")
    
    # ایجاد داده‌های نمونه برای تست
    sample_query = "What is the relationship between TP53 and cancer?"
    
    # شبیه‌سازی داده‌های بازیابی شده
    retrieved_nodes = [
        {"id": "Gene::7157", "name": "TP53", "kind": "Gene", "depth": 0, "score": 1.0},
        {"id": "Disease::DOID:162", "name": "malignant glioma", "kind": "Disease", "depth": 1, "score": 0.8},
        {"id": "Compound::DB00262", "name": "Carmustine", "kind": "Compound", "depth": 2, "score": 0.6}
    ]
    
    retrieved_edges = [
        {"source": "Gene::7157", "target": "Disease::DOID:162", "relation": "DaG", "weight": 1.0},
        {"source": "Compound::DB00262", "target": "Disease::DOID:162", "relation": "CtD", "weight": 0.9}
    ]
    
    # تبدیل به فرمت معنادار
    print("📝 تبدیل داده‌های بازیابی شده به فرمت معنادار:")
    
    enhanced_nodes = []
    for node in retrieved_nodes:
        node_id = node["id"]
        node_info = lookup_system.get_node_info(node_id)
        if node_info:
            enhanced_node = {
                "id": node_id,
                "name": node_info.name,
                "kind": node_info.kind,
                "description": node_info.description,
                "biological_role": node_info.biological_role,
                "significance": node_info.significance,
                "depth": node["depth"],
                "score": node["score"]
            }
            enhanced_nodes.append(enhanced_node)
            print(f"• {node_id} → {node_info.name} ({node_info.kind})")
            if node_info.description:
                print(f"  توضیح: {node_info.description}")
    
    enhanced_edges = []
    for edge in retrieved_edges:
        source_display = lookup_system.format_node_for_display(edge["source"])
        target_display = lookup_system.format_node_for_display(edge["target"])
        relation_desc = lookup_system.format_edge_for_display(edge["source"], edge["target"], edge["relation"])
        
        enhanced_edge = {
            "source": edge["source"],
            "target": edge["target"],
            "relation": edge["relation"],
            "source_display": source_display,
            "target_display": target_display,
            "relation_description": relation_desc,
            "weight": edge["weight"]
        }
        enhanced_edges.append(enhanced_edge)
        print(f"• {relation_desc}")
    
    # 6. ایجاد متن زمینه بهبود یافته
    print("\n📄 ایجاد متن زمینه بهبود یافته:")
    
    context_parts = []
    context_parts.append("🧬 **داده‌های بازیابی شده (فرمت معنادار):**")
    context_parts.append("")
    
    # اطلاعات نودها
    context_parts.append("📋 **نودهای کلیدی:**")
    for node in enhanced_nodes:
        context_parts.append(f"• {node['name']} ({node['kind']})")
        if node['description']:
            context_parts.append(f"  توضیح: {node['description']}")
        if node['biological_role']:
            context_parts.append(f"  نقش زیستی: {node['biological_role']}")
        if node['significance']:
            context_parts.append(f"  اهمیت: {node['significance']}")
        context_parts.append("")
    
    # اطلاعات یال‌ها
    context_parts.append("🔗 **روابط معنادار:**")
    for edge in enhanced_edges:
        context_parts.append(f"• {edge['relation_description']}")
    context_parts.append("")
    
    # تحلیل زیستی
    context_parts.append("🔬 **تحلیل زیستی:**")
    
    # یافتن ژن‌ها
    gene_nodes = [node for node in enhanced_nodes if node['kind'] == 'Gene']
    if gene_nodes:
        context_parts.append("• ژن‌های کلیدی:")
        for gene in gene_nodes:
            context_parts.append(f"  - {gene['name']}: {gene.get('biological_role', 'نقش زیستی مشخص نشده')}")
    
    # یافتن بیماری‌ها
    disease_nodes = [node for node in enhanced_nodes if node['kind'] == 'Disease']
    if disease_nodes:
        context_parts.append("• بیماری‌های مرتبط:")
        for disease in disease_nodes:
            context_parts.append(f"  - {disease['name']}: {disease.get('significance', 'اهمیت بالینی مشخص نشده')}")
    
    # یافتن داروها
    compound_nodes = [node for node in enhanced_nodes if node['kind'] == 'Compound']
    if compound_nodes:
        context_parts.append("• داروهای مرتبط:")
        for compound in compound_nodes:
            context_parts.append(f"  - {compound['name']}: {compound.get('description', 'کاربرد مشخص نشده')}")
    
    enhanced_context = "\n".join(context_parts)
    print(enhanced_context)
    
    print("\n" + "=" * 60)
    print("✅ تست ادغام NodeLookupSystem تکمیل شد")
    
    return {
        "lookup_system": lookup_system,
        "enhanced_nodes": enhanced_nodes,
        "enhanced_edges": enhanced_edges,
        "enhanced_context": enhanced_context
    }

def test_improved_context_generation():
    """تست تولید متن زمینه بهبود یافته"""
    print("\n📝 تست تولید متن زمینه بهبود یافته")
    print("=" * 60)
    
    # راه‌اندازی سیستم‌ها
    lookup_system = NodeLookupSystem()
    
    # داده‌های نمونه
    sample_data = {
        "query": "What is the relationship between TP53 and cancer treatment?",
        "retrieved_nodes": [
            {"id": "Gene::7157", "name": "TP53", "kind": "Gene", "depth": 0, "score": 1.0},
            {"id": "Disease::DOID:162", "name": "malignant glioma", "kind": "Disease", "depth": 1, "score": 0.8},
            {"id": "Compound::DB00262", "name": "Carmustine", "kind": "Compound", "depth": 2, "score": 0.6},
            {"id": "Biological Process::GO:0006915", "name": "apoptosis", "kind": "Biological Process", "depth": 1, "score": 0.7}
        ],
        "retrieved_edges": [
            {"source": "Gene::7157", "target": "Disease::DOID:162", "relation": "DaG", "weight": 1.0},
            {"source": "Compound::DB00262", "target": "Disease::DOID:162", "relation": "CtD", "weight": 0.9},
            {"source": "Gene::7157", "target": "Biological Process::GO:0006915", "relation": "GpBP", "weight": 0.8}
        ]
    }
    
    # ایجاد متن زمینه بهبود یافته
    enhanced_context = create_enhanced_context_with_lookup(sample_data, lookup_system)
    
    print("📄 متن زمینه بهبود یافته:")
    print(enhanced_context)
    
    print("\n" + "=" * 60)
    print("✅ تست تولید متن زمینه بهبود یافته تکمیل شد")
    
    return enhanced_context

def create_enhanced_context_with_lookup(data: Dict, lookup_system: NodeLookupSystem) -> str:
    """ایجاد متن زمینه بهبود یافته با استفاده از NodeLookupSystem"""
    
    context_parts = []
    
    # 1. مقدمه
    context_parts.append(f"🧬 **متن زمینه بهبود یافته برای سوال:** {data['query']}")
    context_parts.append("")
    context_parts.append("این متن شامل اطلاعات معنادار و قابل فهم برای مدل زبان است.")
    context_parts.append("")
    
    # 2. نودهای کلیدی
    context_parts.append("📋 **نودهای کلیدی (با اطلاعات معنادار):**")
    for node in data['retrieved_nodes']:
        node_id = node['id']
        node_info = lookup_system.get_node_info(node_id)
        
        if node_info:
            context_parts.append(f"• {node_info.name} ({node_info.kind})")
            if node_info.description:
                context_parts.append(f"  توضیح: {node_info.description}")
            if node_info.biological_role:
                context_parts.append(f"  نقش زیستی: {node_info.biological_role}")
            if node_info.significance:
                context_parts.append(f"  اهمیت: {node_info.significance}")
        else:
            context_parts.append(f"• {node_id} (اطلاعات کامل در دسترس نیست)")
        context_parts.append("")
    
    # 3. روابط معنادار
    context_parts.append("🔗 **روابط معنادار:**")
    for edge in data['retrieved_edges']:
        relation_desc = lookup_system.format_edge_for_display(
            edge['source'], edge['target'], edge['relation']
        )
        context_parts.append(f"• {relation_desc}")
    context_parts.append("")
    
    # 4. تحلیل زیستی
    context_parts.append("🔬 **تحلیل زیستی و استنتاجات:**")
    
    # گروه‌بندی نودها بر اساس نوع
    gene_nodes = [n for n in data['retrieved_nodes'] if n['kind'] == 'Gene']
    disease_nodes = [n for n in data['retrieved_nodes'] if n['kind'] == 'Disease']
    compound_nodes = [n for n in data['retrieved_nodes'] if n['kind'] == 'Compound']
    process_nodes = [n for n in data['retrieved_nodes'] if n['kind'] == 'Biological Process']
    
    if gene_nodes:
        context_parts.append("• ژن‌های کلیدی و نقش‌های زیستی:")
        for gene in gene_nodes:
            node_info = lookup_system.get_node_info(gene['id'])
            if node_info and node_info.biological_role:
                context_parts.append(f"  - {node_info.name}: {node_info.biological_role}")
    
    if disease_nodes:
        context_parts.append("• بیماری‌های مرتبط و اهمیت بالینی:")
        for disease in disease_nodes:
            node_info = lookup_system.get_node_info(disease['id'])
            if node_info and node_info.significance:
                context_parts.append(f"  - {node_info.name}: {node_info.significance}")
    
    if compound_nodes:
        context_parts.append("• داروهای مرتبط و کاربردها:")
        for compound in compound_nodes:
            node_info = lookup_system.get_node_info(compound['id'])
            if node_info and node_info.description:
                context_parts.append(f"  - {node_info.name}: {node_info.description}")
    
    # 5. استنتاجات زیستی
    context_parts.append("")
    context_parts.append("🧠 **استنتاجات زیستی:**")
    
    # یافتن روابط مهم
    treatment_edges = [e for e in data['retrieved_edges'] if e['relation'] == 'CtD']
    gene_disease_edges = [e for e in data['retrieved_edges'] if e['relation'] == 'DaG']
    gene_process_edges = [e for e in data['retrieved_edges'] if e['relation'] == 'GpBP']
    
    if treatment_edges:
        context_parts.append("• روابط درمانی:")
        for edge in treatment_edges:
            source_info = lookup_system.get_node_info(edge['source'])
            target_info = lookup_system.get_node_info(edge['target'])
            if source_info and target_info:
                context_parts.append(f"  - {source_info.name} برای درمان {target_info.name} استفاده می‌شود")
    
    if gene_disease_edges:
        context_parts.append("• روابط ژن-بیماری:")
        for edge in gene_disease_edges:
            source_info = lookup_system.get_node_info(edge['source'])
            target_info = lookup_system.get_node_info(edge['target'])
            if source_info and target_info:
                context_parts.append(f"  - {source_info.name} با {target_info.name} مرتبط است")
    
    if gene_process_edges:
        context_parts.append("• عملکردهای زیستی:")
        for edge in gene_process_edges:
            source_info = lookup_system.get_node_info(edge['source'])
            target_info = lookup_system.get_node_info(edge['target'])
            if source_info and target_info:
                context_parts.append(f"  - {source_info.name} در فرآیند {target_info.name} مشارکت دارد")
    
    # 6. اهمیت بالینی
    context_parts.append("")
    context_parts.append("🏥 **اهمیت بالینی:**")
    context_parts.append("بر اساس داده‌های ارائه شده، این روابط می‌تواند برای:")
    context_parts.append("• درک مکانیسم‌های بیماری")
    context_parts.append("• شناسایی اهداف درمانی")
    context_parts.append("• توسعه داروهای جدید")
    context_parts.append("• پیش‌بینی پاسخ به درمان")
    context_parts.append("مفید باشد.")
    
    return "\n".join(context_parts)

def main():
    """تابع اصلی"""
    print("🚀 شروع تست سیستم بهبود یافته NodeLookupSystem")
    print("=" * 60)
    
    try:
        # تست 1: ادغام NodeLookupSystem
        result1 = test_node_lookup_integration()
        
        # تست 2: تولید متن زمینه بهبود یافته
        result2 = test_improved_context_generation()
        
        print("\n🎉 تمام تست‌ها با موفقیت تکمیل شدند!")
        print("✅ سیستم NodeLookupSystem آماده استفاده است")
        
        return True
        
    except Exception as e:
        print(f"❌ خطا در تست: {e}")
        return False

if __name__ == "__main__":
    main() 
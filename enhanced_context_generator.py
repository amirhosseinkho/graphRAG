#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ماژول تولید متن زمینه بهبود یافته
با استفاده از NodeLookupSystem برای تبدیل شناسه‌ها به نام‌های معنادار
"""

import sys
import os
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from node_lookup_system import NodeLookupSystem
from graphrag_service import GraphNode, GraphEdge, RetrievalResult

def remove_emojis(text: str) -> str:
    """حذف ایموجی‌ها از متن"""
    # الگوی regex برای شناسایی ایموجی‌ها - شامل تمام انواع ایموجی
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U00002600-\U000026FF"  # miscellaneous symbols
        "\U00002B00-\U00002BFF"  # miscellaneous symbols and arrows
        "\U0001F000-\U0001F02F"  # mahjong tiles
        "\U0001F0A0-\U0001F0FF"  # playing cards
        "\U0001F100-\U0001F64F"  # enclosed alphanumeric supplement
        "\U0001F650-\U0001F67F"  # geometric shapes extended
        "\U0001F680-\U0001F6FF"  # transport and map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # geometric shapes extended
        "\U0001F800-\U0001F8FF"  # supplemental arrows-C
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U00002600-\U000027BF"  # miscellaneous symbols
        "\U00002B00-\U00002BFF"  # miscellaneous symbols and arrows
        "\U00002300-\U000023FF"  # technical symbols
        "\U00002500-\U0000257F"  # box drawing
        "\U00002580-\U0000259F"  # block elements
        "\U000025A0-\U000025FF"  # geometric shapes
        "\U00002600-\U0000267F"  # miscellaneous symbols
        "\U00002680-\U0000269F"  # dingbats
        "\U000026A0-\U000026FF"  # miscellaneous symbols
        "\U00002700-\U000027BF"  # dingbats
        "\U000027C0-\U000027EF"  # miscellaneous mathematical symbols-A
        "\U000027F0-\U000027FF"  # supplemental arrows-A
        "\U00002900-\U0000297F"  # supplemental arrows-B
        "\U00002980-\U000029FF"  # miscellaneous mathematical symbols-B
        "\U00002A00-\U00002AFF"  # supplemental mathematical operators
        "\U00002B00-\U00002BFF"  # miscellaneous symbols and arrows
        "\U00002C60-\U00002C7F"  # latin extended-C
        "\U00002E00-\U00002E7F"  # supplemental punctuation
        "\U00003000-\U0000303F"  # cjk symbols and punctuation
        "\U0000FF00-\U0000FFEF"  # halfwidth and fullwidth forms
        "\U0000FE00-\U0000FE0F"  # variation selectors
        "\U0000FE10-\U0000FE1F"  # vertical forms
        "\U0000FE20-\U0000FE2F"  # combining half marks
        "\U0000FE30-\U0000FE4F"  # cjk compatibility forms
        "\U0000FE50-\U0000FE6F"  # small form variants
        "\U0000FE70-\U0000FEFF"  # arabic presentation forms-B
        "\U0000FF00-\U0000FFEF"  # halfwidth and fullwidth forms
        "\U0000FFF0-\U0000FFFF"  # specials
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub('', text).strip()

@dataclass
class EnhancedNode:
    """نود بهبود یافته با اطلاعات معنادار"""
    id: str
    name: str
    kind: str
    description: str = ""
    biological_role: str = ""
    significance: str = ""
    depth: int = 0
    score: float = 1.0

@dataclass
class EnhancedEdge:
    """یال بهبود یافته با توضیحات معنادار"""
    source: str
    target: str
    relation: str
    source_display: str = ""
    target_display: str = ""
    relation_description: str = ""
    weight: float = 1.0

class EnhancedContextGenerator:
    """تولیدکننده متن زمینه بهبود یافته"""
    
    def __init__(self):
        """راه‌اندازی سیستم"""
        self.lookup_system = NodeLookupSystem()
        print("✅ سیستم تولید متن زمینه بهبود یافته راه‌اندازی شد")
    
    def enhance_retrieval_result(self, retrieval_result: RetrievalResult) -> Dict[str, Any]:
        """بهبود نتیجه بازیابی با اطلاعات معنادار"""
        
        # تبدیل نودها
        enhanced_nodes = []
        for node in retrieval_result.nodes:
            node_info = self.lookup_system.get_node_info(node.id)
            if node_info:
                enhanced_node = EnhancedNode(
                    id=node.id,
                    name=node_info.name,
                    kind=node_info.kind,
                    description=node_info.description,
                    biological_role=node_info.biological_role,
                    significance=node_info.significance,
                    depth=node.depth,
                    score=node.score
                )
                enhanced_nodes.append(enhanced_node)
            else:
                # اگر اطلاعات کامل در دسترس نباشد
                enhanced_node = EnhancedNode(
                    id=node.id,
                    name=node.name,
                    kind=node.kind,
                    depth=node.depth,
                    score=node.score
                )
                enhanced_nodes.append(enhanced_node)
        
        # تبدیل یال‌ها
        enhanced_edges = []
        for edge in retrieval_result.edges:
            # بهبود نمایش منبع و هدف
            source_info = self.lookup_system.get_node_info(edge.source)
            target_info = self.lookup_system.get_node_info(edge.target)
            
            if source_info:
                source_display = f"{source_info.name} ({source_info.kind})"
                if source_info.description:
                    source_display += f" - {source_info.description}"
            else:
                source_display = edge.source
            
            if target_info:
                target_display = f"{target_info.name} ({target_info.kind})"
                if target_info.description:
                    target_display += f" - {target_info.description}"
            else:
                target_display = edge.target
            
            # بهبود توضیح رابطه
            relation_descriptions = {
                "GpBP": "مشارکت در فرآیند زیستی",
                "GpPW": "مشارکت در مسیر زیستی",
                "GpMF": "مشارکت در عملکرد مولکولی",
                "GpCC": "مشارکت در اجزای سلولی",
                "GiG": "تعامل با ژن",
                "Gr>G": "تنظیم ژن",
                "GcG": "هم‌تغییری با ژن",
                "DaG": "ارتباط با ژن",
                "DuG": "تنظیم مثبت ژن",
                "DdG": "تنظیم منفی ژن",
                "DlA": "محل در بافت",
                "DpS": "علائم",
                "CtD": "درمان بیماری",
                "CuG": "تنظیم مثبت ژن",
                "CdG": "تنظیم منفی ژن",
                "CbG": "اتصال به ژن",
                "AeG": "بیان ژن",
                "AuG": "تنظیم مثبت ژن",
                "AdG": "تنظیم منفی ژن",
                "PCiC": "شامل دارو",
                "SEcC": "عوارض جانبی",
                "SpD": "علائم بیماری"
            }
            
            relation_desc = relation_descriptions.get(edge.relation, edge.relation)
            
            enhanced_edge = EnhancedEdge(
                source=edge.source,
                target=edge.target,
                relation=edge.relation,
                source_display=source_display,
                target_display=target_display,
                relation_description=relation_desc,
                weight=edge.weight
            )
            enhanced_edges.append(enhanced_edge)
        
        return {
            "enhanced_nodes": enhanced_nodes,
            "enhanced_edges": enhanced_edges,
            "original_result": retrieval_result
        }
    
    def create_enhanced_context_text(self, retrieval_result: RetrievalResult, 
                                   context_type: str = "INTELLIGENT") -> str:
        """ایجاد متن زمینه بهبود یافته"""
        
        # بهبود داده‌های بازیابی شده
        enhanced_data = self.enhance_retrieval_result(retrieval_result)
        enhanced_nodes = enhanced_data["enhanced_nodes"]
        enhanced_edges = enhanced_data["enhanced_edges"]
        
        # انتخاب نوع متن زمینه
        if context_type == "INTELLIGENT":
            return self._create_intelligent_context(enhanced_nodes, enhanced_edges, retrieval_result.query)
        elif context_type == "SCIENTIFIC_ANALYTICAL":
            return self._create_scientific_context(enhanced_nodes, enhanced_edges, retrieval_result.query)
        elif context_type == "CLINICAL_RELEVANCE":
            return self._create_clinical_context(enhanced_nodes, enhanced_edges, retrieval_result.query)
        elif context_type == "BIOLOGICAL_PATHWAY":
            return self._create_pathway_context(enhanced_nodes, enhanced_edges, retrieval_result.query)
        else:
            return self._create_general_context(enhanced_nodes, enhanced_edges, retrieval_result.query)
    
    def _create_intelligent_context(self, nodes: List[EnhancedNode], 
                                  edges: List[EnhancedEdge], query: str) -> str:
        """ایجاد متن زمینه هوشمند"""
        
        context_parts = []
        
        # 1. مقدمه
        context_parts.append(f"**متن زمینه هوشمند برای سوال:** {query}")
        context_parts.append("")
        context_parts.append("این متن شامل اطلاعات معنادار و قابل فهم برای مدل زبان است.")
        context_parts.append("")
        
        # 2. نودهای کلیدی
        context_parts.append("**نودهای کلیدی (با اطلاعات معنادار):**")
        for node in nodes:
            context_parts.append(f"• {node.name} ({node.kind})")
            if node.description:
                context_parts.append(f"  توضیح: {node.description}")
            if node.biological_role:
                context_parts.append(f"  نقش زیستی: {node.biological_role}")
            if node.significance:
                context_parts.append(f"  اهمیت: {node.significance}")
            context_parts.append("")
        
        # 3. روابط معنادار
        context_parts.append("**روابط معنادار:**")
        for edge in edges:
            context_parts.append(f"• {edge.relation_description}")
        context_parts.append("")
        
        # 4. تحلیل زیستی
        context_parts.append("**تحلیل زیستی و استنتاجات:**")
        
        # گروه‌بندی نودها
        gene_nodes = [n for n in nodes if n.kind == 'Gene']
        disease_nodes = [n for n in nodes if n.kind == 'Disease']
        compound_nodes = [n for n in nodes if n.kind == 'Compound']
        process_nodes = [n for n in nodes if n.kind == 'Biological Process']
        anatomy_nodes = [n for n in nodes if n.kind == 'Anatomy']
        
        if gene_nodes:
            context_parts.append("• ژن‌های کلیدی و نقش‌های زیستی:")
            for gene in gene_nodes:
                role = gene.biological_role if gene.biological_role else "نقش زیستی مشخص نشده"
                context_parts.append(f"  - {gene.name}: {role}")
        
        if disease_nodes:
            context_parts.append("• بیماری‌های مرتبط و اهمیت بالینی:")
            for disease in disease_nodes:
                significance = disease.significance if disease.significance else "اهمیت بالینی مشخص نشده"
                context_parts.append(f"  - {disease.name}: {significance}")
        
        if compound_nodes:
            context_parts.append("• داروهای مرتبط و کاربردها:")
            for compound in compound_nodes:
                description = compound.description if compound.description else "کاربرد مشخص نشده"
                context_parts.append(f"  - {compound.name}: {description}")
        
        if process_nodes:
            context_parts.append("• فرآیندهای زیستی:")
            for process in process_nodes:
                description = process.description if process.description else "توضیح مشخص نشده"
                context_parts.append(f"  - {process.name}: {description}")
        
        if anatomy_nodes:
            context_parts.append("• بافت‌های مرتبط:")
            for anatomy in anatomy_nodes:
                description = anatomy.description if anatomy.description else "توضیح مشخص نشده"
                context_parts.append(f"  - {anatomy.name}: {description}")
        
        # 5. استنتاجات زیستی
        context_parts.append("")
        context_parts.append("**استنتاجات زیستی:**")
        
        # یافتن روابط مهم
        treatment_edges = [e for e in edges if e.relation == 'CtD']
        gene_disease_edges = [e for e in edges if e.relation == 'DaG']
        gene_process_edges = [e for e in edges if e.relation == 'GpBP']
        anatomy_expression_edges = [e for e in edges if e.relation == 'AeG']
        
        if treatment_edges:
            context_parts.append("• روابط درمانی:")
            for edge in treatment_edges:
                context_parts.append(f"  - {edge.source_display} برای درمان {edge.target_display} استفاده می‌شود")
        
        if gene_disease_edges:
            context_parts.append("• روابط ژن-بیماری:")
            for edge in gene_disease_edges:
                context_parts.append(f"  - {edge.source_display} با {edge.target_display} مرتبط است")
        
        if gene_process_edges:
            context_parts.append("• عملکردهای زیستی:")
            for edge in gene_process_edges:
                context_parts.append(f"  - {edge.source_display} در فرآیند {edge.target_display} مشارکت دارد")
        
        if anatomy_expression_edges:
            context_parts.append("• بیان ژن در بافت:")
            for edge in anatomy_expression_edges:
                context_parts.append(f"  - {edge.source_display} در {edge.target_display} بیان می‌شود")
        
        # 6. اهمیت بالینی
        context_parts.append("")
        context_parts.append("**اهمیت بالینی:**")
        context_parts.append("بر اساس داده‌های ارائه شده، این روابط می‌تواند برای:")
        context_parts.append("• درک مکانیسم‌های بیماری")
        context_parts.append("• شناسایی اهداف درمانی")
        context_parts.append("• توسعه داروهای جدید")
        context_parts.append("• پیش‌بینی پاسخ به درمان")
        context_parts.append("• تشخیص و طبقه‌بندی بیماری‌ها")
        context_parts.append("مفید باشد.")
        
        # حذف ایموجی‌ها از متن نهایی
        final_text = "\n".join(context_parts)
        return remove_emojis(final_text)
    
    def _create_scientific_context(self, nodes: List[EnhancedNode], 
                                 edges: List[EnhancedEdge], query: str) -> str:
        """ایجاد متن زمینه علمی-تحلیلی"""
        
        context_parts = []
        
        # 1. مقدمه علمی
        context_parts.append(f"**تحلیل علمی برای سوال:** {query}")
        context_parts.append("")
        context_parts.append("تحلیل علمی بر اساس داده‌های زیستی و روابط مولکولی:")
        context_parts.append("")
        
        # 2. آمار و ارقام
        context_parts.append("**آمار بازیابی:**")
        context_parts.append(f"• تعداد نودها: {len(nodes)}")
        context_parts.append(f"• تعداد روابط: {len(edges)}")
        
        # گروه‌بندی آماری
        node_types = {}
        for node in nodes:
            if node.kind not in node_types:
                node_types[node.kind] = 0
            node_types[node.kind] += 1
        
        context_parts.append("• توزیع انواع نودها:")
        for kind, count in node_types.items():
            context_parts.append(f"  - {kind}: {count} نود")
        
        context_parts.append("")
        
        # 3. تحلیل روابط
        context_parts.append("**تحلیل روابط مولکولی:**")
        relation_types = {}
        for edge in edges:
            if edge.relation not in relation_types:
                relation_types[edge.relation] = 0
            relation_types[edge.relation] += 1
        
        for relation, count in relation_types.items():
            context_parts.append(f"• {relation}: {count} رابطه")
        
        context_parts.append("")
        
        # 4. تحلیل عمیق
        context_parts.append("**تحلیل عمیق زیستی:**")
        
        # تحلیل ژن‌ها
        gene_nodes = [n for n in nodes if n.kind == 'Gene']
        if gene_nodes:
            context_parts.append("• تحلیل ژن‌های کلیدی:")
            for gene in gene_nodes:
                context_parts.append(f"  - {gene.name}")
                if gene.biological_role:
                    context_parts.append(f"    نقش زیستی: {gene.biological_role}")
                if gene.description:
                    context_parts.append(f"    توضیح: {gene.description}")
        
        # تحلیل بیماری‌ها
        disease_nodes = [n for n in nodes if n.kind == 'Disease']
        if disease_nodes:
            context_parts.append("• تحلیل بیماری‌های مرتبط:")
            for disease in disease_nodes:
                context_parts.append(f"  - {disease.name}")
                if disease.significance:
                    context_parts.append(f"    اهمیت: {disease.significance}")
        
        # 5. استنتاجات علمی
        context_parts.append("")
        context_parts.append("**استنتاجات علمی:**")
        context_parts.append("بر اساس تحلیل داده‌ها:")
        
        if gene_nodes and disease_nodes:
            context_parts.append("• روابط ژن-بیماری شناسایی شد")
        
        if compound_nodes := [n for n in nodes if n.kind == 'Compound']:
            context_parts.append("• اهداف دارویی شناسایی شد")
        
        if process_nodes := [n for n in nodes if n.kind == 'Biological Process']:
            context_parts.append("• مسیرهای زیستی شناسایی شد")
        
        # حذف ایموجی‌ها از متن نهایی
        final_text = "\n".join(context_parts)
        return remove_emojis(final_text)
    
    def _create_clinical_context(self, nodes: List[EnhancedNode], 
                               edges: List[EnhancedEdge], query: str) -> str:
        """ایجاد متن زمینه بالینی"""
        
        context_parts = []
        
        # 1. مقدمه بالینی
        context_parts.append(f"**تحلیل بالینی برای سوال:** {query}")
        context_parts.append("")
        context_parts.append("تحلیل بالینی بر اساس روابط درمانی و اهمیت پزشکی:")
        context_parts.append("")
        
        # 2. اطلاعات بالینی
        context_parts.append("**اطلاعات بالینی:**")
        
        # یافتن داروها
        compound_nodes = [n for n in nodes if n.kind == 'Compound']
        if compound_nodes:
            context_parts.append("• داروهای مرتبط:")
            for compound in compound_nodes:
                context_parts.append(f"  - {compound.name}")
                if compound.description:
                    context_parts.append(f"    کاربرد: {compound.description}")
        
        # یافتن بیماری‌ها
        disease_nodes = [n for n in nodes if n.kind == 'Disease']
        if disease_nodes:
            context_parts.append("• بیماری‌های مرتبط:")
            for disease in disease_nodes:
                context_parts.append(f"  - {disease.name}")
                if disease.significance:
                    context_parts.append(f"    اهمیت: {disease.significance}")
        
        # 3. روابط درمانی
        context_parts.append("")
        context_parts.append("**روابط درمانی:**")
        
        treatment_edges = [e for e in edges if e.relation == 'CtD']
        if treatment_edges:
            for edge in treatment_edges:
                context_parts.append(f"• {edge.source_display} برای درمان {edge.target_display}")
        
        # 4. اهمیت بالینی
        context_parts.append("")
        context_parts.append("**اهمیت بالینی:**")
        context_parts.append("این روابط برای موارد زیر اهمیت دارد:")
        context_parts.append("• انتخاب درمان مناسب")
        context_parts.append("• پیش‌بینی پاسخ به درمان")
        context_parts.append("• مدیریت عوارض جانبی")
        context_parts.append("• توسعه پروتکل‌های درمانی")
        
        # حذف ایموجی‌ها از متن نهایی
        final_text = "\n".join(context_parts)
        return remove_emojis(final_text)
    
    def _create_pathway_context(self, nodes: List[EnhancedNode], 
                              edges: List[EnhancedEdge], query: str) -> str:
        """ایجاد متن زمینه مسیر زیستی"""
        
        context_parts = []
        
        # 1. مقدمه مسیر زیستی
        context_parts.append(f"**تحلیل مسیر زیستی برای سوال:** {query}")
        context_parts.append("")
        context_parts.append("تحلیل مسیرهای زیستی و روابط مولکولی:")
        context_parts.append("")
        
        # 2. مسیرهای زیستی
        context_parts.append("**مسیرهای زیستی شناسایی شده:**")
        
        # یافتن فرآیندهای زیستی
        process_nodes = [n for n in nodes if n.kind == 'Biological Process']
        if process_nodes:
            for process in process_nodes:
                context_parts.append(f"• {process.name}")
                if process.description:
                    context_parts.append(f"  توضیح: {process.description}")
        
        # 3. روابط مسیر
        context_parts.append("")
        context_parts.append("**روابط مسیر زیستی:**")
        
        gene_process_edges = [e for e in edges if e.relation == 'GpBP']
        if gene_process_edges:
            for edge in gene_process_edges:
                context_parts.append(f"• {edge.source_display} در مسیر {edge.target_display} مشارکت دارد")
        
        # 4. تحلیل مسیر
        context_parts.append("")
        context_parts.append("**تحلیل مسیر زیستی:**")
        context_parts.append("این مسیرها برای موارد زیر اهمیت دارند:")
        context_parts.append("• درک مکانیسم‌های سلولی")
        context_parts.append("• شناسایی نقاط کنترل")
        context_parts.append("• توسعه درمان‌های هدفمند")
        
        # حذف ایموجی‌ها از متن نهایی
        final_text = "\n".join(context_parts)
        return remove_emojis(final_text)
    
    def _create_general_context(self, nodes: List[EnhancedNode], 
                              edges: List[EnhancedEdge], query: str) -> str:
        """ایجاد متن زمینه عمومی"""
        
        context_parts = []
        
        # 1. مقدمه عمومی
        context_parts.append(f"**اطلاعات عمومی برای سوال:** {query}")
        context_parts.append("")
        
        # 2. نودهای کلیدی
        context_parts.append("**نودهای کلیدی:**")
        for node in nodes:
            context_parts.append(f"• {node.name} ({node.kind})")
            if node.description:
                context_parts.append(f"  توضیح: {node.description}")
        
        # 3. روابط
        context_parts.append("")
        context_parts.append("**روابط:**")
        for edge in edges:
            context_parts.append(f"• {edge.relation_description}")
        
        # حذف ایموجی‌ها از متن نهایی
        final_text = "\n".join(context_parts)
        return remove_emojis(final_text)

def test_enhanced_context_generator():
    """تست ماژول تولید متن زمینه بهبود یافته"""
    print("تست ماژول تولید متن زمینه بهبود یافته")
    print("=" * 60)
    
    # راه‌اندازی
    generator = EnhancedContextGenerator()
    
    # داده‌های نمونه
    sample_nodes = [
        GraphNode(id="Gene::7157", name="TP53", kind="Gene", depth=0, score=1.0),
        GraphNode(id="Disease::DOID:162", name="malignant glioma", kind="Disease", depth=1, score=0.8),
        GraphNode(id="Compound::DB00262", name="Carmustine", kind="Compound", depth=2, score=0.6),
        GraphNode(id="Biological Process::GO:0006915", name="apoptosis", kind="Biological Process", depth=1, score=0.7)
    ]
    
    sample_edges = [
        GraphEdge(source="Gene::7157", target="Disease::DOID:162", relation="DaG", weight=1.0),
        GraphEdge(source="Compound::DB00262", target="Disease::DOID:162", relation="CtD", weight=0.9),
        GraphEdge(source="Gene::7157", target="Biological Process::GO:0006915", relation="GpBP", weight=0.8)
    ]
    
    sample_result = RetrievalResult(
        nodes=sample_nodes,
        edges=sample_edges,
        paths=[],
        context_text="",
        method="Test",
        query="What is the relationship between TP53 and cancer treatment?"
    )
    
    # تست انواع مختلف متن زمینه
    context_types = ["INTELLIGENT", "SCIENTIFIC_ANALYTICAL", "CLINICAL_RELEVANCE", "BIOLOGICAL_PATHWAY", "GENERAL"]
    
    for context_type in context_types:
        print(f"\n📄 تست نوع متن زمینه: {context_type}")
        print("-" * 40)
        
        enhanced_context = generator.create_enhanced_context_text(sample_result, context_type)
        print(enhanced_context[:500] + "..." if len(enhanced_context) > 500 else enhanced_context)
    
    print("\n" + "=" * 60)
    print("✅ تست ماژول تولید متن زمینه بهبود یافته تکمیل شد")

if __name__ == "__main__":
    test_enhanced_context_generator() 
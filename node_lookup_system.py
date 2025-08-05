#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سیستم تبدیل شناسه‌های داخلی به نام‌های معنادار
برای گراف زیستی Hetionet
"""

import pandas as pd
import pickle
import os
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class NodeInfo:
    """اطلاعات کامل یک نود"""
    id: str
    name: str
    kind: str
    description: str = ""
    biological_role: str = ""
    significance: str = ""

class NodeLookupSystem:
    """سیستم تبدیل شناسه‌های داخلی به نام‌های معنادار"""
    
    def __init__(self, nodes_file: str = "hetionet-v1.0-nodes.tsv"):
        self.nodes_file = nodes_file
        self.node_lookup: Dict[str, NodeInfo] = {}
        self.kind_lookup: Dict[str, List[str]] = {}
        self.load_nodes()
        self.enhance_with_biological_info()
    
    def load_nodes(self):
        """بارگذاری نودها از فایل TSV"""
        try:
            df = pd.read_csv(self.nodes_file, sep='\t')
            print(f"📊 بارگذاری {len(df)} نود از فایل {self.nodes_file}")
            
            for _, row in df.iterrows():
                node_id = row['id']
                name = row['name']
                kind = row['kind']
                
                # ایجاد NodeInfo
                node_info = NodeInfo(
                    id=node_id,
                    name=name,
                    kind=kind
                )
                
                self.node_lookup[node_id] = node_info
                
                # گروه‌بندی بر اساس نوع
                if kind not in self.kind_lookup:
                    self.kind_lookup[kind] = []
                self.kind_lookup[kind].append(node_id)
            
            print(f"✅ {len(self.node_lookup)} نود بارگذاری شد")
            print(f"📋 انواع نودها: {list(self.kind_lookup.keys())}")
            
        except Exception as e:
            print(f"❌ خطا در بارگذاری نودها: {e}")
            # ایجاد داده‌های نمونه برای تست
            self._create_sample_data()
    
    def _create_sample_data(self):
        """ایجاد داده‌های نمونه برای تست"""
        sample_nodes = [
            ("Gene::7157", "TP53", "Gene", "سرکوب‌گر تومور و تنظیم‌کننده چرخه سلولی"),
            ("Gene::4087", "SMAD2", "Gene", "تنظیم‌کننده مسیر TGF-beta"),
            ("Gene::675", "BRCA1", "Gene", "ترمیم DNA و سرکوب‌گر تومور"),
            ("Compound::DB00262", "Carmustine", "Compound", "داروی شیمی‌درمانی"),
            ("Compound::DB00316", "Cisplatin", "Compound", "داروی شیمی‌درمانی"),
            ("Disease::DOID:162", "malignant glioma", "Disease", "گلیوم بدخیم مغزی"),
            ("Disease::DOID:10534", "breast cancer", "Disease", "سرطان پستان"),
            ("Anatomy::UBERON:0000955", "brain", "Anatomy", "مغز"),
            ("Biological Process::GO:0006915", "apoptosis", "Biological Process", "مرگ برنامه‌ریزی شده سلول"),
            ("Biological Process::GO:0006396", "RNA processing", "Biological Process", "پردازش RNA"),
            ("Pathway::WP:000000", "p53 pathway", "Pathway", "مسیر p53"),
            ("Molecular Function::GO:0003700", "transcription factor activity", "Molecular Function", "فعالیت فاکتور رونویسی"),
            ("Cellular Component::GO:0005634", "nucleus", "Cellular Component", "هسته سلول"),
            ("Symptom::UMLS:C0000737", "headache", "Symptom", "سردرد"),
            ("Side Effect::UMLS:C0002962", "nausea", "Side Effect", "تهوع"),
            ("Pharmacologic Class::PC:000000", "antineoplastic agents", "Pharmacologic Class", "عوامل ضد سرطان")
        ]
        
        for node_id, name, kind, description in sample_nodes:
            node_info = NodeInfo(
                id=node_id,
                name=name,
                kind=kind,
                description=description
            )
            self.node_lookup[node_id] = node_info
            
            if kind not in self.kind_lookup:
                self.kind_lookup[kind] = []
            self.kind_lookup[kind].append(node_id)
        
        print(f"✅ {len(sample_nodes)} نود نمونه ایجاد شد")
    
    def enhance_with_biological_info(self):
        """افزودن اطلاعات زیستی به نودها"""
        
        # نقش‌های زیستی مهم
        biological_roles = {
            "TP53": "سرکوب‌گر تومور و تنظیم‌کننده چرخه سلولی",
            "BRCA1": "ترمیم DNA و سرکوب‌گر تومور",
            "BRCA2": "ترمیم DNA و سرکوب‌گر تومور",
            "APC": "سرکوب‌گر تومور و تنظیم‌کننده چرخه سلولی",
            "PTEN": "سرکوب‌گر تومور و تنظیم‌کننده مسیر PI3K",
            "RB1": "سرکوب‌گر تومور و تنظیم‌کننده چرخه سلولی",
            "CDKN2A": "سرکوب‌گر تومور و تنظیم‌کننده چرخه سلولی",
            "SMAD2": "تنظیم‌کننده مسیر TGF-beta",
            "SMAD4": "تنظیم‌کننده مسیر TGF-beta",
            "PIK3CA": "انکوژن و تنظیم‌کننده مسیر PI3K",
            "KRAS": "انکوژن و تنظیم‌کننده مسیر MAPK",
            "BRAF": "انکوژن و تنظیم‌کننده مسیر MAPK",
            "EGFR": "گیرنده فاکتور رشد اپیدرمی",
            "HER2": "گیرنده فاکتور رشد اپیدرمی 2",
            "VEGF": "فاکتور رشد اندوتلیال عروقی",
            "MYC": "انکوژن و تنظیم‌کننده رونویسی",
            "BCL2": "تنظیم‌کننده آپوپتوز",
            "BAX": "تنظیم‌کننده آپوپتوز",
            "CASP3": "کاسپاز 3 و تنظیم‌کننده آپوپتوز",
            "CASP9": "کاسپاز 9 و تنظیم‌کننده آپوپتوز"
        }
        
        # اهمیت بیماری‌ها
        disease_significance = {
            "malignant glioma": "گلیوم بدخیم مغزی",
            "glioblastoma": "گلیوبلاستوما",
            "breast cancer": "سرطان پستان",
            "lung cancer": "سرطان ریه",
            "colorectal cancer": "سرطان روده بزرگ",
            "prostate cancer": "سرطان پروستات",
            "ovarian cancer": "سرطان تخمدان",
            "pancreatic cancer": "سرطان لوزالمعده",
            "melanoma": "ملانوم",
            "leukemia": "لوسمی",
            "lymphoma": "لنفوم",
            "cancer": "سرطان"
        }
        
        # کاربرد داروها
        drug_applications = {
            "Carmustine": "داروی شیمی‌درمانی برای درمان گلیوم",
            "Cisplatin": "داروی شیمی‌درمانی برای درمان سرطان‌های مختلف",
            "Doxorubicin": "داروی شیمی‌درمانی آنتراسایکلین",
            "Paclitaxel": "داروی شیمی‌درمانی تاکسول",
            "Tamoxifen": "داروی هورمونی برای درمان سرطان پستان",
            "Imatinib": "داروی هدفمند برای درمان لوسمی",
            "Bevacizumab": "داروی ضد رگ‌زایی",
            "Trastuzumab": "داروی هدفمند برای HER2+",
            "Cetuximab": "داروی هدفمند EGFR",
            "Rituximab": "داروی هدفمند CD20"
        }
        
        # به‌روزرسانی نودها با اطلاعات زیستی
        for node_id, node_info in self.node_lookup.items():
            name = node_info.name
            
            # افزودن نقش زیستی برای ژن‌ها
            if node_info.kind == "Gene" and name in biological_roles:
                node_info.biological_role = biological_roles[name]
            
            # افزودن اهمیت برای بیماری‌ها
            if node_info.kind == "Disease" and name in disease_significance:
                node_info.significance = disease_significance[name]
            
            # افزودن کاربرد برای داروها
            if node_info.kind == "Compound" and name in drug_applications:
                node_info.description = drug_applications[name]
        
        print("✅ اطلاعات زیستی به نودها اضافه شد")
    
    def get_node_info(self, node_id: str) -> Optional[NodeInfo]:
        """دریافت اطلاعات نود بر اساس شناسه"""
        return self.node_lookup.get(node_id)
    
    def get_node_name(self, node_id: str) -> str:
        """دریافت نام نود"""
        node_info = self.get_node_info(node_id)
        return node_info.name if node_info else node_id
    
    def get_node_kind(self, node_id: str) -> str:
        """دریافت نوع نود"""
        node_info = self.get_node_info(node_id)
        return node_info.kind if node_info else "Unknown"
    
    def get_node_description(self, node_id: str) -> str:
        """دریافت توضیح نود"""
        node_info = self.get_node_info(node_id)
        if not node_info:
            return ""
        
        description_parts = []
        
        if node_info.description:
            description_parts.append(node_info.description)
        
        if node_info.biological_role:
            description_parts.append(f"نقش زیستی: {node_info.biological_role}")
        
        if node_info.significance:
            description_parts.append(f"اهمیت: {node_info.significance}")
        
        return " - ".join(description_parts) if description_parts else ""
    
    def format_node_for_display(self, node_id: str) -> str:
        """فرمت کردن نود برای نمایش"""
        node_info = self.get_node_info(node_id)
        if not node_info:
            return node_id
        
        # تبدیل نوع به فارسی
        kind_translations = {
            "Gene": "ژن",
            "Compound": "دارو",
            "Disease": "بیماری",
            "Anatomy": "بافت",
            "Biological Process": "فرآیند زیستی",
            "Pathway": "مسیر زیستی",
            "Molecular Function": "عملکرد مولکولی",
            "Cellular Component": "اجزای سلولی",
            "Symptom": "علائم",
            "Side Effect": "عوارض جانبی",
            "Pharmacologic Class": "طبقه دارویی"
        }
        
        kind_fa = kind_translations.get(node_info.kind, node_info.kind)
        
        if node_info.description or node_info.biological_role or node_info.significance:
            description = self.get_node_description(node_id)
            return f"{node_info.name} ({kind_fa}) - {description}"
        else:
            return f"{node_info.name} ({kind_fa})"
    
    def format_edge_for_display(self, source_id: str, target_id: str, relation: str) -> str:
        """فرمت کردن یال برای نمایش"""
        source_display = self.format_node_for_display(source_id)
        target_display = self.format_node_for_display(target_id)
        
        # توضیح رابطه
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
            "AdG": "تنظیم منفی ژن"
        }
        
        relation_desc = relation_descriptions.get(relation, relation)
        
        return f"{source_display} → {relation_desc} → {target_display}"
    
    def format_path_for_display(self, path: List[str], edges: List[Tuple[str, str, str]]) -> str:
        """فرمت کردن مسیر برای نمایش"""
        if len(path) < 2:
            return " → ".join([self.format_node_for_display(node_id) for node_id in path])
        
        formatted_parts = []
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]
            
            # پیدا کردن رابطه
            relation = None
            for edge_source, edge_target, edge_relation in edges:
                if edge_source == source_id and edge_target == target_id:
                    relation = edge_relation
                    break
            
            source_display = self.format_node_for_display(source_id)
            target_display = self.format_node_for_display(target_id)
            
            if relation:
                relation_descriptions = {
                    "GpBP": "مشارکت در فرآیند زیستی",
                    "GpPW": "مشارکت در مسیر زیستی",
                    "GiG": "تعامل با ژن",
                    "DaG": "ارتباط با ژن",
                    "CtD": "درمان بیماری",
                    "DlA": "محل در بافت"
                }
                relation_desc = relation_descriptions.get(relation, relation)
                formatted_parts.append(f"{source_display} → {relation_desc} → {target_display}")
            else:
                formatted_parts.append(f"{source_display} → {target_display}")
        
        return " و ".join(formatted_parts)
    
    def save_lookup_cache(self, filename: str = "node_lookup_cache.pkl"):
        """ذخیره کش lookup"""
        cache_data = {
            'node_lookup': self.node_lookup,
            'kind_lookup': self.kind_lookup
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✅ کش lookup در {filename} ذخیره شد")
    
    def load_lookup_cache(self, filename: str = "node_lookup_cache.pkl"):
        """بارگذاری کش lookup"""
        try:
            with open(filename, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.node_lookup = cache_data['node_lookup']
            self.kind_lookup = cache_data['kind_lookup']
            
            print(f"✅ کش lookup از {filename} بارگذاری شد")
            return True
        except FileNotFoundError:
            print(f"⚠️ فایل کش {filename} یافت نشد")
            return False
        except Exception as e:
            print(f"❌ خطا در بارگذاری کش: {e}")
            return False

def test_node_lookup_system():
    """تست سیستم lookup"""
    print("🧬 تست سیستم تبدیل شناسه‌ها به نام‌های معنادار")
    print("=" * 60)
    
    # ایجاد سیستم
    lookup_system = NodeLookupSystem()
    
    # تست نودهای مختلف
    test_nodes = [
        "Gene::7157",  # TP53
        "Gene::4087",  # SMAD2
        "Compound::DB00262",  # Carmustine
        "Disease::DOID:162",  # malignant glioma
        "Anatomy::UBERON:0000955",  # brain
        "Biological Process::GO:0006915"  # apoptosis
    ]
    
    print("\n📋 تست نمایش نودها:")
    for node_id in test_nodes:
        display = lookup_system.format_node_for_display(node_id)
        print(f"• {node_id} → {display}")
    
    print("\n🔗 تست نمایش یال‌ها:")
    test_edges = [
        ("Gene::7157", "Gene::4087", "GiG"),
        ("Compound::DB00262", "Disease::DOID:162", "CtD"),
        ("Gene::7157", "Biological Process::GO:0006915", "GpBP")
    ]
    
    for source, target, relation in test_edges:
        display = lookup_system.format_edge_for_display(source, target, relation)
        print(f"• {display}")
    
    print("\n🛤️ تست نمایش مسیرها:")
    test_path = ["Gene::7157", "Gene::4087", "Disease::DOID:162"]
    test_edges_for_path = [
        ("Gene::7157", "Gene::4087", "GiG"),
        ("Gene::4087", "Disease::DOID:162", "DaG")
    ]
    
    path_display = lookup_system.format_path_for_display(test_path, test_edges_for_path)
    print(f"• مسیر: {path_display}")
    
    print("\n" + "=" * 60)
    print("✅ تست سیستم lookup تکمیل شد")

if __name__ == "__main__":
    test_node_lookup_system() 
# -*- coding: utf-8 -*-
"""
GraphRAG Service - سرویس اصلی GraphRAG
این سرویس تمام قابلیت‌های GraphRAG را فراهم می‌کند
"""

import pandas as pd
import networkx as nx
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import deque
import pickle
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class RetrievalMethod(Enum):
    """روش‌های بازیابی"""
    BFS = "BFS"
    DFS = "DFS"
    SHORTEST_PATH = "Shortest Path"
    NEIGHBORS = "Neighbors"
    HYBRID = "Hybrid"
    MULTI_METHOD = "Multi-Method"
    ENSEMBLE = "Ensemble"
    ADAPTIVE = "Adaptive"
    INTELLIGENT = "Intelligent Semantic Search"
    NO_RETRIEVAL = "بدون بازیابی (فقط مدل)"

class GenerationModel(Enum):
    """مدل‌های تولید متن"""
    SIMPLE = "Simple Template"
    GPT_SIMULATION = "GPT Simulation"
    CUSTOM = "Custom Model"
    HUGGINGFACE = "HuggingFace Models"
    OPENAI_GPT = "OpenAI GPT"
    ANTHROPIC_CLAUDE = "Anthropic Claude"
    GOOGLE_GEMINI = "Google Gemini"

@dataclass
class GraphNode:
    """نمایش یک نود گراف"""
    id: str
    name: str
    kind: str
    depth: int = 0
    score: float = 1.0

@dataclass
class GraphEdge:
    """نمایش یک یال گراف"""
    source: str
    target: str
    relation: str
    weight: float = 1.0

@dataclass
class RetrievalResult:
    """نتیجه بازیابی"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    paths: List[List[str]]
    context_text: str
    method: str
    query: str

@dataclass
class GenerationResult:
    """نتیجه تولید متن"""
    answer: str
    model: str
    context_used: str
    confidence: float

class GraphRAGService:
    """سرویس اصلی GraphRAG"""
    
    def __init__(self, graph_data_path: str = None):
        self.G = None
        self.nlp = None
        self.graph_data_path = graph_data_path
        self.initialize()
    
    def initialize(self):
        """راه‌اندازی سرویس"""
        print("🔧 راه‌اندازی GraphRAG Service...")
        
        # بارگذاری مدل spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ مدل spaCy بارگذاری شد")
        except:
            print("❌ خطا در بارگذاری مدل spaCy")
            return
        
        # بارگذاری یا ایجاد گراف
        if self.graph_data_path and os.path.exists(self.graph_data_path):
            self.load_graph_from_file()
        else:
            self.create_sample_graph()
    
    def create_sample_graph(self):
        """ایجاد گراف نمونه بر اساس ساختار واقعی Hetionet"""
        print("🔧 ایجاد گراف نمونه بر اساس Hetionet...")
        
        self.G = nx.DiGraph()
        
        # اضافه کردن نودها بر اساس metanodes واقعی Hetionet
        nodes_data = [
            # Gene nodes (20945 total in Hetionet)
            ('Gene::TP53', {'name': 'TP53', 'kind': 'Gene', 'metanode': 'Gene'}),
            ('Gene::BRCA1', {'name': 'BRCA1', 'kind': 'Gene', 'metanode': 'Gene'}),
            ('Gene::MMP9', {'name': 'MMP9', 'kind': 'Gene', 'metanode': 'Gene'}),
            ('Gene::BID', {'name': 'BID', 'kind': 'Gene', 'metanode': 'Gene'}),
            ('Gene::HMGB3', {'name': 'HMGB3', 'kind': 'Gene', 'metanode': 'Gene'}),
            ('Gene::KCNQ2', {'name': 'KCNQ2', 'kind': 'Gene', 'metanode': 'Gene'}),
            ('Gene::APOE', {'name': 'APOE', 'kind': 'Gene', 'metanode': 'Gene'}),
            ('Gene::CFTR', {'name': 'CFTR', 'kind': 'Gene', 'metanode': 'Gene'}),
            
            # Anatomy nodes (402 total in Hetionet)
            ('Anatomy::Heart', {'name': 'Heart', 'kind': 'Anatomy', 'metanode': 'Anatomy'}),
            ('Anatomy::Brain', {'name': 'Brain', 'kind': 'Anatomy', 'metanode': 'Anatomy'}),
            ('Anatomy::Liver', {'name': 'Liver', 'kind': 'Anatomy', 'metanode': 'Anatomy'}),
            ('Anatomy::Lung', {'name': 'Lung', 'kind': 'Anatomy', 'metanode': 'Anatomy'}),
            ('Anatomy::Kidney', {'name': 'Kidney', 'kind': 'Anatomy', 'metanode': 'Anatomy'}),
            ('Anatomy::Stomach', {'name': 'Stomach', 'kind': 'Anatomy', 'metanode': 'Anatomy'}),
            
            # Disease nodes (137 total in Hetionet)
            ('Disease::Breast Cancer', {'name': 'Breast Cancer', 'kind': 'Disease', 'metanode': 'Disease'}),
            ('Disease::Lung Cancer', {'name': 'Lung Cancer', 'kind': 'Disease', 'metanode': 'Disease'}),
            ('Disease::Heart Disease', {'name': 'Heart Disease', 'kind': 'Disease', 'metanode': 'Disease'}),
            ('Disease::Alzheimer Disease', {'name': 'Alzheimer Disease', 'kind': 'Disease', 'metanode': 'Disease'}),
            ('Disease::Cystic Fibrosis', {'name': 'Cystic Fibrosis', 'kind': 'Disease', 'metanode': 'Disease'}),
            
            # Compound nodes (1552 total in Hetionet)
            ('Compound::Aspirin', {'name': 'Aspirin', 'kind': 'Compound', 'metanode': 'Compound'}),
            ('Compound::Ibuprofen', {'name': 'Ibuprofen', 'kind': 'Compound', 'metanode': 'Compound'}),
            ('Compound::Paracetamol', {'name': 'Paracetamol', 'kind': 'Compound', 'metanode': 'Compound'}),
            ('Compound::Caffeine', {'name': 'Caffeine', 'kind': 'Compound', 'metanode': 'Compound'}),
            ('Compound::Vitamin C', {'name': 'Vitamin C', 'kind': 'Compound', 'metanode': 'Compound'}),
            ('Compound::Metformin', {'name': 'Metformin', 'kind': 'Compound', 'metanode': 'Compound'}),
            
            # Biological Process nodes (11381 total in Hetionet)
            ('BiologicalProcess::Cell Death', {'name': 'Cell Death', 'kind': 'Biological Process', 'metanode': 'Biological Process'}),
            ('BiologicalProcess::DNA Repair', {'name': 'DNA Repair', 'kind': 'Biological Process', 'metanode': 'Biological Process'}),
            ('BiologicalProcess::Cell Cycle', {'name': 'Cell Cycle', 'kind': 'Biological Process', 'metanode': 'Biological Process'}),
            ('BiologicalProcess::Apoptosis', {'name': 'Apoptosis', 'kind': 'Biological Process', 'metanode': 'Biological Process'}),
            
            # Pathway nodes (1822 total in Hetionet)
            ('Pathway::Apoptosis', {'name': 'Apoptosis', 'kind': 'Pathway', 'metanode': 'Pathway'}),
            ('Pathway::Cell Cycle', {'name': 'Cell Cycle', 'kind': 'Pathway', 'metanode': 'Pathway'}),
            ('Pathway::DNA Repair', {'name': 'DNA Repair', 'kind': 'Pathway', 'metanode': 'Pathway'}),
            
            # Symptom nodes (438 total in Hetionet)
            ('Symptom::Pain', {'name': 'Pain', 'kind': 'Symptom', 'metanode': 'Symptom'}),
            ('Symptom::Fever', {'name': 'Fever', 'kind': 'Symptom', 'metanode': 'Symptom'}),
            ('Symptom::Cough', {'name': 'Cough', 'kind': 'Symptom', 'metanode': 'Symptom'}),
            ('Symptom::Fatigue', {'name': 'Fatigue', 'kind': 'Symptom', 'metanode': 'Symptom'}),
            
            # Side Effect nodes (5734 total in Hetionet)
            ('SideEffect::Nausea', {'name': 'Nausea', 'kind': 'Side Effect', 'metanode': 'Side Effect'}),
            ('SideEffect::Headache', {'name': 'Headache', 'kind': 'Side Effect', 'metanode': 'Side Effect'}),
            ('SideEffect::Dizziness', {'name': 'Dizziness', 'kind': 'Side Effect', 'metanode': 'Side Effect'}),
            
            # Molecular Function nodes (2884 total in Hetionet)
            ('MolecularFunction::Enzyme', {'name': 'Enzyme', 'kind': 'Molecular Function', 'metanode': 'Molecular Function'}),
            ('MolecularFunction::Receptor', {'name': 'Receptor', 'kind': 'Molecular Function', 'metanode': 'Molecular Function'}),
            ('MolecularFunction::Transporter', {'name': 'Transporter', 'kind': 'Molecular Function', 'metanode': 'Molecular Function'}),
            
            # Cellular Component nodes (1391 total in Hetionet)
            ('CellularComponent::Nucleus', {'name': 'Nucleus', 'kind': 'Cellular Component', 'metanode': 'Cellular Component'}),
            ('CellularComponent::Mitochondria', {'name': 'Mitochondria', 'kind': 'Cellular Component', 'metanode': 'Cellular Component'}),
            ('CellularComponent::Cell Membrane', {'name': 'Cell Membrane', 'kind': 'Cellular Component', 'metanode': 'Cellular Component'}),
            
            # Pharmacologic Class nodes (345 total in Hetionet)
            ('PharmacologicClass::NSAID', {'name': 'NSAID', 'kind': 'Pharmacologic Class', 'metanode': 'Pharmacologic Class'}),
            ('PharmacologicClass::Antibiotic', {'name': 'Antibiotic', 'kind': 'Pharmacologic Class', 'metanode': 'Pharmacologic Class'}),
            ('PharmacologicClass::Antihypertensive', {'name': 'Antihypertensive', 'kind': 'Pharmacologic Class', 'metanode': 'Pharmacologic Class'})
        ]
        
        for node_id, attrs in nodes_data:
            self.G.add_node(node_id, **attrs)
        
        # اضافه کردن یال‌ها بر اساس metaedges واقعی Hetionet
        edges_data = [
            # Anatomy - expresses - Gene (AeG) - 526407 edges in Hetionet
            ('Anatomy::Heart', 'Gene::MMP9', 'AeG'),
            ('Anatomy::Heart', 'Gene::BID', 'AeG'),
            ('Anatomy::Heart', 'Gene::KCNQ2', 'AeG'),
            ('Anatomy::Brain', 'Gene::APOE', 'AeG'),
            ('Anatomy::Brain', 'Gene::TP53', 'AeG'),
            ('Anatomy::Liver', 'Gene::BRCA1', 'AeG'),
            ('Anatomy::Lung', 'Gene::CFTR', 'AeG'),
            
            # Anatomy - upregulates - Gene (AuG) - 97848 edges in Hetionet
            ('Anatomy::Heart', 'Gene::HMGB3', 'AuG'),
            ('Anatomy::Brain', 'Gene::BRCA1', 'AuG'),
            
            # Anatomy - downregulates - Gene (AdG) - 102240 edges in Hetionet
            ('Anatomy::Liver', 'Gene::MMP9', 'AdG'),
            
            # Disease - associates - Gene (DaG) - 12623 edges in Hetionet
            ('Disease::Breast Cancer', 'Gene::BRCA1', 'DaG'),
            ('Disease::Breast Cancer', 'Gene::TP53', 'DaG'),
            ('Disease::Lung Cancer', 'Gene::MMP9', 'DaG'),
            ('Disease::Alzheimer Disease', 'Gene::APOE', 'DaG'),
            ('Disease::Cystic Fibrosis', 'Gene::CFTR', 'DaG'),
            
            # Disease - upregulates - Gene (DuG) - 7731 edges in Hetionet
            ('Disease::Breast Cancer', 'Gene::BID', 'DuG'),
            ('Disease::Lung Cancer', 'Gene::TP53', 'DuG'),
            
            # Disease - downregulates - Gene (DdG) - 7623 edges in Hetionet
            ('Disease::Heart Disease', 'Gene::KCNQ2', 'DdG'),
            
            # Disease - localizes - Anatomy (DlA) - 3602 edges in Hetionet
            ('Disease::Breast Cancer', 'Anatomy::Stomach', 'DlA'),
            ('Disease::Lung Cancer', 'Anatomy::Lung', 'DlA'),
            ('Disease::Heart Disease', 'Anatomy::Heart', 'DlA'),
            
            # Disease - presents - Symptom (DpS) - 3357 edges in Hetionet
            ('Disease::Breast Cancer', 'Symptom::Pain', 'DpS'),
            ('Disease::Lung Cancer', 'Symptom::Cough', 'DpS'),
            ('Disease::Heart Disease', 'Symptom::Fatigue', 'DpS'),
            
            # Compound - binds - Gene (CbG) - 11571 edges in Hetionet
            ('Compound::Caffeine', 'Gene::TP53', 'CbG'),
            ('Compound::Vitamin C', 'Gene::BRCA1', 'CbG'),
            ('Compound::Metformin', 'Gene::APOE', 'CbG'),
            
            # Compound - treats - Disease (CtD) - 755 edges in Hetionet
            ('Compound::Aspirin', 'Disease::Heart Disease', 'CtD'),
            ('Compound::Metformin', 'Disease::Breast Cancer', 'CtD'),
            
            # Compound - palliates - Disease (CpD) - 390 edges in Hetionet
            ('Compound::Ibuprofen', 'Disease::Breast Cancer', 'CpD'),
            ('Compound::Paracetamol', 'Disease::Lung Cancer', 'CpD'),
            
            # Compound - causes - Side Effect (CcSE) - 138944 edges in Hetionet
            ('Compound::Aspirin', 'SideEffect::Nausea', 'CcSE'),
            ('Compound::Ibuprofen', 'SideEffect::Headache', 'CcSE'),
            ('Compound::Caffeine', 'SideEffect::Dizziness', 'CcSE'),
            
            # Compound - upregulates - Gene (CuG) - 18756 edges in Hetionet
            ('Compound::Vitamin C', 'Gene::TP53', 'CuG'),
            ('Compound::Metformin', 'Gene::BRCA1', 'CuG'),
            
            # Compound - downregulates - Gene (CdG) - 21102 edges in Hetionet
            ('Compound::Caffeine', 'Gene::MMP9', 'CdG'),
            
            # Gene - participates - Biological Process (GpBP) - 559504 edges in Hetionet
            ('Gene::BID', 'BiologicalProcess::Cell Death', 'GpBP'),
            ('Gene::TP53', 'BiologicalProcess::DNA Repair', 'GpBP'),
            ('Gene::BRCA1', 'BiologicalProcess::Apoptosis', 'GpBP'),
            ('Gene::MMP9', 'BiologicalProcess::Cell Cycle', 'GpBP'),
            
            # Gene - participates - Pathway (GpPW) - 84372 edges in Hetionet
            ('Gene::BRCA1', 'Pathway::Apoptosis', 'GpPW'),
            ('Gene::TP53', 'Pathway::Cell Cycle', 'GpPW'),
            ('Gene::BID', 'Pathway::DNA Repair', 'GpPW'),
            
            # Gene - participates - Molecular Function (GpMF) - 97222 edges in Hetionet
            ('Gene::TP53', 'MolecularFunction::Enzyme', 'GpMF'),
            ('Gene::BRCA1', 'MolecularFunction::Receptor', 'GpMF'),
            ('Gene::CFTR', 'MolecularFunction::Transporter', 'GpMF'),
            
            # Gene - participates - Cellular Component (GpCC) - 73566 edges in Hetionet
            ('Gene::BRCA1', 'CellularComponent::Nucleus', 'GpCC'),
            ('Gene::TP53', 'CellularComponent::Mitochondria', 'GpCC'),
            ('Gene::CFTR', 'CellularComponent::Cell Membrane', 'GpCC'),
            
            # Gene - interacts - Gene (GiG) - 147164 edges in Hetionet
            ('Gene::TP53', 'Gene::BRCA1', 'GiG'),
            ('Gene::MMP9', 'Gene::BID', 'GiG'),
            ('Gene::APOE', 'Gene::CFTR', 'GiG'),
            
            # Gene > regulates > Gene (Gr>G) - 265672 edges in Hetionet
            ('Gene::TP53', 'Gene::MMP9', 'Gr>G'),
            ('Gene::BRCA1', 'Gene::BID', 'Gr>G'),
            ('Gene::APOE', 'Gene::KCNQ2', 'Gr>G'),
            
            # Gene - covaries - Gene (GcG) - 61690 edges in Hetionet
            ('Gene::TP53', 'Gene::BRCA1', 'GcG'),
            ('Gene::MMP9', 'Gene::BID', 'GcG'),
            
            # Pharmacologic Class - includes - Compound (PCiC) - 1029 edges in Hetionet
            ('PharmacologicClass::NSAID', 'Compound::Aspirin', 'PCiC'),
            ('PharmacologicClass::NSAID', 'Compound::Ibuprofen', 'PCiC'),
            ('PharmacologicClass::Antibiotic', 'Compound::Metformin', 'PCiC'),
            
            # Compound - resembles - Compound (CrC) - 6486 edges in Hetionet
            ('Compound::Aspirin', 'Compound::Ibuprofen', 'CrC'),
            ('Compound::Caffeine', 'Compound::Vitamin C', 'CrC'),
            
            # Disease - resembles - Disease (DrD) - 543 edges in Hetionet
            ('Disease::Breast Cancer', 'Disease::Lung Cancer', 'DrD'),
            ('Disease::Alzheimer Disease', 'Disease::Cystic Fibrosis', 'DrD')
        ]
        
        for source, target, metaedge in edges_data:
            self.G.add_edge(source, target, metaedge=metaedge, relation=metaedge)
        
        # اضافه کردن یال‌های معکوس برای پشتیبانی از سوالات پیچیده
        reverse_edges_data = [
            # Gene - expressed_in - Anatomy (GeA) - معکوس AeG
            ('Gene::MMP9', 'Anatomy::Heart', 'GeA'),
            ('Gene::BID', 'Anatomy::Heart', 'GeA'),
            ('Gene::KCNQ2', 'Anatomy::Heart', 'GeA'),
            ('Gene::APOE', 'Anatomy::Brain', 'GeA'),
            ('Gene::TP53', 'Anatomy::Brain', 'GeA'),
            ('Gene::BRCA1', 'Anatomy::Liver', 'GeA'),
            ('Gene::CFTR', 'Anatomy::Lung', 'GeA'),
            
            # Gene - upregulates - Anatomy (GuA) - معکوس AuG
            ('Gene::HMGB3', 'Anatomy::Heart', 'GuA'),
            ('Gene::BRCA1', 'Anatomy::Brain', 'GuA'),
            
            # Gene - downregulates - Anatomy (GdA) - معکوس AdG
            ('Gene::MMP9', 'Anatomy::Liver', 'GdA'),
            
            # Gene - associates - Disease (GaD) - معکوس DaG
            ('Gene::BRCA1', 'Disease::Breast Cancer', 'GaD'),
            ('Gene::TP53', 'Disease::Breast Cancer', 'GaD'),
            ('Gene::MMP9', 'Disease::Lung Cancer', 'GaD'),
            ('Gene::APOE', 'Disease::Alzheimer Disease', 'GaD'),
            ('Gene::CFTR', 'Disease::Cystic Fibrosis', 'GaD'),
            
            # Gene - upregulates - Disease (GuD) - معکوس DuG
            ('Gene::BID', 'Disease::Breast Cancer', 'GuD'),
            ('Gene::TP53', 'Disease::Lung Cancer', 'GuD'),
            
            # Gene - downregulates - Disease (GdD) - معکوس DdG
            ('Gene::KCNQ2', 'Disease::Heart Disease', 'GdD'),
            
            # Anatomy - localizes - Disease (AlD) - معکوس DlA
            ('Anatomy::Stomach', 'Disease::Breast Cancer', 'AlD'),
            ('Anatomy::Lung', 'Disease::Lung Cancer', 'AlD'),
            ('Anatomy::Heart', 'Disease::Heart Disease', 'AlD'),
            
            # Symptom - presents - Disease (SpD) - معکوس DpS
            ('Symptom::Pain', 'Disease::Breast Cancer', 'SpD'),
            ('Symptom::Cough', 'Disease::Lung Cancer', 'SpD'),
            ('Symptom::Fatigue', 'Disease::Heart Disease', 'SpD'),
            
            # Gene - binds - Compound (GbC) - معکوس CbG
            ('Gene::TP53', 'Compound::Caffeine', 'GbC'),
            ('Gene::BRCA1', 'Compound::Vitamin C', 'GbC'),
            ('Gene::APOE', 'Compound::Metformin', 'GbC'),
            
            # Disease - treated_by - Compound (DtC) - معکوس CtD
            ('Disease::Heart Disease', 'Compound::Aspirin', 'DtC'),
            ('Disease::Breast Cancer', 'Compound::Metformin', 'DtC'),
            
            # Disease - palliated_by - Compound (DpC) - معکوس CpD
            ('Disease::Breast Cancer', 'Compound::Ibuprofen', 'DpC'),
            ('Disease::Lung Cancer', 'Compound::Paracetamol', 'DpC'),
            
            # Side Effect - caused_by - Compound (SEcC) - معکوس CcSE
            ('SideEffect::Nausea', 'Compound::Aspirin', 'SEcC'),
            ('SideEffect::Headache', 'Compound::Ibuprofen', 'SEcC'),
            ('SideEffect::Dizziness', 'Compound::Caffeine', 'SEcC'),
            
            # Gene - upregulates - Compound (GuC) - معکوس CuG
            ('Gene::TP53', 'Compound::Vitamin C', 'GuC'),
            ('Gene::BRCA1', 'Compound::Metformin', 'GuC'),
            
            # Gene - downregulates - Compound (GdC) - معکوس CdG
            ('Gene::MMP9', 'Compound::Caffeine', 'GdC'),
            
            # Biological Process - participates - Gene (BPpG) - معکوس GpBP
            ('BiologicalProcess::Cell Death', 'Gene::BID', 'BPpG'),
            ('BiologicalProcess::DNA Repair', 'Gene::TP53', 'BPpG'),
            ('BiologicalProcess::Apoptosis', 'Gene::BRCA1', 'BPpG'),
            ('BiologicalProcess::Cell Cycle', 'Gene::MMP9', 'BPpG'),
            
            # Pathway - participates - Gene (PWpG) - معکوس GpPW
            ('Pathway::Apoptosis', 'Gene::BRCA1', 'PWpG'),
            ('Pathway::Cell Cycle', 'Gene::TP53', 'PWpG'),
            ('Pathway::DNA Repair', 'Gene::BID', 'PWpG'),
            
            # Molecular Function - participates - Gene (MFpG) - معکوس GpMF
            ('MolecularFunction::Enzyme', 'Gene::TP53', 'MFpG'),
            ('MolecularFunction::Receptor', 'Gene::BRCA1', 'MFpG'),
            ('MolecularFunction::Transporter', 'Gene::CFTR', 'MFpG'),
            
            # Cellular Component - participates - Gene (CCpG) - معکوس GpCC
            ('CellularComponent::Nucleus', 'Gene::BRCA1', 'CCpG'),
            ('CellularComponent::Mitochondria', 'Gene::TP53', 'CCpG'),
            ('CellularComponent::Cell Membrane', 'Gene::CFTR', 'CCpG'),
            
            # Compound - includes - Pharmacologic Class (CiPC) - معکوس PCiC
            ('Compound::Aspirin', 'PharmacologicClass::NSAID', 'CiPC'),
            ('Compound::Ibuprofen', 'PharmacologicClass::NSAID', 'CiPC'),
            ('Compound::Metformin', 'PharmacologicClass::Antibiotic', 'CiPC')
        ]
        
        for source, target, metaedge in reverse_edges_data:
            self.G.add_edge(source, target, metaedge=metaedge, relation=metaedge)
        
        print(f"✅ گراف نمونه بر اساس Hetionet ایجاد شد: {self.G.number_of_nodes()} نود، {self.G.number_of_edges()} یال")
        print(f"📊 شامل {len([n for n, d in self.G.nodes(data=True) if d.get('metanode') == 'Gene'])} ژن، {len([n for n, d in self.G.nodes(data=True) if d.get('metanode') == 'Anatomy'])} آناتومی")
        print(f"🔗 شامل {len([e for e in self.G.edges(data=True) if e[2].get('metaedge') == 'AeG'])} یال AeG (Anatomy-expresses-Gene)")
        print(f"🔄 شامل {len([e for e in self.G.edges(data=True) if e[2].get('metaedge') == 'GeA'])} یال GeA (Gene-expressed_in-Anatomy) - معکوس")
    
    def load_graph_from_file(self):
        """بارگذاری گراف از فایل"""
        try:
            with open(self.graph_data_path, 'rb') as f:
                self.G = pickle.load(f)
            print(f"✅ گراف از فایل بارگذاری شد: {self.G.number_of_nodes()} نود، {self.G.number_of_edges()} یال")
        except Exception as e:
            print(f"❌ خطا در بارگذاری گراف: {e}")
            self.create_sample_graph()
    
    def extract_keywords(self, text: str) -> List[str]:
        """استخراج کلمات کلیدی از متن"""
        doc = self.nlp(text)
        keywords = set()
        
        # موجودیت‌های نام‌دار
        for ent in doc.ents:
            if ent.label_ not in {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}:
                # حذف علائم نگارشی از موجودیت‌ها
                clean_text = ''.join(c for c in ent.text.lower() if c.isalnum() or c.isspace())
                if clean_text.strip():
                    keywords.add(clean_text.strip())
        
        # اسم‌ها و اسم خاص‌ها
        for token in doc:
            if (token.pos_ in {"NOUN", "PROPN"} and 
                token.text.lower() not in STOP_WORDS and 
                token.is_alpha and len(token.text) > 2):
                # حذف علائم نگارشی
                clean_text = ''.join(c for c in token.text.lower() if c.isalnum() or c.isspace())
                if clean_text.strip():
                    keywords.add(clean_text.strip())
        
        return sorted(keywords)
    
    def analyze_question_intent(self, query: str) -> Dict[str, Any]:
        """تحلیل مفهومی سوال و استخراج قصد کاربر بر اساس جدول نگاشت Hetionet"""
        query_lower = query.lower()
        
        # 1. تشخیص نوع سوال بر اساس جدول نگاشت
        question_patterns = {
            # بیان ژن در بافت
            'anatomy_expression': {
                'patterns': ['expressed in', 'expression in', 'genes in', 'expressed by', 'genes are expressed', 'what genes are expressed'],
                'metaedges': ['AeG'],
                'description': 'کدام ژن‌ها در [بافت] بیان می‌شوند؟'
            },
            # بیان ژن در مکان خاص
            'gene_expression_location': {
                'patterns': ['where is', 'expressed in', 'found in', 'located in', 'where does'],
                'metaedges': ['GeA'],
                'description': 'ژن [X] در کجا بیان می‌شود؟'
            },
            # مشارکت در فرآیند زیستی
            'biological_participation': {
                'patterns': ['participates in', 'involved in', 'role in', 'part of', 'participate'],
                'metaedges': ['GpBP', 'GpMF', 'GpCC'],
                'description': 'ژن‌هایی که در [فرآیند زیستی] شرکت دارند؟'
            },
            # تعامل ژن‌ها
            'gene_interaction': {
                'patterns': ['interacts', 'interaction', 'binds', 'binding', 'interact with', 'which genes interact'],
                'metaedges': ['GiG'],
                'description': 'ژنی که با ژن [X] تعامل دارد؟'
            },
            # تنظیم ژن توسط بیماری
            'disease_gene_regulation': {
                'patterns': ['regulates', 'upregulates', 'downregulates', 'associated', 'associates'],
                'metaedges': ['DuG', 'DdG', 'DaG'],
                'description': 'ژن‌هایی که بیماری [Y] را تنظیم می‌کنند؟'
            },
            # درمان بیماری
            'disease_treatment': {
                'patterns': ['treats', 'treatment', 'therapy', 'therapeutic', 'treat'],
                'metaedges': ['CtD'],
                'description': 'دارویی که بیماری را درمان می‌کند؟'
            },
            # تنظیم ژن توسط دارو
            'compound_gene_regulation': {
                'patterns': ['upregulates', 'downregulates', 'binds to', 'regulates'],
                'metaedges': ['CuG', 'CdG', 'CbG'],
                'description': 'دارویی که ژن را تنظیم می‌کند؟'
            },
            # بیماری‌های مرتبط با بافت
            'anatomy_disease': {
                'patterns': ['diseases in', 'affects', 'localized to', 'disease in'],
                'metaedges': ['DlA'],
                'description': 'بیماری‌هایی که به [بافت] مربوطند؟'
            },
            # اثر بیماری بر بافت‌ها (جدید)
            'disease_tissue_effect': {
                'patterns': ['how does', 'affect', 'affects', 'effect on', 'effects on', 'tissue', 'tissues'],
                'metaedges': ['DlA', 'DuG', 'DdG', 'AeG', 'AuG', 'AdG', 'GpBP'],
                'description': 'چگونه بیماری بر بافت‌های مختلف اثر می‌گذارد؟'
            },
            # علائم بیماری
            'disease_symptom': {
                'patterns': ['symptoms', 'presents', 'signs', 'manifestation', 'symptom'],
                'metaedges': ['DpS'],
                'description': 'علائم بیماری [Z] چیست؟'
            },
            # بیماری‌های مشابه
            'disease_similarity': {
                'patterns': ['similar', 'resembles', 'alike', 'related', 'similar to'],
                'metaedges': ['DrD'],
                'description': 'بیماری‌های شبیه به بیماری [X]؟'
            },
            # عوارض دارو
            'compound_side_effect': {
                'patterns': ['side effect', 'adverse', 'reaction', 'causes', 'side effects'],
                'metaedges': ['CcSE'],
                'description': 'عوارض داروی [X] چیست؟'
            },
            # مسیرهای ژن
            'gene_pathway': {
                'patterns': ['pathway', 'signaling', 'metabolic', 'cascade', 'pathways'],
                'metaedges': ['GpPW'],
                'description': 'فرآیندهایی که ژن در آنها نقش دارد؟'
            },
            # تنظیم ژن توسط ژن دیگر
            'gene_regulation': {
                'patterns': ['regulates', 'controls', 'regulation', 'regulate'],
                'metaedges': ['Gr>G'],
                'description': 'ژن‌هایی که ژن [X] را تنظیم می‌کنند؟'
            },
            # همبستگی ژن‌ها
            'gene_covariation': {
                'patterns': ['covaries', 'correlated', 'correlation', 'evolutionary'],
                'metaedges': ['GcG'],
                'description': 'ژن‌هایی که با ژن [X] همبستگی دارند؟'
            }
        }
        
        # تشخیص نوع سوال
        detected_type = "general"
        detected_metaedges = []
        
        for qtype, config in question_patterns.items():
            for pattern in config['patterns']:
                if pattern in query_lower:
                    detected_type = qtype
                    detected_metaedges.extend(config['metaedges'])
                    break
            if detected_type != "general":
                break
        
        # 2. شناسایی موجودیت‌ها بر اساس metanodes Hetionet
        entity_mapping = {
            'Gene': ['gene', 'genes', 'protein', 'proteins', 'dna', 'rna', 'mrna', 'genetic', 'molecular', 'tp53', 'brca1', 'apoe', 'cftr', 'mmp9', 'bid', 'kcnq2', 'hmgb3'],
            'Anatomy': ['anatomy', 'organ', 'tissue', 'heart', 'brain', 'liver', 'lung', 'kidney', 'stomach', 'breast'],
            'Disease': ['disease', 'disorder', 'condition', 'syndrome', 'cancer', 'tumor', 'alzheimer', 'diabetes', 'cystic fibrosis', 'breast cancer', 'lung cancer', 'heart disease'],
            'Compound': ['compound', 'drug', 'medication', 'medicine', 'chemical', 'molecule', 'aspirin', 'insulin', 'caffeine', 'vitamin c', 'metformin', 'ibuprofen', 'paracetamol'],
            'Biological Process': ['process', 'biological', 'cellular', 'metabolic', 'apoptosis', 'inflammation', 'cell death', 'dna repair', 'cell cycle'],
            'Pathway': ['pathway', 'signaling', 'metabolic', 'cascade', 'wnt', 'notch', 'apoptosis pathway'],
            'Symptom': ['symptom', 'sign', 'manifestation', 'presentation', 'pain', 'fever', 'cough', 'fatigue'],
            'Side Effect': ['side effect', 'adverse', 'reaction', 'toxicity', 'nausea', 'headache', 'dizziness'],
            'Molecular Function': ['function', 'molecular', 'catalytic', 'binding', 'enzyme', 'receptor', 'transporter'],
            'Cellular Component': ['component', 'cellular', 'organelle', 'structure', 'nucleus', 'mitochondria', 'cell membrane'],
            'Pharmacologic Class': ['class', 'pharmacologic', 'therapeutic', 'antibiotic', 'antiviral', 'nsaid']
        }
        
        # شناسایی موجودیت‌ها
        detected_entities = []
        entity_types = []
        
        for entity_type, keywords in entity_mapping.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if entity_type not in entity_types:
                        entity_types.append(entity_type)
                    if keyword not in detected_entities:
                        detected_entities.append(keyword)
        
        # 3. استخراج کلمات کلیدی با استفاده از تابع بهبود یافته
        keywords = self.extract_keywords(query)
        
        # 4. تشخیص جهت رابطه
        direction = "forward"
        if any(word in query_lower for word in ['where', 'location', 'found in', 'expressed in', 'where is', 'where does']):
            direction = "reverse"
        
        return {
            'question_type': detected_type,
            'metaedges': list(set(detected_metaedges)),  # حذف تکرار
            'entities': detected_entities,
            'entity_types': entity_types,
            'keywords': keywords,
            'direction': direction,
            'query': query,
            'query_lower': query_lower,
            'description': question_patterns.get(detected_type, {}).get('description', 'سوال عمومی')
        }
    
    def intelligent_semantic_search(self, query: str, max_depth: int = 3) -> List[Tuple[str, int, float, str]]:
        """جستجوی معنایی هوشمند بر اساس جدول نگاشت Hetionet"""
        if not self.G:
            return []
        
        # تحلیل مفهومی سوال
        intent = self.analyze_question_intent(query)
        print(f"🔍 تحلیل مفهومی سوال: {intent['question_type']}")
        print(f"📊 موجودیت‌ها: {intent['entity_types']}")
        print(f"🔗 metaedges: {intent['metaedges']}")
        print(f"📝 توضیح: {intent['description']}")
        
        # استخراج کلمات کلیدی
        keywords = intent['keywords']
        print(f"🔑 کلمات کلیدی: {keywords}")
        
        # تطبیق توکن‌ها با نودها
        matched_nodes = self.match_tokens_to_nodes(keywords)
        print(f"🎯 نودهای تطبیق یافته: {matched_nodes}")
        
        # اگر هیچ نودی تطبیق نکرد، سعی کن همه توکن‌ها را تطبیق دهی
        if not matched_nodes:
            print("⚠️ هیچ نودی تطبیق نکرد، تلاش برای تطبیق همه توکن‌ها")
            all_tokens = query.lower().split()
            matched_nodes = self.match_tokens_to_nodes(all_tokens)
            print(f"🎯 نودهای تطبیق یافته (تلاش دوم): {matched_nodes}")
        
        results = []
        
        # تشخیص سوالات خاص ژن-سرطان
        if self._is_gene_cancer_question(query, matched_nodes):
            print("🎯 تشخیص سوال ژن-سرطان")
            results = self._search_gene_cancer_relationships(query, matched_nodes, max_depth)
        # بر اساس نوع سوال و metaedges، روش جستجوی مناسب را انتخاب کن
        elif intent['question_type'] == 'anatomy_expression':
            print("🫀 تشخیص نوع سوال: بیان ژن در آناتومی")
            results = self._search_by_metaedges(matched_nodes, intent, ['AeG'], max_depth)
            
        elif intent['question_type'] == 'gene_expression_location':
            print("📍 تشخیص نوع سوال: مکان بیان ژن")
            results = self._search_by_metaedges(matched_nodes, intent, ['GeA'], max_depth)
            
        elif intent['question_type'] == 'biological_participation':
            print("🧬 تشخیص نوع سوال: مشارکت در فرآیند زیستی")
            results = self._search_by_metaedges(matched_nodes, intent, ['GpBP', 'GpMF', 'GpCC'], max_depth)
            
        elif intent['question_type'] == 'gene_interaction':
            print("🔗 تشخیص نوع سوال: تعامل ژن‌ها")
            results = self._search_by_metaedges(matched_nodes, intent, ['GiG'], max_depth)
            
        elif intent['question_type'] == 'disease_gene_regulation':
            print("🏥 تشخیص نوع سوال: تنظیم ژن توسط بیماری")
            results = self._search_by_metaedges(matched_nodes, intent, ['DuG', 'DdG', 'DaG'], max_depth)
            
        elif intent['question_type'] == 'disease_treatment':
            print("💊 تشخیص نوع سوال: درمان بیماری")
            results = self._search_by_metaedges(matched_nodes, intent, ['CtD'], max_depth)
            
        elif intent['question_type'] == 'compound_gene_regulation':
            print("🧪 تشخیص نوع سوال: تنظیم ژن توسط دارو")
            results = self._search_by_metaedges(matched_nodes, intent, ['CuG', 'CdG', 'CbG'], max_depth)
            
        elif intent['question_type'] == 'anatomy_disease':
            print("🏥 تشخیص نوع سوال: بیماری‌های مرتبط با بافت")
            results = self._search_by_metaedges(matched_nodes, intent, ['DlA'], max_depth)
            
        elif intent['question_type'] == 'disease_symptom':
            print("🤒 تشخیص نوع سوال: علائم بیماری")
            results = self._search_by_metaedges(matched_nodes, intent, ['DpS'], max_depth)
            
        elif intent['question_type'] == 'disease_similarity':
            print("🔄 تشخیص نوع سوال: بیماری‌های مشابه")
            results = self._search_by_metaedges(matched_nodes, intent, ['DrD'], max_depth)
            
        elif intent['question_type'] == 'compound_side_effect':
            print("⚠️ تشخیص نوع سوال: عوارض دارو")
            results = self._search_by_metaedges(matched_nodes, intent, ['CcSE'], max_depth)
            
        elif intent['question_type'] == 'gene_pathway':
            print("🛤️ تشخیص نوع سوال: مسیرهای ژن")
            results = self._search_by_metaedges(matched_nodes, intent, ['GpPW'], max_depth)
            
        elif intent['question_type'] == 'gene_regulation':
            print("🎛️ تشخیص نوع سوال: تنظیم ژن توسط ژن دیگر")
            results = self._search_by_metaedges(matched_nodes, intent, ['Gr>G'], max_depth)
            
        elif intent['question_type'] == 'gene_covariation':
            print("📈 تشخیص نوع سوال: همبستگی ژن‌ها")
            results = self._search_by_metaedges(matched_nodes, intent, ['GcG'], max_depth)
            
        else:
            print("🔍 تشخیص نوع سوال: عمومی")
            # استفاده از تمام metaedges موجود
            all_metaedges = ['AeG', 'GeA', 'GpBP', 'GpMF', 'GpCC', 'GpPW', 'GiG', 'Gr>G', 'GcG', 
                           'DuG', 'DdG', 'DaG', 'DlA', 'DpS', 'DrD', 'CtD', 'CuG', 'CdG', 'CbG', 'CcSE']
            results = self._search_by_metaedges(matched_nodes, intent, all_metaedges, max_depth)
        
        # حذف تکرار و مرتب‌سازی بر اساس امتیاز
        unique_results = {}
        for node_id, depth, score, explanation in results:
            if node_id not in unique_results or score > unique_results[node_id][2]:
                unique_results[node_id] = (node_id, depth, score, explanation)
        
        final_results = sorted(unique_results.values(), key=lambda x: x[2], reverse=True)
        
        return final_results
    
    def _is_gene_cancer_question(self, query: str, matched_nodes: Dict[str, str]) -> bool:
        """تشخیص سوالات ژن-سرطان"""
        query_lower = query.lower()
        cancer_keywords = ['cancer', 'tumor', 'malignancy', 'oncology', 'carcinoma', 'sarcoma', 'leukemia', 'lymphoma']
        
        # بررسی وجود کلمات سرطان
        has_cancer = any(keyword in query_lower for keyword in cancer_keywords)
        
        # بررسی وجود ژن‌ها
        has_gene = any(self.G.nodes[node_id].get('kind') == 'Gene' for node_id in matched_nodes.values())
        
        return has_cancer and has_gene
    
    def _search_gene_cancer_relationships(self, query: str, matched_nodes: Dict[str, str], max_depth: int) -> List[Tuple[str, int, float, str]]:
        """جستجوی روابط ژن-سرطان"""
        results = []
        
        # شناسایی ژن‌ها و بیماری‌های سرطان
        gene_nodes = []
        cancer_nodes = []
        
        for token, node_id in matched_nodes.items():
            node_attrs = self.G.nodes[node_id]
            if node_attrs.get('kind') == 'Gene':
                gene_nodes.append((token, node_id))
            elif node_attrs.get('kind') == 'Disease':
                # بررسی اینکه آیا بیماری سرطان است
                node_name_lower = node_attrs['name'].lower()
                cancer_keywords = ['cancer', 'tumor', 'malignancy', 'carcinoma', 'sarcoma', 'leukemia', 'lymphoma']
                if any(keyword in node_name_lower for keyword in cancer_keywords):
                    cancer_nodes.append((token, node_id))
        
        print(f"🧬 ژن‌های یافت شده: {[name for name, _ in gene_nodes]}")
        print(f"🏥 سرطان‌های یافت شده: {[name for name, _ in cancer_nodes]}")
        
        # جستجوی روابط مستقیم ژن-سرطان
        for gene_token, gene_node_id in gene_nodes:
            gene_name = self.G.nodes[gene_node_id]['name']
            print(f"🔍 جستجوی روابط برای ژن: {gene_name}")
            
            # جستجوی همسایه‌های بیماری
            for neighbor in self.G.neighbors(gene_node_id):
                neighbor_attrs = self.G.nodes[neighbor]
                if neighbor_attrs.get('kind') == 'Disease':
                    edge_data = self.G.get_edge_data(gene_node_id, neighbor)
                    if edge_data:
                        metaedge = edge_data.get('metaedge', 'Unknown')
                        # امتیاز بالاتر برای روابط سرطان
                        neighbor_name_lower = neighbor_attrs['name'].lower()
                        cancer_score = 2.0 if any(keyword in neighbor_name_lower for keyword in ['cancer', 'tumor', 'malignancy']) else 1.0
                        
                        score = self._calculate_metaedge_score(metaedge, 1) * cancer_score
                        explanation = f"{gene_name} related to {neighbor_attrs['name']} via {metaedge}"
                        
                        results.append((neighbor, 1, score, explanation))
                        print(f"  ✅ {neighbor_attrs['name']} - {metaedge} (امتیاز: {score})")
            
            # جستجوی معکوس (بیماری‌ها که به ژن متصل هستند)
            for other_node, other_attrs in self.G.nodes(data=True):
                if other_attrs.get('kind') == 'Disease' and other_node != gene_node_id:
                    for neighbor in self.G.neighbors(other_node):
                        if neighbor == gene_node_id:
                            edge_data = self.G.get_edge_data(other_node, neighbor)
                            if edge_data:
                                metaedge = edge_data.get('metaedge', 'Unknown')
                                # امتیاز بالاتر برای روابط سرطان
                                other_name_lower = other_attrs['name'].lower()
                                cancer_score = 2.0 if any(keyword in other_name_lower for keyword in ['cancer', 'tumor', 'malignancy']) else 1.0
                                
                                score = self._calculate_metaedge_score(metaedge, 1) * cancer_score * 0.8  # امتیاز کمتر برای معکوس
                                explanation = f"{other_attrs['name']} related to {gene_name} via {metaedge}"
                                
                                results.append((other_node, 1, score, explanation))
                                print(f"  ✅ {other_attrs['name']} - {metaedge} معکوس (امتیاز: {score})")
        
        # جستجوی عمیق برای روابط غیرمستقیم
        if max_depth > 1:
            for gene_token, gene_node_id in gene_nodes:
                print(f"🔍 جستجوی عمیق برای ژن: {self.G.nodes[gene_node_id]['name']}")
                dfs_results = self.dfs_search(gene_node_id, max_depth)
                for found_node, depth in dfs_results:
                    found_attrs = self.G.nodes[found_node]
                    if found_attrs.get('kind') == 'Disease':
                        # امتیاز بر اساس عمق و نوع بیماری
                        neighbor_name_lower = found_attrs['name'].lower()
                        cancer_score = 1.5 if any(keyword in neighbor_name_lower for keyword in ['cancer', 'tumor', 'malignancy']) else 1.0
                        
                        score = self._calculate_metaedge_score('Unknown', depth) * cancer_score * (1.0 / depth)
                        explanation = f"{found_attrs['name']} found at depth {depth} from {self.G.nodes[gene_node_id]['name']}"
                        
                        results.append((found_node, depth, score, explanation))
                        print(f"  ✅ {found_attrs['name']} در عمق {depth} (امتیاز: {score})")
        
        return results
    
    def _search_genes_expressed_in_anatomy(self, matched_nodes: Dict[str, str], intent: Dict, max_depth: int = 2) -> List[Tuple[str, int, float, str]]:
        """
        جستجو می‌کند که کدام ژن‌ها از طریق روابط مختلف در یک اندام خاص بیان می‌شوند.
        بر اساس Hetionet: AeG (expresses), AuG (upregulates), AdG (downregulates)

        Args:
            matched_nodes (dict): نگاشت توکن‌ها به نودها
            intent (dict): نتیجه intent detection
            max_depth (int): حداکثر عمق جستجو

        Returns:
            List[Tuple[str, int, float, str]]: لیست نودهای ژن پیدا شده با عمق و نمره
        """
        results = []
        
        for token, node_id in matched_nodes.items():
            if self.G.nodes[node_id]['kind'] == 'Anatomy':
                anatomy_name = self.G.nodes[node_id]['name']
                print(f"🔍 جستجوی ژن‌های مرتبط با {anatomy_name} در Hetionet")
                
                # بررسی تمام روابط مرتبط با بیان ژن
                expression_relations = ['AeG', 'AuG', 'AdG']  # Anatomy -> Gene relations
                
                for relation in expression_relations:
                    relation_name = {
                        'AeG': 'expresses',
                        'AuG': 'upregulates', 
                        'AdG': 'downregulates'
                    }.get(relation, relation)
                    
                    print(f"  🔍 بررسی رابطه {relation} ({relation_name})")
                    
                    for neighbor in self.G.neighbors(node_id):
                        if self.G.nodes[neighbor]['kind'] == 'Gene':
                            edge_data = self.G.get_edge_data(node_id, neighbor)
                            if edge_data and edge_data.get('metaedge') == relation:
                                gene_name = self.G.nodes[neighbor]['name']
                                
                                # امتیازدهی بر اساس نوع رابطه
                                if relation == 'AeG':
                                    score = 5.0  # بیان مستقیم
                                    explanation = f"{gene_name} is expressed in {anatomy_name}"
                                elif relation == 'AuG':
                                    score = 4.5  # تنظیم مثبت
                                    explanation = f"{gene_name} is upregulated in {anatomy_name}"
                                elif relation == 'AdG':
                                    score = 4.0  # تنظیم منفی
                                    explanation = f"{gene_name} is downregulated in {anatomy_name}"
                                else:
                                    score = 3.5
                                    explanation = f"{gene_name} is related to {anatomy_name} via {relation}"
                                
                                results.append((neighbor, 1, score, explanation))
                                print(f"    ✅ {gene_name} - {relation_name} در {anatomy_name} (امتیاز: {score})")
                
                # جستجوی معکوس (Gene -> Anatomy) اگر وجود داشته باشد
                print(f"  🔍 بررسی روابط معکوس (Gene -> {anatomy_name})")
                reverse_relations = ['GeA', 'GuA', 'GdA']  # Gene -> Anatomy relations
                
                for gene_node, gene_attrs in self.G.nodes(data=True):
                    if gene_attrs.get('kind') == 'Gene':
                        for neighbor in self.G.neighbors(gene_node):
                            if neighbor == node_id:
                                edge_data = self.G.get_edge_data(gene_node, neighbor)
                                if edge_data:
                                    relation = edge_data.get('metaedge')
                                    if relation in reverse_relations:
                                        gene_name = gene_attrs['name']
                                        
                                        # امتیازدهی برای روابط معکوس
                                        if relation == 'GeA':
                                            score = 4.0
                                            explanation = f"{gene_name} expresses in {anatomy_name}"
                                        elif relation == 'GuA':
                                            score = 3.5
                                            explanation = f"{gene_name} upregulates in {anatomy_name}"
                                        elif relation == 'GdA':
                                            score = 3.0
                                            explanation = f"{gene_name} downregulates in {anatomy_name}"
                                        else:
                                            score = 2.5
                                            explanation = f"{gene_name} related to {anatomy_name} via {relation}"
                                        
                                        results.append((gene_node, 1, score, explanation))
                                        print(f"    ✅ {gene_name} - رابطه معکوس {relation} با {anatomy_name} (امتیاز: {score})")
                
                # جستجوی عمیق با فیلتر روابط بیان
                print(f"  🔍 جستجوی عمیق با فیلتر روابط بیان")
                for depth in range(2, max_depth + 1):
                    for relation in expression_relations:
                        dfs_results = self.dfs_search(node_id, depth, relation_filter=relation)
                        for gene_node, gene_depth in dfs_results:
                            if self.G.nodes[gene_node]['kind'] == 'Gene':
                                gene_name = self.G.nodes[gene_node]['name']
                                score = 4.0 / gene_depth  # کاهش امتیاز با افزایش عمق
                                explanation = f"{gene_name} related to {anatomy_name} via {relation} (depth {gene_depth})"
                                results.append((gene_node, gene_depth, score, explanation))
                                print(f"    ✅ {gene_name} - عمق {gene_depth} با رابطه {relation} (امتیاز: {score:.2f})")
        
        # حذف تکراری‌ها و مرتب‌سازی بر اساس امتیاز
        unique_results = {}
        for node_id, depth, score, explanation in results:
            if node_id not in unique_results or score > unique_results[node_id][1]:
                unique_results[node_id] = (depth, score, explanation)
        
        # مرتب‌سازی بر اساس امتیاز
        sorted_results = sorted(unique_results.items(), key=lambda x: x[1][1], reverse=True)
        
        final_results = [(node_id, depth, score, explanation) for node_id, (depth, score, explanation) in sorted_results]
        
        print(f"📊 مجموع {len(final_results)} ژن منحصر به فرد یافت شد")
        return final_results
    
    def _add_node_if_not_exists(self, node_id: str):
        """اضافه کردن نود به گراف اگر وجود نداشته باشد"""
        if not self.G.has_node(node_id):
            # ایجاد نود با اطلاعات پیش‌فرض
            self.G.add_node(node_id, name=node_id, kind='Unknown')
            print(f"  ➕ نود اضافه شد: {node_id}")
    
    def _add_edge_if_not_exists(self, source: str, target: str, relation: str = 'Unknown'):
        """اضافه کردن یال به گراف اگر وجود نداشته باشد"""
        if not self.G.has_edge(source, target):
            self.G.add_edge(source, target, metaedge=relation, relation=relation)
            print(f"  ➕ یال اضافه شد: {source} → {target} ({relation})")
    
    def _search_anatomy_expression(self, matched_nodes: Dict[str, str], intent: Dict, max_depth: int) -> List[Tuple[str, int, float, str]]:
        """جستجوی بیان ژن در آناتومی با تمرکز بر روابط AeG (Anatomy → expresses → Gene)"""
        results = []
        
        for token, node_id in matched_nodes.items():
            if self.G.nodes[node_id]['kind'] == 'Anatomy':
                anatomy_name = self.G.nodes[node_id]['name']
                print(f"🔍 جستجوی بیان ژن در {anatomy_name} با استفاده از رابطه AeG")
                
                # روش 1: یافتن مستقیم ژن‌های بیان شده (Anatomy → expresses → Gene)
                for neighbor in self.G.neighbors(node_id):
                    if self.G.nodes[neighbor]['kind'] == 'Gene':
                        edge_data = self.G.get_edge_data(node_id, neighbor)
                        if edge_data and edge_data.get('metaedge') == 'AeG':
                            results.append((neighbor, 1, 5.0, f"{self.G.nodes[neighbor]['name']} expressed in {anatomy_name}"))
                            print(f"  ✅ {self.G.nodes[neighbor]['name']} - بیان مستقیم در {anatomy_name} (AeG)")
                
                # روش 2: جستجوی معکوس (Gene → expresses → Anatomy) - اگر وجود داشته باشد
                for gene_node, gene_attrs in self.G.nodes(data=True):
                    if gene_attrs.get('kind') == 'Gene':
                        for neighbor in self.G.neighbors(gene_node):
                            if neighbor == node_id:
                                edge_data = self.G.get_edge_data(gene_node, neighbor)
                                if edge_data and edge_data.get('metaedge') == 'GeA':
                                    results.append((gene_node, 1, 4.5, f"{gene_attrs['name']} expressed in {anatomy_name}"))
                                    print(f"  ✅ {gene_attrs['name']} - بیان معکوس در {anatomy_name} (GeA)")
                
                # روش 3: جستجوی عمیق با فیلتر دقیق AeG
                for depth in range(2, max_depth + 1):
                    # استفاده از DFS با فیلتر دقیق
                    dfs_results = self.dfs_search(node_id, depth, relation_filter='AeG')
                    for gene_node, gene_depth in dfs_results:
                        if self.G.nodes[gene_node]['kind'] == 'Gene':
                            score = 4.0 / gene_depth
                            results.append((gene_node, gene_depth, score, f"{self.G.nodes[gene_node]['name']} expressed in {anatomy_name} (depth {gene_depth})"))
                            print(f"  ✅ {self.G.nodes[gene_node]['name']} - عمق {gene_depth} (AeG)")
                
                # روش 4: جستجوی بر اساس کلمات کلیدی در نام‌ها (برای قلب)
                if 'heart' in token.lower() or 'heart' in anatomy_name.lower():
                    for gene_node, gene_attrs in self.G.nodes(data=True):
                        if gene_attrs.get('kind') == 'Gene':
                            gene_name = gene_attrs['name'].lower()
                            # جستجوی ژن‌های مرتبط با قلب
                            if any(keyword in gene_name for keyword in ['cardiac', 'heart', 'myocardial', 'cardio']):
                                results.append((gene_node, 2, 3.5, f"ژن مرتبط با قلب: {gene_attrs['name']}"))
                                print(f"  ✅ {gene_attrs['name']} - مرتبط با قلب")
        
        # حذف تکراری‌ها و مرتب‌سازی بر اساس امتیاز
        unique_results = {}
        for node_id, depth, score, reason in results:
            if node_id not in unique_results or score > unique_results[node_id][1]:
                unique_results[node_id] = (depth, score, reason)
        
        # مرتب‌سازی بر اساس امتیاز
        sorted_results = sorted(unique_results.items(), key=lambda x: x[1][1], reverse=True)
        
        return [(node_id, depth, score, reason) for node_id, (depth, score, reason) in sorted_results]
    
    def _search_disease_related(self, matched_nodes: Dict[str, str], intent: Dict, max_depth: int) -> List[Tuple[str, int, float, str]]:
        """جستجوی مرتبط با بیماری‌ها"""
        results = []
        
        for token, node_id in matched_nodes.items():
            if self.G.nodes[node_id]['kind'] == 'Disease':
                # یافتن ژن‌های مرتبط با بیماری
                for neighbor in self.G.neighbors(node_id):
                    if self.G.nodes[neighbor]['kind'] == 'Gene':
                        edge_data = self.G.get_edge_data(node_id, neighbor)
                        if edge_data:
                            results.append((neighbor, 1, 5.0, f"مرتبط با {self.G.nodes[node_id]['name']}"))
                
                # یافتن داروهای درمانی
                for neighbor in self.G.neighbors(node_id):
                    if self.G.nodes[neighbor]['kind'] == 'Drug':
                        edge_data = self.G.get_edge_data(node_id, neighbor)
                        if edge_data and 'treats' in edge_data.get('metaedge', ''):
                            results.append((neighbor, 1, 4.5, f"درمان {self.G.nodes[node_id]['name']}"))
                
                # جستجوی عمیق‌تر
                for depth in range(2, max_depth + 1):
                    for path in nx.single_source_shortest_path(self.G, node_id, cutoff=depth).values():
                        if len(path) == depth + 1:
                            target_node = path[-1]
                            score = 4.0 / depth
                            results.append((target_node, depth, score, f"مسیر {depth} سطحی"))
        
        return results
    
    def _search_drug_related(self, matched_nodes: Dict[str, str], intent: Dict, max_depth: int) -> List[Tuple[str, int, float, str]]:
        """جستجوی مرتبط با داروها"""
        results = []
        
        for token, node_id in matched_nodes.items():
            if self.G.nodes[node_id]['kind'] == 'Drug':
                # یافتن بیماری‌هایی که این دارو درمان می‌کند
                for neighbor in self.G.neighbors(node_id):
                    if self.G.nodes[neighbor]['kind'] == 'Disease':
                        edge_data = self.G.get_edge_data(node_id, neighbor)
                        if edge_data and 'treats' in edge_data.get('metaedge', ''):
                            results.append((neighbor, 1, 5.0, f"درمان شده توسط {self.G.nodes[node_id]['name']}"))
                
                # یافتن ژن‌های هدف
                for neighbor in self.G.neighbors(node_id):
                    if self.G.nodes[neighbor]['kind'] == 'Gene':
                        edge_data = self.G.get_edge_data(node_id, neighbor)
                        if edge_data:
                            results.append((neighbor, 1, 4.5, f"هدف {self.G.nodes[node_id]['name']}"))
        
        return results
    
    def _search_gene_function(self, matched_nodes: Dict[str, str], intent: Dict, max_depth: int) -> List[Tuple[str, int, float, str]]:
        """جستجوی عملکرد ژن‌ها"""
        results = []
        
        for token, node_id in matched_nodes.items():
            if self.G.nodes[node_id]['kind'] == 'Gene':
                # یافتن فرآیندهای زیستی مرتبط
                for neighbor in self.G.neighbors(node_id):
                    if self.G.nodes[neighbor]['kind'] == 'Biological_Process':
                        edge_data = self.G.get_edge_data(node_id, neighbor)
                        if edge_data:
                            results.append((neighbor, 1, 4.5, f"فرآیند مرتبط با {self.G.nodes[node_id]['name']}"))
                
                # یافتن ژن‌های تعاملی
                for neighbor in self.G.neighbors(node_id):
                    if self.G.nodes[neighbor]['kind'] == 'Gene':
                        edge_data = self.G.get_edge_data(node_id, neighbor)
                        if edge_data and 'interacts_with' in edge_data.get('metaedge', ''):
                            results.append((neighbor, 1, 4.0, f"تعامل با {self.G.nodes[node_id]['name']}"))
        
        return results
    
    def _search_general(self, matched_nodes: Dict[str, str], intent: Dict, max_depth: int) -> List[Tuple[str, int, float, str]]:
        """جستجوی عمومی"""
        results = []
        
        for token, node_id in matched_nodes.items():
            # جستجوی همسایه‌ها
            for neighbor in self.G.neighbors(node_id):
                score = 4.0
                results.append((neighbor, 1, score, f"همسایه {self.G.nodes[node_id]['name']}"))
            
            # جستجوی عمیق‌تر
            for depth in range(2, max_depth + 1):
                for path in nx.single_source_shortest_path(self.G, node_id, cutoff=depth).values():
                    if len(path) == depth + 1:
                        target_node = path[-1]
                        score = 3.0 / depth
                        results.append((target_node, depth, score, f"مسیر {depth} سطحی"))
        
        return results
    
    def match_tokens_to_nodes(self, tokens: List[str]) -> Dict[str, str]:
        """تطبیق توکن‌ها با نودهای گراف با پشتیبانی از تطبیق نوع موجودیت بر اساس Hetionet"""
        matched = {}
        
        # نگاشت کامل بر اساس metanodes Hetionet
        fallback_kinds = {
            # Gene (20945 nodes)
            'gene': 'Gene', 'genes': 'Gene', 'protein': 'Gene', 'proteins': 'Gene',
            'dna': 'Gene', 'rna': 'Gene', 'mrna': 'Gene', 'genetic': 'Gene',
            
            # Anatomy (402 nodes)
            'anatomy': 'Anatomy', 'anatomical': 'Anatomy', 'organ': 'Anatomy', 'organs': 'Anatomy',
            'tissue': 'Anatomy', 'tissues': 'Anatomy', 'body': 'Anatomy', 'body part': 'Anatomy',
            'heart': 'Anatomy', 'brain': 'Anatomy', 'liver': 'Anatomy', 'lung': 'Anatomy',
            'kidney': 'Anatomy', 'stomach': 'Anatomy', 'muscle': 'Anatomy', 'bone': 'Anatomy',
            
            # Disease (137 nodes)
            'disease': 'Disease', 'diseases': 'Disease', 'disorder': 'Disease', 'disorders': 'Disease',
            'syndrome': 'Disease', 'syndromes': 'Disease', 'cancer': 'Disease', 'cancers': 'Disease',
            'tumor': 'Disease', 'tumors': 'Disease', 'malignancy': 'Disease', 'malignancies': 'Disease',
            'diabetes': 'Disease', 'alzheimer': 'Disease', 'fibrosis': 'Disease',
            
            # Compound (1552 nodes)
            'compound': 'Compound', 'compounds': 'Compound', 'drug': 'Compound', 'drugs': 'Compound',
            'medication': 'Compound', 'medications': 'Compound', 'medicine': 'Compound', 'medicines': 'Compound',
            'chemical': 'Compound', 'chemicals': 'Compound', 'molecule': 'Compound', 'molecules': 'Compound',
            'aspirin': 'Compound', 'ibuprofen': 'Compound', 'caffeine': 'Compound', 'vitamin': 'Compound',
            
            # Biological Process (11381 nodes)
            'process': 'Biological Process', 'processes': 'Biological Process', 'biological': 'Biological Process',
            'pathway': 'Biological Process', 'pathways': 'Biological Process', 'mechanism': 'Biological Process',
            'function': 'Biological Process', 'functions': 'Biological Process', 'activity': 'Biological Process',
            'apoptosis': 'Biological Process', 'cell cycle': 'Biological Process', 'dna repair': 'Biological Process',
            
            # Pathway (1822 nodes)
            'pathway': 'Pathway', 'pathways': 'Pathway', 'signaling': 'Pathway', 'metabolic': 'Pathway',
            'cascade': 'Pathway', 'cascades': 'Pathway', 'network': 'Pathway', 'networks': 'Pathway',
            
            # Symptom (438 nodes)
            'symptom': 'Symptom', 'symptoms': 'Symptom', 'sign': 'Symptom', 'signs': 'Symptom',
            'manifestation': 'Symptom', 'manifestations': 'Symptom', 'indication': 'Symptom',
            'pain': 'Symptom', 'fever': 'Symptom', 'cough': 'Symptom', 'fatigue': 'Symptom',
            
            # Side Effect (5734 nodes)
            'side effect': 'Side Effect', 'side effects': 'Side Effect', 'adverse': 'Side Effect',
            'reaction': 'Side Effect', 'reactions': 'Side Effect', 'toxicity': 'Side Effect',
            'nausea': 'Side Effect', 'headache': 'Side Effect', 'dizziness': 'Side Effect',
            
            # Molecular Function (2884 nodes)
            'molecular': 'Molecular Function', 'function': 'Molecular Function', 'functions': 'Molecular Function',
            'activity': 'Molecular Function', 'activities': 'Molecular Function', 'enzymatic': 'Molecular Function',
            'enzyme': 'Molecular Function', 'receptor': 'Molecular Function', 'transporter': 'Molecular Function',
            
            # Cellular Component (1391 nodes)
            'cellular': 'Cellular Component', 'component': 'Cellular Component', 'components': 'Cellular Component',
            'organelle': 'Cellular Component', 'organelles': 'Cellular Component', 'structure': 'Cellular Component',
            'nucleus': 'Cellular Component', 'mitochondria': 'Cellular Component', 'membrane': 'Cellular Component',
            
            # Pharmacologic Class (345 nodes)
            'pharmacologic': 'Pharmacologic Class', 'pharmacological': 'Pharmacologic Class', 'class': 'Pharmacologic Class',
            'category': 'Pharmacologic Class', 'categories': 'Pharmacologic Class', 'type': 'Pharmacologic Class',
            'nsaid': 'Pharmacologic Class', 'antibiotic': 'Pharmacologic Class', 'antihypertensive': 'Pharmacologic Class'
        }
        
        # نگاشت ژن‌های مشهور و نام‌های مختلف آنها
        famous_genes = {
            'tp53': ['TP53', 'P53', 'p53', 'Tumor Protein P53', 'Tumor Suppressor P53'],
            'brca1': ['BRCA1', 'Breast Cancer 1', 'BRCA1 Gene'],
            'brca2': ['BRCA2', 'Breast Cancer 2', 'BRCA2 Gene'],
            'apoe': ['APOE', 'Apolipoprotein E', 'APOE Gene'],
            'cftr': ['CFTR', 'Cystic Fibrosis Transmembrane Conductance Regulator'],
            'mmp9': ['MMP9', 'Matrix Metallopeptidase 9'],
            'bid': ['BID', 'BH3 Interacting Domain Death Agonist'],
            'kcnq2': ['KCNQ2', 'Potassium Voltage-Gated Channel Subfamily Q Member 2'],
            'hmgb3': ['HMGB3', 'High Mobility Group Box 3']
        }
        
        for token in tokens:
            token_lower = token.lower()
            found = False
            
            # روش 1: تطبیق ژن‌های مشهور
            if token_lower in famous_genes:
                gene_variants = famous_genes[token_lower]
                for variant in gene_variants:
                    for node_id, attrs in self.G.nodes(data=True):
                        if (attrs.get('kind') == 'Gene' and 
                            variant.upper() in attrs['name'].upper()):
                            matched[token] = node_id
                            found = True
                            print(f"🔍 تطبیق ژن مشهور: '{token}' -> {attrs['name']} ({attrs.get('kind', 'Unknown')})")
                            break
                    if found:
                        break
            
            # روش 2: جستجوی مستقیم بر اساس نام
            if not found:
                for node_id, attrs in self.G.nodes(data=True):
                    if token_lower in attrs['name'].lower():
                        matched[token] = node_id
                        found = True
                        print(f"🔍 تطبیق مستقیم: '{token}' -> {attrs['name']} ({attrs.get('kind', 'Unknown')})")
                        break
                    # تطبیق ژن‌های مشهور
                    elif token.upper() in ['TP53', 'P53'] and 'TP53' in attrs['name'].upper():
                        matched[token] = node_id
                        found = True
                        print(f"🔍 تطبیق ژن مشهور: '{token}' -> {attrs['name']} ({attrs.get('kind', 'Unknown')})")
                        break
            
            # روش 3: جستجو بر اساس نوع موجودیت
            if not found and token_lower in fallback_kinds:
                kind = fallback_kinds[token_lower]
                candidates = [(nid, attrs) for nid, attrs in self.G.nodes(data=True)
                            if attrs.get('kind') == kind or attrs.get('metanode') == kind]
                
                if candidates:
                    # انتخاب بهترین کاندید بر اساس شباهت نام
                    best_candidate = None
                    best_score = 0
                    
                    for nid, attrs in candidates:
                        name_lower = attrs['name'].lower()
                        # محاسبه امتیاز شباهت
                        if token_lower in name_lower:
                            score = len(token_lower) / len(name_lower)
                        elif any(word in name_lower for word in token_lower.split()):
                            score = 0.5
                        else:
                            score = 0.1
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = (nid, attrs)
                    
                    if best_candidate:
                        matched[token] = best_candidate[0]
                        print(f"🔍 تطبیق نوع موجودیت: '{token}' -> {kind} (نمونه: {best_candidate[1]['name']})")
                        found = True
            
            # روش 4: جستجوی جزئی برای کلمات چندبخشی
            if not found and ' ' in token_lower:
                words = token_lower.split()
                for node_id, attrs in self.G.nodes(data=True):
                    name_lower = attrs['name'].lower()
                    if all(word in name_lower for word in words):
                        matched[token] = node_id
                        found = True
                        print(f"🔍 تطبیق جزئی: '{token}' -> {attrs['name']} ({attrs.get('kind', 'Unknown')})")
                        break
            
            # روش 5: جستجوی فازی برای ژن‌ها
            if not found and len(token) >= 3:
                for node_id, attrs in self.G.nodes(data=True):
                    if attrs.get('kind') == 'Gene':
                        name_lower = attrs['name'].lower()
                        # تطبیق فازی برای ژن‌ها
                        if (token_lower in name_lower or 
                            name_lower in token_lower or
                            any(word in name_lower for word in token_lower.split())):
                            matched[token] = node_id
                            found = True
                            print(f"🔍 تطبیق فازی ژن: '{token}' -> {attrs['name']} ({attrs.get('kind', 'Unknown')})")
                            break
            
            if not found:
                print(f"❌ تطبیق نشد: '{token}'")
        
        return matched
    
    def bfs_search(self, start_node: str, max_depth: int = 2) -> List[Tuple[str, int]]:
        """جستجوی سطح اول"""
        visited = set()
        queue = deque([(start_node, 0)])
        result = []
        
        while queue:
            node, depth = queue.popleft()
            if node in visited or depth > max_depth:
                continue
            visited.add(node)
            result.append((node, depth))
            for neighbor in self.G.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return result
    
    def dfs_search(self, start_node: str, max_depth: int = 2, relation_filter: str = None) -> List[Tuple[str, int]]:
        """جستجوی عمیق اول با امکان فیلتر بر اساس نوع رابطه"""
        visited = set()
        result = []
        
        def dfs(node, depth):
            if depth > max_depth or node in visited:
                return
            visited.add(node)
            result.append((node, depth))
            
            for neighbor in self.G.neighbors(node):
                if neighbor not in visited:
                    # اگر فیلتر رابطه مشخص شده، فقط یال‌های مرتبط را بررسی کن
                    if relation_filter:
                        edge_data = self.G.get_edge_data(node, neighbor)
                        if edge_data and relation_filter.lower() in edge_data.get('relation', '').lower():
                            dfs(neighbor, depth + 1)
                    else:
                        dfs(neighbor, depth + 1)
        
        dfs(start_node, 0)
        return result
    
    def get_shortest_paths(self, source: str, target: str, max_paths: int = 3) -> List[List[str]]:
        """یافتن کوتاه‌ترین مسیرها"""
        try:
            paths = list(nx.all_shortest_paths(self.G, source=source, target=target))
            return paths[:max_paths]
        except nx.NetworkXNoPath:
            return []
    
    def get_neighbors_by_type(self, node_id: str, kind_filter: str = None) -> List[Tuple[str, str]]:
        """دریافت همسایه‌ها بر اساس نوع"""
        neighbors = []
        for neighbor in self.G.neighbors(node_id):
            kind = self.G.nodes[neighbor].get('kind')
            if kind_filter is None or kind == kind_filter:
                neighbors.append((neighbor, self.G.nodes[neighbor]['name']))
        return neighbors
    
    def hybrid_search(self, nodes: List[str], max_depth: int = 2) -> List[Tuple[str, int]]:
        """جستجوی ترکیبی"""
        all_results = []
        for node in nodes:
            bfs_result = self.bfs_search(node, max_depth)
            all_results.extend(bfs_result)
        
        # حذف تکراری‌ها و مرتب‌سازی بر اساس عمق
        unique_results = {}
        for node, depth in all_results:
            if node not in unique_results or depth < unique_results[node]:
                unique_results[node] = depth
        
        return sorted(unique_results.items(), key=lambda x: x[1])
    
    def multi_method_search(self, nodes: List[str], max_depth: int = 2) -> List[Tuple[str, int, str]]:
        """جستجوی چند روشی - ترکیب BFS، DFS، و همسایه‌ها"""
        all_results = []
        
        for node in nodes:
            # BFS
            bfs_result = self.bfs_search(node, max_depth)
            for n, depth in bfs_result:
                all_results.append((n, depth, 'BFS'))
            
            # DFS
            dfs_result = self.dfs_search(node, max_depth)
            for n, depth in dfs_result:
                all_results.append((n, depth, 'DFS'))
            
            # همسایه‌های مستقیم
            neighbors = self.get_neighbors_by_type(node)
            for nid, name in neighbors:
                all_results.append((nid, 1, 'Neighbors'))
        
        # حذف تکراری‌ها و امتیازدهی
        unique_results = {}
        for node, depth, method in all_results:
            if node not in unique_results:
                unique_results[node] = {'depth': depth, 'methods': [method], 'score': 1.0}
            else:
                unique_results[node]['methods'].append(method)
                unique_results[node]['score'] += 0.5  # امتیاز بیشتر برای روش‌های مختلف
                if depth < unique_results[node]['depth']:
                    unique_results[node]['depth'] = depth
        
        # مرتب‌سازی بر اساس امتیاز و عمق
        sorted_results = []
        for node, info in unique_results.items():
            sorted_results.append((node, info['depth'], ', '.join(info['methods'])))
        
        return sorted(sorted_results, key=lambda x: (x[1], -len(x[2].split(', '))))
    
    def ensemble_search(self, nodes: List[str], max_depth: int = 2) -> List[Tuple[str, int, float]]:
        """جستجوی گروهی - ترکیب همه روش‌ها با وزن‌دهی"""
        method_weights = {
            'BFS': 1.0,
            'DFS': 0.8,
            'SHORTEST_PATH': 1.2,
            'NEIGHBORS': 0.9
        }
        
        all_results = {}
        
        for node in nodes:
            # BFS
            bfs_result = self.bfs_search(node, max_depth)
            for n, depth in bfs_result:
                if n not in all_results:
                    all_results[n] = {'score': 0, 'depth': depth, 'count': 0}
                all_results[n]['score'] += method_weights['BFS'] / (depth + 1)
                all_results[n]['count'] += 1
            
            # DFS
            dfs_result = self.dfs_search(node, max_depth)
            for n, depth in dfs_result:
                if n not in all_results:
                    all_results[n] = {'score': 0, 'depth': depth, 'count': 0}
                all_results[n]['score'] += method_weights['DFS'] / (depth + 1)
                all_results[n]['count'] += 1
            
            # همسایه‌ها
            neighbors = self.get_neighbors_by_type(node)
            for nid, name in neighbors:
                if nid not in all_results:
                    all_results[nid] = {'score': 0, 'depth': 1, 'count': 0}
                all_results[nid]['score'] += method_weights['NEIGHBORS']
                all_results[nid]['count'] += 1
        
        # مرتب‌سازی بر اساس امتیاز
        sorted_results = []
        for node, info in all_results.items():
            final_score = info['score'] * (1 + 0.1 * info['count'])  # پاداش برای تکرار
            sorted_results.append((node, info['depth'], final_score))
        
        return sorted(sorted_results, key=lambda x: x[2], reverse=True)
    
    def adaptive_search(self, nodes: List[str], max_depth: int = 2, query: str = "") -> List[Tuple[str, int, str]]:
        """جستجوی تطبیقی - انتخاب روش بر اساس نوع نود و سوال"""
        all_results = []
        query_lower = query.lower()
        
        # تشخیص نوع سوال
        is_expression_question = any(word in query_lower for word in ['expressed', 'expression', 'express', 'genes'])
        is_relationship_question = any(word in query_lower for word in ['relationship', 'related', 'connection', 'link'])
        is_function_question = any(word in query_lower for word in ['function', 'role', 'purpose', 'effect'])
        
        print(f"🔍 تشخیص نوع سوال: expression={is_expression_question}, relationship={is_relationship_question}, function={is_function_question}")
        
        for node in nodes:
            node_kind = self.G.nodes[node]['kind']
            node_name = self.G.nodes[node]['name']
            print(f"  📍 پردازش نود: {node_name} ({node_kind})")
            
            # انتخاب روش بر اساس نوع نود و سوال
            if node_kind == 'Anatomy' and is_expression_question:
                # برای سوالات بیان در آناتومی، از جستجوی تخصصی استفاده کن
                print(f"    🫀 استفاده از جستجوی تخصصی آناتومی برای {node_name}")
                
                # جستجوی مستقیم ژن‌های بیان شده
                for neighbor in self.G.neighbors(node):
                    if self.G.nodes[neighbor]['kind'] == 'Gene':
                        edge_data = self.G.get_edge_data(node, neighbor)
                        if edge_data:
                            relation = edge_data.get('metaedge', '')
                            if relation == 'AeG':
                                all_results.append((neighbor, 1, 'Expression-Direct'))
                                print(f"      ✅ {self.G.nodes[neighbor]['name']} - بیان مستقیم (AeG)")
                
                # جستجوی معکوس
                for gene_node, gene_attrs in self.G.nodes(data=True):
                    if gene_attrs.get('kind') == 'Gene':
                        for neighbor in self.G.neighbors(gene_node):
                            if neighbor == node:
                                edge_data = self.G.get_edge_data(gene_node, neighbor)
                                if edge_data:
                                    relation = edge_data.get('metaedge', '')
                                    if relation == 'GeA':
                                        all_results.append((gene_node, 1, 'Expression-Reverse'))
                                        print(f"      ✅ {gene_attrs['name']} - بیان معکوس (GeA)")
                
                # جستجوی عمیق با فیلتر
                dfs_result = self.dfs_search(node, max_depth, relation_filter='AeG')
                for n, depth in dfs_result:
                    if self.G.nodes[n]['kind'] == 'Gene':
                        all_results.append((n, depth, 'Expression-DFS'))
                        print(f"      ✅ {self.G.nodes[n]['name']} - عمق {depth}")
            
            elif node_kind in ['Gene', 'Disease']:
                # برای ژن‌ها و بیماری‌ها از BFS و همسایه‌ها
                print(f"    🧬 استفاده از BFS برای {node_name}")
                bfs_result = self.bfs_search(node, max_depth)
                for n, depth in bfs_result:
                    all_results.append((n, depth, 'BFS'))
                
                neighbors = self.get_neighbors_by_type(node)
                for nid, name in neighbors:
                    all_results.append((nid, 1, 'Neighbors'))
            
            elif node_kind in ['Drug', 'Compound']:
                # برای داروها از DFS و کوتاه‌ترین مسیر
                print(f"    💊 استفاده از DFS برای {node_name}")
                dfs_result = self.dfs_search(node, max_depth)
                for n, depth in dfs_result:
                    all_results.append((n, depth, 'DFS'))
            
            elif node_kind in ['Biological Process', 'Pathway']:
                # برای فرآیندهای زیستی از همه روش‌ها
                print(f"    ⚙️ استفاده از روش‌های ترکیبی برای {node_name}")
                bfs_result = self.bfs_search(node, max_depth)
                for n, depth in bfs_result:
                    all_results.append((n, depth, 'BFS'))
                
                dfs_result = self.dfs_search(node, max_depth)
                for n, depth in dfs_result:
                    all_results.append((n, depth, 'DFS'))
            
            else:
                # برای بقیه از روش ترکیبی
                print(f"    🔄 استفاده از روش ترکیبی برای {node_name}")
                hybrid_result = self.hybrid_search([node], max_depth)
                for n, depth in hybrid_result:
                    all_results.append((n, depth, 'Hybrid'))
        
        # حذف تکراری‌ها و امتیازدهی
        unique_results = {}
        for node, depth, method in all_results:
            if node not in unique_results:
                unique_results[node] = (depth, method, 1)
            else:
                # افزایش امتیاز برای تکرار
                unique_results[node] = (min(depth, unique_results[node][0]), 
                                      method, unique_results[node][2] + 1)
        
        # مرتب‌سازی بر اساس عمق و امتیاز
        sorted_results = []
        for node, (depth, method, count) in unique_results.items():
            # امتیازدهی بر اساس روش
            method_score = {
                'Expression-Direct': 5.0,
                'Expression-Reverse': 4.5,
                'Expression-DFS': 4.0,
                'BFS': 3.5,
                'DFS': 3.0,
                'Neighbors': 2.5,
                'Hybrid': 2.0
            }.get(method, 1.0)
            
            final_score = method_score * (1 + 0.1 * count) / (depth + 1)
            sorted_results.append((node, depth, method, final_score))
        
        # مرتب‌سازی بر اساس امتیاز نهایی
        sorted_results.sort(key=lambda x: x[3], reverse=True)
        
        return [(node, depth, method) for node, depth, method, score in sorted_results]
    
    def retrieve_information(self, query: str, method: RetrievalMethod, 
                           max_depth: int = 2, max_nodes: int = 10) -> RetrievalResult:
        """بازیابی اطلاعات از گراف"""
        print(f"🔍 بازیابی اطلاعات با روش {method.value if hasattr(method, 'value') else method}...")
        
        # استخراج کلمات کلیدی
        keywords = self.extract_keywords(query)
        print(f"کلمات کلیدی: {keywords}")
        
        # تطبیق با نودهای گراف
        matches = self.match_tokens_to_nodes(keywords)
        print(f"تطبیق‌های یافت شده: {matches}")
        
        nodes = []
        edges = []
        paths = []
        
        if method == RetrievalMethod.BFS:
            # BFS برای هر نود تطبیق یافته
            for token, node_id in matches.items():
                bfs_result = self.bfs_search(node_id, max_depth)
                for node, depth in bfs_result[:max_nodes]:
                    nodes.append(GraphNode(
                        id=node,
                        name=self.G.nodes[node]['name'],
                        kind=self.G.nodes[node]['kind'],
                        depth=depth
                    ))
        
        elif method == RetrievalMethod.DFS:
            # DFS برای هر نود تطبیق یافته
            for token, node_id in matches.items():
                dfs_result = self.dfs_search(node_id, max_depth)
                for node, depth in dfs_result[:max_nodes]:
                    nodes.append(GraphNode(
                        id=node,
                        name=self.G.nodes[node]['name'],
                        kind=self.G.nodes[node]['kind'],
                        depth=depth
                    ))
        
        elif method == RetrievalMethod.SHORTEST_PATH:
            # کوتاه‌ترین مسیر بین نودها
            if len(matches) >= 2:
                node_ids = list(matches.values())
                for i in range(len(node_ids)):
                    for j in range(i+1, len(node_ids)):
                        paths.extend(self.get_shortest_paths(node_ids[i], node_ids[j]))
                        
                        # اضافه کردن نودهای مسیر
                        for path in paths:
                            for k, node in enumerate(path):
                                nodes.append(GraphNode(
                                    id=node,
                                    name=self.G.nodes[node]['name'],
                                    kind=self.G.nodes[node]['kind'],
                                    depth=k
                                ))
        
        elif method == RetrievalMethod.NEIGHBORS:
            # همسایه‌های مستقیم
            for token, node_id in matches.items():
                neighbors = self.get_neighbors_by_type(node_id)
                for nid, name in neighbors[:max_nodes]:
                    nodes.append(GraphNode(
                        id=nid,
                        name=name,
                        kind=self.G.nodes[nid]['kind'],
                        depth=1
                    ))
        
        elif method == RetrievalMethod.HYBRID:
            # ترکیبی از روش‌ها
            if len(matches) >= 2:
                node_ids = list(matches.values())
                hybrid_result = self.hybrid_search(node_ids, max_depth)
                for node, depth in hybrid_result[:max_nodes]:
                    nodes.append(GraphNode(
                        id=node,
                        name=self.G.nodes[node]['name'],
                        kind=self.G.nodes[node]['kind'],
                        depth=depth
                    ))
        
        elif method == RetrievalMethod.MULTI_METHOD:
            # جستجوی چند روشی
            node_ids = list(matches.values())
            multi_result = self.multi_method_search(node_ids, max_depth)
            for node, depth, methods in multi_result[:max_nodes]:
                nodes.append(GraphNode(
                    id=node,
                    name=self.G.nodes[node]['name'],
                    kind=self.G.nodes[node]['kind'],
                    depth=depth,
                    score=len(methods.split(', '))  # امتیاز بر اساس تعداد روش‌ها
                ))
        
        elif method == RetrievalMethod.ENSEMBLE:
            # جستجوی گروهی
            node_ids = list(matches.values())
            ensemble_result = self.ensemble_search(node_ids, max_depth)
            for node, depth, score in ensemble_result[:max_nodes]:
                nodes.append(GraphNode(
                    id=node,
                    name=self.G.nodes[node]['name'],
                    kind=self.G.nodes[node]['kind'],
                    depth=depth,
                    score=score
                ))
        
        elif method == RetrievalMethod.ADAPTIVE:
            # جستجوی تطبیقی با پاس دادن query
            node_ids = list(matches.values())
            adaptive_result = self.adaptive_search(node_ids, max_depth, query)
            for node, depth, method in adaptive_result[:max_nodes]:
                nodes.append(GraphNode(
                    id=node,
                    name=self.G.nodes[node]['name'],
                    kind=self.G.nodes[node]['kind'],
                    depth=depth
                ))
        
        elif method == RetrievalMethod.INTELLIGENT:
            # جستجوی معنایی هوشمند
            intelligent_result = self.intelligent_semantic_search(query, max_depth)
            
            # تبدیل نتایج به GraphNode
            for node_id, depth, score, reason in intelligent_result[:max_nodes]:
                nodes.append(GraphNode(
                    id=node_id,
                    name=self.G.nodes[node_id]['name'],
                    kind=self.G.nodes[node_id]['kind'],
                    depth=depth,
                    score=score
                ))
            
            # یافتن مسیرهای ارتباطی بین نودها
            if len(nodes) >= 2:
                node_ids = [node.id for node in nodes]
                for i in range(len(node_ids)):
                    for j in range(i+1, len(node_ids)):
                        paths.extend(self.get_shortest_paths(node_ids[i], node_ids[j]))
            
            # یافتن یال‌های مرتبط
            for node in nodes:
                for neighbor in self.G.neighbors(node.id):
                    if any(n.id == neighbor for n in nodes):
                        edge_data = self.G.get_edge_data(node.id, neighbor)
                        if edge_data:
                            edges.append(GraphEdge(
                                source=node.id,
                                target=neighbor,
                                relation=edge_data.get('metaedge', 'related'),
                                weight=edge_data.get('weight', 1.0)
                            ))
        
        elif method == RetrievalMethod.NO_RETRIEVAL:
            # بدون بازیابی - فقط مدل
            print("🔍 بدون بازیابی از گراف - فقط استفاده از مدل")
            # ایجاد یک نود خالی برای حفظ ساختار
            nodes.append(GraphNode(
                id="no_retrieval",
                name="بدون بازیابی",
                kind="System",
                depth=0
            ))
        
        # حذف تکراری‌ها
        unique_nodes = {}
        for node in nodes:
            if node.id not in unique_nodes:
                unique_nodes[node.id] = node
        
        nodes = list(unique_nodes.values())
        
        # ایجاد یال‌ها بین نودهای مرتبط
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                if self.G.has_edge(node1.id, node2.id):
                    edge_data = self.G.get_edge_data(node1.id, node2.id)
                    edges.append(GraphEdge(
                        source=node1.id,
                        target=node2.id,
                        relation=edge_data['metaedge']
                    ))
        
        # ایجاد متن زمینه بهبود یافته
        retrieval_result = RetrievalResult(
            nodes=nodes,
            edges=edges,
            paths=paths,
            context_text="",
            method=method.value,
            query=query
        )
        context_text = self._create_enhanced_context_text(retrieval_result)
        
        return RetrievalResult(
            nodes=nodes,
            edges=edges,
            paths=paths,
            context_text=context_text,
            method=method.value if hasattr(method, 'value') else str(method),
            query=query
        )
    
    def create_context_text(self, nodes: List[GraphNode], edges: List[GraphEdge], 
                           paths: List[List[str]]) -> str:
        """ایجاد متن زمینه بهبود یافته با اطلاعات زیستی غنی شده"""
        # استفاده از تابع جدید برای غنی‌سازی
        retrieval_result = RetrievalResult(
            nodes=nodes,
            edges=edges,
            paths=paths,
            context_text="",
            method="Enhanced",
            query=""
        )
        return self._create_enhanced_context_text(retrieval_result)
    
    def _enrich_retrieved_data(self, nodes: List[GraphNode], edges: List[GraphEdge], query: str) -> Dict[str, Any]:
        """
        غنی‌سازی داده‌های بازیابی شده با اطلاعات زیستی و روابط معنادار
        """
        enriched_data = {
            'biological_context': {},
            'relationship_details': [],
            'tissue_specific_info': {},
            'gene_functions': {},
            'disease_associations': {},
            'pathway_information': {}
        }
        
        # 1. استخراج اطلاعات بافت‌محور
        anatomy_nodes = [n for n in nodes if n.kind == 'Anatomy']
        gene_nodes = [n for n in nodes if n.kind == 'Gene']
        
        for anatomy in anatomy_nodes:
            enriched_data['tissue_specific_info'][anatomy.name] = {
                'genes_expressed': [],
                'genes_upregulated': [],
                'genes_downregulated': [],
                'biological_significance': self._get_anatomy_significance(anatomy.name)
            }
            
            # یافتن ژن‌های مرتبط با این بافت
            for edge in edges:
                if edge.source == anatomy.id and edge.target in [g.id for g in gene_nodes]:
                    gene_name = next((g.name for g in gene_nodes if g.id == edge.target), edge.target)
                    if edge.relation == 'AeG':
                        enriched_data['tissue_specific_info'][anatomy.name]['genes_expressed'].append(gene_name)
                    elif edge.relation == 'AuG':
                        enriched_data['tissue_specific_info'][anatomy.name]['genes_upregulated'].append(gene_name)
                    elif edge.relation == 'AdG':
                        enriched_data['tissue_specific_info'][anatomy.name]['genes_downregulated'].append(gene_name)
        
        # 2. استخراج اطلاعات عملکرد ژن‌ها
        for gene in gene_nodes:
            enriched_data['gene_functions'][gene.name] = {
                'biological_processes': [],
                'molecular_functions': [],
                'cellular_components': [],
                'pathways': [],
                'disease_associations': []
            }
            
            # یافتن فرآیندهای زیستی مرتبط
            for edge in edges:
                if edge.source == gene.id:
                    target_node = next((n for n in nodes if n.id == edge.target), None)
                    if target_node:
                        if edge.relation == 'GpBP':
                            enriched_data['gene_functions'][gene.name]['biological_processes'].append(target_node.name)
                        elif edge.relation == 'GpMF':
                            enriched_data['gene_functions'][gene.name]['molecular_functions'].append(target_node.name)
                        elif edge.relation == 'GpCC':
                            enriched_data['gene_functions'][gene.name]['cellular_components'].append(target_node.name)
                        elif edge.relation == 'GpPW':
                            enriched_data['gene_functions'][gene.name]['pathways'].append(target_node.name)
                        elif edge.relation == 'DaG':
                            enriched_data['gene_functions'][gene.name]['disease_associations'].append(target_node.name)
        
        # 3. ایجاد متن توصیفی زیستی
        enriched_data['biological_context'] = self._create_biological_context(enriched_data, query)
        
        return enriched_data
    
    def _get_anatomy_significance(self, anatomy_name: str) -> str:
        """
        دریافت اهمیت زیستی بافت‌ها
        """
        significance_map = {
            'heart': 'عضله قلب، مسئول پمپاژ خون و عملکرد سیستم قلبی-عروقی',
            'brain': 'مرکز کنترل سیستم عصبی، مسئول تفکر، حافظه و عملکردهای شناختی',
            'liver': 'مرکز متابولیسم بدن، مسئول سم‌زدایی و تولید پروتئین‌های ضروری',
            'kidney': 'تصفیه خون و تنظیم تعادل الکترولیت‌ها',
            'lung': 'تبادل گازهای تنفسی و اکسیژن‌رسانی به بدن',
            'muscle': 'حرکت و انقباض، تولید انرژی و حفظ وضعیت بدن',
            'blood': 'انتقال مواد مغذی، اکسیژن و سلول‌های ایمنی',
            'skin': 'محافظت از بدن، تنظیم دما و حس لمس'
        }
        return significance_map.get(anatomy_name.lower(), f'بافت {anatomy_name} با عملکردهای زیستی متعدد')
    
    def _create_biological_context(self, enriched_data: Dict, query: str) -> str:
        """
        ایجاد متن توصیفی زیستی بر اساس داده‌های غنی شده
        """
        context_parts = []
        
        # تحلیل بافت‌های موجود
        for tissue, info in enriched_data['tissue_specific_info'].items():
            if info['genes_expressed'] or info['genes_upregulated'] or info['genes_downregulated']:
                context_parts.append(f"**{tissue}:** {info['biological_significance']}")
                
                if info['genes_expressed']:
                    context_parts.append(f"ژن‌های بیان شده: {', '.join(info['genes_expressed'][:5])}")
                if info['genes_upregulated']:
                    context_parts.append(f"ژن‌های تنظیم مثبت: {', '.join(info['genes_upregulated'][:3])}")
                if info['genes_downregulated']:
                    context_parts.append(f"ژن‌های تنظیم منفی: {', '.join(info['genes_downregulated'][:3])}")
                context_parts.append("")
        
        # تحلیل عملکرد ژن‌ها
        for gene, functions in enriched_data['gene_functions'].items():
            if any(functions.values()):
                context_parts.append(f"**{gene}:**")
                if functions['biological_processes']:
                    context_parts.append(f"فرآیندهای زیستی: {', '.join(functions['biological_processes'][:3])}")
                if functions['pathways']:
                    context_parts.append(f"مسیرهای زیستی: {', '.join(functions['pathways'][:3])}")
                if functions['disease_associations']:
                    context_parts.append(f"ارتباط با بیماری‌ها: {', '.join(functions['disease_associations'][:3])}")
                context_parts.append("")
        
        return "\n".join(context_parts) if context_parts else "اطلاعات زیستی محدودی در دسترس است."
    
    def _create_enhanced_context_text(self, retrieval_result: RetrievalResult) -> str:
        """
        ایجاد متن زمینه خلاصه و کاربردی
        """
        # استفاده از روش جدید بازیابی هدفمند
        intent = self.analyze_question_intent(retrieval_result.query)
        retrieval_data = self._targeted_retrieval_for_question(retrieval_result.query, intent)
        
        # ایجاد متن ساختاریافته خلاصه
        structured_text = self._create_structured_text_for_model(retrieval_data, retrieval_result.query)
        
        # اضافه کردن اطلاعات آماری کوتاه
        context_parts = []
        context_parts.append("📊 **آمار بازیابی:**")
        context_parts.append(f"• نودها: {len(retrieval_result.nodes)}, روابط: {len(retrieval_result.edges)}")
        context_parts.append(f"• ژن‌های اصلی: {len(retrieval_data['primary_genes'])}")
        context_parts.append("")
        
        # اضافه کردن متن ساختاریافته
        context_parts.append("🧬 **داده‌های کلیدی:**")
        context_parts.append(structured_text)
        
        return "\n".join(context_parts)
    
    def _targeted_retrieval_for_question(self, query: str, intent: Dict) -> Dict[str, Any]:
        """
        بازیابی هدفمند بر اساس نوع سوال و metaedge های مرتبط
        """
        matched_nodes = self.match_tokens_to_nodes(self.extract_keywords(query))
        question_type = intent.get('question_type', 'general')
        
        retrieval_data = {
            'primary_genes': [],
            'secondary_genes': [],
            'biological_processes': [],
            'pathways': [],
            'diseases': [],
            'drugs': [],
            'anatomy': [],
            'metaedges_used': [],
            'relationships': []
        }
        
        # تعیین metaedge های هدف بر اساس نوع سوال
        target_metaedges = self._get_target_metaedges_for_question(question_type, query)
        retrieval_data['metaedges_used'] = target_metaedges
        
        print(f"🎯 بازیابی هدفمند برای سوال: {question_type}")
        print(f"📋 Metaedge های هدف: {target_metaedges}")
        
        # بازیابی اولیه بر اساس metaedge های اصلی
        for metaedge in target_metaedges:
            results = self._search_by_metaedges(matched_nodes, intent, [metaedge], max_depth=2)
            
            for node_id, depth, score, explanation in results:
                node_name = self.G.nodes[node_id]['name']
                node_kind = self.G.nodes[node_id]['kind']
                
                # دسته‌بندی نتایج بر اساس نوع
                if node_kind == 'Gene':
                    if metaedge in ['AeG', 'AuG', 'AdG', 'DaG', 'DuG', 'DdG']:
                        retrieval_data['primary_genes'].append({
                            'name': node_name,
                            'metaedge': metaedge,
                            'score': score,
                            'explanation': explanation
                        })
                    else:
                        retrieval_data['secondary_genes'].append({
                            'name': node_name,
                            'metaedge': metaedge,
                            'score': score,
                            'explanation': explanation
                        })
                elif node_kind == 'Biological Process':
                    retrieval_data['biological_processes'].append({
                        'name': node_name,
                        'metaedge': metaedge,
                        'score': score
                    })
                elif node_kind == 'Pathway':
                    retrieval_data['pathways'].append({
                        'name': node_name,
                        'metaedge': metaedge,
                        'score': score
                    })
                elif node_kind == 'Disease':
                    retrieval_data['diseases'].append({
                        'name': node_name,
                        'metaedge': metaedge,
                        'score': score
                    })
                elif node_kind == 'Compound':
                    retrieval_data['drugs'].append({
                        'name': node_name,
                        'metaedge': metaedge,
                        'score': score
                    })
                elif node_kind == 'Anatomy':
                    retrieval_data['anatomy'].append({
                        'name': node_name,
                        'metaedge': metaedge,
                        'score': score
                    })
        
        # غنی‌سازی با اطلاعات اضافی برای ژن‌های اصلی
        retrieval_data = self._enrich_primary_genes(retrieval_data)
        
        # برای سوالات مربوط به اثر بیماری بر بافت‌ها، مسیرهای ترکیبی را اضافه کن
        if 'DlA' in target_metaedges:
            retrieval_data = self._add_tissue_disease_paths(retrieval_data, matched_nodes)
        
        # برای سوالات مربوط به درمان بیماری، مسیرهای درمانی را اضافه کن
        if 'CtD' in target_metaedges:
            retrieval_data = self._add_treatment_paths(retrieval_data, matched_nodes)
        
        return retrieval_data
    
    def _add_tissue_disease_paths(self, retrieval_data: Dict[str, Any], matched_nodes: Dict[str, str]) -> Dict[str, Any]:
        """
        اضافه کردن مسیرهای ترکیبی بیماری→بافت→ژن برای سوالات مربوط به اثر بیماری بر بافت‌ها
        """
        tissue_disease_paths = []
        
        # یافتن بیماری‌ها در matched_nodes
        disease_nodes = []
        for token, node_id in matched_nodes.items():
            node_attrs = self.G.nodes[node_id]
            if node_attrs.get('kind') == 'Disease':
                disease_nodes.append((node_id, node_attrs['name']))
        
        # برای هر بیماری، بافت‌های مرتبط و ژن‌های بیان شده در آن بافت‌ها را پیدا کن
        for disease_id, disease_name in disease_nodes:
            # یافتن بافت‌های مرتبط با بیماری (DlA)
            for neighbor in self.G.neighbors(disease_id):
                neighbor_attrs = self.G.nodes[neighbor]
                edge_data = self.G.get_edge_data(disease_id, neighbor)
                
                if edge_data and edge_data.get('metaedge') == 'DlA' and neighbor_attrs.get('kind') == 'Anatomy':
                    tissue_name = neighbor_attrs['name']
                    tissue_id = neighbor
                    
                    # یافتن ژن‌های بیان شده در این بافت (AeG)
                    tissue_genes = []
                    for gene_neighbor in self.G.neighbors(tissue_id):
                        gene_attrs = self.G.nodes[gene_neighbor]
                        gene_edge_data = self.G.get_edge_data(tissue_id, gene_neighbor)
                        
                        if gene_edge_data and gene_edge_data.get('metaedge') == 'AeG' and gene_attrs.get('kind') == 'Gene':
                            gene_name = gene_attrs['name']
                            
                            # یافتن فرآیندهای زیستی مرتبط با این ژن (GpBP)
                            biological_processes = []
                            for bp_neighbor in self.G.neighbors(gene_neighbor):
                                bp_attrs = self.G.nodes[bp_neighbor]
                                bp_edge_data = self.G.get_edge_data(gene_neighbor, bp_neighbor)
                                
                                if bp_edge_data and bp_edge_data.get('metaedge') == 'GpBP' and bp_attrs.get('kind') == 'Biological Process':
                                    biological_processes.append(bp_attrs['name'])
                            
                            tissue_genes.append({
                                'gene_name': gene_name,
                                'biological_processes': biological_processes[:2]  # حداکثر 2 فرآیند
                            })
                    
                    # اضافه کردن مسیر کامل
                    if tissue_genes:
                        tissue_disease_paths.append({
                            'disease': disease_name,
                            'tissue': tissue_name,
                            'genes': tissue_genes[:3]  # حداکثر 3 ژن
                        })
        
        retrieval_data['tissue_disease_paths'] = tissue_disease_paths
        return retrieval_data
    
    def _add_treatment_paths(self, retrieval_data: Dict[str, Any], matched_nodes: Dict[str, str]) -> Dict[str, Any]:
        """
        اضافه کردن مسیرهای درمانی Compound→Disease→Gene برای سوالات مربوط به درمان
        """
        treatment_paths = []
        
        # یافتن بیماری‌ها در matched_nodes
        disease_nodes = []
        for token, node_id in matched_nodes.items():
            node_attrs = self.G.nodes[node_id]
            if node_attrs.get('kind') == 'Disease':
                disease_nodes.append((node_id, node_attrs['name']))
        
        # برای هر بیماری، داروهای درمانی و ژن‌های مرتبط را پیدا کن
        for disease_id, disease_name in disease_nodes:
            # یافتن داروهای درمانی (CtD)
            for neighbor in self.G.neighbors(disease_id):
                neighbor_attrs = self.G.nodes[neighbor]
                edge_data = self.G.get_edge_data(disease_id, neighbor)
                
                if edge_data and edge_data.get('metaedge') == 'CtD' and neighbor_attrs.get('kind') == 'Compound':
                    drug_name = neighbor_attrs['name']
                    drug_id = neighbor
                    
                    # یافتن ژن‌های تنظیم شده توسط این دارو (CuG, CdG)
                    drug_genes = []
                    for gene_neighbor in self.G.neighbors(drug_id):
                        gene_attrs = self.G.nodes[gene_neighbor]
                        gene_edge_data = self.G.get_edge_data(drug_id, gene_neighbor)
                        
                        if gene_edge_data and gene_edge_data.get('metaedge') in ['CuG', 'CdG'] and gene_attrs.get('kind') == 'Gene':
                            gene_name = gene_attrs['name']
                            
                            # یافتن فرآیندهای زیستی مرتبط با این ژن (GpBP)
                            biological_processes = []
                            for bp_neighbor in self.G.neighbors(gene_neighbor):
                                bp_attrs = self.G.nodes[bp_neighbor]
                                bp_edge_data = self.G.get_edge_data(gene_neighbor, bp_neighbor)
                                
                                if bp_edge_data and bp_edge_data.get('metaedge') == 'GpBP' and bp_attrs.get('kind') == 'Biological Process':
                                    biological_processes.append(bp_attrs['name'])
                            
                            drug_genes.append({
                                'gene_name': gene_name,
                                'regulation': gene_edge_data.get('metaedge'),
                                'biological_processes': biological_processes[:2]  # حداکثر 2 فرآیند
                            })
                    
                    # اضافه کردن مسیر کامل
                    if drug_genes:
                        treatment_paths.append({
                            'disease': disease_name,
                            'drug': drug_name,
                            'genes': drug_genes[:3]  # حداکثر 3 ژن
                        })
        
        retrieval_data['treatment_paths'] = treatment_paths
        return retrieval_data
    
    def _get_target_metaedges_for_question(self, question_type: str, query: str) -> List[str]:
        """
        تعیین metaedge های هدف بر اساس نوع سوال
        """
        query_lower = query.lower()
        
        # سوالات مربوط به بیان ژن در بافت‌ها
        if any(word in query_lower for word in ['expressed', 'express', 'expression']):
            if any(word in query_lower for word in ['heart', 'cardiac', 'myocardium']):
                return ['AeG', 'AuG', 'AdG']  # بیان، تنظیم مثبت، تنظیم منفی
            elif any(word in query_lower for word in ['brain', 'neural', 'cerebral']):
                return ['AeG', 'AuG', 'AdG']
            elif any(word in query_lower for word in ['liver', 'hepatic']):
                return ['AeG', 'AuG', 'AdG']
            else:
                return ['AeG', 'AuG', 'AdG']
        
        # سوالات مربوط به ژن‌ها و بیماری‌ها
        elif any(word in query_lower for word in ['disease', 'cancer', 'diabetes', 'alzheimer']):
            # بررسی سوالات مربوط به اثر بیماری بر بافت‌ها
            if any(word in query_lower for word in ['tissue', 'tissues', 'affect', 'effect', 'localize']):
                return ['DlA', 'DuG', 'DdG', 'AeG', 'AuG', 'AdG', 'GpBP']  # بیماری→بافت، تنظیم ژن، بیان ژن، فرآیند زیستی
            else:
                return ['DaG', 'DuG', 'DdG']  # مرتبط، تنظیم مثبت، تنظیم منفی
        
        # سوالات مربوط به داروها و درمان
        elif any(word in query_lower for word in ['drug', 'treat', 'compound', 'medication']):
            return ['CtD', 'CuG', 'CdG', 'CbG']  # درمان، تنظیم مثبت، تنظیم منفی، اتصال
        
        # سوالات مربوط به فرآیندهای زیستی
        elif any(word in query_lower for word in ['process', 'function', 'biological']):
            return ['GpBP', 'GpMF', 'GpCC']  # فرآیند، عملکرد، اجزای سلولی
        
        # سوالات مربوط به مسیرهای زیستی
        elif any(word in query_lower for word in ['pathway', 'signaling', 'metabolism']):
            return ['GpPW']  # مسیرهای زیستی
        
        # سوالات مربوط به تعامل ژن‌ها
        elif any(word in query_lower for word in ['interact', 'regulate', 'covary']):
            return ['GiG', 'Gr>G', 'GcG']  # تعامل، تنظیم، همبستگی
        
        # سوالات مربوط به علائم و عوارض
        elif any(word in query_lower for word in ['symptom', 'side effect', 'adverse']):
            return ['DpS', 'CcSE']  # علائم بیماری، عوارض جانبی
        
        # سوالات پیچیده و چندمرحله‌ای
        else:
            return ['AeG', 'DaG', 'GpBP', 'GpPW', 'GiG']  # ترکیبی از روابط مهم
    
    def _enrich_primary_genes(self, retrieval_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        غنی‌سازی ژن‌های اصلی با اطلاعات اضافی
        """
        enriched_genes = []
        
        for gene_info in retrieval_data['primary_genes']:
            gene_name = gene_info['name']
            gene_id = None
            
            # یافتن ID ژن
            for node_id, attrs in self.G.nodes(data=True):
                if attrs.get('name') == gene_name and attrs.get('kind') == 'Gene':
                    gene_id = node_id
                    break
            
            if gene_id:
                enriched_gene = {
                    **gene_info,
                    'biological_processes': [],
                    'pathways': [],
                    'diseases': [],
                    'interacting_genes': [],
                    'molecular_functions': [],
                    'cellular_components': []
                }
                
                # یافتن فرآیندهای زیستی مرتبط
                for neighbor in self.G.neighbors(gene_id):
                    neighbor_attrs = self.G.nodes[neighbor]
                    edge_data = self.G.get_edge_data(gene_id, neighbor)
                    
                    if edge_data:
                        metaedge = edge_data.get('metaedge', '')
                        
                        if metaedge == 'GpBP' and neighbor_attrs.get('kind') == 'Biological Process':
                            enriched_gene['biological_processes'].append(neighbor_attrs['name'])
                        elif metaedge == 'GpPW' and neighbor_attrs.get('kind') == 'Pathway':
                            enriched_gene['pathways'].append(neighbor_attrs['name'])
                        elif metaedge == 'DaG' and neighbor_attrs.get('kind') == 'Disease':
                            enriched_gene['diseases'].append(neighbor_attrs['name'])
                        elif metaedge == 'GiG' and neighbor_attrs.get('kind') == 'Gene':
                            enriched_gene['interacting_genes'].append(neighbor_attrs['name'])
                        elif metaedge == 'GpMF' and neighbor_attrs.get('kind') == 'Molecular Function':
                            enriched_gene['molecular_functions'].append(neighbor_attrs['name'])
                        elif metaedge == 'GpCC' and neighbor_attrs.get('kind') == 'Cellular Component':
                            enriched_gene['cellular_components'].append(neighbor_attrs['name'])
                
                enriched_genes.append(enriched_gene)
        
        retrieval_data['primary_genes'] = enriched_genes
        return retrieval_data
    
    def _create_structured_text_for_model(self, retrieval_data: Dict[str, Any], query: str) -> str:
        """
        ایجاد متن ساختاریافته بهبود یافته برای ارسال به مدل زبانی
        """
        context_parts = []
        
        # 1. سوال اصلی
        context_parts.append(f"🧬 **Query:** {query}")
        context_parts.append("")
        
        # 2. خلاصه آماری دقیق
        total_genes_in_graph = 14010  # تعداد کل ژن‌ها در Hetionet
        primary_genes = len(retrieval_data['primary_genes'])
        secondary_genes = len(retrieval_data['secondary_genes'])
        total_found = primary_genes + secondary_genes
        
        context_parts.append("📊 **Graph Summary:**")
        context_parts.append(f"• Total genes in Hetionet: {total_genes_in_graph:,}")
        context_parts.append(f"• Genes found for this query: {total_found}")
        context_parts.append(f"• Primary genes (direct relationships): {primary_genes}")
        context_parts.append(f"• Secondary genes (indirect relationships): {secondary_genes}")
        
        # نمایش روابط استفاده شده
        if retrieval_data['metaedges_used']:
            metaedge_descriptions = {
                'AeG': 'Anatomy–expresses–Gene',
                'AuG': 'Anatomy–upregulates–Gene',
                'AdG': 'Anatomy–downregulates–Gene',
                'DaG': 'Disease–associates–Gene',
                'DuG': 'Disease–upregulates–Gene',
                'DdG': 'Disease–downregulates–Gene',
                'CtD': 'Compound–treats–Disease',
                'CuG': 'Compound–upregulates–Gene',
                'CdG': 'Compound–downregulates–Gene',
                'CbG': 'Compound–binds–Gene',
                'GpBP': 'Gene–participates–Biological Process',
                'GpPW': 'Gene–participates–Pathway',
                'GpMF': 'Gene–participates–Molecular Function',
                'GpCC': 'Gene–participates–Cellular Component',
                'GiG': 'Gene–interacts–Gene',
                'Gr>G': 'Gene–regulates–Gene',
                'GcG': 'Gene–covaries–Gene',
                'DpS': 'Disease–presents–Symptom',
                'DlA': 'Disease–localizes–Anatomy',
                'CcSE': 'Compound–causes–Side Effect'
            }
            
            relationships_used = []
            for metaedge in retrieval_data['metaedges_used']:
                desc = metaedge_descriptions.get(metaedge, metaedge)
                relationships_used.append(f"{metaedge}: {desc}")
            
            context_parts.append(f"• Relationships used: {len(retrieval_data['metaedges_used'])} ({', '.join(relationships_used)})")
        
        context_parts.append("")
        
        # 3. ژن‌های کلیدی با اطلاعات غنی (حداکثر 3 ژن)
        if retrieval_data['primary_genes']:
            context_parts.append("🔍 **Key Results:**")
            context_parts.append("The following genes were identified:")
            context_parts.append("")
            
            for gene in retrieval_data['primary_genes'][:3]:  # حداکثر 3 ژن
                relation_desc = {
                    'AeG': 'expressed in',
                    'AuG': 'upregulated in',
                    'AdG': 'downregulated in',
                    'DaG': 'associated with disease',
                    'DuG': 'upregulated in disease',
                    'DdG': 'downregulated in disease'
                }.get(gene['metaedge'], gene['metaedge'])
                
                # اطلاعات اصلی ژن (خلاصه‌تر)
                gene_info = f"• **{gene['name']}** – {relation_desc}"
                
                # اضافه کردن مهم‌ترین اطلاعات زیستی
                if gene.get('biological_processes'):
                    gene_info += f" ({gene['biological_processes'][0]})"
                elif gene.get('diseases'):
                    gene_info += f" ({gene['diseases'][0]})"
                elif gene.get('pathways'):
                    gene_info += f" ({gene['pathways'][0]})"
                
                context_parts.append(gene_info)
            context_parts.append("")
        
        # 4. فرآیندهای زیستی مرتبط
        if retrieval_data['biological_processes']:
            context_parts.append("⚙️ **Related Biological Processes:**")
            for process in retrieval_data['biological_processes'][:3]:
                context_parts.append(f"• {process['name']}")
            context_parts.append("")
        
        # 5. مسیرهای زیستی مرتبط
        if retrieval_data['pathways']:
            context_parts.append("🛤️ **Related Pathways:**")
            for pathway in retrieval_data['pathways'][:3]:
                context_parts.append(f"• {pathway['name']}")
            context_parts.append("")
        
        # 6. بیماری‌های مرتبط
        if retrieval_data['diseases']:
            context_parts.append("🏥 **Related Diseases:**")
            for disease in retrieval_data['diseases'][:3]:
                context_parts.append(f"• {disease['name']}")
            context_parts.append("")
        
        # 7. داروهای مرتبط
        if retrieval_data['drugs']:
            context_parts.append("💊 **Related Drugs/Compounds:**")
            for drug in retrieval_data['drugs'][:3]:
                context_parts.append(f"• {drug['name']}")
            context_parts.append("")
        
        # 8. مسیرهای ترکیبی بیماری→بافت→ژن (برای سوالات مربوط به اثر بیماری بر بافت‌ها)
        if retrieval_data.get('tissue_disease_paths'):
            context_parts.append("🔄 **Disease-Tissue-Gene Pathways:**")
            context_parts.append("The following pathways show how diseases affect specific tissues and their genes:")
            context_parts.append("")
            
            for path in retrieval_data['tissue_disease_paths'][:3]:  # حداکثر 3 مسیر
                context_parts.append(f"• **{path['disease']}** → affects → **{path['tissue']}**")
                for gene_info in path['genes']:
                    gene_desc = f"  - **{gene_info['gene_name']}**"
                    if gene_info['biological_processes']:
                        gene_desc += f" ({gene_info['biological_processes'][0]})"
                    context_parts.append(gene_desc)
                context_parts.append("")
        
        # 9. مسیرهای درمانی دارو→بیماری→ژن (برای سوالات مربوط به درمان)
        if retrieval_data.get('treatment_paths'):
            context_parts.append("💊 **Treatment-Disease-Gene Pathways:**")
            context_parts.append("The following pathways show how drugs treat diseases by regulating genes:")
            context_parts.append("")
            
            for path in retrieval_data['treatment_paths'][:3]:  # حداکثر 3 مسیر
                context_parts.append(f"• **{path['drug']}** → treats → **{path['disease']}**")
                for gene_info in path['genes']:
                    regulation = "upregulates" if gene_info['regulation'] == 'CuG' else "downregulates"
                    gene_desc = f"  - **{gene_info['gene_name']}** ({regulation})"
                    if gene_info['biological_processes']:
                        gene_desc += f" ({gene_info['biological_processes'][0]})"
                    context_parts.append(gene_desc)
                context_parts.append("")
        
        # 10. دستورالعمل کوتاه و کاربردی
        context_parts.append("📌 **Instructions:** Analyze biological relevance and clinical importance of these genes.")
        
        return "\n".join(context_parts)
    
    def test_targeted_retrieval(self, query: str) -> Dict[str, Any]:
        """
        تست بازیابی هدفمند و نمایش نتایج
        """
        print(f"🧪 تست بازیابی هدفمند برای سوال: {query}")
        print("=" * 60)
        
        # تحلیل سوال
        intent = self.analyze_question_intent(query)
        print(f"📋 نوع سوال تشخیص داده شده: {intent.get('question_type', 'unknown')}")
        
        # بازیابی هدفمند
        retrieval_data = self._targeted_retrieval_for_question(query, intent)
        
        # نمایش نتایج
        print(f"\n📊 نتایج بازیابی:")
        print(f"• ژن‌های اصلی: {len(retrieval_data['primary_genes'])}")
        print(f"• ژن‌های ثانویه: {len(retrieval_data['secondary_genes'])}")
        print(f"• فرآیندهای زیستی: {len(retrieval_data['biological_processes'])}")
        print(f"• مسیرهای زیستی: {len(retrieval_data['pathways'])}")
        print(f"• بیماری‌ها: {len(retrieval_data['diseases'])}")
        print(f"• داروها: {len(retrieval_data['drugs'])}")
        print(f"• بافت‌ها: {len(retrieval_data['anatomy'])}")
        
        # نمایش ژن‌های اصلی با جزئیات
        if retrieval_data['primary_genes']:
            print(f"\n🧬 ژن‌های اصلی یافت شده:")
            for i, gene in enumerate(retrieval_data['primary_genes'][:5], 1):
                print(f"{i}. {gene['name']} ({gene['metaedge']}) - امتیاز: {gene['score']:.2f}")
                
                if gene.get('biological_processes'):
                    print(f"   فرآیندهای زیستی: {', '.join(gene['biological_processes'][:2])}")
                if gene.get('pathways'):
                    print(f"   مسیرهای زیستی: {', '.join(gene['pathways'][:2])}")
                if gene.get('diseases'):
                    print(f"   بیماری‌های مرتبط: {', '.join(gene['diseases'][:2])}")
                if gene.get('interacting_genes'):
                    print(f"   ژن‌های تعاملی: {', '.join(gene['interacting_genes'][:3])}")
                print()
        
        # نمایش متن ساختاریافته
        structured_text = self._create_structured_text_for_model(retrieval_data, query)
        print(f"📝 متن ساختاریافته برای مدل:")
        print("-" * 40)
        print(structured_text)
        print("-" * 40)
        
        return {
            'query': query,
            'intent': intent,
            'retrieval_data': retrieval_data,
            'structured_text': structured_text
        }
    
    def test_compact_retrieval(self, query: str) -> Dict[str, Any]:
        """
        تست بازیابی خلاصه و نمایش متن کوتاه
        """
        print(f"🧪 تست بازیابی خلاصه برای سوال: {query}")
        print("=" * 50)
        
        # تحلیل سوال
        intent = self.analyze_question_intent(query)
        print(f"📋 نوع سوال: {intent.get('question_type', 'unknown')}")
        
        # بازیابی هدفمند
        retrieval_data = self._targeted_retrieval_for_question(query, intent)
        
        # نمایش خلاصه نتایج
        print(f"\n📊 خلاصه نتایج:")
        print(f"• ژن‌های اصلی: {len(retrieval_data['primary_genes'])}")
        print(f"• فرآیندهای زیستی: {len(retrieval_data['biological_processes'])}")
        print(f"• مسیرهای زیستی: {len(retrieval_data['pathways'])}")
        print(f"• بیماری‌ها: {len(retrieval_data['diseases'])}")
        print(f"• داروها: {len(retrieval_data['drugs'])}")
        
        # نمایش متن ساختاریافته بهبود یافته
        structured_text = self._create_structured_text_for_model(retrieval_data, query)
        print(f"\n📝 متن بهبود یافته برای مدل:")
        print("-" * 50)
        print(structured_text)
        print("-" * 50)
        
        # محاسبه طول متن
        text_length = len(structured_text)
        print(f"\n📏 طول متن: {text_length} کاراکتر")
        
        # تحلیل کیفیت متن
        if text_length > 1500:
            print("⚠️ متن خیلی طولانی است!")
        elif text_length > 800:
            print("⚠️ متن متوسط است")
        elif text_length > 400:
            print("✅ متن مناسب است")
        else:
            print("✅ متن کوتاه و عالی است")
        
        # بررسی کیفیت محتوا
        if retrieval_data['primary_genes']:
            genes_with_info = sum(1 for gene in retrieval_data['primary_genes'] 
                                if gene.get('biological_processes') or gene.get('pathways') or gene.get('diseases'))
            print(f"📊 کیفیت محتوا: {genes_with_info}/{len(retrieval_data['primary_genes'])} ژن با اطلاعات زیستی")
        
        if retrieval_data['metaedges_used']:
            print(f"🔗 روابط استفاده شده: {len(retrieval_data['metaedges_used'])} نوع")
        
        return {
            'query': query,
            'intent': intent,
            'retrieval_data': retrieval_data,
            'structured_text': structured_text,
            'text_length': text_length
        }
    
    def generate_answer(self, retrieval_result: RetrievalResult, 
                       model: GenerationModel) -> GenerationResult:
        """تولید پاسخ بر اساس نتایج بازیابی"""
        print(f"🤖 تولید پاسخ با مدل {model.value}...")
        
        if model == GenerationModel.SIMPLE:
            answer = self.simple_template_generation(retrieval_result)
            confidence = 0.7
        elif model == GenerationModel.GPT_SIMULATION:
            answer = self.gpt_simulation_generation(retrieval_result)
            confidence = 0.85
        elif model == GenerationModel.CUSTOM:
            answer = self.custom_generation(retrieval_result)
            confidence = 0.9
        elif model == GenerationModel.HUGGINGFACE:
            answer = self.huggingface_generation(retrieval_result)
            confidence = 0.92
        elif model == GenerationModel.OPENAI_GPT:
            answer = self.openai_gpt_generation(retrieval_result)
            confidence = 0.95
        elif model == GenerationModel.ANTHROPIC_CLAUDE:
            answer = self.anthropic_claude_generation(retrieval_result)
            confidence = 0.94
        elif model == GenerationModel.GOOGLE_GEMINI:
            answer = self.google_gemini_generation(retrieval_result)
            confidence = 0.93
        else:
            answer = "متأسفانه مدل انتخاب شده در دسترس نیست."
            confidence = 0.0
        
        return GenerationResult(
            answer=answer,
            model=model.value,
            context_used=retrieval_result.context_text,
            confidence=confidence
        )
    
    def simple_template_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ ساده با قالب بهبود یافته"""
        query_lower = retrieval_result.query.lower()
        
        # تحلیل نوع سوال
        question_type = self._analyze_question_type(query_lower)
        
        if question_type == "relationship":
            return self._generate_relationship_answer(retrieval_result)
        elif question_type == "drug_treatment":
            return self._generate_drug_treatment_answer(retrieval_result)
        elif question_type == "gene_function":
            return self._generate_gene_function_answer(retrieval_result)
        elif question_type == "disease_info":
            return self._generate_disease_info_answer(retrieval_result)
        elif question_type == "anatomy_expression":
            return self._generate_anatomy_expression_answer(retrieval_result)
        else:
            return self._generate_general_answer(retrieval_result)
    
    def _analyze_question_type(self, query_lower: str) -> str:
        """تحلیل نوع سوال"""
        if any(word in query_lower for word in ["relationship", "relation", "connect", "link"]):
            return "relationship"
        elif any(word in query_lower for word in ["drug", "treat", "medicine", "therapy"]):
            return "drug_treatment"
        elif any(word in query_lower for word in ["gene", "function", "regulate", "express"]):
            return "gene_function"
        elif any(word in query_lower for word in ["disease", "disorder", "condition", "symptom"]):
            return "disease_info"
        elif any(word in query_lower for word in ["anatomy", "organ", "tissue", "heart", "brain", "liver"]):
            return "anatomy_expression"
        else:
            return "general"
    
    def _generate_relationship_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات رابطه"""
        if not retrieval_result.edges:
            return "❌ No direct relationships found between the specified entities in the knowledge graph."
        
        answer_parts = ["🔗 RELATIONSHIPS FOUND:"]
        
        # دسته‌بندی روابط
        relations_by_type = {}
        for edge in retrieval_result.edges:
            if edge.relation not in relations_by_type:
                relations_by_type[edge.relation] = []
            relations_by_type[edge.relation].append(edge)
        
        for relation, edges in relations_by_type.items():
            answer_parts.append(f"\n📌 {relation.upper()} ({len(edges)} connections):")
            for edge in edges[:5]:  # حداکثر 5 رابطه از هر نوع
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"  • {source_name} → {target_name}")
        
        return "\n".join(answer_parts)
    
    def _generate_drug_treatment_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات درمان دارویی"""
        drug_nodes = [n for n in retrieval_result.nodes if n.kind in ['Drug', 'Compound']]
        disease_nodes = [n for n in retrieval_result.nodes if n.kind == 'Disease']
        
        if not drug_nodes and not disease_nodes:
            return "❌ No drug or disease information found in the retrieved context."
        
        answer_parts = ["💊 DRUG-DISEASE RELATIONSHIPS:"]
        
        # روابط درمان
        treatment_edges = [e for e in retrieval_result.edges if 'treat' in e.relation.lower() or 'therapy' in e.relation.lower()]
        if treatment_edges:
            answer_parts.append(f"\n✅ TREATMENT RELATIONSHIPS ({len(treatment_edges)} found):")
            for edge in treatment_edges[:10]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"  • {source_name} treats {target_name}")
        
        # داروهای یافت شده
        if drug_nodes:
            answer_parts.append(f"\n💊 DRUGS FOUND ({len(drug_nodes)}):")
            for drug in drug_nodes[:10]:
                answer_parts.append(f"  • {drug.name}")
        
        # بیماری‌های یافت شده
        if disease_nodes:
            answer_parts.append(f"\n🏥 DISEASES FOUND ({len(disease_nodes)}):")
            for disease in disease_nodes[:10]:
                answer_parts.append(f"  • {disease.name}")
        
        return "\n".join(answer_parts)
    
    def _generate_gene_function_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات عملکرد ژن"""
        gene_nodes = [n for n in retrieval_result.nodes if n.kind == 'Gene']
        process_nodes = [n for n in retrieval_result.nodes if n.kind == 'Biological Process']
        
        if not gene_nodes:
            return "❌ No gene information found in the retrieved context."
        
        answer_parts = ["🧬 GENE FUNCTION ANALYSIS:"]
        
        # ژن‌های یافت شده
        answer_parts.append(f"\n🧬 GENES FOUND ({len(gene_nodes)}):")
        for gene in gene_nodes[:10]:
            answer_parts.append(f"  • {gene.name}")
        
        # فرآیندهای زیستی مرتبط
        if process_nodes:
            answer_parts.append(f"\n⚙️ BIOLOGICAL PROCESSES ({len(process_nodes)}):")
            for process in process_nodes[:10]:
                answer_parts.append(f"  • {process.name}")
        
        # روابط ژن-فرآیند
        gene_process_edges = [e for e in retrieval_result.edges if 'participate' in e.relation.lower() or 'regulate' in e.relation.lower()]
        if gene_process_edges:
            answer_parts.append(f"\n🔗 GENE-PROCESS RELATIONSHIPS ({len(gene_process_edges)}):")
            for edge in gene_process_edges[:10]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"  • {source_name} → {target_name}")
        
        return "\n".join(answer_parts)
    
    def _generate_disease_info_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات اطلاعات بیماری"""
        disease_nodes = [n for n in retrieval_result.nodes if n.kind == 'Disease']
        gene_nodes = [n for n in retrieval_result.nodes if n.kind == 'Gene']
        symptom_nodes = [n for n in retrieval_result.nodes if n.kind == 'Symptom']
        
        if not disease_nodes:
            return "❌ No disease information found in the retrieved context."
        
        answer_parts = ["🏥 DISEASE ANALYSIS:"]
        
        # بیماری‌های یافت شده
        answer_parts.append(f"\n🏥 DISEASES FOUND ({len(disease_nodes)}):")
        for disease in disease_nodes[:10]:
            answer_parts.append(f"  • {disease.name}")
        
        # ژن‌های مرتبط
        if gene_nodes:
            answer_parts.append(f"\n🧬 ASSOCIATED GENES ({len(gene_nodes)}):")
            for gene in gene_nodes[:10]:
                answer_parts.append(f"  • {gene.name}")
        
        # علائم مرتبط
        if symptom_nodes:
            answer_parts.append(f"\n🤒 ASSOCIATED SYMPTOMS ({len(symptom_nodes)}):")
            for symptom in symptom_nodes[:10]:
                answer_parts.append(f"  • {symptom.name}")
        
        return "\n".join(answer_parts)
    
    def _generate_anatomy_expression_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات بیان ژن در آناتومی"""
        anatomy_nodes = [n for n in retrieval_result.nodes if n.kind == 'Anatomy']
        gene_nodes = [n for n in retrieval_result.nodes if n.kind == 'Gene']
        
        if not anatomy_nodes:
            return "❌ No anatomy information found in the retrieved context."
        
        answer_parts = ["🫀 ANATOMY-GENE EXPRESSION:"]
        
        # اندام‌های یافت شده
        answer_parts.append(f"\n🫀 ANATOMICAL STRUCTURES ({len(anatomy_nodes)}):")
        for anatomy in anatomy_nodes[:10]:
            answer_parts.append(f"  • {anatomy.name}")
        
        # ژن‌های بیان شده
        if gene_nodes:
            answer_parts.append(f"\n🧬 EXPRESSED GENES ({len(gene_nodes)}):")
            for gene in gene_nodes[:10]:
                answer_parts.append(f"  • {gene.name}")
        
        # روابط بیان
        expression_edges = [e for e in retrieval_result.edges if 'express' in e.relation.lower()]
        if expression_edges:
            answer_parts.append(f"\n🔗 EXPRESSION RELATIONSHIPS ({len(expression_edges)}):")
            for edge in expression_edges[:10]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"  • {source_name} expresses {target_name}")
        
        return "\n".join(answer_parts)
    
    def _generate_general_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ عمومی"""
        if not retrieval_result.nodes:
            return "❌ No relevant information found in the knowledge graph for your query."
        
        answer_parts = ["📊 GENERAL INFORMATION FOUND:"]
        
        # دسته‌بندی نودها
        nodes_by_type = {}
        for node in retrieval_result.nodes:
            if node.kind not in nodes_by_type:
                nodes_by_type[node.kind] = []
            nodes_by_type[node.kind].append(node)
        
        for kind, nodes in nodes_by_type.items():
            answer_parts.append(f"\n🔹 {kind.upper()} ({len(nodes)} entities):")
            for node in nodes[:5]:
                answer_parts.append(f"  • {node.name}")
        
        # روابط مهم
        if retrieval_result.edges:
            answer_parts.append(f"\n🔗 KEY RELATIONSHIPS ({len(retrieval_result.edges)}):")
            for edge in retrieval_result.edges[:10]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"  • {source_name} → {target_name} ({edge.relation})")
        
        return "\n".join(answer_parts)
    
    def gpt_simulation_generation(self, retrieval_result: RetrievalResult) -> str:
        """شبیه‌سازی تولید پاسخ GPT بهبود یافته"""
        query = retrieval_result.query
        query_lower = query.lower()
        
        # تحلیل نوع سوال
        question_type = self._analyze_question_type(query_lower)
        
        # استخراج اطلاعات مهم از نتایج بازیابی
        gene_nodes = [n for n in retrieval_result.nodes if n.kind == 'Gene']
        disease_nodes = [n for n in retrieval_result.nodes if n.kind == 'Disease']
        drug_nodes = [n for n in retrieval_result.nodes if n.kind in ['Drug', 'Compound']]
        anatomy_nodes = [n for n in retrieval_result.nodes if n.kind == 'Anatomy']
        process_nodes = [n for n in retrieval_result.nodes if n.kind == 'Biological Process']
        
        # تشخیص سوالات ژن-سرطان
        if self._is_gene_cancer_question_from_context(retrieval_result):
            return self._generate_gene_cancer_answer(retrieval_result, gene_nodes, disease_nodes)
        
        # تولید پاسخ بر اساس نوع سوال
        if question_type == "relationship":
            return self._generate_intelligent_relationship_answer(retrieval_result, gene_nodes, disease_nodes, drug_nodes)
        elif question_type == "drug_treatment":
            return self._generate_intelligent_drug_answer(retrieval_result, drug_nodes, disease_nodes)
        elif question_type == "gene_function":
            return self._generate_intelligent_gene_answer(retrieval_result, gene_nodes, process_nodes)
        elif question_type == "disease_info":
            return self._generate_intelligent_disease_answer(retrieval_result, disease_nodes, gene_nodes)
        elif question_type == "anatomy_expression":
            return self._generate_intelligent_anatomy_answer(retrieval_result, anatomy_nodes, gene_nodes)
        else:
            return self._generate_intelligent_general_answer(retrieval_result, gene_nodes, disease_nodes, drug_nodes, anatomy_nodes, process_nodes)
    
    def _generate_intelligent_relationship_answer(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes, drug_nodes) -> str:
        """تولید پاسخ هوشمند برای سوالات رابطه"""
        if not retrieval_result.edges:
            return "🔍 **تحلیل رابطه:**\n\nمتأسفانه رابطه مستقیمی بین موجودیت‌های مورد نظر در گراف دانش یافت نشد. این ممکن است به دلیل:\n• فاصله زیاد بین نودها در گراف\n• نیاز به جستجوی عمیق‌تر\n• عدم وجود رابطه مستقیم در داده‌های موجود"
        
        answer_parts = ["🔍 **تحلیل رابطه:**\n"]
        
        # تحلیل روابط مهم
        important_relations = {}
        for edge in retrieval_result.edges:
            if edge.relation not in important_relations:
                important_relations[edge.relation] = []
            important_relations[edge.relation].append(edge)
        
        # نمایش مهم‌ترین روابط
        answer_parts.append("**روابط مهم یافت شده:**\n")
        for relation, edges in sorted(important_relations.items(), key=lambda x: len(x[1]), reverse=True)[:3]:
            answer_parts.append(f"• **{relation}** ({len(edges)} رابطه):")
            for edge in edges[:3]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"  - {source_name} → {target_name}")
            answer_parts.append("")
        
        # تحلیل آماری
        answer_parts.append("**آمار کلی:**")
        answer_parts.append(f"• تعداد کل روابط: {len(retrieval_result.edges)}")
        answer_parts.append(f"• انواع روابط: {len(important_relations)}")
        answer_parts.append(f"• نودهای مرتبط: {len(retrieval_result.nodes)}")
        
        return "\n".join(answer_parts)
    
    def _generate_intelligent_drug_answer(self, retrieval_result: RetrievalResult, drug_nodes, disease_nodes) -> str:
        """تولید پاسخ هوشمند برای سوالات دارویی"""
        answer_parts = ["💊 **تحلیل دارویی:**\n"]
        
        if drug_nodes:
            answer_parts.append("**داروهای یافت شده:**")
            for drug in drug_nodes[:5]:
                score_info = f" (امتیاز: {drug.score:.2f})" if hasattr(drug, 'score') and drug.score != 1.0 else ""
                answer_parts.append(f"• {drug.name}{score_info}")
            answer_parts.append("")
        
        if disease_nodes:
            answer_parts.append("**بیماری‌های مرتبط:**")
            for disease in disease_nodes[:5]:
                answer_parts.append(f"• {disease.name}")
            answer_parts.append("")
        
        # روابط درمان
        treatment_edges = [e for e in retrieval_result.edges if 'treat' in e.relation.lower() or 'therapy' in e.relation.lower()]
        if treatment_edges:
            answer_parts.append("**روابط درمانی:**")
            for edge in treatment_edges[:5]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"• {source_name} درمان می‌کند: {target_name}")
        
        if not drug_nodes and not disease_nodes:
            answer_parts.append("❌ اطلاعات دارویی یا بیماری در نتایج یافت نشد.")
        
        return "\n".join(answer_parts)
    
    def _generate_intelligent_gene_answer(self, retrieval_result: RetrievalResult, gene_nodes, process_nodes) -> str:
        """تولید پاسخ هوشمند برای سوالات ژن"""
        answer_parts = ["🧬 **تحلیل ژنتیکی:**\n"]
        
        if gene_nodes:
            answer_parts.append("**ژن‌های مهم یافت شده:**")
            # مرتب‌سازی بر اساس امتیاز
            sorted_genes = sorted(gene_nodes, key=lambda x: getattr(x, 'score', 1.0), reverse=True)
            
            for gene in sorted_genes[:5]:
                score_info = f" (امتیاز: {gene.score:.2f})" if hasattr(gene, 'score') and gene.score != 1.0 else ""
                answer_parts.append(f"• **{gene.name}**{score_info}")
            answer_parts.append("")
        
        if process_nodes:
            answer_parts.append("**فرآیندهای زیستی مرتبط:**")
            for process in process_nodes[:5]:
                answer_parts.append(f"• {process.name}")
            answer_parts.append("")
        
        # روابط ژن-فرآیند
        gene_process_edges = [e for e in retrieval_result.edges 
                            if any(n.id == e.source for n in gene_nodes) and 
                               any(n.id == e.target for n in process_nodes)]
        
        if gene_process_edges:
            answer_parts.append("**روابط ژن-فرآیند:**")
            for edge in gene_process_edges[:5]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"• {source_name} → {target_name} ({edge.relation})")
            answer_parts.append("")
        
        # تحلیل آماری
        total_genes = len(gene_nodes)
        total_processes = len(process_nodes)
        total_relationships = len(retrieval_result.edges)
        
        answer_parts.append("**آمار کلی:**")
        answer_parts.append(f"• ژن‌های یافت شده: {total_genes}")
        answer_parts.append(f"• فرآیندهای زیستی: {total_processes}")
        answer_parts.append(f"• روابط کل: {total_relationships}")
        
        if not gene_nodes:
            answer_parts.append("\n❌ ژن مرتبطی در نتایج یافت نشد.")
        
        return "\n".join(answer_parts)
    
    def _generate_intelligent_disease_answer(self, retrieval_result: RetrievalResult, disease_nodes, gene_nodes) -> str:
        """تولید پاسخ هوشمند برای سوالات بیماری"""
        answer_parts = ["🏥 **تحلیل بیماری:**\n"]
        
        if disease_nodes:
            answer_parts.append("**بیماری‌های یافت شده:**")
            for disease in disease_nodes[:5]:
                answer_parts.append(f"• {disease.name}")
            answer_parts.append("")
        
        if gene_nodes:
            answer_parts.append("**ژن‌های مرتبط:**")
            for gene in gene_nodes[:5]:
                answer_parts.append(f"• {gene.name}")
            answer_parts.append("")
        
        # روابط بیماری-ژن
        disease_gene_edges = [e for e in retrieval_result.edges if 'cause' in e.relation.lower() or 'associate' in e.relation.lower()]
        if disease_gene_edges:
            answer_parts.append("**روابط بیماری-ژن:**")
            for edge in disease_gene_edges[:5]:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"• {source_name} → {target_name}")
        
        if not disease_nodes:
            answer_parts.append("❌ اطلاعات بیماری در نتایج یافت نشد.")
        
        return "\n".join(answer_parts)
    
    def _generate_intelligent_anatomy_answer(self, retrieval_result: RetrievalResult, anatomy_nodes, gene_nodes) -> str:
        """تولید پاسخ مبتنی بر شواهد برای سوالات آناتومی"""
        answer_parts = []
        
        # استخراج نام آناتومی
        anatomy_name = "unknown anatomy"
        if anatomy_nodes:
            anatomy_name = anatomy_nodes[0].name
        
        # استخراج ژن‌های بیان شده
        gene_names = [gene.name for gene in gene_nodes if gene.kind == 'Gene']
        
        if not gene_names:
            return "متأسفانه هیچ ژن بیان شده‌ای برای این آناتومی یافت نشد."
        
        # 📌 بخش 1: پرسش اصلی
        answer_parts.append(f"**📌 پرسش:** {retrieval_result.query}")
        answer_parts.append("")
        
        # ✅ بخش 2: پاسخ کلیدی
        answer_parts.append(f"**✅ پاسخ کلیدی:**")
        answer_parts.append(f"بر اساس تحلیل داده‌های زیستی، {len(gene_names)} ژن در بافت {anatomy_name} بیان می‌شوند:")
        
        for i, gene_name in enumerate(gene_names[:10], 1):  # حداکثر 10 ژن
            answer_parts.append(f"• {gene_name}")
        
        if len(gene_names) > 10:
            answer_parts.append(f"• و {len(gene_names) - 10} ژن دیگر")
        
            answer_parts.append("")
        
        # 🔎 بخش 3: مسیرهای استناد
        answer_parts.append("**🔎 مسیرهای استناد:**")
        
        # یافتن یال‌های AeG
        aeG_edges = [e for e in retrieval_result.edges if e.relation == 'AeG']
        if aeG_edges:
            answer_parts.append("روابط بیان مستقیم (Anatomy → expresses → Gene):")
            for edge in aeG_edges[:5]:  # حداکثر 5 رابطه
                source_name = next((n.name for n in retrieval_result.nodes if n.id == edge.source), edge.source)
                target_name = next((n.name for n in retrieval_result.nodes if n.id == edge.target), edge.target)
                answer_parts.append(f"• {source_name} → {target_name} (AeG)")
        else:
            answer_parts.append("• روابط بیان از طریق جستجوی هوشمند شناسایی شد")
        
        # یافتن یال‌های AuG و AdG
        auG_edges = [e for e in retrieval_result.edges if e.relation == 'AuG']
        adG_edges = [e for e in retrieval_result.edges if e.relation == 'AdG']
        
        if auG_edges:
            answer_parts.append("روابط تنظیم مثبت (Anatomy → upregulates → Gene):")
            for edge in auG_edges[:3]:
                source_name = next((n.name for n in retrieval_result.nodes if n.id == edge.source), edge.source)
                target_name = next((n.name for n in retrieval_result.nodes if n.id == edge.target), edge.target)
                answer_parts.append(f"• {source_name} → {target_name} (AuG)")
        
        if adG_edges:
            answer_parts.append("روابط تنظیم منفی (Anatomy → downregulates → Gene):")
            for edge in adG_edges[:3]:
                source_name = next((n.name for n in retrieval_result.nodes if n.id == edge.source), edge.source)
                target_name = next((n.name for n in retrieval_result.nodes if n.id == edge.target), edge.target)
                answer_parts.append(f"• {source_name} → {target_name} (AdG)")
        
        answer_parts.append("")
        
        # 📚 بخش 4: منبع داده
        answer_parts.append("**📚 منبع داده:**")
        answer_parts.append("• **داده‌های بیان ژن:** پایگاه‌های Bgee و TISSUES")
        answer_parts.append("• **روابط زیستی:** Hetionet (شبکه دانش زیستی)")
        answer_parts.append("• **اطلاعات آناتومیکی:** Uberon (آناتومی یکپارچه)")
        answer_parts.append("• **روش جستجو:** Intelligent Semantic Search با فیلتر metaedge")
        answer_parts.append("")
        
        # 💬 بخش 5: تحلیل زیستی
        answer_parts.append("**💬 تحلیل زیستی:**")
        answer_parts.append(f"• **عملکرد طبیعی:** ژن‌های بیان شده در {anatomy_name} در عملکرد فیزیولوژیک این اندام نقش دارند.")
        answer_parts.append(f"• **اهمیت بالینی:** تغییرات در بیان این ژن‌ها ممکن است با بیماری‌های {anatomy_name} مرتبط باشد.")
        answer_parts.append("• **پتانسیل درمانی:** این ژن‌ها می‌توانند اهداف درمانی جدیدی برای بیماری‌های مرتبط باشند.")
        answer_parts.append("• **نشانگر زیستی:** برخی از این ژن‌ها می‌توانند به عنوان نشانگرهای زیستی برای تشخیص بیماری‌ها استفاده شوند.")
        answer_parts.append("")
        
        # 🔬 بخش 6: پیشنهادات پژوهشی
        answer_parts.append("**🔬 پیشنهادات پژوهشی:**")
        answer_parts.append("• مطالعه مکانیسم‌های تنظیم بیان این ژن‌ها در شرایط مختلف")
        answer_parts.append("• بررسی ارتباط بین تغییرات بیان و بیماری‌های مرتبط")
        answer_parts.append("• توسعه روش‌های درمانی مبتنی بر تنظیم بیان ژن")
        answer_parts.append("• پژوهش در زمینه نشانگرهای زیستی برای تشخیص زودهنگام")
        
        return "\n".join(answer_parts)
    
    def _generate_intelligent_general_answer(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes, drug_nodes, anatomy_nodes, process_nodes) -> str:
        """تولید پاسخ هوشمند عمومی"""
        query_lower = retrieval_result.query.lower()
        
        # تشخیص سوالات ژن-سرطان
        if self._is_gene_cancer_question_from_context(retrieval_result):
            return self._generate_gene_cancer_answer(retrieval_result, gene_nodes, disease_nodes)
        
        answer_parts = ["📊 **تحلیل جامع:**\n"]
        
        # خلاصه آماری
        total_entities = len(retrieval_result.nodes)
        total_relationships = len(retrieval_result.edges)
        
        answer_parts.append(f"**خلاصه آماری:**")
        answer_parts.append(f"• کل موجودیت‌ها: {total_entities}")
        answer_parts.append(f"• کل روابط: {total_relationships}")
        answer_parts.append("")
        
        # دسته‌بندی موجودیت‌ها
        if gene_nodes:
            answer_parts.append(f"**ژن‌ها ({len(gene_nodes)}):**")
            for gene in gene_nodes[:3]:
                score_info = f" (امتیاز: {gene.score:.2f})" if hasattr(gene, 'score') and gene.score != 1.0 else ""
                answer_parts.append(f"• {gene.name}{score_info}")
            answer_parts.append("")
        
        if disease_nodes:
            answer_parts.append(f"**بیماری‌ها ({len(disease_nodes)}):**")
            for disease in disease_nodes[:3]:
                answer_parts.append(f"• {disease.name}")
            answer_parts.append("")
        
        if drug_nodes:
            answer_parts.append(f"**داروها ({len(drug_nodes)}):**")
            for drug in drug_nodes[:3]:
                answer_parts.append(f"• {drug.name}")
            answer_parts.append("")
        
        if anatomy_nodes:
            answer_parts.append(f"**ساختارهای آناتومیک ({len(anatomy_nodes)}):**")
            for anatomy in anatomy_nodes[:3]:
                answer_parts.append(f"• {anatomy.name}")
            answer_parts.append("")
        
        if process_nodes:
            answer_parts.append(f"**فرآیندهای زیستی ({len(process_nodes)}):**")
            for process in process_nodes[:3]:
                answer_parts.append(f"• {process.name}")
            answer_parts.append("")
        
        # مهم‌ترین روابط
        if retrieval_result.edges:
            answer_parts.append("**مهم‌ترین روابط:**")
            # مرتب‌سازی بر اساس نوع رابطه
            relations_count = {}
            for edge in retrieval_result.edges:
                relations_count[edge.relation] = relations_count.get(edge.relation, 0) + 1
            
            for relation, count in sorted(relations_count.items(), key=lambda x: x[1], reverse=True)[:3]:
                answer_parts.append(f"• {relation}: {count} رابطه")
        
        if not retrieval_result.nodes:
            answer_parts.append("❌ اطلاعات مرتبطی در گراف دانش یافت نشد.")
        
        return "\n".join(answer_parts)
    
    def _is_gene_cancer_question_from_context(self, retrieval_result: RetrievalResult) -> bool:
        """تشخیص سوالات ژن-سرطان از محتوای بازیابی شده"""
        query_lower = retrieval_result.query.lower()
        cancer_keywords = ['cancer', 'tumor', 'malignancy', 'oncology', 'carcinoma', 'sarcoma', 'leukemia', 'lymphoma']
        
        # بررسی وجود کلمات سرطان در سوال
        has_cancer_in_query = any(keyword in query_lower for keyword in cancer_keywords)
        
        # بررسی وجود ژن‌ها و بیماری‌های سرطان در نتایج
        gene_nodes = [n for n in retrieval_result.nodes if n.kind == 'Gene']
        disease_nodes = [n for n in retrieval_result.nodes if n.kind == 'Disease']
        
        has_genes = len(gene_nodes) > 0
        has_cancer_diseases = any(
            any(keyword in disease.name.lower() for keyword in cancer_keywords)
            for disease in disease_nodes
        )
        
        return has_cancer_in_query and has_genes and has_cancer_diseases
    
    def _generate_gene_cancer_answer(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes) -> str:
        """تولید پاسخ تخصصی برای سوالات ژن-سرطان"""
        answer_parts = ["🧬 **تحلیل تخصصی ژن-سرطان:**\n"]
        
        # شناسایی ژن‌های اصلی
        primary_genes = []
        for gene in gene_nodes:
            gene_name_lower = gene.name.lower()
            # بررسی ژن‌های مشهور
            famous_genes = ['tp53', 'p53', 'brca1', 'brca2', 'apoe', 'cftr', 'mmp9', 'bid', 'kcnq2', 'hmgb3']
            if any(famous in gene_name_lower for famous in famous_genes):
                primary_genes.append(gene)
        
        if primary_genes:
            answer_parts.append("**ژن‌های اصلی یافت شده:**")
            for gene in primary_genes:
                score_info = f" (امتیاز: {gene.score:.2f})" if hasattr(gene, 'score') and gene.score != 1.0 else ""
                answer_parts.append(f"• **{gene.name}**{score_info}")
            answer_parts.append("")
        
        # شناسایی سرطان‌های مرتبط
        cancer_diseases = []
        other_diseases = []
        
        for disease in disease_nodes:
            disease_name_lower = disease.name.lower()
            cancer_keywords = ['cancer', 'tumor', 'malignancy', 'carcinoma', 'sarcoma', 'leukemia', 'lymphoma']
            if any(keyword in disease_name_lower for keyword in cancer_keywords):
                cancer_diseases.append(disease)
            else:
                other_diseases.append(disease)
        
        if cancer_diseases:
            answer_parts.append("**سرطان‌های مرتبط:**")
            for cancer in cancer_diseases:
                answer_parts.append(f"• {cancer.name}")
            answer_parts.append("")
        
        if other_diseases:
            answer_parts.append("**بیماری‌های دیگر مرتبط:**")
            for disease in other_diseases[:3]:
                answer_parts.append(f"• {disease.name}")
            answer_parts.append("")
        
        # تحلیل روابط
        if retrieval_result.edges:
            answer_parts.append("**روابط مهم یافت شده:**")
            relations_count = {}
            for edge in retrieval_result.edges:
                relations_count[edge.relation] = relations_count.get(edge.relation, 0) + 1
            
            for relation, count in sorted(relations_count.items(), key=lambda x: x[1], reverse=True)[:5]:
                answer_parts.append(f"• {relation}: {count} رابطه")
            answer_parts.append("")
        
        # تحلیل آماری
        total_entities = len(retrieval_result.nodes)
        total_relationships = len(retrieval_result.edges)
        
        answer_parts.append("**آمار کلی:**")
        answer_parts.append(f"• کل موجودیت‌ها: {total_entities}")
        answer_parts.append(f"• کل روابط: {total_relationships}")
        answer_parts.append(f"• ژن‌های اصلی: {len(primary_genes)}")
        answer_parts.append(f"• سرطان‌های مرتبط: {len(cancer_diseases)}")
        
        # پیام راهنما
        answer_parts.append("\n📌 **راهنما:** تحلیل اهمیت زیستی و بالینی این ژن‌ها را بررسی کنید.")
        
        return "\n".join(answer_parts)
    
    def custom_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ سفارشی پیشرفته"""
        query = retrieval_result.query
        query_lower = query.lower()
        
        # تحلیل عمیق‌تر سوال
        question_type = self._analyze_question_type(query_lower)
        
        # استخراج اطلاعات با جزئیات بیشتر
        gene_nodes = [n for n in retrieval_result.nodes if n.kind == 'Gene']
        disease_nodes = [n for n in retrieval_result.nodes if n.kind == 'Disease']
        drug_nodes = [n for n in retrieval_result.nodes if n.kind in ['Drug', 'Compound']]
        anatomy_nodes = [n for n in retrieval_result.nodes if n.kind == 'Anatomy']
        process_nodes = [n for n in retrieval_result.nodes if n.kind == 'Biological Process']
        symptom_nodes = [n for n in retrieval_result.nodes if n.kind == 'Symptom']
        
        # تولید پاسخ سفارشی با تحلیل عمیق
        answer_parts = ["🎯 **تحلیل سفارشی پیشرفته:**\n"]
        
        # تحلیل کیفیت نتایج
        total_score = sum(getattr(n, 'score', 1.0) for n in retrieval_result.nodes)
        avg_score = total_score / len(retrieval_result.nodes) if retrieval_result.nodes else 0
        
        answer_parts.append(f"**کیفیت نتایج:**")
        answer_parts.append(f"• میانگین امتیاز: {avg_score:.2f}")
        answer_parts.append(f"• تعداد نودهای با امتیاز بالا: {len([n for n in retrieval_result.nodes if getattr(n, 'score', 1.0) > 2.0])}")
        answer_parts.append("")
        
        # تحلیل بر اساس نوع سوال
        if question_type == "relationship":
            answer_parts.extend(self._custom_relationship_analysis(retrieval_result, gene_nodes, disease_nodes, drug_nodes))
        elif question_type == "drug_treatment":
            answer_parts.extend(self._custom_drug_analysis(retrieval_result, drug_nodes, disease_nodes))
        elif question_type == "gene_function":
            answer_parts.extend(self._custom_gene_analysis(retrieval_result, gene_nodes, process_nodes))
        elif question_type == "disease_info":
            answer_parts.extend(self._custom_disease_analysis(retrieval_result, disease_nodes, gene_nodes, symptom_nodes))
        elif question_type == "anatomy_expression":
            answer_parts.extend(self._custom_anatomy_analysis(retrieval_result, anatomy_nodes, gene_nodes))
        else:
            answer_parts.extend(self._custom_general_analysis(retrieval_result, gene_nodes, disease_nodes, drug_nodes, anatomy_nodes, process_nodes))
        
        # توصیه‌های کاربردی
        answer_parts.append("**💡 توصیه‌های کاربردی:**")
        if len(retrieval_result.nodes) < 5:
            answer_parts.append("• افزایش عمق جستجو برای یافتن نتایج بیشتر")
        if len(retrieval_result.edges) < 3:
            answer_parts.append("• استفاده از روش‌های ترکیبی برای یافتن روابط بیشتر")
        if avg_score < 2.0:
            answer_parts.append("• استفاده از روش Ensemble برای بهبود کیفیت نتایج")
        
        return "\n".join(answer_parts)
    
    def _custom_relationship_analysis(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes, drug_nodes) -> List[str]:
        """تحلیل سفارشی روابط"""
        parts = []
        parts.append("**🔍 تحلیل روابط:**")
        
        if retrieval_result.edges:
            # تحلیل پیچیدگی روابط
            edge_types = {}
            for edge in retrieval_result.edges:
                edge_types[edge.relation] = edge_types.get(edge.relation, 0) + 1
            
            parts.append(f"• انواع روابط یافت شده: {len(edge_types)}")
            parts.append(f"• پیچیدگی شبکه: {len(retrieval_result.edges)} / {len(retrieval_result.nodes)} = {len(retrieval_result.edges)/len(retrieval_result.nodes):.2f}")
            
            # مهم‌ترین روابط
            most_common = max(edge_types.items(), key=lambda x: x[1])
            parts.append(f"• رایج‌ترین رابطه: {most_common[0]} ({most_common[1]} بار)")
        else:
            parts.append("• هیچ رابطه مستقیمی یافت نشد")
        
        return parts
    
    def _custom_drug_analysis(self, retrieval_result: RetrievalResult, drug_nodes, disease_nodes) -> List[str]:
        """تحلیل سفارشی دارویی"""
        parts = []
        parts.append("**💊 تحلیل دارویی:**")
        
        if drug_nodes:
            # تحلیل داروهای مهم
            high_score_drugs = [d for d in drug_nodes if getattr(d, 'score', 1.0) > 2.0]
            parts.append(f"• داروهای با امتیاز بالا: {len(high_score_drugs)}")
            
            if high_score_drugs:
                parts.append("• مهم‌ترین داروها:")
                for drug in high_score_drugs[:3]:
                    parts.append(f"  - {drug.name} (امتیاز: {drug.score:.2f})")
        
        if disease_nodes:
            parts.append(f"• بیماری‌های مرتبط: {len(disease_nodes)}")
        
        return parts
    
    def _custom_gene_analysis(self, retrieval_result: RetrievalResult, gene_nodes, process_nodes) -> List[str]:
        """تحلیل سفارشی ژنتیکی"""
        parts = []
        parts.append("**🧬 تحلیل ژنتیکی:**")
        
        if gene_nodes:
            # تحلیل ژن‌های مهم
            sorted_genes = sorted(gene_nodes, key=lambda x: getattr(x, 'score', 1.0), reverse=True)
            parts.append(f"• ژن‌های یافت شده: {len(gene_nodes)}")
            parts.append("• مهم‌ترین ژن‌ها:")
            for gene in sorted_genes[:3]:
                parts.append(f"  - {gene.name} (امتیاز: {gene.score:.2f})")
        
        if process_nodes:
            parts.append(f"• فرآیندهای زیستی: {len(process_nodes)}")
        
        return parts
    
    def _custom_disease_analysis(self, retrieval_result: RetrievalResult, disease_nodes, gene_nodes, symptom_nodes) -> List[str]:
        """تحلیل سفارشی بیماری"""
        parts = []
        parts.append("**🏥 تحلیل بیماری:**")
        
        if disease_nodes:
            parts.append(f"• بیماری‌های یافت شده: {len(disease_nodes)}")
        
        if gene_nodes:
            parts.append(f"• ژن‌های مرتبط: {len(gene_nodes)}")
        
        if symptom_nodes:
            parts.append(f"• علائم مرتبط: {len(symptom_nodes)}")
        
        return parts
    
    def _custom_anatomy_analysis(self, retrieval_result: RetrievalResult, anatomy_nodes, gene_nodes) -> List[str]:
        """تحلیل سفارشی آناتومیک"""
        parts = []
        parts.append("**🫀 تحلیل آناتومیک:**")
        
        if anatomy_nodes:
            parts.append(f"• ساختارهای آناتومیک: {len(anatomy_nodes)}")
        
        if gene_nodes:
            parts.append(f"• ژن‌های بیان شده: {len(gene_nodes)}")
        
        return parts
    
    def _custom_general_analysis(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes, drug_nodes, anatomy_nodes, process_nodes) -> List[str]:
        """تحلیل سفارشی عمومی"""
        parts = []
        parts.append("**📊 تحلیل جامع:**")
        
        # آمار کلی
        parts.append(f"• کل موجودیت‌ها: {len(retrieval_result.nodes)}")
        parts.append(f"• کل روابط: {len(retrieval_result.edges)}")
        parts.append(f"• تراکم شبکه: {len(retrieval_result.edges)/max(len(retrieval_result.nodes), 1):.2f}")
        
        # توزیع انواع
        type_distribution = {}
        for node in retrieval_result.nodes:
            type_distribution[node.kind] = type_distribution.get(node.kind, 0) + 1
        
        parts.append("• توزیع انواع:")
        for kind, count in sorted(type_distribution.items(), key=lambda x: x[1], reverse=True):
            parts.append(f"  - {kind}: {count}")
        
        return parts
    
    def process_query(self, query: str, retrieval_method: RetrievalMethod, 
                     generation_model: GenerationModel, max_depth: int = 2) -> Dict[str, Any]:
        """پردازش کامل یک سوال"""
        print(f"🚀 پردازش سوال: {query}")
        
        # مرحله 1: بازیابی
        retrieval_result = self.retrieve_information(query, retrieval_method, max_depth)
        
        # مرحله 2: تولید پاسخ
        generation_result = self.generate_answer(retrieval_result, generation_model)
        
        # آماده‌سازی نتیجه
        result = {
            "query": query,
            "retrieval_method": retrieval_method.value,
            "generation_model": generation_model.value,
            "keywords": self.extract_keywords(query),
            "matched_nodes": {k: self.G.nodes[v]['name'] for k, v in self.match_tokens_to_nodes(self.extract_keywords(query)).items()},
            "retrieved_nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "kind": node.kind,
                    "depth": node.depth,
                    "score": node.score
                } for node in retrieval_result.nodes
            ],
            "retrieved_edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "weight": edge.weight
                } for edge in retrieval_result.edges
            ],
            "paths": retrieval_result.paths,
            "context_text": retrieval_result.context_text,
            "answer": generation_result.answer,
            "confidence": generation_result.confidence,
            "process_steps": [
                "1. استخراج کلمات کلیدی از سوال",
                "2. تطبیق کلمات کلیدی با نودهای گراف",
                f"3. بازیابی اطلاعات با روش {retrieval_method.value}",
                "4. ایجاد متن زمینه از نتایج",
                f"5. تولید پاسخ با مدل {generation_model.value}"
            ]
        }
        
        return result
    
    def huggingface_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ با مدل‌های HuggingFace (رایگان)"""
        try:
            # استفاده از مدل‌های رایگان HuggingFace
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # مدل‌های رایگان قدرتمند
            models = [
                "microsoft/DialoGPT-medium",  # چت‌بات
                "gpt2",  # GPT-2
                "distilgpt2",  # GPT-2 سبک
                "EleutherAI/gpt-neo-125M",  # GPT-Neo کوچک
                "microsoft/DialoGPT-small"  # چت‌بات کوچک
            ]
            
            # انتخاب بهترین مدل موجود
            selected_model = None
            for model_name in models:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    selected_model = model_name
                    break
                except:
                    continue
            
            if selected_model is None:
                return self._fallback_generation(retrieval_result, "HuggingFace")
            
            # آماده‌سازی متن ورودی
            prompt = self._create_advanced_prompt(retrieval_result)
            
            # تولید پاسخ
            inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=300,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # استخراج پاسخ از خروجی
            if len(response) > len(prompt):
                answer = response[len(prompt):].strip()
            else:
                answer = response.strip()
            
            return answer if answer else self._fallback_generation(retrieval_result, "HuggingFace")
            
        except Exception as e:
            print(f"خطا در HuggingFace: {e}")
            return self._fallback_generation(retrieval_result, "HuggingFace")
    
    def openai_gpt_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ با OpenAI GPT (نیاز به API Key)"""
        try:
            from openai import OpenAI
            
            # بررسی وجود API Key
            if not hasattr(self, 'openai_api_key') or not self.openai_api_key:
                return "🔑 برای استفاده از OpenAI GPT، لطفاً API Key را تنظیم کنید.\n\n" + self._fallback_generation(retrieval_result, "OpenAI")
            
            # ایجاد کلاینت OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            # آماده‌سازی متن ورودی
            prompt = self._create_advanced_prompt(retrieval_result)
            
            # انتخاب مدل بر اساس هزینه و کیفیت
            # gpt-3.5-turbo: ارزان و سریع
            # gpt-4: گران‌تر اما کیفیت بهتر
            # gpt-4-turbo-preview: جدیدترین و بهترین
            model_choice = "gpt-3.5-turbo"  # می‌توانید به gpt-4 تغییر دهید
            
            # درخواست به OpenAI
            response = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": "You are a biomedical expert analyzing knowledge graph data. Provide detailed, accurate, and well-structured answers in Persian with proper formatting and emojis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,  # افزایش تعداد توکن‌ها
                temperature=0.7,
                presence_penalty=0.1,  # تشویق به تنوع
                frequency_penalty=0.1   # کاهش تکرار
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"خطا در OpenAI: {e}")
            return self._fallback_generation(retrieval_result, "OpenAI")
    
    def anthropic_claude_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ با Anthropic Claude (نیاز به API Key)"""
        try:
            import anthropic
            
            # بررسی وجود API Key
            if not hasattr(self, 'anthropic_api_key') or not self.anthropic_api_key:
                return "🔑 برای استفاده از Claude، لطفاً API Key را تنظیم کنید.\n\n" + self._fallback_generation(retrieval_result, "Claude")
            
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            # آماده‌سازی متن ورودی
            prompt = self._create_advanced_prompt(retrieval_result)
            
            # درخواست به Claude
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            print(f"خطا در Claude: {e}")
            return self._fallback_generation(retrieval_result, "Claude")
    
    def google_gemini_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ با Google Gemini (نیاز به API Key)"""
        try:
            import google.generativeai as genai
            
            # بررسی وجود API Key
            if not hasattr(self, 'gemini_api_key') or not self.gemini_api_key:
                return "🔑 برای استفاده از Gemini، لطفاً API Key را تنظیم کنید.\n\n" + self._fallback_generation(retrieval_result, "Gemini")
            
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # آماده‌سازی متن ورودی
            prompt = self._create_advanced_prompt(retrieval_result)
            
            # درخواست به Gemini
            response = model.generate_content(prompt)
            
            return response.text.strip()
            
        except Exception as e:
            print(f"خطا در Gemini: {e}")
            return self._fallback_generation(retrieval_result, "Gemini")
    
    def _create_advanced_prompt(self, retrieval_result: RetrievalResult) -> str:
        """ایجاد متن ورودی پیشرفته برای مدل‌های AI با اطلاعات زیستی غنی شده"""
        query = retrieval_result.query
        context = retrieval_result.context_text
        method = retrieval_result.method
        
        # غنی‌سازی داده‌ها با اطلاعات زیستی
        enriched_data = self._enrich_retrieved_data(retrieval_result.nodes, retrieval_result.edges, query)
        
        # تحلیل نوع سوال
        question_type = self._analyze_question_type(query.lower())
        
        # بررسی اینکه آیا از گراف استفاده شده یا نه
        if method == "بدون بازیابی (فقط مدل)":
            # فقط مدل - بدون اطلاعات گراف
            system_prompt = """You are an expert biomedical AI assistant with comprehensive knowledge of:
            - Molecular biology and genetics
            - Drug discovery and pharmacology
            - Disease mechanisms and pathology
            - Biological pathways and networks
            - Clinical medicine and therapeutics
            
            Your task is to provide detailed, accurate, and well-structured answers to biomedical questions
            based on your training knowledge. Focus on:
            - Scientific accuracy and current understanding
            - Comprehensive analysis and insights
            - Practical implications and applications
            - Research directions and future possibilities
            
            Always answer in Persian with proper formatting and structure your response with clear sections.
            Do not use emojis in your response."""
            
            user_prompt = f"""
            **سوال پزشکی-زیستی:**
            {query}
            
            **دستورالعمل پاسخ‌دهی:**
            لطفاً بر اساس دانش تخصصی خود در زمینه علوم زیستی و پزشکی، پاسخ جامع و دقیقی ارائه دهید که شامل:
            
            1. **تحلیل موضوع:** بررسی دقیق سوال و جنبه‌های مختلف آن
            2. **مبانی علمی:** توضیح مکانیزم‌ها و فرآیندهای زیستی مرتبط
            3. **جنبه‌های درمانی:** در صورت مرتبط بودن، روش‌های درمانی و دارویی
            4. **تحقیقات:** وضعیت فعلی تحقیقات و مطالعات مرتبط
            5. **چشم‌انداز آینده:** مسیرهای تحقیقاتی و پیشرفت‌های آینده
            6. **توصیه‌های کاربردی:** نکات مهم برای پژوهشگران و پزشکان
            
            پاسخ را به صورت ساختاریافته و با فرمت‌بندی مناسب ارائه دهید.
            """
            
        else:
            # با اطلاعات گراف و داده‌های غنی شده
            system_prompt = """You are a biomedical knowledge graph expert analyzing data from Hetionet, a comprehensive
                biological knowledge graph containing information about:
                - Genes, proteins, and their functions
                - Diseases and their molecular mechanisms
                - Drugs, compounds, and their therapeutic effects
                - Biological processes and pathways
                - Anatomical structures and gene expression
                - Clinical relationships and treatment outcomes

                Your task is to provide precise, actionable analysis based on the retrieved graph data:
                - Evaluate biological relevance of genes to the specific query context
                - Assess clinical significance and potential therapeutic applications
                - Identify genes that are most likely to be clinically actionable
                - Provide specific insights rather than generic statements
                - Consider pathway involvement and disease associations
                - Focus on actionable insights and specific biological relevance

                IMPORTANT: 
                - Base your analysis primarily on the provided graph data
                - Supplement with your biomedical knowledge when needed
                - Provide specific, actionable insights rather than general statements
                - Focus on clinical relevance and therapeutic potential
                - Be precise about biological functions and mechanisms

                Always answer in Persian with proper formatting and structure your response with clear sections.
                Do not use emojis in your response."""
            
            user_prompt = f"""
            **سوال:** {query}
            
            **داده‌های بازیابی شده:**
            {context}
            
            **دستورالعمل:** بر اساس داده‌های بالا، پاسخ دقیق و کاربردی ارائه دهید. بر اهمیت زیستی و بالینی تمرکز کنید.
            """
        
        return f"{system_prompt}\n\n{user_prompt}"
    
    def _fallback_generation(self, retrieval_result: RetrievalResult, model_name: str) -> str:
        """تولید پاسخ پشتیبان در صورت خطا"""
        # استفاده از تحلیل داخلی به جای پیام ساده
        return f"""🤖 **تحلیل با {model_name} (پاسخ پشتیبان):**

{self.gpt_simulation_generation(retrieval_result)}

---
💡 **نکته:** برای استفاده از مدل‌های پیشرفته‌تر، لطفاً API Key مربوطه را تنظیم کنید.
"""
    
    def set_openai_api_key(self, api_key: str):
        """تنظیم API Key برای OpenAI"""
        self.openai_api_key = api_key
        print("✅ OpenAI API Key تنظیم شد")
    
    def set_anthropic_api_key(self, api_key: str):
        """تنظیم API Key برای Anthropic"""
        self.anthropic_api_key = api_key
        print("✅ Anthropic API Key تنظیم شد")
    
    def set_gemini_api_key(self, api_key: str):
        """تنظیم API Key برای Google Gemini"""
        self.gemini_api_key = api_key
        print("✅ Gemini API Key تنظیم شد")
    
    def _search_by_metaedges(self, matched_nodes: Dict[str, str], intent: Dict, target_metaedges: List[str], max_depth: int = 2) -> List[Tuple[str, int, float, str]]:
        """
        جستجو بر اساس metaedges مشخص شده
        """
        results = []
        
        print(f"🔍 جستجو با metaedges: {target_metaedges}")
        
        for token, node_id in matched_nodes.items():
            node_name = self.G.nodes[node_id]['name']
            node_kind = self.G.nodes[node_id]['kind']
            print(f"  بررسی نود: {node_name} ({node_kind})")
            
            # جستجوی مستقیم بر اساس metaedges
            for metaedge in target_metaedges:
                print(f"    بررسی metaedge: {metaedge}")
                
                # جستجوی همسایه‌ها با metaedge مشخص
                for neighbor in self.G.neighbors(node_id):
                    edge_data = self.G.get_edge_data(node_id, neighbor)
                    if edge_data and edge_data.get('metaedge') == metaedge:
                        neighbor_name = self.G.nodes[neighbor]['name']
                        neighbor_kind = self.G.nodes[neighbor]['kind']
                        
                        # امتیازدهی بر اساس نوع metaedge
                        score = self._calculate_metaedge_score(metaedge, 1)
                        explanation = f"{neighbor_name} ({neighbor_kind}) connected to {node_name} via {metaedge}"
                        
                        results.append((neighbor, 1, score, explanation))
                        print(f"      ✅ {neighbor_name} - {metaedge} (امتیاز: {score})")
                
                # جستجوی معکوس (اگر metaedge معکوس وجود دارد)
                reverse_metaedges = self._get_reverse_metaedges(metaedge)
                for reverse_metaedge in reverse_metaedges:
                    print(f"    بررسی metaedge معکوس: {reverse_metaedge}")
                    for other_node, other_attrs in self.G.nodes(data=True):
                        if other_node != node_id:
                            for neighbor in self.G.neighbors(other_node):
                                if neighbor == node_id:
                                    edge_data = self.G.get_edge_data(other_node, neighbor)
                                    if edge_data and edge_data.get('metaedge') == reverse_metaedge:
                                        other_name = other_attrs['name']
                                        other_kind = other_attrs['kind']
                                        
                                        score = self._calculate_metaedge_score(reverse_metaedge, 1) * 0.8  # امتیاز کمتر برای معکوس
                                        explanation = f"{other_name} ({other_kind}) connected to {node_name} via {reverse_metaedge}"
                                        
                                        results.append((other_node, 1, score, explanation))
                                        print(f"      ✅ {other_name} - {reverse_metaedge} معکوس (امتیاز: {score})")
                
                # جستجوی معکوس (اگر metaedge معکوس وجود دارد)
                reverse_metaedges = self._get_reverse_metaedges(metaedge)
                for reverse_metaedge in reverse_metaedges:
                    print(f"    بررسی metaedge معکوس: {reverse_metaedge}")
                    for other_node, other_attrs in self.G.nodes(data=True):
                        if other_node != node_id:
                            for neighbor in self.G.neighbors(other_node):
                                if neighbor == node_id:
                                    edge_data = self.G.get_edge_data(other_node, neighbor)
                                    if edge_data and edge_data.get('metaedge') == reverse_metaedge:
                                        other_name = other_attrs['name']
                                        other_kind = other_attrs['kind']
                                        
                                        score = self._calculate_metaedge_score(reverse_metaedge, 1) * 0.8  # امتیاز کمتر برای معکوس
                                        explanation = f"{other_name} ({other_kind}) connected to {node_name} via {reverse_metaedge}"
                                        
                                        results.append((other_node, 1, score, explanation))
                                        print(f"      ✅ {other_name} - {reverse_metaedge} معکوس (امتیاز: {score})")
            
            # جستجوی عمیق با فیلتر metaedges
            if max_depth > 1:
                print(f"    جستجوی عمیق تا عمق {max_depth}")
                for metaedge in target_metaedges:
                    dfs_results = self.dfs_search(node_id, max_depth, relation_filter=metaedge)
                    for found_node, depth in dfs_results:
                        if found_node != node_id:
                            found_name = self.G.nodes[found_node]['name']
                            found_kind = self.G.nodes[found_node]['kind']
                            
                            score = self._calculate_metaedge_score(metaedge, depth)
                            explanation = f"{found_name} ({found_kind}) related to {node_name} via {metaedge} (depth {depth})"
                            
                            results.append((found_node, depth, score, explanation))
                            print(f"      ✅ {found_name} - عمق {depth} با {metaedge} (امتیاز: {score:.2f})")
        
        # حذف تکرار و مرتب‌سازی
        unique_results = {}
        for node_id, depth, score, explanation in results:
            if node_id not in unique_results or score > unique_results[node_id][2]:
                unique_results[node_id] = (node_id, depth, score, explanation)
        
        final_results = sorted(unique_results.values(), key=lambda x: x[2], reverse=True)
        
        print(f"  📊 نتایج نهایی: {len(final_results)} نود منحصر به فرد")
        return final_results
    
    def _calculate_metaedge_score(self, metaedge: str, depth: int) -> float:
        """
        محاسبه امتیاز بر اساس نوع metaedge و عمق - بهبود یافته
        """
        # امتیازات پایه بر اساس اهمیت و فراوانی در Hetionet
        base_scores = {
            # بیان ژن در آناتومی - بسیار مهم
            'AeG': 6.0,  # Anatomy expresses Gene (526,407 edges)
            'GeA': 5.5,  # Gene expressed in Anatomy
            
            # تعاملات ژن‌ها - مهم
            'GiG': 5.0,  # Gene interacts with Gene (147,164 edges)
            'Gr>G': 4.5, # Gene regulates Gene (265,672 edges)
            'GcG': 4.0,  # Gene covaries with Gene (61,690 edges)
            
            # مشارکت در فرآیندهای زیستی - مهم
            'GpBP': 5.0, # Gene participates in Biological Process (559,504 edges)
            'GpPW': 4.5, # Gene participates in Pathway (84,372 edges)
            'GpMF': 4.0, # Gene participates in Molecular Function (97,222 edges)
            'GpCC': 4.0, # Gene participates in Cellular Component (73,566 edges)
            
            # تنظیم ژن توسط آناتومی
            'AuG': 4.5,  # Anatomy upregulates Gene (97,848 edges)
            'AdG': 4.5,  # Anatomy downregulates Gene (102,240 edges)
            'GuA': 4.0,  # Gene upregulates Anatomy
            'GdA': 4.0,  # Gene downregulates Anatomy
            
            # بیماری‌ها و ژن‌ها
            'DaG': 4.5,  # Disease associates with Gene (12,623 edges)
            'DuG': 4.0,  # Disease upregulates Gene (7,731 edges)
            'DdG': 4.0,  # Disease downregulates Gene (7,623 edges)
            'GaD': 4.0,  # Gene associates Disease
            'GuD': 3.5,  # Gene upregulates Disease
            'GdD': 3.5,  # Gene downregulates Disease
            
            # داروها و درمان
            'CtD': 4.5,  # Compound treats Disease (755 edges)
            'CpD': 4.0,  # Compound palliates Disease (390 edges)
            'DtC': 4.0,  # Disease treats Compound
            'DpC': 3.5,  # Disease palliates Compound
            
            # تنظیم ژن توسط دارو
            'CuG': 4.0,  # Compound upregulates Gene (18,756 edges)
            'CdG': 4.0,  # Compound downregulates Gene (21,102 edges)
            'CbG': 4.5,  # Compound binds Gene (11,571 edges)
            'GuC': 3.5,  # Gene upregulates Compound
            'GdC': 3.5,  # Gene downregulates Compound
            'GbC': 4.0,  # Gene binds Compound
            
            # بیماری‌ها و آناتومی
            'DlA': 4.0,  # Disease localizes to Anatomy (3,602 edges)
            'AlD': 3.5,  # Anatomy localizes Disease
            
            # علائم و عوارض
            'DpS': 4.0,  # Disease presents Symptom (3,357 edges)
            'SpD': 3.5,  # Symptom presents Disease
            'CcSE': 3.5, # Compound causes Side Effect (138,944 edges)
            'SEcC': 3.0, # Side Effect causes Compound
            
            # تشابه‌ها
            'DrD': 3.5,  # Disease resembles Disease (543 edges)
            'CrC': 3.5,  # Compound resembles Compound (6,486 edges)
            
            # کلاس‌های دارویی
            'PCiC': 3.0, # Pharmacologic Class includes Compound (1,029 edges)
            'CiPC': 2.5  # Compound includes Pharmacologic Class
        }
        
        base_score = base_scores.get(metaedge, 2.5)
        
        # بهبود محاسبه جریمه عمق
        if depth == 1:
            depth_penalty = 1.0
        elif depth == 2:
            depth_penalty = 0.7
        elif depth == 3:
            depth_penalty = 0.5
        else:
            depth_penalty = 0.3
        
        # اضافه کردن بونوس برای metaedges مهم
        importance_bonus = 1.0
        if metaedge in ['AeG', 'GiG', 'GpBP', 'DaG', 'CtD']:
            importance_bonus = 1.2
        elif metaedge in ['Gr>G', 'GpPW', 'CbG']:
            importance_bonus = 1.1
        
        return base_score * depth_penalty * importance_bonus
    
    def _get_reverse_metaedges(self, metaedge: str) -> List[str]:
        """
        دریافت metaedges معکوس
        """
        reverse_mapping = {
            'AeG': ['GeA'],  # Anatomy expresses Gene ↔ Gene expressed in Anatomy
            'GeA': ['AeG'],
            'AuG': ['GuA'],  # Anatomy upregulates Gene ↔ Gene upregulates Anatomy
            'GuA': ['AuG'],
            'AdG': ['GdA'],  # Anatomy downregulates Gene ↔ Gene downregulates Anatomy
            'GdA': ['AdG'],
            'DaG': ['GaD'],  # Disease associates Gene ↔ Gene associates Disease
            'GaD': ['DaG'],
            'DuG': ['GuD'],  # Disease upregulates Gene ↔ Gene upregulates Disease
            'GuD': ['DuG'],
            'DdG': ['GdD'],  # Disease downregulates Gene ↔ Gene downregulates Disease
            'GdD': ['DdG'],
            'CtD': ['DtC'],  # Compound treats Disease ↔ Disease treats Compound
            'DtC': ['CtD'],
            'CuG': ['GuC'],  # Compound upregulates Gene ↔ Gene upregulates Compound
            'GuC': ['CuG'],
            'CdG': ['GdC'],  # Compound downregulates Gene ↔ Gene downregulates Compound
            'GdC': ['CdG'],
            'CbG': ['GbC'],  # Compound binds Gene ↔ Gene binds Compound
            'GbC': ['CbG'],
            'DlA': ['AlD'],  # Disease localizes Anatomy ↔ Anatomy localizes Disease
            'AlD': ['DlA'],
            'DpS': ['SpD'],  # Disease presents Symptom ↔ Symptom presents Disease
            'SpD': ['DpS'],
            'CcSE': ['SEcC'], # Compound causes Side Effect ↔ Side Effect causes Compound
            'SEcC': ['CcSE'],
            'GpBP': ['BPpG'], # Gene participates Biological Process ↔ Biological Process participates Gene
            'BPpG': ['GpBP'],
            'GpMF': ['MFpG'], # Gene participates Molecular Function ↔ Molecular Function participates Gene
            'MFpG': ['GpMF'],
            'GpCC': ['CCpG'], # Gene participates Cellular Component ↔ Cellular Component participates Gene
            'CCpG': ['GpCC'],
            'GpPW': ['PWpG'], # Gene participates Pathway ↔ Pathway participates Gene
            'PWpG': ['GpPW'],
            'PCiC': ['CiPC'], # Pharmacologic Class includes Compound ↔ Compound includes Pharmacologic Class
            'CiPC': ['PCiC']
        }
        
        return reverse_mapping.get(metaedge, [])
    
    def multi_hop_search(self, query: str, max_depth: int = 3) -> List[Tuple[str, int, float, str, List[str]]]:
        """
        جستجوی چندمرحله‌ای برای سوالات پیچیده
        Returns: (node_id, depth, score, explanation, path)
        """
        print(f"🔄 جستجوی چندمرحله‌ای: {query}")
        
        # تحلیل سوال
        intent = self.analyze_question_intent(query)
        print(f"  تشخیص نوع: {intent['question_type']}")
        print(f"  Metaedges: {intent['metaedges']}")
        
        # استخراج کلمات کلیدی
        keywords = intent['keywords']
        matched_nodes = self.match_tokens_to_nodes(keywords)
        
        if not matched_nodes:
            print("  ❌ هیچ نودی تطبیق نکرد")
            return []
        
        results = []
        
        # بر اساس نوع سوال، مسیرهای چندمرحله‌ای را تعریف کن
        multi_hop_patterns = {
            'anatomy_expression': [
                # Anatomy → AeG → Gene
                ['AeG'],
                # Anatomy → AuG → Gene  
                ['AuG'],
                # Anatomy → AdG → Gene
                ['AdG']
            ],
            'compound_gene_regulation': [
                # Compound → CuG → Gene
                ['CuG'],
                # Compound → CdG → Gene
                ['CdG'],
                # Compound → CbG → Gene
                ['CbG']
            ],
            'disease_gene_regulation': [
                # Disease → DuG → Gene
                ['DuG'],
                # Disease → DdG → Gene
                ['DdG'],
                # Disease → DaG → Gene
                ['DaG']
            ],
            'complex_expression': [
                # Anatomy → AeG → Gene → CuG → Compound
                ['AeG', 'CuG'],
                # Anatomy → AeG → Gene → CdG → Compound
                ['AeG', 'CdG'],
                # Gene → GeA → Anatomy → Compound (معکوس)
                ['GeA', 'GuC'],
                # Gene → GeA → Anatomy → Compound (معکوس)
                ['GeA', 'GdC'],
                # Compound → CdG → Gene → GeA → Anatomy (معکوس)
                ['CdG', 'GeA'],
                # Compound → CuG → Gene → GeA → Anatomy (معکوس)
                ['CuG', 'GeA']
            ],
            'complex_disease': [
                # Disease → DaG → Gene → GiG → Gene
                ['DaG', 'GiG'],
                # Disease → DuG → Gene → GpBP → Biological Process
                ['DuG', 'GpBP'],
                # Disease → DlA → Anatomy → AeG → Gene
                ['DlA', 'AeG'],
                # Gene → GaD → Disease → GpBP → Biological Process
                ['GaD', 'GpBP']
            ],
            'complex_treatment': [
                # Compound → CtD → Disease → DaG → Gene
                ['CtD', 'DaG'],
                # Compound → CuG → Gene → GaD → Disease
                ['CuG', 'GaD'],
                # Compound → CdG → Gene → GaD → Disease
                ['CdG', 'GaD'],
                # Disease → DtC → Compound → CuG → Gene
                ['DtC', 'CuG'],
                # Gene → GuC → Compound → CtD → Disease
                ['GuC', 'CtD']
            ],
            'complex_function': [
                # Gene → GpBP → Biological Process → BPpG → Gene
                ['GpBP', 'BPpG'],
                # Gene → GpPW → Pathway → PWpG → Gene
                ['GpPW', 'PWpG'],
                # Gene → GiG → Gene → GpBP → Biological Process
                ['GiG', 'GpBP'],
                # Gene → Gr>G → Gene → GpMF → Molecular Function
                ['Gr>G', 'GpMF']
            ]
        }
        
        # تشخیص نوع سوال پیچیده
        complex_type = self._detect_complex_question_type(intent)
        print(f"  نوع سوال پیچیده: {complex_type}")
        
        if complex_type in multi_hop_patterns:
            patterns = multi_hop_patterns[complex_type]
            
            for pattern in patterns:
                print(f"  بررسی الگو: {' → '.join(pattern)}")
                pattern_results = self._search_multi_hop_pattern(matched_nodes, pattern, max_depth)
                results.extend(pattern_results)
        
        # حذف تکرار و مرتب‌سازی
        unique_results = {}
        for node_id, depth, score, explanation, path in results:
            if node_id not in unique_results or score > unique_results[node_id][2]:
                unique_results[node_id] = (node_id, depth, score, explanation, path)
        
        final_results = sorted(unique_results.values(), key=lambda x: x[2], reverse=True)
        
        print(f"  ✅ {len(final_results)} نتیجه چندمرحله‌ای یافت شد")
        return final_results
    
    def _detect_complex_question_type(self, intent: Dict) -> str:
        """تشخیص نوع سوال پیچیده"""
        query_lower = intent['query_lower']
        
        # سوالات پیچیده که نیاز به چند مرحله دارند
        if any(word in query_lower for word in ['upregulate', 'downregulate', 'regulate']) and \
           any(word in query_lower for word in ['expressed', 'expression']):
            return 'complex_expression'
        
        if any(word in query_lower for word in ['interact', 'interaction']) and \
           any(word in query_lower for word in ['disease', 'associated']):
            return 'complex_disease'
        
        if any(word in query_lower for word in ['treat', 'treatment', 'therapy']) and \
           any(word in query_lower for word in ['compound', 'drug', 'medicine']):
            return 'complex_treatment'
        
        if any(word in query_lower for word in ['function', 'process', 'pathway']) and \
           any(word in query_lower for word in ['gene', 'protein']):
            return 'complex_function'
        
        # تشخیص بر اساس metaedges موجود
        metaedges = intent.get('metaedges', [])
        if 'AeG' in metaedges and ('CuG' in metaedges or 'CdG' in metaedges):
            return 'complex_expression'
        if 'DaG' in metaedges and ('GiG' in metaedges or 'GpBP' in metaedges):
            return 'complex_disease'
        if 'CtD' in metaedges and ('DaG' in metaedges or 'CuG' in metaedges):
            return 'complex_treatment'
        if 'GpBP' in metaedges or 'GpPW' in metaedges:
            return 'complex_function'
        
        # سوالات ساده‌تر
        if intent['question_type'] == 'anatomy_expression':
            return 'anatomy_expression'
        elif intent['question_type'] == 'compound_gene_regulation':
            return 'compound_gene_regulation'
        elif intent['question_type'] == 'disease_gene_regulation':
            return 'disease_gene_regulation'
        
        return 'general'
    
    def _search_multi_hop_pattern(self, matched_nodes: Dict[str, str], pattern: List[str], max_depth: int) -> List[Tuple[str, int, float, str, List[str]]]:
        """جستجو بر اساس الگوی چندمرحله‌ای"""
        results = []
        
        for token, start_node in matched_nodes.items():
            print(f"    شروع از نود: {self.G.nodes[start_node]['name']}")
            
            # جستجوی مسیرهای چندمرحله‌ای
            paths = self._find_paths_with_pattern(start_node, pattern, max_depth)
            
            for path, path_metaedges in paths:
                if len(path) > 1:  # حداقل 2 نود
                    target_node = path[-1]
                    depth = len(path) - 1
                    
                    # محاسبه امتیاز بر اساس الگو
                    score = self._calculate_pattern_score(pattern, path_metaedges, depth)
                    
                    # ایجاد توضیح
                    path_names = [self.G.nodes[node]['name'] for node in path]
                    explanation = f"مسیر: {' → '.join(path_names)} (الگو: {' → '.join(pattern)})"
                    
                    results.append((target_node, depth, score, explanation, path))
        
        return results
    
    def _find_paths_with_pattern(self, start_node: str, pattern: List[str], max_depth: int) -> List[Tuple[List[str], List[str]]]:
        """یافتن مسیرهایی که با الگوی مشخص شده مطابقت دارند"""
        paths = []
        
        def dfs_with_pattern(node: str, current_path: List[str], current_metaedges: List[str], depth: int):
            if depth >= max_depth:
                return
            
            current_path.append(node)
            
            # بررسی اینکه آیا مسیر فعلی با الگو مطابقت دارد
            if len(current_metaedges) == len(pattern):
                paths.append((current_path.copy(), current_metaedges.copy()))
            
            # جستجوی همسایه‌ها
            for neighbor in self.G.neighbors(node):
                if neighbor not in current_path:  # جلوگیری از حلقه
                    edge_data = self.G.get_edge_data(node, neighbor)
                    if edge_data and edge_data.get('relation'):
                        metaedge = edge_data.get('relation')
                        
                        # بررسی اینکه آیا این metaedge در الگو است
                        if len(current_metaedges) < len(pattern) and metaedge == pattern[len(current_metaedges)]:
                            new_metaedges = current_metaedges + [metaedge]
                            dfs_with_pattern(neighbor, current_path, new_metaedges, depth + 1)
                        elif metaedge in pattern:  # جستجوی آزادتر
                            new_metaedges = current_metaedges + [metaedge]
                            dfs_with_pattern(neighbor, current_path, new_metaedges, depth + 1)
            
            current_path.pop()
        
        # شروع از نود اول
        dfs_with_pattern(start_node, [], [], 0)
        
        # اگر مسیری پیدا نشد، سعی کن از نودهای دیگر شروع کنی
        if not paths:
            print(f"    ⚠️ هیچ مسیری از {start_node} پیدا نشد، تلاش از نودهای دیگر...")
            
            # برای الگوهای چندمرحله‌ای، سعی کن از نودهای میانی شروع کنی
            if len(pattern) > 1:
                # برای الگوهای AeG → CuG/CdG، از نودهای Compound شروع کن
                if 'CuG' in pattern or 'CdG' in pattern:
                    compound_nodes = [nid for nid, attrs in self.G.nodes(data=True) 
                                    if attrs.get('kind') == 'Compound' or attrs.get('metanode') == 'Compound']
                    
                    for compound_node in compound_nodes[:3]:  # 3 نود اول
                        if compound_node != start_node:
                            print(f"    تلاش از نود: {self.G.nodes[compound_node]['name']}")
                            dfs_with_pattern(compound_node, [], [], 0)
                
                # از نودهای ژن شروع کن (برای الگوهای دیگر)
                else:
                    gene_nodes = [nid for nid, attrs in self.G.nodes(data=True) 
                                 if attrs.get('kind') == 'Gene' or attrs.get('metanode') == 'Gene']
                    
                    for gene_node in gene_nodes[:5]:  # 5 نود اول
                        if gene_node != start_node:
                            print(f"    تلاش از نود: {self.G.nodes[gene_node]['name']}")
                            dfs_with_pattern(gene_node, [], [], 0)
        
        return paths
    
    def _calculate_pattern_score(self, pattern: List[str], path_metaedges: List[str], depth: int) -> float:
        """محاسبه امتیاز بر اساس تطابق الگو"""
        base_score = 5.0
        
        # امتیاز برای تطابق کامل الگو
        if path_metaedges == pattern:
            base_score += 2.0
        elif all(me in path_metaedges for me in pattern):
            base_score += 1.0
        
        # کاهش امتیاز با افزایش عمق
        depth_penalty = 1.0 / (depth + 1)
        
        # امتیاز برای طول مسیر
        length_bonus = len(path_metaedges) * 0.5
        
        return (base_score + length_bonus) * depth_penalty

# نمونه استفاده
if __name__ == "__main__":
    service = GraphRAGService()
    
    # تست سرویس
    result = service.process_query(
        query="What is the relationship between HMGB3 and diabetes?",
        retrieval_method=RetrievalMethod.BFS,
        generation_model=GenerationModel.GPT_SIMULATION
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False)) 
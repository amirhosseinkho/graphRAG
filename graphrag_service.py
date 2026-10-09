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
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import new modules
try:
    from graphrag_new.search import KGSearch
    from graphrag_new.utils import get_entity_type2sampels, get_llm_cache, set_llm_cache, get_relation
    from graphrag_new.query_analyze_prompt import PROMPTS
    from graphrag_new.entity_resolution import EntityResolution
    from rag_new.llm.chat_model import GptTurbo, MoonshotChat, AzureChat, QWenChat, ZhipuChat, OllamaChat, GeminiChat, AnthropicChat
    from rag_new.utils import REDIS_CONN
    NEW_MODULES_AVAILABLE = True
except ImportError:
    NEW_MODULES_AVAILABLE = False
    print("Warning: New GraphRAG modules not available. Using classic methods only.")

from enhanced_context_generator import EnhancedContextGenerator

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

# دیکشنری کامل توضیحات metaedge برای استفاده در متن زمینه‌ای
METAEDGE_DESCRIPTIONS = {
    # Anatomy relationships
    "AdG": "Anatomy–downregulates–Gene: تنظیم منفی ژن در بافت",
    "AeG": "Anatomy–expresses–Gene: بیان ژن در بافت",
    "AlD": "Anatomy–localizes–Disease: محل بیماری در بافت",
    "AuG": "Anatomy–upregulates–Gene: تنظیم مثبت ژن در بافت",
    
    # Biological Process relationships
    "BPpG": "Biological Process–participates–Gene: مشارکت ژن در فرآیند زیستی",
    "GpBP": "Gene–participates–Biological Process: مشارکت ژن در فرآیند زیستی",
    
    # Cellular Component relationships
    "CCpG": "Cellular Component–participates–Gene: مشارکت ژن در اجزای سلولی",
    "GpCC": "Gene–participates–Cellular Component: مشارکت ژن در اجزای سلولی",
    
    # Compound relationships
    "CbG": "Compound–binds–Gene: اتصال دارو به ژن",
    "CcSE": "Compound–causes–Side Effect: عوارض جانبی دارو",
    "CdG": "Compound–downregulates–Gene: تنظیم منفی ژن توسط دارو",
    "CiPC": "Compound–includes–Pharmacologic Class: طبقه‌بندی دارویی",
    "CpD": "Compound–palliates–Disease: تسکین بیماری توسط دارو",
    "CrC": "Compound–resembles–Compound: شباهت دارویی",
    "CtD": "Compound–treats–Disease: درمان بیماری توسط دارو",
    "CuG": "Compound–upregulates–Gene: تنظیم مثبت ژن توسط دارو",
    
    # Disease relationships
    "DaG": "Disease–associates–Gene: ارتباط بیماری با ژن",
    "DdG": "Disease–downregulates–Gene: تنظیم منفی ژن در بیماری",
    "DlA": "Disease–localizes–Anatomy: محل بیماری در بافت",
    "DpC": "Disease–palliates–Compound: تسکین بیماری",
    "DpS": "Disease–presents–Symptom: علائم بیماری",
    "DrD": "Disease–resembles–Disease: شباهت بیماری‌ها",
    "DtC": "Disease–treats–Compound: درمان بیماری",
    "DuG": "Disease–upregulates–Gene: تنظیم مثبت ژن در بیماری",
    
    # Gene relationships
    "GaD": "Gene–associates–Disease: ارتباط ژن با بیماری",
    "GbC": "Gene–binds–Compound: اتصال ژن به دارو",
    "GcG": "Gene–covaries–Gene: هم‌تغییری ژن‌ها",
    "GdA": "Gene–downregulates–Anatomy: تنظیم منفی بافت توسط ژن",
    "GdC": "Gene–downregulates–Compound: تنظیم منفی دارو توسط ژن",
    "GdD": "Gene–downregulates–Disease: تنظیم منفی بیماری توسط ژن",
    "GeA": "Gene–expresses–Anatomy: بیان ژن در بافت",
    "GiG": "Gene–interacts–Gene: تعامل ژن‌ها",
    "GuA": "Gene–upregulates–Anatomy: تنظیم مثبت بافت توسط ژن",
    "GuC": "Gene–upregulates–Compound: تنظیم مثبت دارو توسط ژن",
    "GuD": "Gene–upregulates–Disease: تنظیم مثبت بیماری توسط ژن",
    "Gr>G": "Gene→regulates→Gene: تنظیم ژن توسط ژن",
    "G<rG": "Gene←regulates←Gene: تنظیم ژن توسط ژن",
    
    # Molecular Function relationships
    "MFpG": "Molecular Function–participates–Gene: مشارکت ژن در عملکرد مولکولی",
    "GpMF": "Gene–participates–Molecular Function: مشارکت ژن در عملکرد مولکولی",
    
    # Pathway relationships
    "PWpG": "Pathway–participates–Gene: مشارکت ژن در مسیر زیستی",
    "GpPW": "Gene–participates–Pathway: مشارکت ژن در مسیر زیستی",
    
    # Pharmacologic Class relationships
    "PCiC": "Pharmacologic Class–includes–Compound: شامل دارو",
    
    # Side Effect relationships
    "SEcC": "Side Effect–causes–Compound: عوارض جانبی",
    
    # Symptom relationships
    "SpD": "Symptom–presents–Disease: علائم بیماری"
}

# نقش‌های زیستی مهم برای تحلیل
BIOLOGICAL_ROLES = {
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

# بیماری‌های مهم برای تحلیل
DISEASE_SIGNIFICANCE = {
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

class RetrievalMethod(Enum):
    """روش‌های بازیابی"""
    # الگوریتم‌های کلاسیک
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
    
    # الگوریتم‌های جدید GraphRAG
    KG_SEARCH = "KGSearch (الگوریتم اصلی جدید)"
    N_HOP_RETRIEVAL = "N-Hop Retrieval (بازیابی چندمرحله‌ای)"
    PAGERANK_BASED = "PageRank-Based (بر اساس اهمیت)"
    SEMANTIC_SIMILARITY = "Semantic Similarity (شباهت معنایی)"
    COMMUNITY_DETECTION = "Community Detection (تشخیص جامعه‌ها)"
    ENTITY_RESOLUTION = "Entity Resolution (حل موجودیت‌ها)"
    HYBRID_NEW = "Hybrid New (ترکیب روش‌های جدید)"

class TokenExtractionMethod(Enum):
    """روش‌های استخراج توکن"""
    CLASSIC = "کلاسیک (روش قبلی)"
    LLM_BASED = "LLM-Based (هوشمند)"

class TokenExtractionModel(Enum):
    """مدل‌های استخراج توکن"""
    # OpenAI Models
    OPENAI_GPT_4O = "GPT-4o (بهترین کیفیت)"
    OPENAI_GPT_4O_MINI = "GPT-4o Mini (سریع و اقتصادی)"
    OPENAI_GPT_3_5_TURBO = "GPT-3.5 Turbo (سریع)"
    
    # Anthropic Models
    ANTHROPIC_CLAUDE_3_5_SONNET = "Claude 3.5 Sonnet"
    ANTHROPIC_CLAUDE_3_5_HAIKU = "Claude 3.5 Haiku"
    
    # Google Models
    GOOGLE_GEMINI_1_5_PRO = "Gemini 1.5 Pro"
    GOOGLE_GEMINI_1_5_FLASH = "Gemini 1.5 Flash"

class ContextTextType(Enum):
    """انواع متن زمینه"""
    SIMPLE = "متن ساده عمومی"
    INTELLIGENT = "متن تخصصی هوشمند"
    SCIENTIFIC_ANALYTICAL = "متن علمی-تحلیلی (تحقیقاتی)"
    NARRATIVE = "متن روایی (ساده و توصیفی)"
    DATA_DRIVEN = "متن داده‌محور (رابطه‌ها به صورت لیست)"
    STEP_BY_STEP = "متن توضیح قبل از سؤال (استدلال گام به گام)"
    COMPACT_DIRECT = "متن فشرده و مستقیم"
    BIOLOGICAL_PATHWAY = "متن مسیر زیستی (تخصصی)"
    CLINICAL_RELEVANCE = "متن ارتباط بالینی"
    MECHANISTIC_DETAILED = "متن مکانیسمی تفصیلی"

class GenerationModel(Enum):
    """مدل‌های تولید متن"""
    # مدل‌های محلی و رایگان
    GENERAL_SIMPLE = "General Simple (پاسخ ساده و عمومی)"
    SIMPLE = "Simple Template"
    GPT_SIMULATION = "GPT Simulation"
    CUSTOM = "Custom Model"
    HUGGINGFACE = "HuggingFace Models"
    
    # OpenAI GPT Models
    OPENAI_GPT_4O = "OpenAI GPT-4o (جدیدترین و قوی‌ترین)"
    OPENAI_GPT_4O_MINI = "OpenAI GPT-4o Mini (سریع و اقتصادی)"
    OPENAI_GPT_4_TURBO = "OpenAI GPT-4 Turbo (تعادل سرعت و کیفیت)"
    OPENAI_GPT_4 = "OpenAI GPT-4 (کیفیت بالا)"
    OPENAI_GPT_3_5_TURBO = "OpenAI GPT-3.5 Turbo (سریع و اقتصادی)"
    OPENAI_GPT_3_5_TURBO_16K = "OpenAI GPT-3.5 Turbo 16K (متن طولانی)"
    
    # Anthropic Claude Models
    ANTHROPIC_CLAUDE_3_5_SONNET = "Anthropic Claude 3.5 Sonnet (جدیدترین)"
    ANTHROPIC_CLAUDE_3_5_HAIKU = "Anthropic Claude 3.5 Haiku (سریع)"
    ANTHROPIC_CLAUDE_3_OPUS = "Anthropic Claude 3 Opus (قوی‌ترین)"
    ANTHROPIC_CLAUDE_3_SONNET = "Anthropic Claude 3 Sonnet (تعادل)"
    ANTHROPIC_CLAUDE_3_HAIKU = "Anthropic Claude 3 Haiku (سریع)"
    
    # Google Gemini Models
    GOOGLE_GEMINI_1_5_PRO = "Google Gemini 1.5 Pro (جدیدترین)"
    GOOGLE_GEMINI_1_5_FLASH = "Google Gemini 1.5 Flash (سریع)"
    GOOGLE_GEMINI_1_0_PRO = "Google Gemini 1.0 Pro (پایدار)"
    GOOGLE_GEMINI_1_0_FLASH = "Google Gemini 1.0 Flash (سریع)"
    
    # سایر مدل‌های پیشرفته
    META_LLAMA_3_1 = "Meta Llama 3.1 (محلی)"
    MISTRAL_AI = "Mistral AI (کیفیت بالا)"
    COHERE_COMMAND = "Cohere Command (تخصصی)"
    PERPLEXITY_SONAR = "Perplexity Sonar (جستجوگر)"
    
    # مدل‌های قدیمی برای سازگاری
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
        """راه‌اندازی سرویس GraphRAG"""
        self.graph_data_path = graph_data_path or "hetionet_graph.pkl"
        self.G = None
        self.nlp = None
        # ایندکس‌ها و کش‌ها
        self._name_to_ids = {}
        self._id_to_name = {}
        self._kind_to_ids = {}
        self._name_entries = []  # [(lower_name, node_id)] برای fallback فازی سبک
        self._pagerank = {}
        self._keyword_cache = {}
        self._last_intent = None
        # ژنراتور متن زمینه بهبود یافته
        try:
            self.context_generator = EnhancedContextGenerator()
        except Exception:
            self.context_generator = None
        
        # تنظیمات قابل تغییر برای محدودیت‌ها
        self.config = {
            'max_nodes': 10,           # حداکثر تعداد نودهای بازیابی شده
            'max_edges': 20,           # حداکثر تعداد یال‌های بازیابی شده
            'max_depth': 3,            # حداکثر عمق جستجو
            'max_paths': 5,            # حداکثر تعداد مسیرها
            'max_context_length': 2000, # حداکثر طول متن زمینه (کاراکتر)
            'max_answer_tokens': 1000,  # حداکثر توکن‌های پاسخ
            'max_prompt_tokens': 4000,  # حداکثر توکن‌های ورودی
            'enable_verbose_logging': True,  # نمایش جزئیات
            'enable_biological_enrichment': True,  # غنی‌سازی زیستی
            'enable_smart_filtering': True,  # فیلتر هوشمند
        }
        
        # API Keys
        self.openai_api_key = None
        # self.anthropic_api_key = None
        # self.gemini_api_key = None
        
        self.initialize()
    
    def set_config(self, **kwargs):
        """تغییر تنظیمات سیستم"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                print(f" تنظیم {key} = {value}")
            else:
                print(f" تنظیم نامعتبر: {key}")
    
    def get_config(self):
        """دریافت تنظیمات فعلی"""
        return self.config.copy()
    
    def initialize(self):
        """راه‌اندازی سرویس"""
        print(" راه‌اندازی GraphRAG Service...")
        
        # بارگذاری مدل spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print(" مدل spaCy بارگذاری شد")
        except:
            print(" خطا در بارگذاری مدل spaCy - استفاده از استخراج کلیدواژه ساده")
            self.nlp = None
        
        # بارگذاری یا ایجاد گراف
        if self.graph_data_path and os.path.exists(self.graph_data_path):
            self.load_graph_from_file()
        else:
            self.create_sample_graph()

    def _post_graph_loaded(self):
        """اقدامات پس از بارگذاری/ایجاد گراف: ساخت ایندکس‌ها و محاسبه PageRank تنبل"""
        self._build_node_indices()
        # PageRank را به‌صورت تنبل نگه می‌داریم؛ اینجا اگر گراف کوچک باشد حساب می‌کنیم
        try:
            if self.G and self.G.number_of_nodes() <= 5000:
                import networkx as nx
                self._pagerank = nx.pagerank(self.G, alpha=0.85)
        except Exception as e:
            print(f"⚠️ خطا در محاسبه اولیه PageRank: {e}")

    def _build_node_indices(self):
        """ساخت ایندکس‌های کم‌حجم برای تطبیق سریع توکن‌ها با نودها"""
        self._name_to_ids.clear()
        self._id_to_name.clear()
        self._kind_to_ids.clear()
        self._name_entries.clear()
        if not self.G:
            return
        for node_id, attrs in self.G.nodes(data=True):
            name = str(attrs.get('name', node_id))
            kind = str(attrs.get('kind', 'Unknown'))
            self._id_to_name[node_id] = name
            lower_name = name.lower()
            self._name_to_ids.setdefault(lower_name, []).append(node_id)
            self._kind_to_ids.setdefault(kind, []).append(node_id)
            # ورودی برای جستجوی شامل ساده
            self._name_entries.append((lower_name, node_id))

    def _display_node(self, node_id: str) -> str:
        """نمایش انسانی یک نود بر اساس نام و نوع (در صورت وجود)"""
        try:
            name = self._id_to_name.get(node_id) or self.G.nodes[node_id].get('name', node_id)
            kind = self.G.nodes[node_id].get('kind')
            return f"{name} ({kind})" if kind else str(name)
        except Exception:
            return str(node_id)

    def _ensure_pagerank(self):
        if not self._pagerank and self.G:
            try:
                import networkx as nx
                self._pagerank = nx.pagerank(self.G, alpha=0.85)
            except Exception:
                self._pagerank = {}
    
    def create_sample_graph(self):
        """ایجاد گراف نمونه بر اساس ساختار واقعی Hetionet"""
        print(" ایجاد گراف نمونه بر اساس Hetionet...")
        
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
        
        print(f" گراف نمونه بر اساس Hetionet ایجاد شد: {self.G.number_of_nodes()} نود، {self.G.number_of_edges()} یال")
        print(f" شامل {len([n for n, d in self.G.nodes(data=True) if d.get('metanode') == 'Gene'])} ژن، {len([n for n, d in self.G.nodes(data=True) if d.get('metanode') == 'Anatomy'])} آناتومی")
        print(f" شامل {len([e for e in self.G.edges(data=True) if e[2].get('metaedge') == 'AeG'])} یال AeG (Anatomy-expresses-Gene)")
        print(f" شامل {len([e for e in self.G.edges(data=True) if e[2].get('metaedge') == 'GeA'])} یال GeA (Gene-expressed_in-Anatomy) - معکوس")
        self._post_graph_loaded()
    
    def load_graph_from_file(self):
        """بارگذاری گراف از فایل"""
        try:
            with open(self.graph_data_path, 'rb') as f:
                self.G = pickle.load(f)
            print(f" گراف از فایل بارگذاری شد: {self.G.number_of_nodes()} نود، {self.G.number_of_edges()} یال")
            self._post_graph_loaded()
        except Exception as e:
            print(f" خطا در بارگذاری گراف: {e}")
            self.create_sample_graph()
    
    def extract_keywords(self, text: str) -> List[str]:
        """استخراج کلمات کلیدی از متن با بهبود برای ژن‌ها و اصطلاحات تخصصی"""
        # کش ساده برای سرعت
        if text in self._keyword_cache:
            return self._keyword_cache[text]
        if self.nlp is None:
            # fallback ساده بدون spaCy
            import re as _re
            tokens = _re.sub(r"[^\w\s]", " ", text.lower()).split()
            keywords = sorted(set(t for t in tokens if len(t) >= 2))
            self._keyword_cache[text] = keywords
            return keywords
        doc = self.nlp(text)
        keywords = set()
        
        # نگاشت فارسی به انگلیسی برای کلمات کلیدی مهم
        persian_to_english = {
            # ژن‌ها
            'ژن': 'gene', 'ژن‌ها': 'genes', 'پروتئین': 'protein', 'پروتئین‌ها': 'proteins',
            'دی‌ان‌ای': 'dna', 'آر‌ان‌ای': 'rna', 'ام‌آر‌ان‌ای': 'mrna',
            
            # بافت‌ها و اندام‌ها
            'کبد': 'liver', 'مغز': 'brain', 'قلب': 'heart', 'ریه': 'lung', 'کلیه': 'kidney',
            'معده': 'stomach', 'ماهیچه': 'muscle', 'استخوان': 'bone', 'خون': 'blood',
            'بافت': 'tissue', 'بافت‌ها': 'tissues', 'اندام': 'organ', 'اندام‌ها': 'organs',
            'بدن': 'body', 'بخش بدن': 'body part',
            
            # بیماری‌ها
            'سرطان': 'cancer', 'سرطان‌ها': 'cancers', 'تومور': 'tumor', 'تومورها': 'tumors',
            'بیماری': 'disease', 'بیماری‌ها': 'diseases', 'اختلال': 'disorder', 'اختلالات': 'disorders',
            'سندرم': 'syndrome', 'سندرم‌ها': 'syndromes', 'بدخیمی': 'malignancy', 'بدخیمی‌ها': 'malignancies',
            'سرطان سینه': 'breast cancer', 'سرطان ریه': 'lung cancer', 'سرطان کبد': 'liver cancer',
            'سرطان مغز': 'brain cancer', 'سرطان خون': 'blood cancer', 'سرطان معده': 'stomach cancer',
            'دیابت': 'diabetes', 'آلزایمر': 'alzheimer', 'فیبروز': 'fibrosis',
            
            # داروها
            'دارو': 'drug', 'داروها': 'drugs', 'داروی': 'drug', 'دارویی': 'drug',
            'داروهای': 'drugs', 'دارویی': 'drug', 'داروها': 'drugs',
            'آسپرین': 'aspirin', 'ایبوپروفن': 'ibuprofen', 'کافئین': 'caffeine',
            'ویتامین': 'vitamin', 'ویتامین‌ها': 'vitamins', 'شیمیایی': 'chemical',
            'شیمیایی‌ها': 'chemicals', 'مولکول': 'molecule', 'مولکول‌ها': 'molecules',
            'ترکیب': 'compound', 'ترکیبات': 'compounds', 'دارو': 'medication',
            'داروها': 'medications', 'دارو': 'medicine', 'داروها': 'medicines',
            
            # فرآیندهای زیستی
            'فرآیند': 'process', 'فرآیندها': 'processes', 'زیستی': 'biological',
            'مسیر': 'pathway', 'مسیرها': 'pathways', 'مکانیسم': 'mechanism',
            'عملکرد': 'function', 'عملکردها': 'functions', 'فعالیت': 'activity',
            'فعالیت‌ها': 'activities', 'آپوپتوز': 'apoptosis', 'چرخه سلولی': 'cell cycle',
            'ترمیم دی‌ان‌ای': 'dna repair', 'تقسیم سلولی': 'cell division',
            
            # علائم
            'علائم': 'symptom', 'علائم': 'symptoms', 'نشانه': 'sign', 'نشانه‌ها': 'signs',
            'تجلی': 'manifestation', 'تجلیات': 'manifestations', 'نشانه': 'indication',
            'درد': 'pain', 'تب': 'fever', 'سرفه': 'cough', 'خستگی': 'fatigue',
            
            # عوارض جانبی
            'عوارض جانبی': 'side effect', 'عوارض جانبی': 'side effects', 'عوارض': 'adverse',
            'واکنش': 'reaction', 'واکنش‌ها': 'reactions', 'سمیت': 'toxicity',
            'تهوع': 'nausea', 'سردرد': 'headache', 'سرگیجه': 'dizziness',
            
            # عملکرد مولکولی
            'مولکولی': 'molecular', 'آنزیمی': 'enzymatic', 'آنزیم': 'enzyme',
            'گیرنده': 'receptor', 'حامل': 'transporter', 'حامل‌ها': 'transporters',
            
            # اجزای سلولی
            'سلولی': 'cellular', 'جزء': 'component', 'اجزا': 'components',
            'اندامک': 'organelle', 'اندامک‌ها': 'organelles', 'ساختار': 'structure',
            'هسته': 'nucleus', 'میتوکندری': 'mitochondria', 'غشاء': 'membrane',
            
            # طبقه‌بندی دارویی
            'دارویی': 'pharmacologic', 'داروشناختی': 'pharmacological', 'طبقه': 'class',
            'طبقات': 'classes', 'دسته': 'category', 'دسته‌ها': 'categories',
            'نوع': 'type', 'انواع': 'types', 'آنتی‌بیوتیک': 'antibiotic',
            'ضد فشار خون': 'antihypertensive',
            
            # کلمات عمومی
            'کدام': 'which', 'چه': 'what', 'کجا': 'where', 'چگونه': 'how',
            'چرا': 'why', 'چه زمانی': 'when', 'چه کسی': 'who',
            'مرتبط': 'related', 'مرتبط با': 'related to', 'مربوط': 'associated',
            'مربوط به': 'associated with', 'متصل': 'connected', 'متصل به': 'connected to',
            'بیان': 'expression', 'بیان می‌شود': 'expressed', 'بیان می‌شوند': 'expressed',
            'درمان': 'treatment', 'درمان می‌کند': 'treats', 'درمان می‌کنند': 'treat',
            'استفاده': 'used', 'استفاده می‌شود': 'used', 'استفاده می‌شوند': 'used',
            'نقش': 'role', 'نقش دارد': 'plays role', 'نقش دارند': 'play role',
            'شرکت': 'participate', 'شرکت می‌کند': 'participates', 'شرکت می‌کنند': 'participate',
            'تعامل': 'interaction', 'تعامل دارد': 'interacts', 'تعامل دارند': 'interact',
            'تنظیم': 'regulation', 'تنظیم می‌کند': 'regulates', 'تنظیم می‌کنند': 'regulate',
            'افزایش': 'upregulation', 'افزایش می‌دهد': 'upregulates', 'کاهش': 'downregulation',
            'کاهش می‌دهد': 'downregulates', 'محل': 'location', 'محل است': 'located',
            'یافت': 'found', 'یافت می‌شود': 'found', 'یافت می‌شوند': 'found'
        }
        
        # نگاشت ژن‌های مشهور و نام‌های مختلف آنها (با ترجیح کامل-نام برای پرهیز از تطبیق اشتباه TP53RK)
        famous_genes = {
            'tp53': ['TP53', 'Tumor Protein P53', 'Tumor Suppressor P53', 'P53'],
            'brca1': ['BRCA1', 'Breast Cancer 1', 'BRCA1 Gene'],
            'brca2': ['BRCA2', 'Breast Cancer 2', 'BRCA2 Gene'],
            'apoe': ['APOE', 'Apolipoprotein E', 'APOE Gene'],
            'cftr': ['CFTR', 'Cystic Fibrosis Transmembrane Conductance Regulator'],
            'mmp9': ['MMP9', 'Matrix Metallopeptidase 9'],
            'bid': ['BID', 'BH3 Interacting Domain Death Agonist'],
            'kcnq2': ['KCNQ2', 'Potassium Voltage-Gated Channel Subfamily Q Member 2'],
            'hmgb3': ['HMGB3', 'High Mobility Group Box 3']
        }
        
        # بررسی ژن‌های مشهور در متن
        text_lower = text.lower()
        for gene_key, gene_variants in famous_genes.items():
            if gene_key in text_lower:
                keywords.add(gene_key)
                # اضافه کردن نام اصلی ژن
                keywords.add(gene_variants[0])
        
        # تبدیل کلمات فارسی به انگلیسی
        for persian_word, english_word in persian_to_english.items():
            if persian_word in text:
                keywords.add(english_word)
                print(f"🔄 تبدیل فارسی به انگلیسی: '{persian_word}' -> '{english_word}'")
        
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
        
        # اضافه کردن کلمات کلیدی تخصصی
        technical_terms = [
            'cancer', 'tumor', 'malignancy', 'oncology', 'carcinoma', 'sarcoma', 
            'leukemia', 'lymphoma', 'gene', 'protein', 'dna', 'rna', 'mrna',
            'apoptosis', 'cell cycle', 'dna repair', 'mutation', 'expression',
            'regulation', 'pathway', 'signaling', 'metabolic', 'cascade'
        ]
        
        for term in technical_terms:
            if term in text_lower:
                keywords.add(term)
        
        # حذف کلمات خیلی کوتاه و عمومی
        filtered_keywords = set()
        for keyword in keywords:
            if len(keyword) >= 2 and keyword not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                filtered_keywords.add(keyword)
        
        result = sorted(filtered_keywords)
        # محدود کردن اندازه کش
        if len(self._keyword_cache) > 1024:
            self._keyword_cache.clear()
        self._keyword_cache[text] = result
        return result
    
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
                'patterns': [
                    'covary', 'covaries', 'co-vary', 'co-varies',
                    'coexpression', 'co-expression', 'coexpressed',
                    'correlated', 'correlation',
                    'هم‌واریانس', 'همواریانس', 'هم‌بروز', 'همبروز', 'هم‌تغییر', 'همتغییر'
                ],
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
            print(" تشخیص نوع سوال: علائم بیماری")
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
        
        # اضافه کردن ژن‌های اصلی به نتایج
        for gene_token, gene_node_id in gene_nodes:
            gene_name = self.G.nodes[gene_node_id]['name']
            # امتیاز بالاتر برای ژن‌های اصلی
            score = 10.0  # امتیاز بالاتر برای ژن‌های اصلی
            explanation = f"Primary gene: {gene_name}"
            results.append((gene_node_id, 0, score, explanation))
            print(f"  ✅ ژن اصلی: {gene_name} (امتیاز: {score})")
        
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
    
    # def _add_node_if_not_exists(self, node_id: str):
    #     """اضافه کردن نود به گراف اگر وجود نداشته باشد"""
    #     if not self.G.has_node(node_id):
    #         # ایجاد نود با اطلاعات پیش‌فرض
    #         self.G.add_node(node_id, name=node_id, kind='Unknown')
    #         print(f"  ➕ نود اضافه شد: {node_id}")
    
    # def _add_edge_if_not_exists(self, source: str, target: str, relation: str = 'Unknown'):
    #     """اضافه کردن یال به گراف اگر وجود نداشته باشد"""
    #     if not self.G.has_edge(source, target):
    #         self.G.add_edge(source, target, metaedge=relation, relation=relation)
    #         print(f"  ➕ یال اضافه شد: {source} → {target} ({relation})")
    
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
            'blood': 'Anatomy',
            
            # Disease (137 nodes)
            'disease': 'Disease', 'diseases': 'Disease', 'disorder': 'Disease', 'disorders': 'Disease',
            'syndrome': 'Disease', 'syndromes': 'Disease', 'cancer': 'Disease', 'cancers': 'Disease',
            'tumor': 'Disease', 'tumors': 'Disease', 'malignancy': 'Disease', 'malignancies': 'Disease',
            'diabetes': 'Disease', 'alzheimer': 'Disease', 'fibrosis': 'Disease',
            'breast cancer': 'Disease', 'lung cancer': 'Disease', 'liver cancer': 'Disease',
            'brain cancer': 'Disease', 'blood cancer': 'Disease', 'stomach cancer': 'Disease',
            
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
                # ابتدا تطبیق دقیق نام کامل ژن، سپس تطبیق‌های شامل
                for node_id, attrs in self.G.nodes(data=True):
                    if attrs.get('kind') == 'Gene' and attrs.get('name', '').upper() == 'TP53' and token_lower == 'tp53':
                        matched[token] = node_id
                        found = True
                        print(f"🔍 تطبیق ژن مشهور (قفل دقیق): '{token}' -> {attrs['name']} ({attrs.get('kind', 'Unknown')})")
                        break
                if not found:
                    for variant in gene_variants:
                        for node_id, attrs in self.G.nodes(data=True):
                            if (attrs.get('kind') == 'Gene' and 
                                variant.upper() == attrs.get('name', '').upper()):
                                matched[token] = node_id
                                found = True
                                print(f"🔍 تطبیق ژن مشهور (دقیق): '{token}' -> {attrs['name']} ({attrs.get('kind', 'Unknown')})")
                                break
                        if found:
                            break
                if not found:
                    for variant in gene_variants:
                        for node_id, attrs in self.G.nodes(data=True):
                            if (attrs.get('kind') == 'Gene' and 
                                variant.upper() in attrs.get('name', '').upper()):
                                matched[token] = node_id
                                found = True
                                print(f"🔍 تطبیق ژن مشهور (شامل): '{token}' -> {attrs['name']} ({attrs.get('kind', 'Unknown')})")
                                break
                        if found:
                            break
                    if found:
                        break
            
            # روش 2: جستجوی مستقیم بر اساس نام
            if not found:
                import re
                gene_symbol_like = bool(re.fullmatch(r"[A-Za-z0-9\-]{2,10}", token)) and sum(1 for c in token if c.isalpha() and c.isupper()) >= 2
                for node_id, attrs in self.G.nodes(data=True):
                    name = attrs.get('name', '')
                    name_lower = name.lower()
                    # اگر شبیه نماد ژنی است، برای نودهای Gene فقط تطبیق دقیق را قبول کن
                    if gene_symbol_like and attrs.get('kind') == 'Gene':
                        if name_upper := name.upper():
                            if token.upper() == name_upper:
                                matched[token] = node_id
                                found = True
                                print(f"🔍 تطبیق مستقیم دقیق ژن: '{token}' -> {name}")
                                break
                        continue  # از تطبیق‌های شامل مثل TP53RK جلوگیری کن
                    # برای سایر انواع، تطبیق شامل مجاز است
                    if token_lower in name_lower:
                        matched[token] = node_id
                        found = True
                        print(f"🔍 تطبیق مستقیم: '{token}' -> {name} ({attrs.get('kind', 'Unknown')})")
                        break
            
            # روش 3: جستجوی فازی برای کلمات مشابه (بهینه‌سازی شده با ایندکس)
            if not found and len(token) >= 3:
                # 3.1 تطبیق دقیق از ایندکس نام‌ها
                if token_lower in self._name_to_ids:
                    matched[token] = self._name_to_ids[token_lower][0]
                    found = True
                else:
                    # 3.2 شامل بودن سبک روی ورودی‌های ایندکس‌شده (محدود برای کارایی)
                    limit_scan = min(len(self._name_entries), 10000)
                    for name_lower, node_id in self._name_entries[:limit_scan]:
                        if token_lower in name_lower:
                            matched[token] = node_id
                            found = True
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
            
            # روش 5: تطبیق کلمات فارسی با نودهای مشابه
            if not found and any('\u0600' <= c <= '\u06FF' for c in token):  # کاراکترهای فارسی
                # نگاشت کلمات فارسی به انگلیسی برای تطبیق بهتر
                persian_mapping = {
                    'سرطان': 'cancer',
                    'کبد': 'liver', 
                    'مغز': 'brain',
                    'قلب': 'heart',
                    'ریه': 'lung',
                    'کلیه': 'kidney',
                    'معده': 'stomach',
                    'ماهیچه': 'muscle',
                    'استخوان': 'bone',
                    'خون': 'blood',
                    'بافت': 'tissue',
                    'اندام': 'organ',
                    'بدن': 'body',
                    'بیماری': 'disease',
                    'دارو': 'drug',
                    'آسپرین': 'aspirin',
                    'ژن': 'gene',
                    'پروتئین': 'protein',
                    'فرآیند': 'process',
                    'آپوپتوز': 'apoptosis'
                }
                
                if token in persian_mapping:
                    english_word = persian_mapping[token]
                    # جستجوی نود با نام انگلیسی
                    for node_id, attrs in self.G.nodes(data=True):
                        if english_word in attrs['name'].lower():
                            matched[token] = node_id
                            found = True
                            print(f"🔍 تطبیق فارسی-انگلیسی: '{token}' -> {attrs['name']} ({attrs.get('kind', 'Unknown')})")
                            break
            
            # روش 5: جستجوی فازی ویژه ژن‌ها با ایندکس نوع
            if not found and len(token) >= 3 and 'Gene' in self._kind_to_ids:
                for node_id in self._kind_to_ids['Gene'][: min(5000, len(self._kind_to_ids['Gene']))]:
                    attrs = self.G.nodes[node_id]
                    name_lower = attrs.get('name', '').lower()
                    if not name_lower:
                        continue
                    if (token_lower in name_lower or name_lower in token_lower or any(word in name_lower for word in token_lower.split())):
                        matched[token] = node_id
                        found = True
                        break
            
            if not found:
                print(f"❌ تطبیق نشد: '{token}'")
        
        return matched

    def _preferred_core_kinds_for_question(self, question_type: str) -> List[str]:
        mapping = {
            'biological_participation': ['Gene', 'Pathway', 'Biological Process'],
            'gene_interaction': ['Gene'],
            'disease_gene_regulation': ['Gene', 'Disease'],
            'disease_treatment': ['Disease', 'Compound'],
            'compound_gene_regulation': ['Gene', 'Compound'],
            'anatomy_expression': ['Anatomy', 'Gene'],
            'anatomy_disease': ['Disease', 'Anatomy'],
            'gene_pathway': ['Gene', 'Pathway'],
            'gene_regulation': ['Gene'],
            'gene_covariation': ['Gene'],
            'disease_symptom': ['Disease', 'Symptom'],
            'disease_similarity': ['Disease'],
        }
        return mapping.get(question_type, ['Gene', 'Disease', 'Pathway'])

    def _extract_core_nodes(self, query: str, matched_nodes: Dict[str, str], intent: Dict[str, Any]) -> List[str]:
        """انتخاب هسته‌های دقیق بر اساس سوال، نیت و تطبیق‌ها.
        قواعد:
        - برای نمادهای شبیه ژن، فقط تطبیق دقیق نام ژن به‌عنوان هسته پذیرفته می‌شود.
        - انواع هسته بر اساس نوع سوال محدود می‌شوند.
        - در صورت نبود تطبیق دقیق، از تطبیق‌های عبارتی کامل استفاده می‌شود.
        """
        ql = (query or '').lower()
        tokens = set([t.strip() for t in re.split(r"[^A-Za-z0-9]+", ql) if t.strip()])
        preferred_kinds = set(self._preferred_core_kinds_for_question(intent.get('question_type', 'general')))

        core_nodes: List[str] = []
        # مرحله 1: تطبیق دقیق نماد ژن
        for token, node_id in matched_nodes.items():
            attrs = self.G.nodes[node_id]
            name = attrs.get('name', '')
            kind = attrs.get('kind')
            # فقط انواع ترجیحی
            if kind not in preferred_kinds:
                continue
            # ژن: نیاز به تطبیق دقیق نماد
            if kind == 'Gene':
                gene_symbol_like = bool(re.fullmatch(r"[A-Za-z0-9\-]{2,10}", token)) and sum(1 for c in token if c.isalpha() and c.isupper()) >= 2
                if gene_symbol_like and token.upper() == name.upper():
                    core_nodes.append(node_id)
            else:
                # غیر ژن: تطبیق عین عبارت کامل
                if token == name.lower():
                    core_nodes.append(node_id)

        # مرحله 2: اگر هنوز خالی بود، از قید عبارت کامل در متن سوال استفاده کن
        if not core_nodes:
            for token, node_id in matched_nodes.items():
                attrs = self.G.nodes[node_id]
                kind = attrs.get('kind')
                name_lower = attrs.get('name', '').lower()
                if kind in preferred_kinds and name_lower in ql:
                    # جلوگیری از حالات حاوی پسوند/پیشوند برای ژن‌ها
                    if kind == 'Gene':
                        continue
                    core_nodes.append(node_id)

        # حذف تکراری‌ها با حفظ ترتیب
        seen = set()
        unique_core_nodes = []
        for nid in core_nodes:
            if nid not in seen:
                seen.add(nid)
                unique_core_nodes.append(nid)
        return unique_core_nodes
    
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
            # فیلتر یال‌ها برای پرهیز از نویز: فقط یال‌های دارای metaedge/relation معتبر
            for neighbor in self.G.neighbors(node):
                ed = self.G.get_edge_data(node, neighbor) or {}
                if not ed.get('metaedge') and not ed.get('relation'):
                    continue
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
                    edge_data = self.G.get_edge_data(node, neighbor) or {}
                    if relation_filter:
                        rel = (edge_data.get('relation') or edge_data.get('metaedge') or '').lower()
                        if relation_filter.lower() not in rel:
                            continue
                    # حذف یال‌های بدون متاداده
                    if not edge_data.get('relation') and not edge_data.get('metaedge'):
                        continue
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
                           max_depth: int = None, max_nodes: int = None) -> RetrievalResult:
        """بازیابی اطلاعات از گراف"""
        # استفاده از تنظیمات پیش‌فرض اگر مقدار داده نشده
        if max_depth is None:
            max_depth = self.config['max_depth']
        if max_nodes is None:
            max_nodes = self.config['max_nodes']
        """بازیابی اطلاعات از گراف"""
        print(f"🔍 بازیابی اطلاعات با روش {method}...")
        
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
            else:
                # اگر کمتر از 2 نود پیدا شد، از BFS استفاده کن
                print("⚠️ کمتر از 2 نود برای SHORTEST_PATH پیدا شد. استفاده از BFS...")
                for token, node_id in matches.items():
                    bfs_result = self.bfs_search(node_id, max_depth)
                    for node, depth in bfs_result[:max_nodes]:
                        nodes.append(GraphNode(
                            id=node,
                            name=self.G.nodes[node]['name'],
                            kind=self.G.nodes[node]['kind'],
                            depth=depth
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
            else:
                # اگر کمتر از 2 نود پیدا شد، از BFS استفاده کن
                print("⚠️ کمتر از 2 نود برای HYBRID پیدا شد. استفاده از BFS...")
                for token, node_id in matches.items():
                    bfs_result = self.bfs_search(node_id, max_depth)
                    for node, depth in bfs_result[:max_nodes]:
                        nodes.append(GraphNode(
                            id=node,
                            name=self.G.nodes[node]['name'],
                            kind=self.G.nodes[node]['kind'],
                            depth=depth
                        ))
        
        elif method == RetrievalMethod.MULTI_METHOD:
            # جستجوی چند روشی
            if len(matches) >= 1:
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
            else:
                print("⚠️ هیچ نودی برای MULTI_METHOD پیدا نشد.")
        
        elif method == RetrievalMethod.ENSEMBLE:
            # جستجوی گروهی
            if len(matches) >= 1:
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
            else:
                print("⚠️ هیچ نودی برای ENSEMBLE پیدا نشد.")
        
        elif method == RetrievalMethod.ADAPTIVE:
            # جستجوی تطبیقی با پاس دادن query
            if len(matches) >= 1:
                node_ids = list(matches.values())
                adaptive_result = self.adaptive_search(node_ids, max_depth, query)
                for node, depth, method in adaptive_result[:max_nodes]:
                    nodes.append(GraphNode(
                        id=node,
                        name=self.G.nodes[node]['name'],
                        kind=self.G.nodes[node]['kind'],
                        depth=depth
                    ))
            else:
                print("⚠️ هیچ نودی برای ADAPTIVE پیدا نشد.")
        
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
                        spaths = self.get_shortest_paths(node_ids[i], node_ids[j])
                        if not spaths:
                            continue
                        paths.extend(spaths)
                        # افزودن نودهای مسیر
                        for path in spaths:
                            for k, pid in enumerate(path):
                                if pid not in [n.id for n in nodes]:
                                    nodes.append(GraphNode(
                                        id=pid,
                                        name=self.G.nodes[pid]['name'],
                                        kind=self.G.nodes[pid]['kind'],
                                        depth=k
                                    ))
                            # افزودن یال‌های مسیر
                            for k in range(len(path) - 1):
                                ed = self.G.get_edge_data(path[k], path[k+1])
                                if ed:
                                    edges.append(GraphEdge(
                                        source=path[k],
                                        target=path[k+1],
                                        relation=ed.get('metaedge', 'related'),
                                        weight=ed.get('weight', 1.0)
                                    ))
            
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
        
        elif method == RetrievalMethod.KG_SEARCH:
            # جستجوی دانش‌گراف (Knowledge Graph Search)
            print("🔍 استفاده از الگوریتم KG_SEARCH")
            # Intent-aware: اگر سوال هم‌واریانس ژن است، فقط GcG با خروجی مینیمال را برگردان
            intent = self.analyze_question_intent(query)
            if intent.get('question_type') == 'gene_covariation':
                # core lock: یافتن هسته ژن از توکن‌ها
                matched_nodes = self.match_tokens_to_nodes(intent.get('keywords', []))
                core_nodes = self._extract_core_nodes(query, matched_nodes, intent)
                core_gene = None
                for nid in core_nodes:
                    if self.G.nodes[nid].get('kind') == 'Gene':
                        core_gene = nid
                        break
                if core_gene is None and matched_nodes:
                    # fallback ساده
                    for nid in matched_nodes.values():
                        if self.G.nodes[nid].get('kind') == 'Gene':
                            core_gene = nid
                            break
                if core_gene:
                    # فقط همسایه‌های GcG (Gene–covaries–Gene)
                    covary_genes = []
                    for nbr in self.G.neighbors(core_gene):
                        ed = self.G.get_edge_data(core_gene, nbr) or {}
                        if (ed.get('metaedge') or ed.get('relation')) == 'GcG' and self.G.nodes[nbr].get('kind') == 'Gene':
                            covary_genes.append(nbr)
                    # ساخت خروجی مینیمال: core + همسایه‌های ژنی، یال‌ها فقط GcG، مسیرهای یک‌پرش
                    nodes.append(GraphNode(id=core_gene,
                                           name=self.G.nodes[core_gene]['name'],
                                           kind=self.G.nodes[core_gene]['kind'],
                                           depth=0,
                                           score=1.0))
                    for gid in covary_genes[:max_nodes-1]:
                        nodes.append(GraphNode(id=gid,
                                               name=self.G.nodes[gid]['name'],
                                               kind=self.G.nodes[gid]['kind'],
                                               depth=1,
                                               score=1.0))
                        edges.append(GraphEdge(source=core_gene,
                                               target=gid,
                                               relation='GcG',
                                               weight=(self.G.get_edge_data(core_gene, gid) or {}).get('weight', 1.0)))
                        paths.append([core_gene, gid])
                else:
                    # اگر ژن پیدا نشد، به نسخه traceable جدید سوئیچ کن
                    hits, _ = self.kgsearch_traceable(query, top_k=min(10, max_nodes))
                    # تبدیل hits به nodes/edges/paths
                    nid_set = set()
                    for h in hits:
                        seq = h.get('path', [])
                        last_node = None
                        for elem in seq:
                            if 'id' in elem:
                                nid = elem['id']
                                nid_set.add(nid)
                                if not any(n.id == nid for n in nodes):
                                    nodes.append(GraphNode(id=nid,
                                                           name=self.G.nodes[nid].get('name', nid),
                                                           kind=self.G.nodes[nid].get('kind', 'Unknown'),
                                                           depth=0))
                                last_node = nid
                            elif 'edge_id' in elem and last_node is not None:
                                # edge follows between last_node and next node in sequence; will be added when next node arrives
                                pass
                        # بازسازی مسیرهای کوتاه
                        path_nodes = [e['id'] for e in seq if 'id' in e]
                        if len(path_nodes) >= 2:
                            paths.append(path_nodes)
                            for i in range(len(path_nodes)-1):
                                ed = self.G.get_edge_data(path_nodes[i], path_nodes[i+1]) or {}
                                edges.append(GraphEdge(source=path_nodes[i],
                                                       target=path_nodes[i+1],
                                                       relation=ed.get('metaedge', ed.get('relation', 'related')),
                                                       weight=ed.get('weight', 1.0)))
            else:
                # مسیر پیش‌فرض: نسخه traceable با محدودیت Schema
                hits, _ = self.kgsearch_traceable(query, top_k=min(10, max_nodes))
                # تبدیل hits به nodes/edges/paths
                nid_set = set()
                for h in hits:
                    seq = h.get('path', [])
                    last_node = None
                    current_path = []
                    for elem in seq:
                        if 'id' in elem:
                            nid = elem['id']
                            current_path.append(nid)
                            if nid not in nid_set:
                                nid_set.add(nid)
                                nodes.append(GraphNode(id=nid,
                                                       name=self.G.nodes[nid].get('name', nid),
                                                       kind=self.G.nodes[nid].get('kind', 'Unknown'),
                                                       depth=0))
                            last_node = nid
                        elif 'edge_id' in elem and last_node is not None:
                            pass
                    if len(current_path) >= 2:
                        paths.append(current_path)
                        for i in range(len(current_path)-1):
                            ed = self.G.get_edge_data(current_path[i], current_path[i+1]) or {}
                            edges.append(GraphEdge(source=current_path[i],
                                                   target=current_path[i+1],
                                                   relation=ed.get('metaedge', ed.get('relation', 'related')),
                                                   weight=ed.get('weight', 1.0)))
        
        elif method == RetrievalMethod.N_HOP_RETRIEVAL:
            # بازیابی چندمرحله‌ای
            print("🔍 استفاده از الگوریتم N_HOP_RETRIEVAL")
            multi_hop_result = self.multi_hop_search(query, max_depth)
            
            # تبدیل نتایج به GraphNode
            for node_id, depth, score, reason, path in multi_hop_result[:max_nodes]:
                nodes.append(GraphNode(
                    id=node_id,
                    name=self.G.nodes[node_id]['name'],
                    kind=self.G.nodes[node_id]['kind'],
                    depth=depth,
                    score=score
                ))
                if path:
                    paths.append(path)
            
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
        
        elif method == RetrievalMethod.PAGERANK_BASED:
            # جستجو بر اساس PageRank
            print("🔍 استفاده از الگوریتم PAGERANK_BASED")
            try:
                import networkx as nx
                # محاسبه PageRank
                pagerank_scores = nx.pagerank(self.G, alpha=0.85, max_iter=100)
                
                # مرتب‌سازی نودها بر اساس PageRank
                sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
                
                # انتخاب نودهای برتر
                for node_id, score in sorted_nodes[:max_nodes]:
                    if node_id in self.G.nodes:
                        nodes.append(GraphNode(
                            id=node_id,
                            name=self.G.nodes[node_id]['name'],
                            kind=self.G.nodes[node_id]['kind'],
                            depth=0,
                            score=score
                        ))
                
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
            except Exception as e:
                print(f"⚠️ خطا در محاسبه PageRank: {e}")
                # استفاده از روش جایگزین
                intelligent_result = self.intelligent_semantic_search(query, max_depth)
                for node_id, depth, score, reason in intelligent_result[:max_nodes]:
                    nodes.append(GraphNode(
                        id=node_id,
                        name=self.G.nodes[node_id]['name'],
                        kind=self.G.nodes[node_id]['kind'],
                        depth=depth,
                        score=score
                    ))
        
        elif method == RetrievalMethod.SEMANTIC_SIMILARITY:
            # جستجو بر اساس شباهت معنایی
            print("🔍 استفاده از الگوریتم SEMANTIC_SIMILARITY")
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
                        spaths = self.get_shortest_paths(node_ids[i], node_ids[j])
                        if not spaths:
                            continue
                        paths.extend(spaths)
                        # افزودن نودهای مسیر
                        for path in spaths:
                            for k, pid in enumerate(path):
                                if pid not in [n.id for n in nodes]:
                                    nodes.append(GraphNode(
                                        id=pid,
                                        name=self.G.nodes[pid]['name'],
                                        kind=self.G.nodes[pid]['kind'],
                                        depth=k
                                    ))
                            # افزودن یال‌های مسیر
                            for k in range(len(path) - 1):
                                ed = self.G.get_edge_data(path[k], path[k+1])
                                if ed:
                                    edges.append(GraphEdge(
                                        source=path[k],
                                        target=path[k+1],
                                        relation=ed.get('metaedge', 'related'),
                                        weight=ed.get('weight', 1.0)
                                    ))
            
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
        
        elif method == RetrievalMethod.COMMUNITY_DETECTION:
            # تشخیص جامعه‌ها
            print("🔍 استفاده از الگوریتم COMMUNITY_DETECTION")
            try:
                import networkx as nx
                from community import community_louvain
                
                # تشخیص جامعه‌ها با الگوریتم Louvain
                communities = community_louvain.best_partition(self.G)
                
                # گروه‌بندی نودها بر اساس جامعه
                community_nodes = {}
                for node_id, community_id in communities.items():
                    if community_id not in community_nodes:
                        community_nodes[community_id] = []
                    community_nodes[community_id].append(node_id)
                
                # انتخاب نودهای جامعه‌های مختلف
                selected_nodes = []
                for community_id, node_list in community_nodes.items():
                    # انتخاب نودهای برتر از هر جامعه
                    for node_id in node_list[:max(1, max_nodes // len(community_nodes))]:
                        if node_id in self.G.nodes:
                            selected_nodes.append(node_id)
                
                # تبدیل به GraphNode
                for node_id in selected_nodes[:max_nodes]:
                    nodes.append(GraphNode(
                        id=node_id,
                        name=self.G.nodes[node_id]['name'],
                        kind=self.G.nodes[node_id]['kind'],
                        depth=0,
                        score=1.0
                    ))
                
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
            except ImportError:
                print("⚠️ کتابخانه community در دسترس نیست، استفاده از روش جایگزین")
                intelligent_result = self.intelligent_semantic_search(query, max_depth)
                for node_id, depth, score, reason in intelligent_result[:max_nodes]:
                    nodes.append(GraphNode(
                        id=node_id,
                        name=self.G.nodes[node_id]['name'],
                        kind=self.G.nodes[node_id]['kind'],
                        depth=depth,
                        score=score
                    ))
            except Exception as e:
                print(f"⚠️ خطا در تشخیص جامعه‌ها: {e}")
                intelligent_result = self.intelligent_semantic_search(query, max_depth)
                for node_id, depth, score, reason in intelligent_result[:max_nodes]:
                    nodes.append(GraphNode(
                        id=node_id,
                        name=self.G.nodes[node_id]['name'],
                        kind=self.G.nodes[node_id]['kind'],
                        depth=depth,
                        score=score
                    ))
        
        elif method == RetrievalMethod.ENTITY_RESOLUTION:
            # حل موجودیت‌ها
            print("🔍 استفاده از الگوریتم ENTITY_RESOLUTION")
            try:
                if NEW_MODULES_AVAILABLE:
                    # استفاده از ماژول جدید EntityResolution
                    entity_resolver = EntityResolution()
                    resolved_entities = entity_resolver.resolve_entities(query)
                    
                    # تطبیق موجودیت‌های حل شده با نودهای گراف
                    for entity in resolved_entities:
                        for node_id, node_attrs in self.G.nodes(data=True):
                            if entity.lower() in node_attrs['name'].lower():
                                nodes.append(GraphNode(
                                    id=node_id,
                                    name=node_attrs['name'],
                                    kind=node_attrs['kind'],
                                    depth=0,
                                    score=1.0
                                ))
                                break
                else:
                    # استفاده از روش جایگزین
                    intelligent_result = self.intelligent_semantic_search(query, max_depth)
                    for node_id, depth, score, reason in intelligent_result[:max_nodes]:
                        nodes.append(GraphNode(
                            id=node_id,
                            name=self.G.nodes[node_id]['name'],
                            kind=self.G.nodes[node_id]['kind'],
                            depth=depth,
                            score=score
                        ))
            except Exception as e:
                print(f"⚠️ خطا در حل موجودیت‌ها: {e}")
                intelligent_result = self.intelligent_semantic_search(query, max_depth)
                for node_id, depth, score, reason in intelligent_result[:max_nodes]:
                    nodes.append(GraphNode(
                        id=node_id,
                        name=self.G.nodes[node_id]['name'],
                        kind=self.G.nodes[node_id]['kind'],
                        depth=depth,
                        score=score
                    ))
        
        elif method == RetrievalMethod.HYBRID_NEW:
            # ترکیب روش‌های جدید با قفل موجودیت (Entity Locking) و فیلتر نویز
            print("🔍 استفاده از الگوریتم HYBRID_NEW")

            # تحلیل نیت سوال و استخراج توکن‌ها برای قفل موجودیت
            intent = self.analyze_question_intent(query)
            keywords = self.extract_keywords(query)
            matched_nodes = self.match_tokens_to_nodes(keywords)

            # تشخیص سناریوی ژن-سرطان و اعمال عمق محدودتر
            is_gene_cancer = self._is_gene_cancer_question(query, matched_nodes)
            local_max_depth = 2 if is_gene_cancer else max_depth

            # انتخاب دقیق هسته‌ها متناسب با نوع سوال و تطبیق‌ها
            try:
                core_candidates = self._extract_core_nodes(query, matched_nodes, intent)
            except Exception:
                core_candidates = []
            core_node_id = core_candidates[0] if core_candidates else None

            # ترکیب چندین روش
            methods_results = []

            # 0. اگر نود هسته یافت شد، آن را با امتیاز بالا اضافه کن
            if core_node_id is not None:
                methods_results.append((core_node_id, 0, 100.0, 'Core Entity Lock'))

            # 1. جستجوی معنایی هوشمند (با عمق محلی)
            intelligent_result = self.intelligent_semantic_search(query, local_max_depth)
            methods_results.extend(intelligent_result)

            # 2. جستجوی چندمرحله‌ای (با عمق محلی)
            multi_hop_result = self.multi_hop_search(query, local_max_depth)
            for node_id, depth, score, reason, path in multi_hop_result:
                methods_results.append((node_id, depth, score, reason))

            # 3. PageRank (اگر در دسترس باشد)
            try:
                import networkx as nx
                pagerank_scores = nx.pagerank(self.G, alpha=0.85, max_iter=100)
                sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
                for node_id, score in sorted_nodes[:max_nodes//3]:
                    if node_id in self.G.nodes:
                        methods_results.append((node_id, 0, score, 'PageRank'))
            except Exception:
                pass

            # بازنمره‌دهی چندمعیاره: نزدیکی به نود هسته + امتیاز پایه
            if core_node_id is not None:
                for idx in range(len(methods_results)):
                    node_id, depth, score, reason = methods_results[idx]
                    try:
                        dist = nx.shortest_path_length(self.G, core_node_id, node_id)
                        if dist == 0:
                            score += 20.0
                        elif dist <= 2:
                            score += 5.0
                        else:
                            score += 0.0
                    except Exception:
                        pass
                    methods_results[idx] = (node_id, depth, score, reason)

            # ترکیب و مرتب‌سازی نتایج
            unique_results = {}
            for node_id, depth, score, reason in methods_results:
                if node_id not in unique_results or score > unique_results[node_id][2]:
                    unique_results[node_id] = (node_id, depth, score, reason)

            final_results = sorted(unique_results.values(), key=lambda x: x[2], reverse=True)

            # تبدیل نتایج به GraphNode
            for node_id, depth, score, reason in final_results[:max_nodes]:
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
                if core_node_id is not None and core_node_id in node_ids:
                    # فقط مسیرهای از نود هسته به سایر نودها برای کاهش نویز
                    for nid in node_ids:
                        if nid != core_node_id:
                            paths.extend(self.get_shortest_paths(core_node_id, nid))
                else:
                    for i in range(len(node_ids)):
                        for j in range(i+1, len(node_ids)):
                            paths.extend(self.get_shortest_paths(node_ids[i], node_ids[j]))

            # یافتن یال‌های مرتبط با فیلتر نویز (حذف DrD/CrC مگر سوال شباهت بیماری‌ها باشد)
            disease_similarity = intent.get('question_type') == 'disease_similarity'
            for node in nodes:
                for neighbor in self.G.neighbors(node.id):
                    if any(n.id == neighbor for n in nodes):
                        edge_data = self.G.get_edge_data(node.id, neighbor)
                        if edge_data:
                            metaedge = edge_data.get('metaedge', 'related')
                            if not disease_similarity and metaedge in ['DrD', 'CrC']:
                                continue
                            edges.append(GraphEdge(
                                source=node.id,
                                target=neighbor,
                                relation=metaedge,
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
            method=str(method),
            query=query
        )
        # تولید متن زمینه بهبود یافته
        if self.context_generator:
            context_text = self.context_generator.create_enhanced_context_text(retrieval_result, context_type="INTELLIGENT")
        else:
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
        """
        تابع قدیمی تولید متن زمینه - استفاده نمی‌شود
        برای تولید متن زمینه بهبود یافته از EnhancedContextGenerator استفاده کنید
        """
        return "برای استفاده از سیستم بهبود یافته، از IntegratedGraphRAGService استفاده کنید"
    
    def _enrich_retrieved_data(self, nodes: List[GraphNode], edges: List[GraphEdge], query: str) -> Dict[str, Any]:
        """
        تابع قدیمی غنی‌سازی داده‌ها - استفاده نمی‌شود
        """
        return {}
    
    def _get_anatomy_significance(self, anatomy_name: str) -> str:
        """
        تابع قدیمی - استفاده نمی‌شود
        """
        return ""
    
    def _create_biological_context(self, enriched_data: Dict, query: str) -> str:
        """
        تابع قدیمی - استفاده نمی‌شود
        """
        return ""
    
    def _create_enhanced_context_text(self, retrieval_result: RetrievalResult) -> str:
        """سازگاری: اگر ژنراتور در دسترس نبود، متن ساده بساز."""
        parts = []
        if retrieval_result.nodes:
            parts.append("نودها:")
            parts.extend([f"• {n.name} ({n.kind})" for n in retrieval_result.nodes[:10]])
        if retrieval_result.edges:
            parts.append("\nروابط:")
            parts.extend([f"• {e.source} → {e.target} ({e.relation})" for e in retrieval_result.edges[:10]])
        return "\n".join(parts) if parts else "اطلاعات کافی یافت نشد."

    def _create_advanced_context_text(self, retrieval_result: RetrievalResult) -> str:
        """
        تابع قدیمی تولید متن زمینه پیشرفته - استفاده نمی‌شود
        """
        return "برای استفاده از سیستم بهبود یافته، از IntegratedGraphRAGService استفاده کنید"

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
                    retrieval_data['compound'].append({
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
        context_parts.append(f"**Query:** {query}")
        context_parts.append("")
        
        # 2. خلاصه آماری دقیق
        total_genes_in_graph = 14010  # تعداد کل ژن‌ها در Hetionet
        primary_genes = len(retrieval_data['primary_genes'])
        secondary_genes = len(retrieval_data['secondary_genes'])
        total_found = primary_genes + secondary_genes
        
        context_parts.append("**Graph Summary:**")
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
            context_parts.append("**Key Results:**")
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
            context_parts.append("**Related Biological Processes:**")
            for process in retrieval_data['biological_processes'][:3]:
                context_parts.append(f"• {process['name']}")
            context_parts.append("")
        
        # 5. مسیرهای زیستی مرتبط
        if retrieval_data['pathways']:
            context_parts.append("**Related Pathways:**")
            for pathway in retrieval_data['pathways'][:3]:
                context_parts.append(f"• {pathway['name']}")
            context_parts.append("")
        
        # 6. بیماری‌های مرتبط
        if retrieval_data['diseases']:
            context_parts.append("**Related Diseases:**")
            for disease in retrieval_data['diseases'][:3]:
                context_parts.append(f"• {disease['name']}")
            context_parts.append("")
        
        # 7. داروهای مرتبط
        if retrieval_data['compound']:
            context_parts.append("**Related Drugs/Compounds:**")
            for drug in retrieval_data['compound'][:3]:
                context_parts.append(f"• {drug['name']}")
            context_parts.append("")
        
        # 8. مسیرهای ترکیبی بیماری→بافت→ژن (برای سوالات مربوط به اثر بیماری بر بافت‌ها)
        if retrieval_data.get('tissue_disease_paths'):
            context_parts.append("**Disease-Tissue-Gene Pathways:**")
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
            context_parts.append("**Treatment-Disease-Gene Pathways:**")
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
        context_parts.append("**Instructions:** Analyze biological relevance and clinical importance of these genes.")
        
        # حذف ایموجی‌ها از متن نهایی
        final_text = "\n".join(context_parts)
        return remove_emojis(final_text)
    
    def test_targeted_retrieval(self, query: str) -> Dict[str, Any]:
        """
        تست بازیابی هدفمند و نمایش نتایج
        """
        print(f"تست بازیابی هدفمند برای سوال: {query}")
        print("=" * 60)
        
        # تحلیل سوال
        intent = self.analyze_question_intent(query)
        print(f"نوع سوال تشخیص داده شده: {intent.get('question_type', 'unknown')}")
        
        # بازیابی هدفمند
        retrieval_data = self._targeted_retrieval_for_question(query, intent)
        
        # نمایش نتایج
        print(f"\nنتایج بازیابی:")
        print(f"• ژن‌های اصلی: {len(retrieval_data['primary_genes'])}")
        print(f"• ژن‌های ثانویه: {len(retrieval_data['secondary_genes'])}")
        print(f"• فرآیندهای زیستی: {len(retrieval_data['biological_processes'])}")
        print(f"• مسیرهای زیستی: {len(retrieval_data['pathways'])}")
        print(f"• بیماری‌ها: {len(retrieval_data['diseases'])}")
        print(f"• داروها: {len(retrieval_data['compound'])}")
        print(f"• بافت‌ها: {len(retrieval_data['anatomy'])}")
        
        # نمایش ژن‌های اصلی با جزئیات
        if retrieval_data['primary_genes']:
            print(f"\nژن‌های اصلی یافت شده:")
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
        print(f"• داروها: {len(retrieval_data['compound'])}")
        
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
                       model: GenerationModel, text_generation_type: str = 'INTELLIGENT') -> GenerationResult:
        """تولید پاسخ بر اساس نتایج بازیابی"""
        print(f"🤖 تولید پاسخ با مدل {model.value} و نوع {text_generation_type}...")
        # اطمینان از آماده بودن PageRank برای استفاده در امتیازدهی ضمنی
        self._ensure_pagerank()
        
        # انتخاب نوع تولید متن
        if text_generation_type == 'SIMPLE':
            # استفاده از روش‌های ساده
            if model == GenerationModel.GENERAL_SIMPLE:
                answer = self.general_simple_generation(retrieval_result)
                confidence = 0.8
            elif model == GenerationModel.SIMPLE:
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
            # OpenAI GPT Models
            elif model in [GenerationModel.OPENAI_GPT_4O, GenerationModel.OPENAI_GPT_4O_MINI, 
                          GenerationModel.OPENAI_GPT_4_TURBO, GenerationModel.OPENAI_GPT_4,
                          GenerationModel.OPENAI_GPT_3_5_TURBO, GenerationModel.OPENAI_GPT_3_5_TURBO_16K,
                          GenerationModel.OPENAI_GPT]:  # سازگاری با مدل قدیمی
                answer = self.openai_gpt_generation(retrieval_result, model)
                confidence = 0.95
            # Anthropic Claude Models
            elif model in [GenerationModel.ANTHROPIC_CLAUDE_3_5_SONNET, GenerationModel.ANTHROPIC_CLAUDE_3_5_HAIKU,
                          GenerationModel.ANTHROPIC_CLAUDE_3_OPUS, GenerationModel.ANTHROPIC_CLAUDE_3_SONNET,
                          GenerationModel.ANTHROPIC_CLAUDE_3_HAIKU, GenerationModel.ANTHROPIC_CLAUDE]:  # سازگاری
                answer = self.anthropic_claude_generation(retrieval_result, model)
                confidence = 0.94
            # Google Gemini Models
            elif model in [GenerationModel.GOOGLE_GEMINI_1_5_PRO, GenerationModel.GOOGLE_GEMINI_1_5_FLASH,
                          GenerationModel.GOOGLE_GEMINI_1_0_PRO, GenerationModel.GOOGLE_GEMINI_1_0_FLASH,
                          GenerationModel.GOOGLE_GEMINI]:  # سازگاری
                answer = self.google_gemini_generation(retrieval_result, model)
                confidence = 0.93
            # سایر مدل‌های پیشرفته
            elif model == GenerationModel.META_LLAMA_3_1:
                answer = self.meta_llama_generation(retrieval_result)
                confidence = 0.91
            elif model == GenerationModel.MISTRAL_AI:
                answer = self.mistral_ai_generation(retrieval_result)
                confidence = 0.90
            elif model == GenerationModel.COHERE_COMMAND:
                answer = self.cohere_command_generation(retrieval_result)
                confidence = 0.89
            elif model == GenerationModel.PERPLEXITY_SONAR:
                answer = self.perplexity_sonar_generation(retrieval_result)
                confidence = 0.88
            else:
                answer = "متأسفانه مدل انتخاب شده در دسترس نیست."
                confidence = 0.0
        else:  # INTELLIGENT
            # استفاده از روش‌های هوشمند و تخصصی (همان روش‌های عادی با متن زمینه هوشمند)
            if model == GenerationModel.GENERAL_SIMPLE:
                answer = self.general_simple_generation(retrieval_result)
                confidence = 0.85
            elif model == GenerationModel.SIMPLE:
                answer = self.simple_template_generation(retrieval_result)
                confidence = 0.8
            elif model == GenerationModel.GPT_SIMULATION:
                answer = self.gpt_simulation_generation(retrieval_result)
                confidence = 0.9
            elif model == GenerationModel.CUSTOM:
                answer = self.custom_generation(retrieval_result)
                confidence = 0.95
            elif model == GenerationModel.HUGGINGFACE:
                answer = self.huggingface_generation(retrieval_result)
                confidence = 0.92
            # OpenAI GPT Models (Intelligent)
            elif model in [GenerationModel.OPENAI_GPT_4O, GenerationModel.OPENAI_GPT_4O_MINI, 
                          GenerationModel.OPENAI_GPT_4_TURBO, GenerationModel.OPENAI_GPT_4,
                          GenerationModel.OPENAI_GPT_3_5_TURBO, GenerationModel.OPENAI_GPT_3_5_TURBO_16K,
                          GenerationModel.OPENAI_GPT]:  # سازگاری با مدل قدیمی
                answer = self.openai_gpt_generation(retrieval_result, model)
                confidence = 0.97
            # Anthropic Claude Models (Intelligent)
            elif model in [GenerationModel.ANTHROPIC_CLAUDE_3_5_SONNET, GenerationModel.ANTHROPIC_CLAUDE_3_5_HAIKU,
                          GenerationModel.ANTHROPIC_CLAUDE_3_OPUS, GenerationModel.ANTHROPIC_CLAUDE_3_SONNET,
                          GenerationModel.ANTHROPIC_CLAUDE_3_HAIKU, GenerationModel.ANTHROPIC_CLAUDE]:  # سازگاری
                answer = self.anthropic_claude_generation(retrieval_result, model)
                confidence = 0.96
            # Google Gemini Models (Intelligent)
            elif model in [GenerationModel.GOOGLE_GEMINI_1_5_PRO, GenerationModel.GOOGLE_GEMINI_1_5_FLASH,
                          GenerationModel.GOOGLE_GEMINI_1_0_PRO, GenerationModel.GOOGLE_GEMINI_1_0_FLASH,
                          GenerationModel.GOOGLE_GEMINI]:  # سازگاری
                answer = self.google_gemini_generation(retrieval_result, model)
                confidence = 0.95
            # سایر مدل‌های پیشرفته (Intelligent)
            elif model == GenerationModel.META_LLAMA_3_1:
                answer = self.meta_llama_generation(retrieval_result)
                confidence = 0.93
            elif model == GenerationModel.MISTRAL_AI:
                answer = self.mistral_ai_generation(retrieval_result)
                confidence = 0.92
            elif model == GenerationModel.COHERE_COMMAND:
                answer = self.cohere_command_generation(retrieval_result)
                confidence = 0.91
            elif model == GenerationModel.PERPLEXITY_SONAR:
                answer = self.perplexity_sonar_generation(retrieval_result)
                confidence = 0.90
            else:
                answer = "متأسفانه مدل انتخاب شده در دسترس نیست."
                confidence = 0.0
        
        # به‌روزرسانی context_text بر اساس نوع تولید متن
        if text_generation_type == 'SIMPLE':
            retrieval_result.context_text = self._create_simple_context_text(retrieval_result)
        elif text_generation_type == 'ADVANCED':
            retrieval_result.context_text = self._create_advanced_context_text(retrieval_result)
        elif text_generation_type == 'SCIENTIFIC_ANALYTICAL':
            retrieval_result.context_text = self._create_scientific_analytical_context(retrieval_result)
        elif text_generation_type == 'NARRATIVE_DESCRIPTIVE':
            retrieval_result.context_text = self._create_narrative_context(retrieval_result)
        elif text_generation_type == 'DATA_DRIVEN':
            retrieval_result.context_text = self._create_data_driven_context(retrieval_result)
        elif text_generation_type == 'STEP_BY_STEP':
            retrieval_result.context_text = self._create_step_by_step_context(retrieval_result)
        elif text_generation_type == 'CONCISE_DIRECT':
            retrieval_result.context_text = self._create_compact_direct_context(retrieval_result)
        else:  # INTELLIGENT
            retrieval_result.context_text = self._create_intelligent_context_text(retrieval_result)
        
        return GenerationResult(
            answer=answer,
            model=model.value,
            context_used=retrieval_result.context_text,
            confidence=confidence
        )
    
    def general_simple_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ ساده و عمومی برای همه نوع سوالات"""
        query = retrieval_result.query
        context = retrieval_result.context_text
        method = retrieval_result.method
        
        # اگر بدون بازیابی باشد
        if method == "بدون بازیابی (فقط مدل)":
            return f"""🤖 **پاسخ عمومی به سوال شما:**

**سوال:** {query}

بر اساس دانش عمومی در حوزه زیست‌پزشکی، پاسخ شما به شرح زیر است:

{self._generate_general_knowledge_answer(query)}

---
💡 **نکته:** این پاسخ بر اساس دانش عمومی مدل تولید شده است. برای اطلاعات دقیق‌تر و مبتنی بر داده‌های گراف، از روش‌های بازیابی استفاده کنید."""

        # اگر با بازیابی باشد
        if not context or context.strip() == "":
            return f"""❌ **اطلاعات کافی یافت نشد**

**سوال:** {query}

متأسفانه اطلاعات مرتبط با سوال شما در گراف دانش یافت نشد. 

💡 **پیشنهادات:**
• کلمات کلیدی را تغییر دهید
• از روش بازیابی دیگری استفاده کنید
• سوال را به شکل دیگری مطرح کنید"""

        # ایجاد متن زمینه ساده بهینه شده
        simple_context = self._create_simple_context_text(retrieval_result)
        
        # تحلیل نوع سوال برای پاسخ مناسب
        query_lower = query.lower()
        
        # تشخیص نوع سوال
        if any(word in query_lower for word in ["cancer", "tumor", "malignant"]):
            return self._generate_cancer_related_answer(retrieval_result)
        elif any(word in query_lower for word in ["gene", "protein", "express"]):
            return self._generate_gene_related_answer(retrieval_result)
        elif any(word in query_lower for word in ["drug", "medicine", "treatment", "therapy"]):
            return self._generate_drug_related_answer(retrieval_result)
        elif any(word in query_lower for word in ["disease", "disorder", "condition"]):
            return self._generate_disease_related_answer(retrieval_result)
        elif any(word in query_lower for word in ["tissue", "organ", "anatomy", "heart", "brain", "liver"]):
            return self._generate_tissue_related_answer(retrieval_result)
        else:
            return self._generate_general_structured_answer(retrieval_result)

    def _create_simple_context_text(self, retrieval_result: RetrievalResult) -> str:
        """ایجاد متن زمینه ساده و عمومی"""
        nodes = retrieval_result.nodes
        edges = retrieval_result.edges
        query = retrieval_result.query
        
        if not nodes:
            return "اطلاعات کافی یافت نشد."
        
        # گروه‌بندی نودها بر اساس نوع
        gene_nodes = [n for n in nodes if n.kind == 'Gene']
        disease_nodes = [n for n in nodes if n.kind == 'Disease']
        drug_nodes = [n for n in nodes if n.kind == 'Drug']
        anatomy_nodes = [n for n in nodes if n.kind == 'Anatomy']
        
        context_parts = []
        
        if gene_nodes:
            gene_names = [n.name for n in gene_nodes]
            context_parts.append(f"ژن‌های مرتبط: {', '.join(gene_names)}")
        
        if disease_nodes:
            disease_names = [n.name for n in disease_nodes]
            context_parts.append(f"بیماری‌های مرتبط: {', '.join(disease_names)}")
        
        if drug_nodes:
            drug_names = [n.name for n in drug_nodes]
            context_parts.append(f"داروهای مرتبط: {', '.join(drug_names)}")
        
        if anatomy_nodes:
            anatomy_names = [n.name for n in anatomy_nodes]
            context_parts.append(f"بافت‌های مرتبط: {', '.join(anatomy_names)}")
        
        # اضافه کردن یال‌های مهم
        if edges:
            important_edges = edges[:5]  # حداکثر 5 یال
            edge_descriptions = []
            for edge in important_edges:
                sdisp = self._display_node(edge.source)
                tdisp = self._display_node(edge.target)
                edge_descriptions.append(f"{sdisp} → {tdisp} ({edge.relation})")
            if edge_descriptions:
                context_parts.append(f"رابطه‌های مهم: {'; '.join(edge_descriptions)}")
        
        context_text = "\n".join(context_parts) if context_parts else "اطلاعات کافی یافت نشد."
        return remove_emojis(context_text)

    def _create_scientific_analytical_context(self, retrieval_result: RetrievalResult) -> str:
        """ایجاد متن زمینه علمی-تحلیلی (تحقیقاتی)"""
        nodes = retrieval_result.nodes
        edges = retrieval_result.edges
        query = retrieval_result.query
        
        if not nodes:
            return "اطلاعات کافی برای تحلیل علمی یافت نشد."
        
        # تحلیل علمی پیشرفته
        gene_nodes = [n for n in nodes if n.kind == 'Gene']
        disease_nodes = [n for n in nodes if n.kind == 'Disease']
        
        context_parts = []
        
        if gene_nodes and disease_nodes:
            context_parts.append("تحلیل علمی رابطه ژن-بیماری:")
            for gene in gene_nodes[:3]:  # حداکثر 3 ژن
                for disease in disease_nodes[:3]:  # حداکثر 3 بیماری
                    context_parts.append(f"• ژن {gene.name} با بیماری {disease.name} مرتبط است")
        
        # تحلیل یال‌ها
        if edges:
            context_parts.append("\nتحلیل روابط:")
            for edge in edges[:5]:
                sdisp = self._display_node(edge.source)
                tdisp = self._display_node(edge.target)
                context_parts.append(f"• {sdisp} {edge.relation} {tdisp}")
        
        context_text = "\n".join(context_parts) if context_parts else "اطلاعات کافی برای تحلیل علمی یافت نشد."
        return remove_emojis(context_text)

    def _create_narrative_context(self, retrieval_result: RetrievalResult) -> str:
        """ایجاد متن زمینه روایی (ساده و توصیفی)"""
        nodes = retrieval_result.nodes
        edges = retrieval_result.edges
        query = retrieval_result.query
        
        if not nodes:
            return "اطلاعات کافی برای توصیف یافت نشد."
        
        # ایجاد داستان روایی
        gene_nodes = [n for n in nodes if n.kind == 'Gene']
        disease_nodes = [n for n in nodes if n.kind == 'Disease']
        drug_nodes = [n for n in nodes if n.kind == 'Drug']
        
        narrative_parts = []
        
        if gene_nodes and disease_nodes:
            gene_names = [n.name for n in gene_nodes[:2]]
            disease_names = [n.name for n in disease_nodes[:2]]
            narrative_parts.append(f"ژن‌های {', '.join(gene_names)} در ارتباط با بیماری‌های {', '.join(disease_names)} هستند.")
        
        if drug_nodes:
            drug_names = [n.name for n in drug_nodes[:2]]
            narrative_parts.append(f"داروهای {', '.join(drug_names)} برای درمان این بیماری‌ها استفاده می‌شوند.")
        
        if edges:
            narrative_parts.append("این روابط نشان‌دهنده شبکه پیچیده‌ای از تعاملات زیستی است.")
        
        context_text = " ".join(narrative_parts) if narrative_parts else "اطلاعات کافی برای توصیف یافت نشد."
        return remove_emojis(context_text)

    def _create_data_driven_context(self, retrieval_result: RetrievalResult) -> str:
        """ایجاد متن زمینه داده‌محور (رابطه‌ها به صورت لیست)"""
        nodes = retrieval_result.nodes
        edges = retrieval_result.edges
        query = retrieval_result.query
        
        if not nodes and not edges:
            return "داده‌ای یافت نشد."
        
        data_parts = []
        
        # لیست نودها
        if nodes:
            data_parts.append("موجودیت‌های یافت شده:")
            for node in nodes:
                data_parts.append(f"• {node.name} ({node.kind})")
        
        # لیست یال‌ها
        if edges:
            data_parts.append("\nرابطه‌های یافت شده:")
            for edge in edges:
                data_parts.append(f"• {edge.source} → {edge.target} ({edge.relation})")
        
        context_text = "\n".join(data_parts) if data_parts else "داده‌ای یافت نشد."
        return remove_emojis(context_text)

    def _create_step_by_step_context(self, retrieval_result: RetrievalResult) -> str:
        """
        ایجاد متن توضیح قبل از سؤال (استدلال گام به گام)
        مناسب برای مدل‌هایی که به دنبال استدلال خطی هستند
        """
        nodes = retrieval_result.nodes
        edges = retrieval_result.edges
        paths = retrieval_result.paths
        query = retrieval_result.query
        
        context_parts = []
        
        # 1. مقدمه استدلالی
        context_parts.append(f"**استدلال گام به گام برای سوال:** {query}")
        context_parts.append("")
        context_parts.append("برای پاسخ به این سوال، مراحل زیر را دنبال می‌کنیم:")
        context_parts.append("")
        
        # 2. گام‌های استدلال
        context_parts.append("**مراحل استدلال:**")
        
        if paths:
            main_path = paths[0] if paths else []
            if len(main_path) >= 2:
                context_parts.append("**گام 1: شناسایی مسیر اصلی**")
                path_elements = []
                from node_lookup_system import NodeLookupSystem
                lookup = NodeLookupSystem()
                for i, node in enumerate(main_path):
                    if i < len(main_path) - 1:
                        # پیدا کردن رابطه بین این نود و نود بعدی
                        relation = "→"
                        for edge in edges:
                            if edge.source == node and edge.target == main_path[i + 1]:
                                relation = edge.relation
                                break
                        # تبدیل آیدی نود به نام معنادار
                        node_info = lookup.get_node_info(node)
                        node_display = f"{node_info.name} ({node_info.kind})" if node_info else node
                        path_elements.append(f"{i+1}. {node_display}")
                        path_elements.append(f"   ↓ {relation}")
                    else:
                        # تبدیل آیدی نود به نام معنادار
                        node_info = lookup.get_node_info(node)
                        node_display = f"{node_info.name} ({node_info.kind})" if node_info else node
                        path_elements.append(f"{i+1}. {node_display}")
                
                context_parts.append("\n".join(path_elements))
                context_parts.append("")
        
        # 3. استنتاج منطقی
        context_parts.append("**استنتاج منطقی:**")
        context_parts.append("بر اساس این مسیر، می‌توانیم ارتباطات زیستی را درک کنیم.")
        context_parts.append("")
        
        # 4. دستورالعمل استدلالی
        context_parts.append("**دستورالعمل:**")
        context_parts.append("مراحل استدلال را دنبال کرده و پاسخ منطقی ارائه دهید.")
        
        # حذف ایموجی‌ها از متن نهایی
        final_text = "\n".join(context_parts)
        return remove_emojis(final_text)

    def _create_compact_direct_context(self, retrieval_result: RetrievalResult) -> str:
        """
        ایجاد متن فشرده و مستقیم
        مفید برای تست مدل‌هایی که پاسخ‌های خلاصه ولی دقیق تولید می‌کنند
        """
        nodes = retrieval_result.nodes
        edges = retrieval_result.edges
        paths = retrieval_result.paths
        query = retrieval_result.query
        
        context_parts = []
        
        # 1. مقدمه فشرده
        context_parts.append(f"**اطلاعات فشرده برای:** {query}")
        context_parts.append("")
        
        # 2. مسیر مستقیم
        if paths:
            main_path = paths[0] if paths else []
            if len(main_path) >= 2:
                context_parts.append("**مسیر کلیدی:**")
                from node_lookup_system import NodeLookupSystem
                lookup = NodeLookupSystem()
                path_elements = []
                for node in main_path:
                    node_info = lookup.get_node_info(node)
                    node_display = f"{node_info.name} ({node_info.kind})" if node_info else node
                    path_elements.append(node_display)
                path_str = " → ".join(path_elements)
                context_parts.append(f"• {path_str}")
                context_parts.append("")
        
        # 3. دستورالعمل فشرده
        context_parts.append("**دستورالعمل:** پاسخ کوتاه و دقیق ارائه دهید.")
        
        # حذف ایموجی‌ها از متن نهایی
        final_text = "\n".join(context_parts)
        return remove_emojis(final_text)

    def _create_biological_pathway_context(self, retrieval_result: RetrievalResult) -> str:
        """
        ایجاد متن مسیر زیستی (تخصصی)
        برای تحلیل‌های تخصصی زیستی
        """
        nodes = retrieval_result.nodes
        edges = retrieval_result.edges
        paths = retrieval_result.paths
        query = retrieval_result.query
        
        context_parts = []
        
        # 1. مقدمه تخصصی با تمرکز روی ژن مرکزی
        context_parts.append(f"🧬 **تحلیل مسیر زیستی برای:** {query}")
        context_parts.append("")
        
        # شناسایی ژن مرکزی
        central_gene = self._identify_central_gene(nodes, query)
        if central_gene:
            biological_role = BIOLOGICAL_ROLES.get(central_gene, "ژن مهم زیستی")
            context_parts.append(f"🔬 **ژن مرکزی:** {central_gene} - {biological_role}")
            context_parts.append("")
        
        context_parts.append("**مسیرهای زیستی شناسایی شده در Hetionet:**")
        context_parts.append("")
        
        # 2. تحلیل مسیرها با توضیحات توصیفی
        if paths:
            context_parts.append("🛤️ **مسیرهای زیستی:**")
            for i, path in enumerate(paths[:3]):
                context_parts.append(f"**مسیر {i+1}:**")
                for j, node in enumerate(path):
                    if j < len(path) - 1:
                        context_parts.append(f"  {self._display_node(node)} →")
                    else:
                        context_parts.append(f"  {self._display_node(node)}")
                
                # اضافه کردن توضیح توصیفی برای مسیر
                path_description = self._create_path_description(path, edges)
                if path_description:
                    context_parts.append(f"  **توضیح زیستی:** {path_description}")
                context_parts.append("")
        else:
            context_parts.append("⚠️ **هشدار:** هیچ مسیر زیستی مستقیمی یافت نشد.")
            context_parts.append("این ممکن است به دلیل محدودیت عمق جستجو یا عدم وجود مسیر مستقیم باشد.")
            context_parts.append("")
        
        # 3. مکانیسم‌های زیستی با توضیحات کامل
        context_parts.append("⚙️ **مکانیسم‌های زیستی:**")
        if edges:
            edge_types = {}
            for edge in edges:
                if edge.relation not in edge_types:
                    edge_types[edge.relation] = []
                edge_types[edge.relation].append(f"{self._display_node(edge.source)} → {self._display_node(edge.target)}")
            
            for relation, connections in sorted(edge_types.items(), key=lambda x: len(x[1]), reverse=True)[:3]:
                desc = METAEDGE_DESCRIPTIONS.get(relation, relation)
                context_parts.append(f"• {desc} ({len(connections)} مورد)")
                
                # نمایش نمونه‌ای از روابط
                for connection in connections[:2]:
                    context_parts.append(f"  - {connection}")
        context_parts.append("")
        
        # 4. تحلیل زیستی پیشرفته
        biological_inference = self._create_biological_inference(nodes, edges, paths, query)
        if biological_inference:
            context_parts.append("🔬 **تحلیل زیستی:**")
            context_parts.append(biological_inference)
            context_parts.append("")
        
        # 5. دستورالعمل تخصصی
        context_parts.append("🔬 **دستورالعمل تخصصی:**")
        context_parts.append("تحلیل عمیق مسیرهای زیستی و مکانیسم‌های مولکولی ارائه دهید.")
        context_parts.append("تمرکز بر:")
        context_parts.append("• مکانیسم‌های تنظیمی")
        context_parts.append("• مسیرهای سیگنالینگ")
        context_parts.append("• اهمیت بالینی")
        
        context_text = "\n".join(context_parts)
        return remove_emojis(context_text)

    def _create_clinical_relevance_context(self, retrieval_result: RetrievalResult) -> str:
        """
        ایجاد متن ارتباط بالینی
        برای تحلیل‌های مرتبط با پزشکی و درمان
        """
        nodes = retrieval_result.nodes
        edges = retrieval_result.edges
        paths = retrieval_result.paths
        query = retrieval_result.query
        
        context_parts = []
        
        # 1. مقدمه بالینی
        context_parts.append(f"🏥 **تحلیل بالینی برای:** {query}")
        context_parts.append("")
        context_parts.append("**اطلاعات بالینی استخراج شده:**")
        context_parts.append("")
        
        # 2. عناصر بالینی
        clinical_elements = []
        for node in nodes:
            if node.kind in ['Disease', 'Compound', 'Gene', 'Anatomy']:
                clinical_elements.append(f"• {node.kind}: {node.name}")
        
        if clinical_elements:
            context_parts.append("🏥 **عناصر بالینی:**")
            context_parts.extend(clinical_elements[:5])
            context_parts.append("")
        
        # 3. روابط درمانی
        therapeutic_relations = []
        for edge in edges:
            if edge.relation in ['CtD', 'CuG', 'CdG', 'DaG']:
                therapeutic_relations.append(f"• {edge.source} → {edge.relation} → {edge.target}")
        
        if therapeutic_relations:
            context_parts.append("💊 **روابط درمانی:**")
            context_parts.extend(therapeutic_relations[:3])
            context_parts.append("")
        
        # 4. اهمیت بالینی
        context_parts.append("📋 **اهمیت بالینی:**")
        context_parts.append("این روابط می‌تواند برای درک مکانیسم‌های درمانی مفید باشد.")
        context_parts.append("")
        
        # 5. دستورالعمل بالینی
        context_parts.append("🎯 **دستورالعمل بالینی:**")
        context_parts.append("تحلیل بالینی و کاربردهای درمانی را بررسی کنید.")
        
        context_text = "\n".join(context_parts)
        return remove_emojis(context_text)

    def _create_mechanistic_detailed_context(self, retrieval_result: RetrievalResult) -> str:
        """
        ایجاد متن مکانیسمی تفصیلی
        برای تحلیل‌های عمیق مکانیسم‌های زیستی
        """
        nodes = retrieval_result.nodes
        edges = retrieval_result.edges
        paths = retrieval_result.paths
        query = retrieval_result.query
        
        context_parts = []
        
        # 1. مقدمه مکانیسمی با تمرکز روی ژن مرکزی
        context_parts.append(f"⚙️ **تحلیل مکانیسمی تفصیلی برای:** {query}")
        context_parts.append("")
        
        # شناسایی ژن مرکزی
        central_gene = self._identify_central_gene(nodes, query)
        if central_gene:
            biological_role = BIOLOGICAL_ROLES.get(central_gene, "ژن مهم زیستی")
            context_parts.append(f"🔬 **ژن مرکزی:** {central_gene} - {biological_role}")
            context_parts.append("")
        
        context_parts.append("**مکانیسم‌های زیستی شناسایی شده:**")
        context_parts.append("")
        
        # 2. تحلیل عمیق روابط با توضیحات کامل
        if edges:
            context_parts.append("🔬 **تحلیل مکانیسمی روابط:**")
            edge_analysis = {}
            for edge in edges:
                if edge.relation not in edge_analysis:
                    edge_analysis[edge.relation] = []
                edge_analysis[edge.relation].append(f"{edge.source} → {edge.target}")
            
            for relation, connections in edge_analysis.items():
                desc = METAEDGE_DESCRIPTIONS.get(relation, relation)
                context_parts.append(f"**مکانیسم {desc}:**")
                for connection in connections[:3]:  # حداکثر 3 نمونه
                    context_parts.append(f"  - {connection}")
                context_parts.append("")
        
        # 3. مسیرهای مکانیسمی با توضیحات توصیفی
        if paths:
            context_parts.append("🛤️ **مسیرهای مکانیسمی:**")
            for i, path in enumerate(paths[:3]):
                context_parts.append(f"**مسیر مکانیسمی {i+1}:**")
                for j, node in enumerate(path):
                    if j < len(path) - 1:
                        context_parts.append(f"  {node} →")
                    else:
                        context_parts.append(f"  {node}")
                
                # اضافه کردن توضیح مکانیسمی برای مسیر
                path_description = self._create_path_description(path, edges)
                if path_description:
                    context_parts.append(f"  **مکانیسم:** {path_description}")
                context_parts.append("")
        else:
            context_parts.append("⚠️ **هشدار:** هیچ مسیر مکانیسمی مستقیمی یافت نشد.")
            context_parts.append("")
        
        # 4. تحلیل مکانیسمی پیشرفته
        biological_inference = self._create_biological_inference(nodes, edges, paths, query)
        if biological_inference:
            context_parts.append("🔬 **تحلیل مکانیسمی:**")
            context_parts.append(biological_inference)
            context_parts.append("")
        
        # 5. دستورالعمل مکانیسمی
        context_parts.append("🔬 **دستورالعمل مکانیسمی:**")
        context_parts.append("تحلیل عمیق مکانیسم‌های مولکولی و زیستی ارائه دهید.")
        context_parts.append("تمرکز بر:")
        context_parts.append("• مکانیسم‌های تنظیمی")
        context_parts.append("• مسیرهای سیگنالینگ")
        context_parts.append("• تعاملات پروتئین-پروتئین")
        context_parts.append("• اهمیت بالینی")
        
        context_text = "\n".join(context_parts)
        return remove_emojis(context_text)

    def _create_intelligent_context_text(self, retrieval_result: RetrievalResult) -> str:
        """
        ایجاد متن زمینه هوشمند پیشرفته با تحلیل عمیق و استنتاجات زیستی
        """
        nodes = retrieval_result.nodes
        edges = retrieval_result.edges
        paths = retrieval_result.paths
        query = retrieval_result.query
        
        context_parts = []
        
        # 1. مقدمه هوشمند با تمرکز روی گره مرکزی
        context_parts.append(f"🧠 **متن زمینه هوشمند برای سوال. از این اطلاعات استفاده کن و در نهایت پاسخ سوال را با اطلاعاتی که به نظرت به جواب سوال کمک میکنه استفاده کن و سوال رو به بهترین شکل پاسخ بده** {query}")
        context_parts.append("")
        
        # شناسایی گره مرکزی و نقش زیستی آن
        central_gene = self._identify_central_gene(nodes, query)
        if central_gene:
            biological_role = BIOLOGICAL_ROLES.get(central_gene, "ژن مهم زیستی")
            context_parts.append(f"🔬 **ژن مرکزی:** {central_gene} - {biological_role}")
            context_parts.append("")
        
        context_parts.append("🔬 **تحلیل هوشمند داده‌های گراف:**")
        context_parts.append("این متن شامل تحلیل عمیق، استنتاجات زیستی و روابط معنادار است.")
        context_parts.append("")
        
        # 2. تحلیل آماری پیشرفته با توضیحات
        context_parts.append("📊 **تحلیل آماری پیشرفته:**")
        context_parts.append(f"• نودهای بازیابی شده: {len(nodes)}")
        context_parts.append(f"• یال‌های بازیابی شده: {len(edges)}")
        context_parts.append(f"• مسیرهای بازیابی شده: {len(paths)}")
        
        # محاسبه تراکم روابط
        if nodes and edges:
            avg_connections = len(edges) / len(nodes)
            context_parts.append(f"• تراکم متوسط روابط: {avg_connections:.2f} یال به ازای هر نود")
        
        # 3. تحلیل نوع‌شناسی نودها با توضیحات زیستی
        if nodes:
            context_parts.append("")
            context_parts.append("🏷️ **تحلیل نوع‌شناسی نودها:**")
            node_kinds = {}
            for node in nodes:
                if node.kind not in node_kinds:
                    node_kinds[node.kind] = []
                node_kinds[node.kind].append(node.name)
            
            for kind, names in node_kinds.items():
                context_parts.append(f"• {kind}: {len(names)} نود")
                # نمایش تمام نودها
                for i, name in enumerate(names):
                    context_parts.append(f"  {i+1}. {name}")
                
                # اضافه کردن توضیحات زیستی برای ژن‌های مهم
                if kind == "Gene":
                    for gene_name in names:
                        if gene_name in BIOLOGICAL_ROLES:
                            context_parts.append(f"    - {gene_name}: {BIOLOGICAL_ROLES[gene_name]}")
        
        # 4. تحلیل روابط معنادار با توضیحات کامل
        if edges:
            context_parts.append("")
            context_parts.append("🔗 **تحلیل روابط معنادار:**")
            from node_lookup_system import NodeLookupSystem
            lookup = NodeLookupSystem()
            edge_analysis = {}
            for edge in edges:
                if edge.relation not in edge_analysis:
                    edge_analysis[edge.relation] = []
                source_info = lookup.get_node_info(edge.source)
                target_info = lookup.get_node_info(edge.target)
                source_display = f"{source_info.name} ({source_info.kind})" if source_info else edge.source
                target_display = f"{target_info.name} ({target_info.kind})" if target_info else edge.target
                edge_analysis[edge.relation].append(f"{source_display} → {target_display}")
            
            # مرتب‌سازی بر اساس فراوانی
            sorted_relations = sorted(edge_analysis.items(), key=lambda x: len(x[1]), reverse=True)
            for relation, connections in sorted_relations[:5]:  # 5 رابطه برتر
                desc = METAEDGE_DESCRIPTIONS.get(relation, relation)
                context_parts.append(f"• {desc} ({len(connections)} رابطه)")
                
                # نمایش نمونه‌ای از روابط
                for connection in connections[:2]:  # حداکثر 2 نمونه
                    context_parts.append(f"  - {connection}")
        
        # 5. تحلیل مسیرهای زیستی با توضیحات توصیفی
        if paths:
            context_parts.append("")
            context_parts.append("🛤️ **تحلیل مسیرهای زیستی:**")
            context_parts.append("مسیرهای شناسایی شده نشان‌دهنده روابط پیچیده زیستی هستند:")
            
            for i, path in enumerate(paths[:3]):
                path_length = len(path)
                context_parts.append(f"• مسیر {i+1}: {path_length} گام زیستی")
                context_parts.append(f"  مسیر: {' → '.join([self._display_node(n) for n in path])}")
                
                # تولید توضیح توصیفی برای مسیر
                path_description = self._create_path_description(path, edges)
                if path_description:
                    context_parts.append(f"  توضیح: {path_description}")
        else:
            context_parts.append("")
            context_parts.append("⚠️ **هشدار:** هیچ مسیر مستقیمی یافت نشد.")
            context_parts.append("این ممکن است به دلیل محدودیت عمق جستجو یا عدم وجود مسیر مستقیم باشد.")
            context_parts.append("پیشنهاد: افزایش عمق جستجو یا استفاده از روش‌های بازیابی دیگر")
        
        # 6. استنتاجات زیستی پیشرفته
        context_parts.append("")
        context_parts.append("🧬 **استنتاجات زیستی:**")
        
        # تشخیص نوع سوال و استنتاج مناسب
        query_lower = query.lower()
        if any(word in query_lower for word in ["gene", "express", "protein"]):
            context_parts.append("• سوال مربوط به بیان ژن و عملکرد پروتئین‌ها")
            context_parts.append("• تمرکز بر روابط:")
            context_parts.append(f"  - {METAEDGE_DESCRIPTIONS.get('AeG', 'AeG')}")
            context_parts.append(f"  - {METAEDGE_DESCRIPTIONS.get('AuG', 'AuG')}")
            context_parts.append(f"  - {METAEDGE_DESCRIPTIONS.get('GpBP', 'GpBP')}")
        elif any(word in query_lower for word in ["disease", "cancer", "disorder"]):
            context_parts.append("• سوال مربوط به بیماری‌ها و مکانیسم‌های پاتولوژیک")
            context_parts.append("• تمرکز بر روابط:")
            context_parts.append(f"  - {METAEDGE_DESCRIPTIONS.get('DaG', 'DaG')}")
            context_parts.append(f"  - {METAEDGE_DESCRIPTIONS.get('DuG', 'DuG')}")
            context_parts.append(f"  - {METAEDGE_DESCRIPTIONS.get('DlA', 'DlA')}")
        elif any(word in query_lower for word in ["drug", "treatment", "therapy"]):
            context_parts.append("• سوال مربوط به درمان و داروها")
            context_parts.append("• تمرکز بر روابط:")
            context_parts.append(f"  - {METAEDGE_DESCRIPTIONS.get('CtD', 'CtD')}")
            context_parts.append(f"  - {METAEDGE_DESCRIPTIONS.get('CuG', 'CuG')}")
            context_parts.append(f"  - {METAEDGE_DESCRIPTIONS.get('CdG', 'CdG')}")
        elif any(word in query_lower for word in ["tissue", "anatomy", "organ"]):
            context_parts.append("• سوال مربوط به بافت‌ها و آناتومی")
            context_parts.append("• تمرکز بر روابط:")
            context_parts.append(f"  - {METAEDGE_DESCRIPTIONS.get('AeG', 'AeG')}")
            context_parts.append(f"  - {METAEDGE_DESCRIPTIONS.get('AuG', 'AuG')}")
        else:
            context_parts.append("• سوال عمومی - تحلیل جامع تمام روابط")
        
        # اضافه کردن استنتاج زیستی بر اساس داده‌های واقعی
        biological_inference = self._create_biological_inference(nodes, edges, paths, query)
        if biological_inference:
            context_parts.append("")
            context_parts.append("🔬 **استنتاج زیستی بر اساس داده‌ها:**")
            context_parts.append(biological_inference)
        
        # 7. دستورالعمل هوشمند
        context_parts.append("")
        context_parts.append("🎯 **دستورالعمل هوشمند:**")
        context_parts.append("بر اساس تحلیل عمیق داده‌های گراف و استنتاجات زیستی،")
        context_parts.append("پاسخ جامع و تخصصی ارائه دهید که شامل:")
        context_parts.append("• تحلیل روابط معنادار")
        context_parts.append("• استنتاجات زیستی")
        context_parts.append("• اهمیت بالینی")
        context_parts.append("• کاربردهای عملی")
        
        context_text = "\n".join(context_parts)
        return remove_emojis(context_text)

    def _identify_central_gene(self, nodes: List[GraphNode], query: str) -> Optional[str]:
        """
        شناسایی ژن مرکزی بر اساس سوال و نودهای بازیابی‌شده
        """
        query_lower = query.lower()
        
        # جستجوی ژن‌های مهم در سوال
        for gene in BIOLOGICAL_ROLES.keys():
            if gene.lower() in query_lower:
                return gene
        
        # جستجو در نودهای بازیابی‌شده
        gene_nodes = [node for node in nodes if node.kind == "Gene"]
        if gene_nodes:
            # اولویت به ژن‌های مهم
            for gene_node in gene_nodes:
                if gene_node.name in BIOLOGICAL_ROLES:
                    return gene_node.name
            # اگر ژن مهمی نبود، اولین ژن را برگردان
            return gene_nodes[0].name
        
        return None

    def _create_path_description(self, path: List[str], edges: List[GraphEdge]) -> str:
        """
        تولید توضیح توصیفی برای یک مسیر
        """
        if len(path) < 2:
            return ""
        
        from node_lookup_system import NodeLookupSystem
        lookup = NodeLookupSystem()
        descriptions = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            # تبدیل آیدی نودها به نام معنادار
            source_info = lookup.get_node_info(source)
            target_info = lookup.get_node_info(target)
            source_display = f"{source_info.name} ({source_info.kind})" if source_info else source
            target_display = f"{target_info.name} ({target_info.kind})" if target_info else target
            
            # پیدا کردن رابطه بین این دو نود
            relation = None
            for edge in edges:
                if edge.source == source and edge.target == target:
                    relation = edge.relation
                    break
            
            if relation:
                desc = METAEDGE_DESCRIPTIONS.get(relation, relation)
                descriptions.append(f"{source_display} {desc} {target_display}")
            else:
                descriptions.append(f"{source_display} → {target_display}")
        
        if descriptions:
            return " و ".join(descriptions)
        return ""

    def _create_biological_inference(self, nodes: List[GraphNode], edges: List[GraphEdge], 
                                   paths: List[List[str]], query: str) -> str:
        """
        تولید استنتاج زیستی بر اساس داده‌های واقعی
        """
        query_lower = query.lower()
        inferences = []
        
        # تحلیل ژن‌های مهم
        gene_nodes = [node for node in nodes if node.kind == "Gene"]
        important_genes = [gene for gene in gene_nodes if gene.name in BIOLOGICAL_ROLES]
        
        if important_genes:
            gene_names = [gene.name for gene in important_genes[:3]]
            gene_roles = [BIOLOGICAL_ROLES[gene.name] for gene in important_genes[:3]]
            
            if len(gene_names) == 1:
                inferences.append(f"ژن {gene_names[0]} که {gene_roles[0]} است، در این سوال نقش کلیدی دارد.")
            else:
                gene_list = "، ".join(gene_names)
                inferences.append(f"ژن‌های {gene_list} که نقش‌های زیستی مهمی دارند، در این تحلیل شناسایی شدند.")
        
        # تحلیل بیماری‌های مرتبط
        disease_nodes = [node for node in nodes if node.kind == "Disease"]
        if disease_nodes:
            disease_names = [node.name for node in disease_nodes[:3]]
            disease_desc = [DISEASE_SIGNIFICANCE.get(name, name) for name in disease_names]
            
            if len(disease_names) == 1:
                inferences.append(f"بیماری {disease_names[0]} ({disease_desc[0]}) در این تحلیل مورد بررسی قرار گرفته است.")
            else:
                disease_list = "، ".join(disease_names)
                inferences.append(f"بیماری‌های {disease_list} در این تحلیل شناسایی شدند.")
        
        # تحلیل مسیرها
        if paths:
            path_count = len(paths)
            if path_count == 1:
                inferences.append("یک مسیر زیستی مستقیم بین اجزای مختلف شناسایی شد.")
            else:
                inferences.append(f"{path_count} مسیر زیستی مختلف شناسایی شد که نشان‌دهنده پیچیدگی روابط زیستی است.")
        
        # تحلیل روابط غالب
        if edges:
            relation_counts = {}
            for edge in edges:
                relation_counts[edge.relation] = relation_counts.get(edge.relation, 0) + 1
            
            most_common_relation = max(relation_counts.items(), key=lambda x: x[1])
            relation_desc = METAEDGE_DESCRIPTIONS.get(most_common_relation[0], most_common_relation[0])
            inferences.append(f"رابطه غالب در این تحلیل {relation_desc} است که {most_common_relation[1]} بار مشاهده شده است.")
        
        if inferences:
            return " ".join(inferences)
        return ""



    def simple_template_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ ساده با قالب بهبود یافته"""
        query_lower = retrieval_result.query.lower()
        
        # ایجاد متن زمینه ساده بهینه شده
        simple_context = self._create_simple_context_text(retrieval_result)
        
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
            for edge in edges:  # نمایش تمام روابط
                answer_parts.append(f"  • {self._display_node(edge.source)} → {self._display_node(edge.target)}")
        
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
            for edge in treatment_edges:
                answer_parts.append(f"  • {self._display_node(edge.source)} treats {self._display_node(edge.target)}")
        
        # داروهای یافت شده
        if drug_nodes:
            answer_parts.append(f"\n💊 DRUGS FOUND ({len(drug_nodes)}):")
            for drug in drug_nodes:
                answer_parts.append(f"  • {drug.name}")
        
        # بیماری‌های یافت شده
        if disease_nodes:
            answer_parts.append(f"\n🏥 DISEASES FOUND ({len(disease_nodes)}):")
            for disease in disease_nodes:
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
        for gene in gene_nodes:
            answer_parts.append(f"  • {gene.name}")
        
        # فرآیندهای زیستی مرتبط
        if process_nodes:
            answer_parts.append(f"\n⚙️ BIOLOGICAL PROCESSES ({len(process_nodes)}):")
            for process in process_nodes:
                answer_parts.append(f"  • {process.name}")
        
        # روابط ژن-فرآیند
        gene_process_edges = [e for e in retrieval_result.edges if 'participate' in e.relation.lower() or 'regulate' in e.relation.lower()]
        if gene_process_edges:
            answer_parts.append(f"\n🔗 GENE-PROCESS RELATIONSHIPS ({len(gene_process_edges)}):")
            for edge in gene_process_edges:
                answer_parts.append(f"  • {self._display_node(edge.source)} → {self._display_node(edge.target)}")
        
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
        for disease in disease_nodes:
            answer_parts.append(f"  • {disease.name}")
        
        # ژن‌های مرتبط
        if gene_nodes:
            answer_parts.append(f"\n🧬 ASSOCIATED GENES ({len(gene_nodes)}):")
            for gene in gene_nodes:
                answer_parts.append(f"  • {gene.name}")
        
        # علائم مرتبط
        if symptom_nodes:
            answer_parts.append(f"\n🤒 ASSOCIATED SYMPTOMS ({len(symptom_nodes)}):")
            for symptom in symptom_nodes:
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
        for anatomy in anatomy_nodes:
            answer_parts.append(f"  • {anatomy.name}")
        
        # ژن‌های بیان شده
        if gene_nodes:
            answer_parts.append(f"\n🧬 EXPRESSED GENES ({len(gene_nodes)}):")
            for gene in gene_nodes:
                answer_parts.append(f"  • {gene.name}")
        
        # روابط بیان
        expression_edges = [e for e in retrieval_result.edges if 'express' in e.relation.lower()]
        if expression_edges:
            answer_parts.append(f"\n🔗 EXPRESSION RELATIONSHIPS ({len(expression_edges)}):")
            for edge in expression_edges:
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
            for node in nodes:
                answer_parts.append(f"  • {node.name}")
        
        # روابط مهم
        if retrieval_result.edges:
            answer_parts.append(f"\n🔗 KEY RELATIONSHIPS ({len(retrieval_result.edges)}):")
            for edge in retrieval_result.edges:
                answer_parts.append(f"  • {self._display_node(edge.source)} → {self._display_node(edge.target)} ({edge.relation})")
        
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
        for relation, edges in sorted(important_relations.items(), key=lambda x: len(x[1]), reverse=True):
            answer_parts.append(f"• **{relation}** ({len(edges)} رابطه):")
            for edge in edges:
                source_name = next(n.name for n in retrieval_result.nodes if n.id == edge.source)
                target_name = next(n.name for n in retrieval_result.nodes if n.id == edge.target)
                answer_parts.append(f"  - {self._display_node(edge.source)} → {self._display_node(edge.target)}")
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
            for drug in drug_nodes:
                score_info = f" (امتیاز: {drug.score:.2f})" if hasattr(drug, 'score') and drug.score != 1.0 else ""
                answer_parts.append(f"• {drug.name}{score_info}")
            answer_parts.append("")
        
        if disease_nodes:
            answer_parts.append("**بیماری‌های مرتبط:**")
            for disease in disease_nodes:
                answer_parts.append(f"• {disease.name}")
            answer_parts.append("")
        
        # روابط درمان
        treatment_edges = [e for e in retrieval_result.edges if 'treat' in e.relation.lower() or 'therapy' in e.relation.lower()]
        if treatment_edges:
            answer_parts.append("**روابط درمانی:**")
            for edge in treatment_edges[:5]:
                answer_parts.append(f"• {self._display_node(edge.source)} درمان می‌کند: {self._display_node(edge.target)}")
        
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
                answer_parts.append(f"• {self._display_node(edge.source)} → {self._display_node(edge.target)} ({edge.relation})")
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
                answer_parts.append(f"• {self._display_node(edge.source)} → {self._display_node(edge.target)}")
        
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
        
        # تحلیل تخصصی برای TP53
        if any('tp53' in gene.name.lower() for gene in primary_genes):
            answer_parts.append("\n🔬 **تحلیل تخصصی TP53:**")
            answer_parts.append("TP53 (Tumor Protein P53) یکی از مهم‌ترین ژن‌های سرکوبگر تومور است که:")
            answer_parts.append("• در بیش از 50% سرطان‌های انسانی جهش یافته است")
            answer_parts.append("• نقش کلیدی در تنظیم چرخه سلولی و آپوپتوز دارد")
            answer_parts.append("• به عنوان 'نگهبان ژنوم' شناخته می‌شود")
            answer_parts.append("• اختلال در عملکرد آن منجر به تکثیر غیرقابل کنترل سلول‌ها می‌شود")
            answer_parts.append("")
        
        # تحلیل روابط خاص
        if retrieval_result.edges:
            answer_parts.append("**تحلیل روابط یافت شده:**")
            gene_cancer_edges = []
            for edge in retrieval_result.edges:
                source_node = next((n for n in retrieval_result.nodes if n.id == edge.source), None)
                target_node = next((n for n in retrieval_result.nodes if n.id == edge.target), None)
                if source_node and target_node:
                    if (source_node.kind == 'Gene' and target_node.kind == 'Disease') or \
                       (source_node.kind == 'Disease' and target_node.kind == 'Gene'):
                        gene_cancer_edges.append((source_node, target_node, edge.relation))
            
            if gene_cancer_edges:
                answer_parts.append("روابط ژن-سرطان یافت شده:")
                for source, target, relation in gene_cancer_edges[:5]:
                    answer_parts.append(f"• {source.name} → {target.name} ({relation})")
                answer_parts.append("")
        
        # پیام راهنما
        answer_parts.append("📌 **راهنما:** تحلیل اهمیت زیستی و بالینی این ژن‌ها را بررسی کنید.")
        
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
                     generation_model: GenerationModel, text_generation_type: str = 'INTELLIGENT', 
                     max_depth: int = 2) -> Dict[str, Any]:
        """پردازش کامل یک سوال"""
        print(f"🚀 پردازش سوال: {query}")
        print(f"📝 نوع تولید متن: {text_generation_type}")
        
        # مرحله 1: بازیابی
        retrieval_result = self.retrieve_information(query, retrieval_method, max_depth)
        
        # مرحله 2: تولید پاسخ
        generation_result = self.generate_answer(retrieval_result, generation_model, text_generation_type)
        
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
    
    def openai_gpt_generation(self, retrieval_result: RetrievalResult, model: GenerationModel = None) -> str:
        """تولید پاسخ با OpenAI GPT (نیاز به API Key)"""
        try:
            from openai import OpenAI
            
            # بررسی وجود API Key
            if not hasattr(self, 'openai_api_key') or not self.openai_api_key:
                return "🔑 برای استفاده از OpenAI GPT، لطفاً API Key را تنظیم کنید.\n\n" + self._fallback_generation(retrieval_result, "OpenAI")
            
            # تعیین مدل بر اساس انتخاب کاربر
            model_mapping = {
                GenerationModel.OPENAI_GPT_4O: "gpt-4o",
                GenerationModel.OPENAI_GPT_4O_MINI: "gpt-4o-mini",
                GenerationModel.OPENAI_GPT_4_TURBO: "gpt-4-turbo",
                GenerationModel.OPENAI_GPT_4: "gpt-4",
                GenerationModel.OPENAI_GPT_3_5_TURBO: "gpt-3.5-turbo",
                GenerationModel.OPENAI_GPT_3_5_TURBO_16K: "gpt-3.5-turbo-16k",
                GenerationModel.OPENAI_GPT: "gpt-4o"  # مدل پیش‌فرض جدیدترین
            }
            
            model_choice = model_mapping.get(model, "gpt-4o")  # پیش‌فرض جدیدترین مدل
            max_tokens = 1500 if "4o" in model_choice else 1000
            
            # ایجاد کلاینت OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            # آماده‌سازی متن ورودی
            prompt = self._create_advanced_prompt(retrieval_result)
            
            # درخواست به OpenAI
            response = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": "You are a biomedical expert analyzing knowledge graph data. Provide detailed, accurate, and well-structured answers in Persian with proper formatting and emojis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                presence_penalty=0.1,  # تشویق به تنوع
                frequency_penalty=0.1   # کاهش تکرار
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"خطا در OpenAI ({model_choice}): {e}")
            return self._fallback_generation(retrieval_result, f"OpenAI ({model_choice})")
    
    def anthropic_claude_generation(self, retrieval_result: RetrievalResult, model: GenerationModel = None) -> str:
        """تولید پاسخ با Anthropic Claude (نیاز به API Key)"""
        try:
            import anthropic
            
            # بررسی وجود API Key
            if not hasattr(self, 'anthropic_api_key') or not self.anthropic_api_key:
                return "🔑 برای استفاده از Claude، لطفاً API Key را تنظیم کنید.\n\n" + self._fallback_generation(retrieval_result, "Claude")
            
            # تعیین مدل بر اساس انتخاب کاربر
            model_mapping = {
                GenerationModel.ANTHROPIC_CLAUDE_3_5_SONNET: "claude-3-5-sonnet-20241022",
                GenerationModel.ANTHROPIC_CLAUDE_3_5_HAIKU: "claude-3-5-haiku-20241022",
                GenerationModel.ANTHROPIC_CLAUDE_3_OPUS: "claude-3-opus-20240229",
                GenerationModel.ANTHROPIC_CLAUDE_3_SONNET: "claude-3-sonnet-20240229",
                GenerationModel.ANTHROPIC_CLAUDE_3_HAIKU: "claude-3-haiku-20240307",
                GenerationModel.ANTHROPIC_CLAUDE: "claude-3-5-sonnet-20241022"  # مدل پیش‌فرض جدیدترین
            }
            
            model_choice = model_mapping.get(model, "claude-3-5-sonnet-20241022")  # پیش‌فرض جدیدترین مدل
            max_tokens = 1500 if "3-5" in model_choice else 1000
            
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            # آماده‌سازی متن ورودی
            prompt = self._create_advanced_prompt(retrieval_result)
            
            # درخواست به Claude
            response = client.messages.create(
                model=model_choice,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            print(f"خطا در Claude ({model_choice}): {e}")
            return self._fallback_generation(retrieval_result, f"Claude ({model_choice})")
    
    def google_gemini_generation(self, retrieval_result: RetrievalResult, model: GenerationModel = None) -> str:
        """تولید پاسخ با Google Gemini (نیاز به API Key)"""
        try:
            import google.generativeai as genai
            
            # بررسی وجود API Key
            if not hasattr(self, 'gemini_api_key') or not self.gemini_api_key:
                return "🔑 برای استفاده از Gemini، لطفاً API Key را تنظیم کنید.\n\n" + self._fallback_generation(retrieval_result, "Gemini")
            
            # تعیین مدل بر اساس انتخاب کاربر
            model_mapping = {
                GenerationModel.GOOGLE_GEMINI_1_5_PRO: "gemini-1.5-pro",
                GenerationModel.GOOGLE_GEMINI_1_5_FLASH: "gemini-1.5-flash",
                GenerationModel.GOOGLE_GEMINI_1_0_PRO: "gemini-1.0-pro",
                GenerationModel.GOOGLE_GEMINI_1_0_FLASH: "gemini-1.0-flash",
                GenerationModel.GOOGLE_GEMINI: "gemini-1.5-pro"  # مدل پیش‌فرض جدیدترین
            }
            
            model_choice = model_mapping.get(model, "gemini-1.5-pro")  # پیش‌فرض جدیدترین مدل
            
            genai.configure(api_key=self.gemini_api_key)
            model_instance = genai.GenerativeModel(model_choice)
            
            # آماده‌سازی متن ورودی
            prompt = self._create_advanced_prompt(retrieval_result)
            
            # درخواست به Gemini
            response = model_instance.generate_content(prompt)
            
            return response.text.strip()
            
        except Exception as e:
            print(f"خطا در Gemini ({model_choice}): {e}")
            return self._fallback_generation(retrieval_result, f"Gemini ({model_choice})")
    
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
        
        # کاهش وزن روابط شباهت برای جلوگیری از نویز در سوالات مکانیزمی
        base_score = base_scores.get(metaedge, 2.5)
        if metaedge in ['DrD', 'CrC']:
            base_score *= 0.6
        
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
    
    # ==================== KGSearch (Intent-Aware + Schema-Aware for Hetionet) ====================
    def kgsearch_traceable(self, query: str, top_k: int = 10) -> Tuple[List[Dict[str, Any]], str]:
        """
        اجرای kgsearch مبتنی بر Hetionet با توجه به Intent/Schema.
        خروجی: (hits, summary)

        هر hit شامل یک مسیر traceable با نود/یال‌ها و متادیتا است.
        """
        if not self.G:
            return [], "گراف بارگذاری نشده است"

        intent_cfg = self._detect_intent_schema_map(query)
        allow = intent_cfg["allow"]
        deny = intent_cfg["deny"]
        end_type = intent_cfg["end_type"]
        hop_limit = intent_cfg["hop_limit"]
        constraints = intent_cfg.get("constraints", {})

        # 1) Canonicalization & Core Lock
        intent = self.analyze_question_intent(query)
        tokens = intent.get("keywords", [])
        matched = self.match_tokens_to_nodes(tokens)
        core_nodes = self._extract_core_nodes(query, matched, intent)
        if not core_nodes and matched:
            core_nodes = list(dict.fromkeys(matched.values()))[:3]

        # 2) Retrieval constrained by schema (allowlist/denylist + end-type)
        paths_with_meta = self._find_paths_allowlist(
            core_nodes=core_nodes,
            allow_metaedges=allow,
            deny_metaedges=deny,
            end_kind=end_type,
            hop_limit=hop_limit,
            max_results_per_hop=100,
            require_unique_nodes=True,
            extra_constraints=constraints,
            query=query,
        )

        # 3) Ranking
        # اگر intent دقیقاً هم‌واریانس ژن‌هاست، خروجی را مینیمال و ۱-هاپ روی GcG نگه‌دار
        if intent_cfg.get('intent') == 'G-G_covary':
            paths_with_meta = [p for p in paths_with_meta if len(p.get('path_nodes', [])) == 2 and all(m == 'GcG' for m in p.get('metaedges', []) if m)]
        ranked = self._rank_paths(paths_with_meta, query, intent_cfg)
        hits = []
        for rank, item in enumerate(ranked[:top_k], start=1):
            path_nodes = item["path_nodes"]
            path_edges = item["path_edges"]
            score = item["score"]
            hop_count = max(0, len(path_nodes) - 1)

            # ساخت JSON مسیر مطابق فرمت خواسته‌شده
            json_path: List[Dict[str, Any]] = []
            for i, nid in enumerate(path_nodes):
                json_path.append({
                    "id": nid,
                    "label": self.G.nodes[nid].get("name", nid),
                    "type": self.G.nodes[nid].get("kind", "Unknown")
                })
                if i < len(path_nodes) - 1:
                    src, dst = nid, path_nodes[i+1]
                    ed = self.G.get_edge_data(src, dst) or {}
                    metaedge = ed.get("metaedge") or ed.get("relation") or "related"
                    # ساخت شناسه یال پایدار
                    edge_id = f"Edge::{metaedge}::{src}__{dst}"
                    evidence_count = ed.get("evidence_count") or ed.get("evidence") or None
                    source_count = ed.get("source_count") or (len(ed.get("sources", [])) if isinstance(ed.get("sources"), list) else None)
                    unbiased = ed.get("unbiased") if "unbiased" in ed else None
                    extra = {}
                    for k in ("cov_metric", "weight", "score"):
                        if k in ed:
                            extra[k] = ed[k]
                    json_path.append({
                        "edge_id": edge_id,
                        "edge_type": metaedge,
                        "unbiased": unbiased,
                        "evidence_count": evidence_count,
                        "source_count": source_count,
                        **extra
                    })

            notes = item.get("notes", "")
            hits.append({
                "rank": rank,
                "path": json_path,
                "end_type": end_type,
                "hop_count": hop_count,
                "score": round(score, 4),
                "notes": notes
            })

        # 4) Fallback اگر نتیجه تهی شد
        used_fallback = False
        if not hits:
            fb = self._fallback_from_intent(intent_cfg.get("intent"))
            if fb:
                used_fallback = True
                allow_fb = fb["allow"]
                end_type_fb = fb["end_type"] or end_type
                hop_limit_fb = fb["hop_limit"] or hop_limit
                paths_with_meta = self._find_paths_allowlist(
                    core_nodes=core_nodes,
                    allow_metaedges=allow_fb,
                    deny_metaedges=deny,
                    end_kind=end_type_fb,
                    hop_limit=hop_limit_fb,
                    max_results_per_hop=100,
                    require_unique_nodes=True,
                    extra_constraints=fb.get("constraints", {}),
                    query=query,
                )
                ranked = self._rank_paths(paths_with_meta, query, fb)
                for rank, item in enumerate(ranked[:top_k], start=1):
                    path_nodes = item["path_nodes"]
                    hop_count = max(0, len(path_nodes) - 1)
                    json_path: List[Dict[str, Any]] = []
                    for i, nid in enumerate(path_nodes):
                        json_path.append({
                            "id": nid,
                            "label": self.G.nodes[nid].get("name", nid),
                            "type": self.G.nodes[nid].get("kind", "Unknown")
                        })
                        if i < len(path_nodes) - 1:
                            src, dst = nid, path_nodes[i+1]
                            ed = self.G.get_edge_data(src, dst) or {}
                            metaedge = ed.get("metaedge") or ed.get("relation") or "related"
                            edge_id = f"Edge::{metaedge}::{src}__{dst}"
                            json_path.append({
                                "edge_id": edge_id,
                                "edge_type": metaedge,
                                "unbiased": ed.get("unbiased") if "unbiased" in ed else None,
                                "evidence_count": ed.get("evidence_count") or ed.get("evidence") or None,
                                "source_count": ed.get("source_count") or (len(ed.get("sources", [])) if isinstance(ed.get("sources"), list) else None),
                            })
                    hits.append({
                        "rank": rank,
                        "path": json_path,
                        "end_type": end_type_fb,
                        "hop_count": hop_count,
                        "score": round(item["score"], 4),
                        "notes": (item.get("notes", "") + " | fallback")[:200]
                    })

        # 5) Summary کوتاه فارسی
        if hits:
            sum_lines = []
            sum_lines.append(f"نتایج بر اساس Intent='{intent_cfg.get('intent')}', با metaedgeهای مجاز: {', '.join(allow)}؛ end-type='{end_type}' و hop≤{hop_limit}.")
            if used_fallback:
                sum_lines.append("از fallback طبق قواعد استفاده شد؛ این روابط proxy هستند.")
            sum_lines.append(f"تعداد مسیرهای برتر: {min(top_k, len(hits))}، با تمرکز بر مسیرهای کوتاه و شواهد قوی.")
            summary = "\n".join(sum_lines)
        else:
            summary = "چیزی یافت نشد: برای این Intent، یال مرتبط موجود نبود. پیشنهاد: fallback یا افزایش hop-limit را امتحان کنید."

        return hits[:top_k], summary

    def _detect_intent_schema_map(self, query: str) -> Dict[str, Any]:
        q = (query or "").lower()
        # پیش‌فرض
        cfg = {
            "intent": "general",
            "allow": [],
            "deny": ["DrD", "CrC"],
            "end_type": None,
            "hop_limit": 2,
            "constraints": {}
        }

        # فعال‌سازی شباهت تنها در صورت ذکر
        resembles = any(k in q for k in ["resembles", "similar", "similarity", "alike"])

        # I. Gene→Gene
        if any(k in q for k in ["co-expression", "coexpression", "covary", "covaries", "هم‌واریانس", "هم‌بروز", "هم‌تغییر"]):
            cfg.update({"intent": "G-G_covary", "allow": ["GcG"], "end_type": "Gene", "hop_limit": 1})
        elif any(k in q for k in ["interaction", "interacts", "ppi", "تعامل"]):
            cfg.update({"intent": "G-G_interact", "allow": ["GiG"], "end_type": "Gene", "hop_limit": 1})
        elif any(k in q for k in ["regulates", "regulation", "تنظیم"]):
            cfg.update({"intent": "G-G_regulates", "allow": ["Gr>G"], "end_type": "Gene", "hop_limit": 1})

        # II. Disease→Drug/Class
        elif any(k in q for k in ["treats", "treatment", "therapy", "therapeutic", "درمان", "پالیتیو"]):
            allow = ["CtD", "CpD", "PCiC", "CbG"]
            if resembles:
                pass
            cfg.update({
                "intent": "D→(C|PC)",
                "allow": allow,
                "end_type": ("Compound|Pharmacologic Class"),
                "hop_limit": 3,
                "constraints": {"require_any_edge": ["CtD", "CpD"], "require_edge_to": "Disease"}
            })

        # III. Gene→Drug/Class
        elif any(k in q for k in ["drug", "compound", "pharmacologic class", "mechanism", "target"]):
            cfg.update({
                "intent": "G→(C|PC)",
                "allow": ["GiG", "Gr>G", "CbG", "PCiC"],
                "end_type": ("Compound|Pharmacologic Class"),
                "hop_limit": 4
            })

        # IV. Gene→Disease
        elif any(k in q for k in ["disease", "associated", "association", "بیماری"]):
            cfg.update({"intent": "G→D", "allow": ["DaG"], "end_type": "Disease", "hop_limit": 2})

        # V. Disease→Symptom / Anatomy
        if any(k in q for k in ["symptom", "علائم", "signs"]):
            cfg.update({"intent": "D→S", "allow": ["DpS"], "end_type": "Symptom", "hop_limit": 1})
        if any(k in q for k in ["anatomy", "tissue", "بافت", "anatomical", "localized"]):
            cfg.update({"intent": "D→A", "allow": ["DlA"], "end_type": "Anatomy", "hop_limit": 1})

        # VI. Drug→Target/Mechanism/Side-effect
        if any(k in q for k in ["side effect", "adverse", "عوارض"]):
            cfg.update({"intent": "C→SE", "allow": ["CcSE"], "end_type": "Side Effect", "hop_limit": 1})
        elif any(k in q for k in ["mechanism", "target", "binds", "regulates"]):
            cfg.update({"intent": "C→(G|BP|PW)", "allow": ["CbG", "PCiC", "GiG", "Gr>G", "GpPW", "GpBP"], "end_type": ("Gene|BP|PW"), "hop_limit": 2})

        # VII. Anatomy→Gene
        if any(k in q for k in ["expressed in", "expression", "بیان"]):
            cfg.update({"intent": "A→G_expression", "allow": ["AeG"], "end_type": "Gene", "hop_limit": 1})
        if any(k in q for k in ["upregulates", "downregulates", "regulates"]):
            # Anatomy regulation of Gene
            cfg.update({"intent": "A→G_regulation", "allow": ["AuG", "AdG"], "end_type": "Gene", "hop_limit": 1})

        # VIII. Pathway/BP/MF membership
        if any(k in q for k in ["pathway", "biological process", "molecular function", "go:"]):
            cfg.update({"intent": "G↔(PW|BP|MF)", "allow": ["GpPW", "GpBP", "GpMF"], "end_type": ("Gene|PW|BP|MF"), "hop_limit": 1})

        # Denylist for similarity unless explicitly requested
        if resembles:
            cfg["deny"] = [m for m in cfg["deny"] if m not in ("DrD", "CrC")]
        return cfg

    def _fallback_from_intent(self, intent: Optional[str]) -> Optional[Dict[str, Any]]:
        if not intent:
            return None
        # قواعد fallback
        if intent == "G-G_covary":
            return {"intent": "G-G_interact", "allow": ["GiG"], "end_type": "Gene", "hop_limit": 1}
        if intent == "D→(C|PC)":
            return {"intent": "D→(C|PC)_palliative", "allow": ["CpD", "PCiC"], "end_type": ("Compound|Pharmacologic Class"), "hop_limit": 3,
                    "constraints": {"require_any_edge": ["CtD", "CpD"], "require_edge_to": "Disease"}}
        if intent == "C→(G|BP|PW)":
            return {"intent": "C→PC→C", "allow": ["PCiC"], "end_type": ("Gene|BP|PW|Compound|Pharmacologic Class"), "hop_limit": 3}
        if intent == "G→(C|PC)":
            return {"intent": "G→G→(C|PC)", "allow": ["GiG", "Gr>G", "CbG", "PCiC"], "end_type": ("Compound|Pharmacologic Class"), "hop_limit": 4}
        return None

    def _find_paths_allowlist(
        self,
        core_nodes: List[str],
        allow_metaedges: List[str],
        deny_metaedges: List[str],
        end_kind: Optional[str],
        hop_limit: int,
        max_results_per_hop: int,
        require_unique_nodes: bool,
        extra_constraints: Dict[str, Any],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        جستجوی مسیرها فقط با metaedgeهای مجاز، با اعمال end-type و قیود.
        خروجی: لیست دیکشنری شامل path_nodes, path_edges و متادیتا برای رتبه‌بندی.
        """
        if not core_nodes:
            return []
        allow_set = set(allow_metaedges or [])
        deny_set = set(deny_metaedges or [])

        def valid_edge(u, v) -> Optional[str]:
            ed = self.G.get_edge_data(u, v) or {}
            meta = ed.get("metaedge") or ed.get("relation")
            if not meta:
                return None
            if meta in deny_set:
                return None
            if allow_set and meta not in allow_set:
                return None
            return meta

        results: List[Dict[str, Any]] = []
        seen_paths: set = set()

        for start in core_nodes:
            if not self.G.has_node(start):
                continue
            # DFS محدود به hop_limit و allowlist
            stack: List[Tuple[str, List[str]]] = [(start, [start])]
            per_hop_counts = [0] * (hop_limit + 1)
            while stack:
                node, path = stack.pop()
                depth = len(path) - 1
                if depth > hop_limit:
                    continue
                # انتهایی معتبر؟
                if depth >= 1:
                    if end_kind:
                        k = self.G.nodes[path[-1]].get("kind")
                        if k == end_kind or (isinstance(end_kind, str) and any(et.strip() == k for et in end_kind.split("|"))):
                            # قیود خاص Intent (مثل وجود CtD/CpD روی Disease)
                            if self._path_satisfies_constraints(path, extra_constraints):
                                key = tuple(path)
                                if key not in seen_paths:
                                    seen_paths.add(key)
                                    results.append({
                                        "path_nodes": path.copy(),
                                        "path_edges": self._edges_for_path(path),
                                        "metaedges": [valid_edge(path[i], path[i+1]) for i in range(len(path)-1)]
                                    })
                if depth == hop_limit:
                    continue
                # کنترل max_results_per_hop
                if per_hop_counts[depth] >= max_results_per_hop:
                    continue
                per_hop_counts[depth] += 1

                for nbr in self.G.neighbors(node):
                    if require_unique_nodes and nbr in path:
                        continue
                    meta = valid_edge(node, nbr)
                    if not meta:
                        continue
                    # enforce end-kind at final hop only
                    next_depth = depth + 1
                    if next_depth == hop_limit and end_kind:
                        k = self.G.nodes[nbr].get("kind")
                        if isinstance(end_kind, str):
                            end_ok = any(et.strip() == k for et in end_kind.split("|")) or (k == end_kind)
                        else:
                            end_ok = (k == end_kind)
                        if not end_ok:
                            continue
                    stack.append((nbr, path + [nbr]))

        return results

    def _edges_for_path(self, path: List[str]) -> List[Tuple[str, str, str]]:
        edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            ed = self.G.get_edge_data(u, v) or {}
            meta = ed.get("metaedge") or ed.get("relation") or "related"
            edges.append((u, v, meta))
        return edges

    def _path_satisfies_constraints(self, path: List[str], constraints: Dict[str, Any]) -> bool:
        if not constraints:
            return True
        # مثال: require_any_edge=[CtD,CpD] که به Disease متصل باشد
        req_any = constraints.get("require_any_edge") or []
        req_to = constraints.get("require_edge_to")  # نوع نودی که یال باید به آن وصل شود
        if req_any:
            ok = False
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                ed = self.G.get_edge_data(u, v) or {}
                meta = ed.get("metaedge") or ed.get("relation")
                if meta in req_any:
                    if not req_to:
                        ok = True
                        break
                    # بررسی نوع مقصد/مبدأ
                    if self.G.nodes[v].get("kind") == req_to or self.G.nodes[u].get("kind") == req_to:
                        ok = True
                        break
            if not ok:
                return False
        return True

    def _rank_paths(self, paths: List[Dict[str, Any]], query: str, intent_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        # وزن‌ها
        w_match = 0.35
        w_edge = 0.25
        w_schema = 0.20
        w_hop = 0.10
        w_hub = 0.05
        w_div = 0.05

        ql = (query or "").lower()
        allow = set(intent_cfg.get("allow", []))
        end_type = intent_cfg.get("end_type")

        def match_score(metaedges: List[Optional[str]]) -> float:
            # تطبیق ساده کلیدواژه با انواع یال
            s = 0.0
            if any(m == "GcG" for m in metaedges) and any(k in ql for k in ["covary", "co-expression", "coexpression", "هم‌واریانس", "هم‌بروز"]):
                s += 1.0
            if any(m == "GiG" for m in metaedges) and any(k in ql for k in ["interaction", "interacts", "ppi", "تعامل"]):
                s += 1.0
            if any(m == "Gr>G" for m in metaedges) and any(k in ql for k in ["regulates", "regulation", "تنظیم"]):
                s += 1.0
            if any(m == "CtD" for m in metaedges) and any(k in ql for k in ["treats", "درمان"]):
                s += 1.0
            if any(m == "CbG" for m in metaedges) and any(k in ql for k in ["binds", "target", "mechanism"]):
                s += 1.0
            return min(s, 1.0)

        def edge_strength_score(path_edges: List[Tuple[str, str, str]]) -> float:
            # شواهد/بی‌طرفی اگر موجود باشد، در غیر این صورت از نوع یال امتیاز بگیر
            total = 0.0
            for u, v, meta in path_edges:
                ed = self.G.get_edge_data(u, v) or {}
                evc = ed.get("evidence_count") or ed.get("evidence") or 0
                unbiased = 1.0 if ed.get("unbiased") else 0.0
                base = 1.0
                if meta == "CtD":
                    base = 1.2
                elif meta == "CpD":
                    base = 1.0
                elif meta == "GcG":
                    # اگر متریک کوواریانس موجود بود (مثل cov_metric/weight)، در امتیاز اثر بده
                    cov = ed.get("cov_metric") or ed.get("weight") or 0
                    try:
                        cov = float(cov)
                    except Exception:
                        cov = 0
                    base = 1.0 + 0.5 * max(0.0, min(1.0, cov))
                total += base + 0.05 * float(evc) + 0.1 * unbiased
            return total / max(1, len(path_edges))

        def schema_fit_score(metaedges: List[Optional[str]]) -> float:
            if not metaedges:
                return 0.0
            ok = sum(1 for m in metaedges if m in allow)
            return ok / len(metaedges)

        def hop_penalty(num_hops: int) -> float:
            # بیشینه 1.0 برای کوتاه‌ترین‌ها
            if num_hops <= 1:
                return 1.0
            if num_hops == 2:
                return 0.8
            if num_hops == 3:
                return 0.6
            return 0.4

        def hub_penalty(path_nodes: List[str]) -> float:
            # جریمه برای ژن‌های با درجه بالا
            import math
            gene_nodes = [n for n in path_nodes if self.G.nodes[n].get("kind") == "Gene"]
            if not gene_nodes:
                return 1.0
            vals = []
            for n in gene_nodes:
                d = self.G.degree(n)
                vals.append(1.0 / (1.0 + math.log(1 + d)))
            return sum(vals) / len(vals)

        # نرمال‌سازی تنوع روی end-type
        seen_ends: set = set()
        ranked = []
        for item in paths:
            nodes = item["path_nodes"]
            edges = item["path_edges"]
            metas = item.get("metaedges", [])
            num_hops = max(0, len(nodes) - 1)
            ms = match_score(metas)
            es = edge_strength_score(edges)
            ss = schema_fit_score(metas)
            hp = hop_penalty(num_hops)
            hb = hub_penalty(nodes)
            base = w_match * ms + w_edge * es + w_schema * ss + w_hop * hp + w_hub * hb

            end_id = nodes[-1] if nodes else None
            end_kind_ok = (self.G.nodes[end_id].get("kind") if end_id and self.G.has_node(end_id) else None)
            div_bonus = 0.0
            if end_id:
                key = (end_kind_ok, end_id)
                if key not in seen_ends:
                    div_bonus = w_div * 1.0
                    seen_ends.add(key)
            score = base + div_bonus

            ranked.append({**item, "score": float(score), "notes": f"{num_hops} hops; schema-fit={ss:.2f}"})

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked
    
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

    # توابع کمکی برای تولید متن ساده و عمومی
    def _generate_general_knowledge_answer(self, query: str) -> str:
        """تولید پاسخ عمومی بر اساس دانش مدل"""
        query_lower = query.lower()
        
        if "cancer" in query_lower and "tissue" in query_lower:
            return """سرطان می‌تواند به روش‌های مختلف بر بافت‌های بدن تأثیر بگذارد:

**مکانیسم‌های اصلی:**
• **تهاجم مستقیم:** سلول‌های سرطانی به بافت‌های مجاور نفوذ می‌کنند
• **متاستاز:** گسترش سرطان به بافت‌های دورتر از طریق خون یا لنف
• **تغییرات متابولیک:** افزایش مصرف انرژی و تغییر در متابولیسم بافت
• **التهاب:** پاسخ ایمنی که می‌تواند به بافت آسیب برساند
• **فشار مکانیکی:** تومورها می‌توانند بر بافت‌های مجاور فشار وارد کنند

**اثرات بر بافت‌های مختلف:**
• **بافت‌های نرم:** تغییر در ساختار و عملکرد
• **استخوان:** ضعیف شدن و شکستگی
• **عروق خونی:** رشد عروق جدید برای تغذیه تومور
• **سیستم ایمنی:** تغییر در پاسخ‌های ایمنی

**علائم بالینی:**
• درد، تورم، تغییر در عملکرد عضو
• کاهش وزن، خستگی، ضعف عمومی"""

        elif "gene" in query_lower:
            return """ژن‌ها واحدهای وراثتی هستند که اطلاعات ژنتیکی را حمل می‌کنند:

**نقش‌های اصلی:**
• **کدگذاری پروتئین:** تولید پروتئین‌های مورد نیاز سلول
• **تنظیم فرآیندها:** کنترل متابولیسم و رشد سلولی
• **پاسخ به محیط:** تنظیم پاسخ‌های سلولی به تغییرات محیطی

**انواع ژن‌ها:**
• **ژن‌های ساختاری:** تولید پروتئین‌های ساختاری
• **ژن‌های تنظیمی:** کنترل بیان سایر ژن‌ها
• **ژن‌های آنزیمی:** تولید آنزیم‌های متابولیک"""

        else:
            return """بر اساس سوال شما، اطلاعات مرتبط در حوزه زیست‌پزشکی وجود دارد. برای پاسخ دقیق‌تر، لطفاً سوال خود را به شکل مشخص‌تری مطرح کنید یا از روش‌های بازیابی استفاده کنید."""

    def _generate_cancer_related_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات مرتبط با سرطان"""
        query = retrieval_result.query
        context = retrieval_result.context_text
        
        # استخراج اطلاعات مرتبط از context
        cancer_info = self._extract_cancer_info_from_context(context)
        
        return f"""🏥 **تحلیل سرطان و اثرات آن**

**سوال:** {query}

**اطلاعات یافت شده:**
{cancer_info}

**تحلیل کلی:**
سرطان می‌تواند از طریق مکانیسم‌های مختلف بر بافت‌ها تأثیر بگذارد:

• **تغییرات ژنتیکی:** جهش‌های ژنتیکی که منجر به رشد غیرطبیعی سلول‌ها می‌شود
• **تغییرات متابولیک:** افزایش مصرف انرژی و تغییر در مسیرهای متابولیک
• **اثرات بر بافت:** تغییر در ساختار و عملکرد بافت‌های درگیر
• **پاسخ ایمنی:** تغییر در پاسخ‌های سیستم ایمنی

💡 **نکته:** این اطلاعات بر اساس داده‌های گراف دانش استخراج شده است."""

    def _generate_gene_related_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات مرتبط با ژن‌ها"""
        query = retrieval_result.query
        context = retrieval_result.context_text
        
        gene_info = self._extract_gene_info_from_context(context)
        
        return f"""🧬 **تحلیل ژن‌ها و عملکرد آنها**

**سوال:** {query}

**ژن‌های یافت شده:**
{gene_info}

**نقش‌های بیولوژیکی:**
• **تنظیم بیان ژن:** کنترل فرآیندهای سلولی
• **متابولیسم:** شرکت در مسیرهای متابولیک
• **سیگنالینگ:** انتقال پیام‌های سلولی
• **ساختار سلولی:** حفظ ساختار و عملکرد سلول

💡 **نکته:** این اطلاعات بر اساس روابط موجود در گراف دانش استخراج شده است."""

    def _generate_drug_related_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات مرتبط با داروها"""
        query = retrieval_result.query
        context = retrieval_result.context_text
        
        drug_info = self._extract_drug_info_from_context(context)
        
        return f"""💊 **تحلیل داروها و درمان‌ها**

**سوال:** {query}

**داروهای یافت شده:**
{drug_info}

**مکانیسم‌های درمانی:**
• **مهار رشد سلولی:** جلوگیری از تکثیر سلول‌های سرطانی
• **تحریک سیستم ایمنی:** تقویت پاسخ‌های ایمنی
• **مهار آنژیوژنز:** جلوگیری از رشد عروق خونی تومور
• **القای آپوپتوز:** مرگ برنامه‌ریزی شده سلول‌های سرطانی

💡 **نکته:** این اطلاعات بر اساس روابط دارو-بیماری در گراف دانش استخراج شده است."""

    def _generate_disease_related_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات مرتبط با بیماری‌ها"""
        query = retrieval_result.query
        context = retrieval_result.context_text
        
        disease_info = self._extract_disease_info_from_context(context)
        
        return f"""🏥 **تحلیل بیماری‌ها و علل آنها**

**سوال:** {query}

**بیماری‌های یافت شده:**
{disease_info}

**مکانیسم‌های بیماری‌زایی:**
• **تغییرات ژنتیکی:** جهش‌های ژنتیکی مؤثر در بیماری
• **اختلالات متابولیک:** تغییر در مسیرهای متابولیک
• **التهاب:** پاسخ‌های التهابی غیرطبیعی
• **اختلالات ساختاری:** تغییر در ساختار بافت‌ها

💡 **نکته:** این اطلاعات بر اساس روابط بیماری-ژن در گراف دانش استخراج شده است."""

    def _generate_tissue_related_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ برای سوالات مرتبط با بافت‌ها"""
        query = retrieval_result.query
        context = retrieval_result.context_text
        
        tissue_info = self._extract_tissue_info_from_context(context)
        
        return f"""🔬 **تحلیل بافت‌ها و عملکرد آنها**

**سوال:** {query}

**بافت‌های یافت شده:**
{tissue_info}

**نقش‌های بیولوژیکی:**
• **ساختار و پشتیبانی:** حفظ شکل و ساختار اندام‌ها
• **متابولیسم:** شرکت در فرآیندهای متابولیک
• **سیگنالینگ:** انتقال پیام‌های سلولی
• **محافظت:** محافظت از اندام‌های داخلی

💡 **نکته:** این اطلاعات بر اساس روابط بافت-ژن در گراف دانش استخراج شده است."""

    def _generate_general_structured_answer(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ ساختاریافته عمومی"""
        query = retrieval_result.query
        context = retrieval_result.context_text
        
        # خلاصه اطلاعات یافت شده
        summary = self._create_context_summary(context)
        
        return f"""📊 **تحلیل اطلاعات یافت شده**

**سوال:** {query}

**خلاصه اطلاعات:**
{summary}

**تحلیل کلی:**
بر اساس داده‌های گراف دانش، اطلاعات مرتبط با سوال شما یافت شده است. این اطلاعات شامل روابط بین موجودیت‌های زیستی مختلف است که می‌تواند به درک بهتر موضوع کمک کند.

💡 **نکته:** برای اطلاعات دقیق‌تر، می‌توانید از روش‌های بازیابی پیشرفته استفاده کنید."""

    def _extract_cancer_info_from_context(self, context: str) -> str:
        """استخراج اطلاعات سرطان از context"""
        if not context:
            return "اطلاعات کافی یافت نشد."
        
        # استخراج ژن‌ها و بیماری‌های مرتبط
        lines = context.split('\n')
        cancer_related = []
        
        for line in lines:
            if any(word in line.lower() for word in ['cancer', 'tumor', 'malignant']):
                cancer_related.append(line.strip())
        
        if cancer_related:
            return "\n".join(cancer_related[:10])  # حداکثر 10 خط
        else:
            return "اطلاعات مرتبط با سرطان در داده‌های یافت شده محدود است."

    def _extract_gene_info_from_context(self, context: str) -> str:
        """استخراج اطلاعات ژن از context"""
        if not context:
            return "اطلاعات کافی یافت نشد."
        
        lines = context.split('\n')
        gene_related = []
        
        for line in lines:
            if 'gene' in line.lower() or any(word in line.lower() for word in ['express', 'regulate', 'function']):
                gene_related.append(line.strip())
        
        if gene_related:
            return "\n".join(gene_related[:10])
        else:
            return "اطلاعات مرتبط با ژن‌ها در داده‌های یافت شده محدود است."

    def _extract_drug_info_from_context(self, context: str) -> str:
        """استخراج اطلاعات دارو از context"""
        if not context:
            return "اطلاعات کافی یافت نشد."
        
        lines = context.split('\n')
        drug_related = []
        
        for line in lines:
            if any(word in line.lower() for word in ['drug', 'compound', 'medicine', 'treat']):
                drug_related.append(line.strip())
        
        if drug_related:
            return "\n".join(drug_related[:10])
        else:
            return "اطلاعات مرتبط با داروها در داده‌های یافت شده محدود است."

    def _extract_disease_info_from_context(self, context: str) -> str:
        """استخراج اطلاعات بیماری از context"""
        if not context:
            return "اطلاعات کافی یافت نشد."
        
        lines = context.split('\n')
        disease_related = []
        
        for line in lines:
            if any(word in line.lower() for word in ['disease', 'disorder', 'condition', 'symptom']):
                disease_related.append(line.strip())
        
        if disease_related:
            return "\n".join(disease_related[:10])
        else:
            return "اطلاعات مرتبط با بیماری‌ها در داده‌های یافت شده محدود است."

    def _extract_tissue_info_from_context(self, context: str) -> str:
        """استخراج اطلاعات بافت از context"""
        if not context:
            return "اطلاعات کافی یافت نشد."
        
        lines = context.split('\n')
        tissue_related = []
        
        for line in lines:
            if any(word in line.lower() for word in ['tissue', 'organ', 'anatomy', 'heart', 'brain', 'liver']):
                tissue_related.append(line.strip())
        
        if tissue_related:
            return "\n".join(tissue_related[:10])
        else:
            return "اطلاعات مرتبط با بافت‌ها در داده‌های یافت شده محدود است."

    def _create_context_summary(self, context: str) -> str:
        """خلاصه‌سازی context"""
        if not context:
            return "اطلاعات کافی یافت نشد."
        
        lines = context.split('\n')
        summary_lines = []
        
        # شمارش انواع موجودیت‌ها
        gene_count = sum(1 for line in lines if 'gene' in line.lower())
        disease_count = sum(1 for line in lines if 'disease' in line.lower())
        drug_count = sum(1 for line in lines if 'drug' in line.lower() or 'compound' in line.lower())
        
        if gene_count > 0:
            summary_lines.append(f"• ژن‌های مرتبط: {gene_count} مورد")
        if disease_count > 0:
            summary_lines.append(f"• بیماری‌های مرتبط: {disease_count} مورد")
        if drug_count > 0:
            summary_lines.append(f"• داروهای مرتبط: {drug_count} مورد")
        
        if summary_lines:
            return "\n".join(summary_lines)
        else:
            return "اطلاعات کلی در دسترس است."

    # ========================================
    # روش‌های تولید متن هوشمند
    # ========================================
    
    def meta_llama_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ با Meta Llama 3.1 (محلی)"""
        try:
            # این متد نیاز به نصب Llama 3.1 دارد
            # برای استفاده نیاز به API Key یا نصب محلی است
            return "🔧 Meta Llama 3.1 در حال توسعه است.\n\n" + self._fallback_generation(retrieval_result, "Meta Llama 3.1")
        except Exception as e:
            print(f"خطا در Meta Llama 3.1: {e}")
            return self._fallback_generation(retrieval_result, "Meta Llama 3.1")
    
    def mistral_ai_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ با Mistral AI (کیفیت بالا)"""
        try:
            # این متد نیاز به API Key از Mistral AI دارد
            return "🔧 Mistral AI در حال توسعه است.\n\n" + self._fallback_generation(retrieval_result, "Mistral AI")
        except Exception as e:
            print(f"خطا در Mistral AI: {e}")
            return self._fallback_generation(retrieval_result, "Mistral AI")
    
    def cohere_command_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ با Cohere Command (تخصصی)"""
        try:
            # این متد نیاز به API Key از Cohere دارد
            return "🔧 Cohere Command در حال توسعه است.\n\n" + self._fallback_generation(retrieval_result, "Cohere Command")
        except Exception as e:
            print(f"خطا در Cohere Command: {e}")
            return self._fallback_generation(retrieval_result, "Cohere Command")
    
    def perplexity_sonar_generation(self, retrieval_result: RetrievalResult) -> str:
        """تولید پاسخ با Perplexity Sonar (جستجوگر)"""
        try:
            # این متد نیاز به API Key از Perplexity دارد
            return "🔧 Perplexity Sonar در حال توسعه است.\n\n" + self._fallback_generation(retrieval_result, "Perplexity Sonar")
        except Exception as e:
            print(f"خطا در Perplexity Sonar: {e}")
            return self._fallback_generation(retrieval_result, "Perplexity Sonar")

# نمونه استفاده
if __name__ == "__main__":
    service = GraphRAGService()
    
    # تست سرویس
    result = service.process_query(
        query="What is the relationship between HMGB3 and diabetes?",
        retrieval_method=RetrievalMethod.BFS,
        generation_model=GenerationModel.GPT_SIMULATION,
        text_generation_type='INTELLIGENT'
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False)) 
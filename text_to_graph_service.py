# -*- coding: utf-8 -*-
"""
Text to Graph Service - سرویس تبدیل متن به گراف دانش
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from urllib.parse import quote
import networkx as nx
import os

# Try to import spaCy
try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

# Try to import HuggingFace
try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    InferenceClient = None

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Import graph prompts
try:
    from graphrag_new.general.graph_prompt import GRAPH_PROMPTS, CONTINUE_PROMPT, LOOP_PROMPT
except ImportError:
    GRAPH_PROMPTS = {}
    CONTINUE_PROMPT = "MANY entities were missed in the last extraction. Add them below using the same format:\n"
    LOOP_PROMPT = "It appears some entities may have still been missed. Answer Y if there are still entities that need to be added, or N if there are none. Please answer with a single letter Y or N.\n"

# Import entity resolution
try:
    from graphrag_new.entity_resolution import EntityResolution
    ENTITY_RESOLUTION_AVAILABLE = True
except ImportError:
    ENTITY_RESOLUTION_AVAILABLE = False
    EntityResolution = None

# Import Persian normalizer and language detection
try:
    from persian_normalizer import PersianNormalizer, detect_language, is_persian
    PERSIAN_NORMALIZER_AVAILABLE = True
except ImportError:
    PERSIAN_NORMALIZER_AVAILABLE = False
    PersianNormalizer = None
    detect_language = None
    is_persian = None

# Import new extraction modules
try:
    from smart_chunker import SmartChunker, ChunkingStrategy, SlidingWindowProcessor
    from hierarchical_merger import HierarchicalMerger
    from modular_pipeline import ModularExtractionPipeline
    from span_based_extractor import SpanBasedExtractor, BioBERTExtractor, SciBERTExtractor
    from bert_relation_extractor import BERTRelationExtractor
    from coreference_resolver import CoreferenceResolver
    from persian_models import PersianNERModel, PersianRelationExtractor
    NEW_MODULES_AVAILABLE = True
except ImportError as e:
    NEW_MODULES_AVAILABLE = False
    logging.warning(f"Some new modules not available: {e}")


class TextToGraphService:
    """سرویس تبدیل متن به گراف دانش"""
    
    def __init__(self, openai_api_key: Optional[str] = None, spacy_model: str = "en_core_web_sm", hf_token: Optional[str] = None):
        """
        Initialize the service
        
        Args:
            openai_api_key: API key for OpenAI (optional)
            spacy_model: spaCy model name (default: en_core_web_sm)
            hf_token: HuggingFace token (optional, can also be set via environment variable)
        """
        self.openai_api_key = openai_api_key
        self.spacy_model = spacy_model
        self._hf_token_param = hf_token  # Store token parameter
        self.nlp = None
        self.nlp_fa = None  # Persian spaCy model
        
        # Initialize Persian normalizer if available
        self.persian_normalizer = None
        if PERSIAN_NORMALIZER_AVAILABLE:
            try:
                self.persian_normalizer = PersianNormalizer(enable_spell_check=False)
            except Exception as e:
                logging.warning(f"Failed to initialize PersianNormalizer: {e}")
        
        # Initialize spaCy if available
        if SPACY_AVAILABLE:
            # Load English model
            try:
                self.nlp = spacy.load(spacy_model)
                logging.info(f"Loaded spaCy model: {spacy_model}")
            except OSError:
                try:
                    # Try biomedical model
                    self.nlp = spacy.load("en_ner_bionlp13cg_md")
                    logging.info("Loaded biomedical spaCy model")
                except OSError:
                    logging.warning(f"spaCy model {spacy_model} not found. spaCy extraction will be disabled.")
                    self.nlp = None
            
            # Try to load Persian model
            try:
                self.nlp_fa = spacy.load("fa_core_news_sm")
                logging.info("Loaded Persian spaCy model: fa_core_news_sm")
            except OSError:
                logging.warning("Persian spaCy model (fa_core_news_sm) not found. Install with: python -m spacy download fa_core_news_sm")
                self.nlp_fa = None
        
        # Initialize OpenAI if available (currently disabled due to token issues)
        self.openai_client = None
        # if OPENAI_AVAILABLE and openai_api_key:
        #     try:
        #         self.openai_client = openai.OpenAI(api_key=openai_api_key)
        #     except Exception as e:
        #         logging.warning(f"Failed to initialize OpenAI client: {e}")
        #         self.openai_client = None
        
        # Initialize HuggingFace if available
        self.hf_client = None
        self.hf_api_key = None
        if HUGGINGFACE_AVAILABLE:
            # Reload environment variables to ensure .env is loaded
            try:
                from dotenv import load_dotenv
                load_dotenv(override=True)
            except Exception:
                pass

            # Try to get HF token from parameter, then environment
            hf_token = self._hf_token_param or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
            if hf_token:
                try:
                    self.hf_client = InferenceClient(provider="auto", api_key=hf_token)
                    self.hf_api_key = hf_token
                    logging.info("HuggingFace InferenceClient initialized successfully")
                except Exception as e:
                    logging.warning(f"Failed to initialize HuggingFace client: {e}")
                    self.hf_client = None
            else:
                logging.warning("HF_TOKEN not found in environment variables. HuggingFace features will be disabled.")
                logging.warning(
                    f"Current environment: HF_TOKEN={os.environ.get('HF_TOKEN', 'NOT SET')}, "
                    f"HUGGINGFACE_API_KEY={os.environ.get('HUGGINGFACE_API_KEY', 'NOT SET')}"
                )
        else:
            logging.warning("huggingface_hub not available. Please install it with: pip install huggingface_hub")
        
        # Initialize entity resolution if available
        self.entity_resolution = None
        if ENTITY_RESOLUTION_AVAILABLE:
            try:
                self.entity_resolution = EntityResolution(similarity_threshold=0.8)
            except Exception as e:
                logging.warning(f"Failed to initialize EntityResolution: {e}")
                self.entity_resolution = None
    
    def _detect_text_language(self, text: str) -> str:
        """
        تشخیص زبان متن
        
        Args:
            text: متن ورودی
            
        Returns:
            'fa' برای فارسی، 'en' برای انگلیسی، 'mixed' برای ترکیبی
        """
        if PERSIAN_NORMALIZER_AVAILABLE and detect_language:
            return detect_language(text)
        
        # Fallback: simple detection
        persian_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = persian_chars + latin_chars
        
        if total_chars == 0:
            return 'en'
        
        persian_ratio = persian_chars / total_chars if total_chars > 0 else 0
        
        if persian_ratio > 0.5:
            return 'fa'
        elif persian_ratio > 0.1:
            return 'mixed'
        return 'en'
    
    def _get_spacy_model(self, language: str = "auto"):
        """
        دریافت مدل spaCy مناسب بر اساس زبان
        
        Args:
            language: زبان (auto/fa/en)
            
        Returns:
            مدل spaCy مناسب یا None
        """
        if language == "auto":
            return self.nlp  # Default to English
        
        if language == "fa" and self.nlp_fa:
            return self.nlp_fa
        
        return self.nlp  # Fallback to English
    
    def _preprocess_text_for_graph(self, text: str, language: str = "auto", 
                                    remove_stop_words: bool = True) -> str:
        """
        پیش‌پردازش متن برای استخراج گراف
        
        نکته مهم: این تابع stop words را حذف می‌کند تا در گراف ظاهر نشوند،
        اما متن اصلی باید برای مدل‌های زبانی (LLM) حفظ شود تا معنی کامل حفظ شود.
        
        Args:
            text: متن ورودی
            language: زبان متن (auto/fa/en)
            remove_stop_words: آیا stop words حذف شوند؟
            
        Returns:
            متن پیش‌پردازش شده
        """
        if not remove_stop_words:
            return text
        
        # تشخیص خودکار زبان اگر لازم باشد
        if language == "auto":
            language = self._detect_text_language(text)
        
        # اگر spaCy در دسترس است، از آن استفاده کن
        if SPACY_AVAILABLE:
            nlp_model = self._get_spacy_model(language)
            if nlp_model:
                try:
                    doc = nlp_model(text)
                    # حذف stop words و علائم نگارشی
                    filtered_tokens = [
                        token.text for token in doc 
                        if not token.is_stop and not token.is_punct and not token.is_space
                    ]
                    return " ".join(filtered_tokens)
                except Exception as e:
                    logging.warning(f"Error in spaCy preprocessing: {e}")
        
        # Fallback: حذف ساده stop words با regex
        # Stop words فارسی رایج
        persian_stop_words = {
            'از', 'به', 'در', 'که', 'این', 'آن', 'با', 'برای', 'تا', 'را', 'هم', 'یا',
            'اما', 'ولی', 'اگر', 'چون', 'چرا', 'چگونه', 'چه', 'کجا', 'کی', 'چی',
            'است', 'هست', 'بود', 'باش', 'شود', 'می', 'می‌شود', 'می‌کند', 'می‌کنند',
            'یک', 'دو', 'سه', 'چند', 'همه', 'هیچ', 'بعضی', 'هر', 'همین', 'همان'
        }
        
        # Stop words انگلیسی
        english_stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its',
            'our', 'their', 'what', 'which', 'who', 'whom', 'whose', 'where',
            'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'now'
        }
        
        # انتخاب stop words مناسب
        if language == "fa":
            stop_words = persian_stop_words
        elif language == "en":
            stop_words = english_stop_words
        else:  # mixed
            stop_words = persian_stop_words | english_stop_words
        
        # تقسیم متن به کلمات و حذف stop words
        words = text.split()
        filtered_words = [
            word for word in words 
            if word.lower().strip('.,!?;:()[]{}"\'-') not in stop_words
            and len(word.strip('.,!?;:()[]{}"\'-')) > 1  # حذف کاراکترهای تکی
        ]
        
        return " ".join(filtered_words)
    
    def extract_simple(self, text: str, max_entities: int = 100, max_relationships: int = 200) -> Dict[str, Any]:
        """
        استخراج موجودیت‌ها و روابط با روش ساده (Rule-based)
        
        Args:
            text: متن ورودی
            max_entities: حداکثر تعداد موجودیت‌ها
            max_relationships: حداکثر تعداد روابط
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not text or not text.strip():
            raise ValueError("متن ورودی نمی‌تواند خالی باشد")
        
        if max_entities < 1 or max_entities > 10000:
            raise ValueError("حداکثر موجودیت‌ها باید بین 1 تا 10000 باشد")
        
        if max_relationships < 1 or max_relationships > 20000:
            raise ValueError("حداکثر روابط باید بین 1 تا 20000 باشد")
        
        entities = []
        relationships = []
        entity_map = {}  # Map entity names to IDs
        
        # Patterns for entity extraction
        # Gene patterns (uppercase letters and numbers)
        gene_pattern = r'\b[A-Z][A-Z0-9]+\b'
        # Disease patterns (common disease keywords)
        disease_keywords = ['cancer', 'disease', 'syndrome', 'disorder', 'tumor', 'carcinoma']
        # Compound patterns (drug names often end with specific suffixes)
        compound_pattern = r'\b[a-z]+(?:ib|mab|zumab|umab|tinib|olol|pril|sartan)\b'
        
        # Extract potential genes
        genes = re.findall(gene_pattern, text)
        for gene in set(genes):
            if len(gene) >= 2 and gene not in entity_map:
                entity_id = f"GENE_{len(entities)}"
                entities.append({
                    "id": entity_id,
                    "name": gene,
                    "type": "Gene",
                    "attributes": {}
                })
                entity_map[gene] = entity_id
        
        # Extract diseases (simple keyword matching)
        text_lower = text.lower()
        for keyword in disease_keywords:
            pattern = rf'\b[\w\s]+{keyword}\b'
            matches = re.findall(pattern, text_lower)
            for match in matches:
                disease_name = match.strip()
                if disease_name and disease_name not in entity_map:
                    entity_id = f"DISEASE_{len(entities)}"
                    entities.append({
                        "id": entity_id,
                        "name": disease_name.title(),
                        "type": "Disease",
                        "attributes": {}
                    })
                    entity_map[disease_name] = entity_id
        
        # Extract compounds
        compounds = re.findall(compound_pattern, text_lower)
        for compound in set(compounds):
            if compound not in entity_map:
                entity_id = f"COMPOUND_{len(entities)}"
                entities.append({
                    "id": entity_id,
                    "name": compound,
                    "type": "Compound",
                    "attributes": {}
                })
                entity_map[compound] = entity_id
        
        # Extract relationships using simple patterns
        # Pattern: "X regulates Y", "X interacts with Y", "X treats Y"
        relationship_patterns = [
            (r'(\w+)\s+(?:regulates?|controls?)\s+(\w+)', 'Gr>G'),
            (r'(\w+)\s+(?:interacts?\s+with|binds?\s+to)\s+(\w+)', 'GiG'),
            (r'(\w+)\s+(?:treats?|cures?)\s+([\w\s]+)', 'CtD'),
            (r'(\w+)\s+(?:participates?\s+in|involved\s+in)\s+([\w\s]+)', 'GpBP'),
            (r'(\w+)\s+(?:associates?\s+with|related\s+to)\s+(\w+)', 'DaG'),
        ]
        
        for pattern, relation_type in relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                source = match.group(1)
                target = match.group(2).strip()
                
                # Check if both entities exist
                source_id = None
                target_id = None
                
                # Try to find matching entities
                for entity_name, entity_id in entity_map.items():
                    if source.lower() in entity_name.lower() or entity_name.lower() in source.lower():
                        source_id = entity_id
                    if target.lower() in entity_name.lower() or entity_name.lower() in target.lower():
                        target_id = entity_id
                
                # If entities don't exist, create them
                if not source_id:
                    source_id = f"ENTITY_{len(entities)}"
                    entities.append({
                        "id": source_id,
                        "name": source,
                        "type": "Gene",  # Default type
                        "attributes": {}
                    })
                    entity_map[source] = source_id
                
                if not target_id:
                    target_id = f"ENTITY_{len(entities)}"
                    entities.append({
                        "id": target_id,
                        "name": target,
                        "type": "Gene",  # Default type
                        "attributes": {}
                    })
                    entity_map[target] = target_id
                
                # استخراج نام رابطه از متن (از pattern matched)
                relation_name = match.group(0).strip()  # کل pattern matched (مثلاً "regulates", "interacts with")
                
                # Add relationship
                if source_id and target_id:
                    relationships.append({
                        "source": source_id,
                        "target": target_id,
                        "metaedge": relation_type,
                        "relation": relation_name,  # نام واقعی رابطه از متن
                        "attributes": {
                            "pattern": match.group(0),
                            "description": relation_name
                        }
                    })
        
        # Limit results
        entities = entities[:max_entities]
        relationships = relationships[:max_relationships]
        
        return {
            "entities": entities,
            "relationships": relationships,
            "method": "simple",
            "stats": {
                "num_entities": len(entities),
                "num_relationships": len(relationships)
            }
        }
    
    def extract_spacy(self, text: str, max_entities: int = 100, max_relationships: int = 200) -> Dict[str, Any]:
        """
        استخراج موجودیت‌ها و روابط با استفاده از spaCy
        
        Args:
            text: متن ورودی
            max_entities: حداکثر تعداد موجودیت‌ها
            max_relationships: حداکثر تعداد روابط
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not text or not text.strip():
            raise ValueError("متن ورودی نمی‌تواند خالی باشد")
        
        if max_entities < 1 or max_entities > 10000:
            raise ValueError("حداکثر موجودیت‌ها باید بین 1 تا 10000 باشد")
        
        if max_relationships < 1 or max_relationships > 20000:
            raise ValueError("حداکثر روابط باید بین 1 تا 20000 باشد")
        
        if not self.nlp:
            raise ValueError("spaCy model not loaded. Please install spaCy and download a model.")
        
        entities = []
        relationships = []
        entity_map = {}
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract named entities
        for ent in doc.ents:
            entity_name = ent.text.strip()
            if not entity_name or entity_name in entity_map:
                continue
            
            # Determine entity type based on spaCy label
            entity_type = self._map_spacy_label_to_type(ent.label_)
            
            entity_id = f"{entity_type}_{len(entities)}"
            entities.append({
                "id": entity_id,
                "name": entity_name,
                "type": entity_type,
                "attributes": {
                    "spacy_label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char
                }
            })
            entity_map[entity_name] = entity_id
        
        # Extract noun chunks as potential entities
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if len(chunk_text.split()) <= 5 and chunk_text not in entity_map:
                # Check if it's not a stop word
                if not all(word.lower() in STOP_WORDS for word in chunk_text.split()):
                    entity_id = f"ENTITY_{len(entities)}"
                    entities.append({
                        "id": entity_id,
                        "name": chunk_text,
                        "type": "Gene",  # Default type
                        "attributes": {
                            "source": "noun_chunk"
                        }
                    })
                    entity_map[chunk_text] = entity_id
        
        # Extract relationships using dependency parsing
        # Process sentence by sentence for better accuracy
        for sent in doc.sents:
            # Method 1: Extract relationships from verb patterns
            for token in sent:
                if token.pos_ == "VERB" and not token.is_stop:
                    # Get subject and object
                    subject_phrase = self._extract_subject_phrase(token, doc)
                    object_phrase = self._extract_object_phrase(token, doc)
                    
                    if subject_phrase and object_phrase:
                        # Map verb to relationship type
                        verb_lower = token.text.lower()
                        relation_type = self._map_verb_to_relation(verb_lower)
                        
                        # If no specific relation type, use generic
                        if not relation_type:
                            relation_type = "GiG"  # Generic interaction
                        
                        # Find entities by matching phrases
                        subject_id = self._find_entity_by_phrase(subject_phrase, entity_map)
                        obj_id = self._find_entity_by_phrase(object_phrase, entity_map)
                        
                        if subject_id and obj_id:
                            # Check if relationship already exists
                            if not any(r['source'] == subject_id and r['target'] == obj_id 
                                      and r['metaedge'] == relation_type for r in relationships):
                                # Extract relation name from text (verb or verb phrase)
                                relation_name = self._extract_relation_name_from_text(token, sent)
                                
                                relationships.append({
                                    "source": subject_id,
                                    "target": obj_id,
                                    "metaedge": relation_type,
                                    "relation": relation_name,  # نام واقعی رابطه از متن
                                    "attributes": {
                                        "verb": token.text,
                                        "verb_lemma": token.lemma_,
                                        "confidence": 0.7,
                                        "sentence": sent.text[:100],
                                        "description": relation_name
                                    }
                                })
            
            # Method 2: Preposition-based patterns (X with Y, X in Y, etc.)
            for token in sent:
                if token.pos_ == "ADP" and token.head.pos_ == "VERB":
                    verb = token.head
                    verb_text = verb.text.lower()
                    
                    if any(word in verb_text for word in ['partner', 'collaborate', 'work', 'cooperate', 
                                                           'interact', 'participate', 'associate', 'relate',
                                                           'develop', 'sign', 'agree', 'expand']):
                        # Find the subject of the verb
                        subject_phrase = self._extract_subject_phrase(verb, doc)
                        # Find the object of the preposition
                        object_phrase = None
                        for child in token.children:
                            if child.dep_ == "pobj":
                                object_phrase = self._get_full_noun_phrase(child, doc)
                                break
                        
                        if subject_phrase and object_phrase:
                            relation_type = self._map_verb_to_relation(verb_text)
                            if not relation_type:
                                relation_type = "GiG"
                            
                            subject_id = self._find_entity_by_phrase(subject_phrase, entity_map)
                            obj_id = self._find_entity_by_phrase(object_phrase, entity_map)
                            
                            if subject_id and obj_id:
                                if not any(r['source'] == subject_id and r['target'] == obj_id 
                                          and r['metaedge'] == relation_type for r in relationships):
                                    # Extract relation name from text
                                    relation_name = self._extract_relation_name_from_text(verb, sent, preposition=token.text)
                                    
                                    relationships.append({
                                        "source": subject_id,
                                        "target": obj_id,
                                        "metaedge": relation_type,
                                        "relation": relation_name,  # نام واقعی رابطه از متن
                                        "attributes": {
                                            "verb": verb.text,
                                            "verb_lemma": verb.lemma_,
                                            "preposition": token.text,
                                            "confidence": 0.65,
                                            "sentence": sent.text[:100],
                                            "description": relation_name
                                        }
                                    })
            
            # Method 3: Entity co-occurrence in same sentence with action verbs
            # Find all entities in this sentence
            entities_in_sentence = []
            for ent in sent.ents:
                entity_name = ent.text.strip()
                if entity_name in entity_map:
                    entities_in_sentence.append((entity_map[entity_name], entity_name, ent.start))
            
            # If we have 2+ entities in sentence, try to find relationships
            if len(entities_in_sentence) >= 2:
                # Look for verbs between entities
                for token in sent:
                    if token.pos_ == "VERB" and not token.is_stop:
                        verb_text = token.text.lower()
                        relation_type = self._map_verb_to_relation(verb_text)
                        if not relation_type:
                            # Use generic for common verbs
                            if any(word in verb_text for word in ['partner', 'collaborate', 'develop', 
                                                                   'sign', 'agree', 'expand', 'hire', 
                                                                   'supply', 'provide', 'migrate']):
                                relation_type = "GiG"
                        
                        if relation_type:
                            # Find entities before and after verb
                            verb_idx = token.i
                            before_entities = [(eid, name) for eid, name, start in entities_in_sentence 
                                             if start < verb_idx]
                            after_entities = [(eid, name) for eid, name, start in entities_in_sentence 
                                            if start > verb_idx]
                            
                            # Create relationships between before and after entities
                            for subj_id, subj_name in before_entities:
                                for obj_id, obj_name in after_entities:
                                    if not any(r['source'] == subj_id and r['target'] == obj_id 
                                              and r['metaedge'] == relation_type for r in relationships):
                                        # Extract relation name from text
                                        relation_name = self._extract_relation_name_from_text(token, sent)
                                        
                                        relationships.append({
                                            "source": subj_id,
                                            "target": obj_id,
                                            "metaedge": relation_type,
                                            "relation": relation_name,  # نام واقعی رابطه از متن
                                            "attributes": {
                                                "verb": token.text,
                                                "verb_lemma": token.lemma_,
                                                "confidence": 0.6,
                                                "sentence": sent.text[:100],
                                                "description": relation_name
                                            }
                                        })
        
        # Limit results
        entities = entities[:max_entities]
        relationships = relationships[:max_relationships]
        
        return {
            "entities": entities,
            "relationships": relationships,
            "method": "spacy",
            "stats": {
                "num_entities": len(entities),
                "num_relationships": len(relationships)
            }
        }
    
    def extract_llm(self, text: str, model: str = "gpt-4o", max_entities: int = 100, 
                    max_relationships: int = 200, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        استخراج موجودیت‌ها و روابط با استفاده از LLM
        
        Args:
            text: متن ورودی
            model: نام مدل LLM (gpt-4o, gpt-3.5-turbo, etc.)
            max_entities: حداکثر تعداد موجودیت‌ها
            max_relationships: حداکثر تعداد روابط
            confidence_threshold: حداقل confidence برای روابط
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not text or not text.strip():
            raise ValueError("متن ورودی نمی‌تواند خالی باشد")
        
        if max_entities < 1 or max_entities > 10000:
            raise ValueError("حداکثر موجودیت‌ها باید بین 1 تا 10000 باشد")
        
        if max_relationships < 1 or max_relationships > 20000:
            raise ValueError("حداکثر روابط باید بین 1 تا 20000 باشد")
        
        if confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError("آستانه اطمینان باید بین 0 تا 1 باشد")
        
        # Use HuggingFace by default (OpenAI is currently disabled)
        # Try to initialize if not already done
        if not self.hf_client:
            # Try to get token from environment
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
            if hf_token and HUGGINGFACE_AVAILABLE:
                try:
                    self.hf_client = InferenceClient(provider="auto", api_key=hf_token)
                    self.hf_api_key = hf_token
                    logging.info("HuggingFace InferenceClient initialized from environment")
                except Exception as e:
                    logging.error(f"Failed to initialize HuggingFace client: {e}")
        
        return self._extract_llm_huggingface(text, model, max_entities, max_relationships, confidence_threshold, "auto")
        
    def _extract_llm_huggingface(self, text: str, model: str, max_entities: int, max_relationships: int, 
                                 confidence_threshold: float, provider: str = "auto", hf_token: Optional[str] = None) -> Dict[str, Any]:
        """استخراج با استفاده از HuggingFace Inference API (الهام از knowledgegraph.ipynb)"""
        
        # Check if we have a client, if not try to initialize with provided token
        if not self.hf_client:
            # Try to initialize with provided token or re-check environment
            token_to_use = hf_token or self._hf_token_param or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
            if token_to_use and HUGGINGFACE_AVAILABLE:
                try:
                    self.hf_client = InferenceClient(provider="auto", api_key=token_to_use)
                    self.hf_api_key = token_to_use
                    logging.info("HuggingFace InferenceClient initialized successfully with provided token")
                except Exception as e:
                    logging.error(f"Failed to initialize HuggingFace client with token: {str(e)}")
                    raise ValueError(f"نمی‌توان HuggingFace client را راه‌اندازی کرد. لطفاً توکن خود را بررسی کنید. خطا: {str(e)}")
            else:
                error_msg = "HuggingFace client not initialized. "
                if not HUGGINGFACE_AVAILABLE:
                    error_msg += "کتابخانه huggingface_hub نصب نشده است. لطفاً با دستور 'pip install huggingface_hub' نصب کنید."
                elif not token_to_use:
                    error_msg += "توکن HuggingFace تنظیم نشده است. لطفاً توکن خود را در فیلد مربوطه وارد کنید یا متغیر محیطی HF_TOKEN را تنظیم کنید."
                else:
                    error_msg += "لطفاً توکن خود را بررسی کنید."
                logging.error(error_msg)
                raise ValueError(error_msg)
        
        # Get the extraction prompt (prefer generic prompt for general text graphs)
        prompt_template = GRAPH_PROMPTS.get("extract_entities_generic", GRAPH_PROMPTS.get("extract_entities", ""))
        if not prompt_template:
            prompt_template = """Extract entities and relationships from the following text as JSON.
Format:
{{
  "entities": [
    {{"id": "unique_id", "name": "entity_name", "type": "Gene|Disease|Compound|PW|BP|MF|A|S|PC|CC|PERSON|LOCATION|ORGANIZATION|EVENT|CONCEPT", "attributes": {{}}}}
  ],
  "relationships": [
    {{"source": "source_entity_id", "target": "target_entity_id", "metaedge": "GcG|GiG|Gr>G|CtD|...", "relation": "actual_relation_name_from_text", "attributes": {{}}}}
  ]
}}

CRITICAL INSTRUCTIONS FOR RELATIONSHIPS:
1. The "relation" field MUST contain the actual verb/phrase from the text in a readable, meaningful format.
2. Examples of good relation names:
   - "علی به مدرسه رفت" → relation: "رفتن به" or "goes to" or "attends"
   - "TP53 regulates apoptosis" → relation: "regulates" or "regulates"
   - "دارو بیماری را درمان می‌کند" → relation: "درمان می‌کند" or "treats"
3. DO NOT use metaedge codes (like "GoA", "GiG", "CtD") as the relation name.
4. The relation should be in the same language as the text (Persian/Farsi or English).
5. Use natural, descriptive phrases that clearly express the relationship.

Text:
{text}
Response:"""
        
        system_prompt = (
            "You are a PRECISE information extraction system that builds a knowledge graph from a SINGLE text.\n"
            "Top priority: ONLY extract entities and relations that are EXPLICITLY present in the text.\n"
            "Do NOT add background knowledge, assumptions, or hallucinated facts.\n"
            "For every entity and relation you output, there must be clear supporting words or phrases in the text.\n"
            "If the text is short and mentions only a few things, return only those and nothing more.\n"
            "Output ONLY valid JSON (no markdown, no commentary).\n"
        )
        
        # Use string replacement instead of format() to avoid KeyError with braces in text
        user_prompt = prompt_template.replace('{text}', text)
        
        try:
            # Call HuggingFace Inference API
            logging.info(f"🔵 Calling HuggingFace API with model: {model}")
            logging.info(f"🔵 Using token: {self.hf_api_key[:10]}..." if self.hf_api_key else "❌ No token")
            logging.info(f"🔵 Client type: {type(self.hf_client)}")
            logging.info(f"🔵 Client available: {self.hf_client is not None}")
            
            # Test connection first
            if not self.hf_client:
                raise ValueError("❌ HuggingFace client not initialized. لطفاً توکن را بررسی کنید.")
            
            try:
                response = self.hf_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=4096,
                )
                logging.info(f"✅ API call successful, response type: {type(response)}")
            except Exception as api_error:
                error_type = type(api_error).__name__
                error_msg = str(api_error)
                logging.error(f"❌ API call failed: {error_type} - {error_msg}")
                
                # Provide specific error messages
                if "401" in error_msg or "Unauthorized" in error_msg:
                    raise ValueError("❌ خطای احراز هویت: توکن HuggingFace نامعتبر است. لطفاً توکن خود را از https://huggingface.co/settings/tokens دریافت و بررسی کنید.")
                elif "404" in error_msg or "not found" in error_msg.lower():
                    raise ValueError(f"❌ مدل '{model}' یافت نشد. لطفاً نام مدل را بررسی کنید یا از یک مدل رایگان دیگر استفاده کنید.")
                elif "rate limit" in error_msg.lower() or "429" in error_msg:
                    raise ValueError("❌ محدودیت نرخ درخواست: لطفاً چند لحظه صبر کنید و دوباره تلاش کنید.")
                elif "timeout" in error_msg.lower():
                    raise ValueError("❌ زمان اتصال به HuggingFace به پایان رسید. لطفاً اتصال اینترنت خود را بررسی کنید.")
                else:
                    raise ValueError(f"❌ خطا در ارتباط با HuggingFace API: {error_msg}\nلطفاً توکن و مدل را بررسی کنید.")
            
            if not response or not hasattr(response, 'choices') or not response.choices:
                logging.error(f"❌ Invalid response structure: {response}")
                raise ValueError("❌ پاسخ نامعتبر از HuggingFace API دریافت شد. لطفاً توکن و مدل را بررسی کنید.")
            
            response_text = response.choices[0].message.content.strip()
            logging.info(f"✅ Received response from HuggingFace (length: {len(response_text)} chars)")
            logging.info(f"📝 Response starts with: {repr(response_text[:200])}")
            logging.info(f"📝 Response preview: {response_text[:300]}")
            
            result = self._parse_llm_response(response_text)
            
            if not result:
                # Log the full response for debugging
                logging.error(f"Could not parse JSON from HuggingFace response")
                logging.error(f"Response length: {len(response_text)} chars")
                logging.error(f"Response type: {type(response_text)}")
                logging.error(f"Response repr (first 500): {repr(response_text[:500])}")
                logging.error(f"Response text (first 500): {response_text[:500]}")
                logging.error(f"Full response: {response_text}")
                
                # Analyze the response to provide better error message
                has_entities = '"entities"' in response_text
                has_relationships = '"relationships"' in response_text
                has_opening_brace = '{' in response_text
                has_closing_brace = '}' in response_text
                
                error_details = []
                error_details.append(f"📊 تحلیل پاسخ مدل:")
                error_details.append(f"  - طول پاسخ: {len(response_text)} کاراکتر")
                error_details.append(f"  - شامل 'entities': {'✅' if has_entities else '❌'}")
                error_details.append(f"  - شامل 'relationships': {'✅' if has_relationships else '❌'}")
                error_details.append(f"  - شامل '{{': {'✅' if has_opening_brace else '❌'}")
                error_details.append(f"  - شامل '}}': {'✅' if has_closing_brace else '❌'}")
                
                # Show preview of response
                error_preview = response_text[:400] if len(response_text) > 400 else response_text
                error_details.append(f"\n📝 شروع پاسخ مدل:")
                error_details.append(f"{error_preview}")
                if len(response_text) > 400:
                    error_details.append(f"... (بقیه {len(response_text) - 400} کاراکتر)")
                
                error_details.append(f"\n💡 راه‌حل‌های پیشنهادی:")
                error_details.append(f"1. مدل را تغییر دهید (TinyLlama یا Phi-3 معمولاً بهتر کار می‌کنند)")
                error_details.append(f"2. از روش‌های دیگر استفاده کنید: spaCy یا spaCy SVO Enhanced")
                error_details.append(f"3. متن ورودی را کوتاه‌تر کنید")
                error_details.append(f"4. دوباره تلاش کنید (مدل‌ها گاهی پاسخ‌های متفاوتی می‌دهند)")
                
                error_msg = "\n".join(error_details)
                raise ValueError(error_msg)
            
            return self._process_llm_result(result, max_entities, max_relationships, confidence_threshold, "llm", model)
            
        except ValueError as e:
            # Re-raise ValueError as is (these are our custom errors)
            raise
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logging.error(f"Error in HuggingFace extraction ({error_type}): {error_msg}")
            
            # Provide more helpful error messages
            if "401" in error_msg or "Unauthorized" in error_msg or "authentication" in error_msg.lower():
                raise ValueError("خطای احراز هویت: توکن HuggingFace نامعتبر است. لطفاً توکن خود را از https://huggingface.co/settings/tokens دریافت و بررسی کنید.")
            elif "404" in error_msg or "not found" in error_msg.lower():
                raise ValueError(f"مدل '{model}' یافت نشد. لطفاً نام مدل را بررسی کنید یا از یک مدل رایگان دیگر استفاده کنید.")
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                raise ValueError("محدودیت نرخ درخواست: لطفاً چند لحظه صبر کنید و دوباره تلاش کنید.")
            elif "timeout" in error_msg.lower():
                raise ValueError("زمان اتصال به HuggingFace به پایان رسید. لطفاً اتصال اینترنت خود را بررسی کنید و دوباره تلاش کنید.")
            else:
                raise ValueError(f"خطا در استخراج با HuggingFace: {error_msg}. لطفاً توکن و مدل را بررسی کنید.")
    
    def _process_llm_result(self, result: Dict[str, Any], max_entities: int, max_relationships: int, 
                           confidence_threshold: float, method: str, model: str) -> Dict[str, Any]:
        """پردازش نتیجه LLM (مشترک برای OpenAI و HuggingFace)"""
        # Validate and clean result
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        
        # Ensure each relationship has a "relation" field extracted from text
        for rel in relationships:
            relation_name = rel.get("relation", "")
            metaedge = rel.get("metaedge", "")
            
            # If relation is empty or is just a metaedge code, try to get a better name
            if not relation_name or relation_name == metaedge or len(relation_name) <= 3:
                # Try to extract from attributes
                verb = rel.get("attributes", {}).get("verb", "")
                description = rel.get("attributes", {}).get("description", "")
                pattern = rel.get("attributes", {}).get("pattern", "")
                
                if verb:
                    relation_name = verb
                elif description:
                    relation_name = description
                elif pattern:
                    relation_name = pattern
                else:
                    # Last resort: map metaedge to readable name
                    relation_name = self._map_metaedge_to_readable_name(metaedge)
            
            # Clean up relation name - remove metaedge codes if present
            if relation_name and len(relation_name) <= 5 and relation_name.isupper():
                # Probably a metaedge code, use mapping
                relation_name = self._map_metaedge_to_readable_name(relation_name)
            
            rel["relation"] = relation_name
        
        # Filter relationships by confidence if available
        if confidence_threshold > 0:
            filtered_relationships = []
            for rel in relationships:
                confidence = rel.get("attributes", {}).get("confidence", 1.0)
                if confidence >= confidence_threshold:
                    filtered_relationships.append(rel)
            relationships = filtered_relationships
        
        # Limit results
        entities = entities[:max_entities]
        relationships = relationships[:max_relationships]
        
        return {
            "entities": entities,
            "relationships": relationships,
            "method": method,
            "model": model,
            "stats": {
                "num_entities": len(entities),
                "num_relationships": len(relationships)
            }
        }
    
    def extract_spacy_svo_enhanced(self, text: str, max_entities: int = 100, max_relationships: int = 200) -> Dict[str, Any]:
        """
        استخراج موجودیت‌ها و روابط با استفاده از spaCy SVO Enhanced (الهام از knowledgegraph.ipynb)
        
        Args:
            text: متن ورودی
            max_entities: حداکثر تعداد موجودیت‌ها
            max_relationships: حداکثر تعداد روابط
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not text or not text.strip():
            raise ValueError("متن ورودی نمی‌تواند خالی باشد")
        
        if not self.nlp:
            raise ValueError("spaCy model not loaded. Please install spaCy and download a model.")
        
        def normalize(s: str) -> str:
            s = re.sub(r"\s+", " ", s.strip())
            s = s.replace("'", "'")
            return s
        
        def token_span_char(token) -> Tuple[int, int]:
            return token.idx, token.idx + len(token.text)
        
        def pick_entity_for_span(ents_by_char: List[Tuple[int, int, str]], start: int, end: int) -> Optional[str]:
            for s, e, text in ents_by_char:
                if not (end <= s or start >= e):  # overlap
                    return text
            return None
        
        doc = self.nlp(text)
        entities = []
        relationships = []
        entity_map = {}
        
        # Extract entities
        ents_by_char = []
        for ent in doc.ents:
            et = normalize(ent.text)
            if not et or et in entity_map:
                continue
            entity_id = f"ENTITY_{len(entities)}"
            entity_type = self._map_spacy_label_to_type(ent.label_)
            entities.append({
                "id": entity_id,
                "name": et,
                "type": entity_type,
                "attributes": {
                    "spacy_label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "mention_count": 1
                }
            })
            entity_map[et] = entity_id
            ents_by_char.append((ent.start_char, ent.end_char, et))
        
        # Extract noun chunks as potential entities
        for chunk in doc.noun_chunks:
            chunk_text = normalize(chunk.text)
            if len(chunk_text.split()) <= 5 and chunk_text not in entity_map:
                if not all(word.lower() in STOP_WORDS for word in chunk_text.split()):
                    entity_id = f"ENTITY_{len(entities)}"
                    entities.append({
                        "id": entity_id,
                        "name": chunk_text,
                        "type": "Gene",  # Default
                        "attributes": {"source": "noun_chunk"}
                    })
                    entity_map[chunk_text] = entity_id
        
        # Extract SVO relations
        for token in doc:
            if token.pos_ != "VERB":
                continue
            
            subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
            objects = [c for c in token.children if c.dep_ in ("dobj", "obj", "attr", "dative", "oprd")]
            
            # Prepositional objects
            for prep in [c for c in token.children if c.dep_ == "prep"]:
                for pobj in prep.children:
                    if pobj.dep_ == "pobj":
                        objects.append(pobj)
            
            if not subjects or not objects:
                continue
            
            rel = normalize(token.lemma_)
            
            for s in subjects:
                s_start, s_end = token_span_char(s)
                subj_ent = pick_entity_for_span(ents_by_char, s_start, s_end)
                subj = subj_ent if subj_ent else normalize(s.text)
                
                for o in objects:
                    o_start, o_end = token_span_char(o)
                    obj_ent = pick_entity_for_span(ents_by_char, o_start, o_end)
                    obj = obj_ent if obj_ent else normalize(o.text)
                    
                    if len(subj) < 2 or len(obj) < 2:
                        continue
                    if subj.lower() in ("it", "this", "that") or obj.lower() in ("it", "this", "that"):
                        continue
                    
                    # Map to entity IDs
                    subj_id = entity_map.get(subj)
                    obj_id = entity_map.get(obj)
                    
                    if subj_id and obj_id:
                        relation_type = self._map_verb_to_relation(token.lemma_)
                        if not relation_type:
                            relation_type = "GiG"  # Default
                        
                        # استخراج نام رابطه از متن
                        relation_name = self._extract_relation_name_from_text(token, token.sent if hasattr(token, 'sent') and token.sent else None)
                        
                        relationships.append({
                            "source": subj_id,
                            "target": obj_id,
                            "metaedge": relation_type,
                            "relation": relation_name,  # نام واقعی رابطه از متن
                            "attributes": {
                                "verb": token.text,
                                "verb_lemma": token.lemma_,
                                "confidence": 0.7,
                                "sentence": token.sent.text[:100] if hasattr(token, 'sent') and token.sent else "",
                                "description": relation_name
                            }
                        })
        
        # Limit results
        entities = entities[:max_entities]
        relationships = relationships[:max_relationships]
        
        return {
            "entities": entities,
            "relationships": relationships,
            "method": "spacy_svo_enhanced",
            "stats": {
                "num_entities": len(entities),
                "num_relationships": len(relationships)
            }
        }
    
    def extract_llm_multipass(self, text: str, model: str = "mistralai/Mistral-7B-Instruct-v0.2", max_entities: int = 100,
                             max_relationships: int = 200, max_gleanings: int = 2,
                             confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        استخراج موجودیت‌ها و روابط با استفاده از LLM Multi-pass (الهام از ragflow)
        با iterative gleaning برای حداکثر کردن recall
        
        Args:
            text: متن ورودی
            model: نام مدل LLM (HuggingFace models by default)
            max_entities: حداکثر تعداد موجودیت‌ها
            max_relationships: حداکثر تعداد روابط
            max_gleanings: تعداد پاس‌های iterative (پیش‌فرض: 2)
            confidence_threshold: حداقل confidence برای روابط
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not text or not text.strip():
            raise ValueError("متن ورودی نمی‌تواند خالی باشد")
        
        # Try to initialize if not already done
        if not self.hf_client:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or self._hf_token_param
            if hf_token and HUGGINGFACE_AVAILABLE:
                try:
                    self.hf_client = InferenceClient(provider="auto", api_key=hf_token)
                    self.hf_api_key = hf_token
                    logging.info("HuggingFace InferenceClient initialized for multipass")
                except Exception as e:
                    logging.error(f"Failed to initialize HuggingFace client: {e}")
                    raise ValueError(f"نمی‌توان HuggingFace client را راه‌اندازی کرد: {str(e)}")
            else:
                raise ValueError("HuggingFace client not initialized. لطفاً توکن HuggingFace را در فیلد مربوطه وارد کنید یا متغیر محیطی HF_TOKEN را تنظیم کنید.")
        
        # Get the extraction prompt
        prompt_template = GRAPH_PROMPTS.get("extract_entities", "")
        if not prompt_template:
            prompt_template = """Extract entities and relationships from the following text as JSON.
Format:
{{
  "entities": [
    {{"id": "unique_id", "name": "entity_name", "type": "Gene|Disease|Compound|PW|BP|MF|A|S|PC|CC|PERSON|LOCATION|ORGANIZATION|EVENT|CONCEPT", "attributes": {{}}}}
  ],
  "relationships": [
    {{"source": "source_entity_id", "target": "target_entity_id", "metaedge": "GcG|GiG|Gr>G|CtD|...", "relation": "relation_name_from_text", "attributes": {{}}}}
  ]
}

CRITICAL INSTRUCTIONS FOR RELATIONSHIPS:
1. The "relation" field MUST contain the actual verb/phrase from the text in a readable, meaningful format.
2. Examples of good relation names:
   - "علی به مدرسه رفت" → relation: "رفتن به" or "goes to" or "attends"
   - "TP53 regulates apoptosis" → relation: "regulates"
   - "دارو بیماری را درمان می‌کند" → relation: "درمان می‌کند" or "treats"
3. DO NOT use metaedge codes (like "GoA", "GiG", "CtD") as the relation name.
4. The relation should be in the same language as the text (Persian/Farsi or English).
5. Use natural, descriptive phrases that clearly express the relationship.
}}

Text:
{text}
Response:"""
        
        # Use string replacement instead of format() to avoid KeyError with braces in text
        prompt = prompt_template.replace('{text}', text)
        all_entities = []
        all_relationships = []
        seen_entity_ids = set()
        seen_relationship_keys = set()
        
        system_prompt = (
            "You are an information extraction system that generates a DETAILED knowledge graph from text.\n"
            "Primary goal: MAXIMIZE RECALL. Do not summarize or select only main facts.\n"
            "Extract ALL entities explicitly mentioned and EVERY explicit relation in the text.\n"
            "Output ONLY valid JSON (no markdown, no commentary).\n"
        )
        
        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Initial extraction
            response = self.hf_client.chat.completions.create(
                model=model,
                messages=history,
                temperature=0.2,
                max_tokens=4096,
            )
            
            response_text = response.choices[0].message.content.strip()
            history.append({"role": "assistant", "content": response_text})
            
            # Parse initial response
            result = self._parse_llm_response(response_text)
            if result:
                for ent in result.get("entities", []):
                    ent_id = ent.get("id")
                    if ent_id and ent_id not in seen_entity_ids:
                        all_entities.append(ent)
                        seen_entity_ids.add(ent_id)
                
                for rel in result.get("relationships", []):
                    # بهبود نام رابطه
                    relation_name = rel.get("relation", "")
                    metaedge = rel.get("metaedge", "")
                    
                    # اگر relation خالی است یا فقط یک metaedge code است
                    if not relation_name or relation_name == metaedge or (len(relation_name) <= 5 and relation_name.isupper()):
                        verb = rel.get("attributes", {}).get("verb", "")
                        description = rel.get("attributes", {}).get("description", "")
                        
                        if verb:
                            relation_name = verb
                        elif description:
                            relation_name = description
                        else:
                            # Map metaedge to readable name
                            relation_name = self._map_metaedge_to_readable_name(metaedge)
                    
                    rel["relation"] = relation_name
                    
                    rel_key = (rel.get("source"), rel.get("target"), rel.get("metaedge"))
                    if rel_key not in seen_relationship_keys:
                        all_relationships.append(rel)
                        seen_relationship_keys.add(rel_key)
            
            # Iterative gleaning (additional passes focused only on truly missed items)
            for i in range(max_gleanings):
                followup_prompt = (
                    "Review the SAME text again.\n"
                    "In your previous JSON response(s) above, you already listed some entities and relationships.\n"
                    "Now return ONLY additional entities and relationships that are EXPLICITLY present in the text but were NOT included before.\n"
                    "If there are no additional items, respond with:\n"
                    '{"entities": [], "relationships": []}\n'
                    "Output ONLY valid JSON.\n"
                )
                history.append({"role": "user", "content": followup_prompt})
                response = self.hf_client.chat.completions.create(
                    model=model,
                    messages=history,
                    temperature=0.2,
                    max_tokens=4096,
                )
                
                response_text = response.choices[0].message.content.strip()
                history.append({"role": "assistant", "content": response_text})
                
                # Parse additional entities
                result = self._parse_llm_response(response_text)
                if result:
                    added_any = False
                    for ent in result.get("entities", []):
                        ent_id = ent.get("id")
                        if ent_id and ent_id not in seen_entity_ids:
                            all_entities.append(ent)
                            seen_entity_ids.add(ent_id)
                            added_any = True
                    
                    for rel in result.get("relationships", []):
                        # بهبود نام رابطه
                        relation_name = rel.get("relation", "")
                        metaedge = rel.get("metaedge", "")
                        
                        # اگر relation خالی است یا فقط یک metaedge code است
                        if not relation_name or relation_name == metaedge or (len(relation_name) <= 5 and relation_name.isupper()):
                            verb = rel.get("attributes", {}).get("verb", "")
                            description = rel.get("attributes", {}).get("description", "")
                            
                            if verb:
                                relation_name = verb
                            elif description:
                                relation_name = description
                            else:
                                # Map metaedge to readable name
                                relation_name = self._map_metaedge_to_readable_name(metaedge)
                        
                        rel["relation"] = relation_name
                        
                        rel_key = (rel.get("source"), rel.get("target"), rel.get("metaedge"))
                        if rel_key not in seen_relationship_keys:
                            all_relationships.append(rel)
                            seen_relationship_keys.add(rel_key)
                            added_any = True
                    
                    if not added_any:
                        # No new entities/relationships found; stop early
                        break
            
            # Post-filter entities/relationships against original text to reduce hallucinations.
            # For very short texts (e.g., single-sentence queries), we skip this filter to avoid
            # accidentally dropping valid but slightly rephrased/translated entities (e.g., "school" vs "مدرسه").
            try:
                token_count = len(str(text).split())
            except Exception:
                token_count = 0

            if token_count > 20:
                all_entities, all_relationships = self._filter_extractions_by_text(text, all_entities, all_relationships)
            
            # Filter relationships by confidence
            if confidence_threshold > 0:
                filtered_relationships = []
                for rel in all_relationships:
                    confidence = rel.get("attributes", {}).get("confidence", 1.0)
                    if confidence >= confidence_threshold:
                        filtered_relationships.append(rel)
                all_relationships = filtered_relationships
            
            # Limit results
            all_entities = all_entities[:max_entities]
            all_relationships = all_relationships[:max_relationships]
            
            return {
                "entities": all_entities,
                "relationships": all_relationships,
                "method": "llm_multipass",
                "model": model,
                "stats": {
                    "num_entities": len(all_entities),
                    "num_relationships": len(all_relationships)
                }
            }
            
        except Exception as e:
            logging.error(f"Error in LLM multipass extraction: {e}")
            raise ValueError(f"LLM multipass extraction failed: {str(e)}")
    
    def _normalize_for_match(self, text: str) -> str:
        """Normalize text for fuzzy substring matching (simple, language-agnostic)."""
        if not text:
            return ""
        # Lowercase
        t = text.lower()
        # Normalize Persian/Arabic variants of ye/ke
        t = t.replace("ي", "ی").replace("ك", "ک")
        # Replace half-space with normal space
        t = t.replace("\u200c", " ")
        # Remove basic punctuation
        t = re.sub(r"[^\w\s]", " ", t)
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _filter_extractions_by_text(
        self,
        original_text: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Remove entities/relationships that are not clearly supported by the original text.

        This is a defensive, conservative filter intended to cut hallucinated nodes/edges
        when using LLM-based extraction on general text (e.g., preventing extra people
        like 'محمد' or 'معلم' when the text only mentions 'علی' and 'مدرسه').
        """
        if not original_text:
            return entities, relationships

        norm_text = self._normalize_for_match(original_text)
        if not norm_text:
            return entities, relationships

        # Filter entities: keep only if name (or id) appears in text
        filtered_entities: List[Dict[str, Any]] = []
        valid_entity_ids: set = set()

        for ent in entities or []:
            name = str(ent.get("name") or "").strip()
            ent_id = str(ent.get("id") or "").strip()

            keep = False
            if name:
                norm_name = self._normalize_for_match(name)
                if norm_name and norm_name in norm_text:
                    keep = True
            # Fallback: if id looks like surface form and appears in text
            if not keep and ent_id and ent_id.lower() != name.lower():
                norm_id = self._normalize_for_match(ent_id)
                if norm_id and norm_id in norm_text:
                    keep = True

            if keep:
                filtered_entities.append(ent)
                if ent_id:
                    valid_entity_ids.add(ent_id)

        # Filter relationships: keep only if both endpoints kept AND relation appears in text (or its verb/description)
        filtered_relationships: List[Dict[str, Any]] = []

        for rel in relationships or []:
            src = rel.get("source")
            tgt = rel.get("target")
            if not src or not tgt:
                continue
            if src not in valid_entity_ids or tgt not in valid_entity_ids:
                continue

            relation_name = str(rel.get("relation") or "").strip()
            verb = str(rel.get("attributes", {}).get("verb") or "").strip()
            description = str(rel.get("attributes", {}).get("description") or "").strip()

            supported = False
            for candidate in (relation_name, verb, description):
                if candidate:
                    norm_cand = self._normalize_for_match(candidate)
                    if norm_cand and norm_cand in norm_text:
                        supported = True
                        break

            # If we cannot find any textual support, drop the relationship
            if supported:
                filtered_relationships.append(rel)

        return filtered_entities, filtered_relationships

    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response and extract JSON with improved error handling"""
        try:
            if not response_text or not response_text.strip():
                logging.warning("Empty response text")
                return None
            
            original_text = response_text
            logging.debug(f"Parsing response (length: {len(response_text)})")
            
            # Remove markdown code blocks
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'```\s*', '', response_text)
            response_text = response_text.strip()
            
            # First, try to find a complete JSON object
            json_start = response_text.find('{')
            
            if json_start < 0:
                # No { found, try to construct JSON from partial response
                logging.warning("⚠️ No opening brace found, trying to construct JSON")
                logging.warning(f"⚠️ Response starts with: {repr(response_text[:100])}")
                
                # Look for entities or relationships keys (more flexible pattern)
                entities_match = re.search(r'["\']?entities["\']?\s*:', response_text, re.IGNORECASE)
                relationships_match = re.search(r'["\']?relationships["\']?\s*:', response_text, re.IGNORECASE)
                
                if entities_match or relationships_match:
                    # Find the earliest key
                    positions = []
                    if entities_match:
                        positions.append(entities_match.start())
                    if relationships_match:
                        positions.append(relationships_match.start())
                    
                    if positions:
                        start_pos = min(positions)
                        logging.info(f"✅ Found key at position {start_pos}")
                        
                        # Go back to find the opening quote
                        # Handle case like: '\n "entities": [...]'
                        quote_start = start_pos
                        found_quote = False
                        
                        # Go backwards to find the opening quote
                        while quote_start > 0:
                            char = response_text[quote_start]
                            if char == '"':
                                found_quote = True
                                break
                            elif char in ['\n', '\r']:
                                # Skip newlines
                                quote_start -= 1
                                continue
                            elif char == '{':
                                # Found a brace, use it
                                break
                            quote_start -= 1
                        
                        # Extract content
                        if found_quote and quote_start >= 0:
                            # Start from the quote
                            json_content = response_text[quote_start:]
                            logging.info(f"✅ Extracted from quote at position {quote_start}")
                        else:
                            # Start from the key name itself
                            json_content = response_text[start_pos:]
                            # Remove any leading whitespace/newlines
                            json_content = json_content.lstrip()
                            
                            # Ensure it starts with a quote
                            if not json_content.startswith('"'):
                                # Find the actual key name
                                key_match = re.match(r'["\']?(\w+)["\']?\s*:', json_content)
                                if key_match:
                                    key_name = key_match.group(1)
                                    json_content = f'"{key_name}":' + json_content[key_match.end():]
                                    logging.info(f"✅ Reconstructed key: {key_name}")
                                else:
                                    json_content = '"' + json_content
                        
                        # Wrap in braces
                        json_str = '{' + json_content.strip()
                        logging.info(f"✅ Constructed JSON (first 200 chars): {json_str[:200]}")
                        
                        # Try to find the end - look for the last }
                        last_brace = json_str.rfind('}')
                        if last_brace < 0:
                            # No closing brace, try to add one
                            json_str = json_str.rstrip()
                            # Remove trailing commas
                            json_str = re.sub(r',\s*$', '', json_str)
                            # Add closing brace
                            json_str += '}'
                            logging.info("✅ Added missing closing brace")
                        elif json_str.count('{') > json_str.count('}'):
                            # Missing closing braces
                            missing = json_str.count('{') - json_str.count('}')
                            json_str += '}' * missing
                            logging.info(f"✅ Added {missing} missing closing brace(s)")
                    else:
                        logging.warning("❌ Could not find entities or relationships keys")
                        return None
                else:
                    # Try to find any JSON-like structure
                    json_match = re.search(r'["\']?\w+["\']?\s*:', response_text)
                    if json_match:
                        start_pos = json_match.start()
                        json_content = response_text[start_pos:].strip()
                        json_str = '{' + json_content
                        if json_str.count('{') > json_str.count('}'):
                            json_str += '}'
                    else:
                        logging.warning("No JSON-like structure found")
                        return None
            else:
                json_str = response_text[json_start:]
            
            # Clean up JSON string
            json_str = json_str.strip()
            
            # Ensure it starts with {
            if not json_str.startswith('{'):
                json_match = re.search(r'\{[\s\S]*\}', json_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logging.warning("Could not find JSON object boundaries")
                    return None
            
            # Try to parse JSON
            try:
                parsed = json.loads(json_str)
                logging.info("Successfully parsed JSON")
                return parsed
            except json.JSONDecodeError as e:
                logging.warning(f"First JSON parse attempt failed: {e}")
                logging.warning(f"JSON string (first 300 chars): {json_str[:300]}")
                
                # Try to fix common issues
                # Remove trailing commas before } or ]
                json_str_fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                # Remove comments (if any)
                json_str_fixed = re.sub(r'//.*?$', '', json_str_fixed, flags=re.MULTILINE)
                json_str_fixed = re.sub(r'/\*.*?\*/', '', json_str_fixed, flags=re.DOTALL)
                json_str_fixed = json_str_fixed.strip()
                
                try:
                    parsed = json.loads(json_str_fixed)
                    logging.info("Successfully parsed JSON after fixing commas/comments")
                    return parsed
                except json.JSONDecodeError as e2:
                    logging.warning(f"Second JSON parse attempt failed: {e2}")
                    
                    # Try to find valid JSON with regex (more aggressive)
                    json_match = re.search(r'\{[\s\S]*\}', json_str_fixed, re.DOTALL)
                    if json_match:
                        try:
                            cleaned_json = json_match.group(0).strip()
                            parsed = json.loads(cleaned_json)
                            logging.info("Successfully parsed JSON after regex extraction")
                            return parsed
                        except json.JSONDecodeError as e3:
                            logging.warning(f"Third JSON parse attempt failed: {e3}")
                            # Last resort: try to fix common JSON issues
                            # Remove any text before first {
                            cleaned_json = re.sub(r'^[^{]*', '', cleaned_json)
                            # Remove any text after last }
                            cleaned_json = re.sub(r'[^}]*$', '', cleaned_json)
                            try:
                                parsed = json.loads(cleaned_json)
                                logging.info("Successfully parsed JSON after aggressive cleaning")
                                return parsed
                            except json.JSONDecodeError as e4:
                                logging.error(f"All JSON parse attempts failed. Last error: {e4}")
                                logging.error(f"Original response (first 500 chars): {original_text[:500]}")
                                logging.error(f"Final cleaned JSON (first 500 chars): {cleaned_json[:500]}")
                                return None
                    else:
                        logging.error(f"Could not find JSON object in response")
                        logging.error(f"Original response (first 500 chars): {original_text[:500]}")
                        return None
        except Exception as e:
            logging.warning(f"Error parsing LLM response: {e}")
            logging.warning(f"Response text (first 500 chars): {response_text[:500] if 'response_text' in locals() else 'N/A'}")
            return None
    
    def extract_hybrid(self, text: str, max_entities: int = 100, max_relationships: int = 200,
                      methods: List[str] = None, confidence_threshold: float = 0.5,
                      original_text: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        استخراج ترکیبی: ترکیب نتایج از چندین روش
        
        Args:
            text: متن ورودی (ممکن است پیش‌پردازش شده باشد)
            max_entities: حداکثر تعداد موجودیت‌ها
            max_relationships: حداکثر تعداد روابط
            methods: لیست روش‌های استخراج (پیش‌فرض: ['spacy', 'llm'])
            confidence_threshold: حداقل confidence برای روابط
            original_text: متن اصلی (برای استفاده در LLM - اگر preprocessing فعال باشد)
            **kwargs: پارامترهای اضافی
            
        Returns:
            Dictionary containing entities and relationships
        """
        if methods is None:
            methods = kwargs.get("hybrid_methods", ['spacy', 'llm'])
        
        # استفاده از متن اصلی برای LLM (اگر موجود باشد)
        llm_text = original_text if original_text else text
        
        all_entities = []
        all_relationships = []
        entity_map = {}  # name -> entity
        relationship_map = {}  # (source, target, metaedge) -> relationship
        
        # Run each method
        for method in methods:
            try:
                # برای LLM: از متن اصلی استفاده کن (حتی اگر preprocessing فعال باشد)
                # برای بقیه: از متن پیش‌پردازش شده استفاده کن
                text_for_method = llm_text if method in ["llm", "llm_multipass"] else text
                
                if method == "simple":
                    result = self.extract_simple(text_for_method, max_entities=max_entities, max_relationships=max_relationships)
                elif method == "spacy":
                    result = self.extract_spacy(text_for_method, max_entities=max_entities, max_relationships=max_relationships)
                elif method == "spacy_svo_enhanced":
                    result = self.extract_spacy_svo_enhanced(text_for_method, max_entities=max_entities, max_relationships=max_relationships)
                elif method == "llm":
                    result = self.extract_llm(llm_text, max_entities=max_entities, max_relationships=max_relationships, 
                                            confidence_threshold=confidence_threshold, **kwargs)
                elif method == "llm_multipass":
                    result = self.extract_llm_multipass(llm_text, max_entities=max_entities, max_relationships=max_relationships,
                                                      confidence_threshold=confidence_threshold, **kwargs)
                else:
                    logging.warning(f"Unknown method: {method}, skipping")
                    continue
                
                # Merge entities (prefer higher confidence)
                for ent in result.get("entities", []):
                    ent_name = ent.get("name", "")
                    ent_id = ent.get("id", "")
                    
                    if ent_name in entity_map:
                        # Merge: keep entity with higher confidence or more attributes
                        existing = entity_map[ent_name]
                        existing_conf = existing.get("attributes", {}).get("confidence", 0.5)
                        new_conf = ent.get("attributes", {}).get("confidence", 0.5)
                        
                        if new_conf > existing_conf or len(ent.get("attributes", {})) > len(existing.get("attributes", {})):
                            entity_map[ent_name] = ent
                            # Update in all_entities
                            for i, e in enumerate(all_entities):
                                if e.get("name") == ent_name:
                                    all_entities[i] = ent
                                    break
                    else:
                        entity_map[ent_name] = ent
                        all_entities.append(ent)
                
                # Merge relationships
                for rel in result.get("relationships", []):
                    source = rel.get("source")
                    target = rel.get("target")
                    metaedge = rel.get("metaedge", "GiG")
                    key = (source, target, metaedge)
                    
                    if key in relationship_map:
                        # Merge: combine attributes, increase confidence
                        existing = relationship_map[key]
                        existing_weight = existing.get("attributes", {}).get("weight", 1.0)
                        new_weight = rel.get("attributes", {}).get("weight", 1.0)
                        
                        # Combine descriptions
                        existing_desc = existing.get("attributes", {}).get("description", "")
                        new_desc = rel.get("attributes", {}).get("description", "")
                        if new_desc and new_desc not in existing_desc:
                            existing.get("attributes", {})["description"] = f"{existing_desc}; {new_desc}"
                        
                        # Update weight
                        existing.get("attributes", {})["weight"] = existing_weight + new_weight
                        existing.get("attributes", {})["confidence"] = min(1.0, (existing.get("attributes", {}).get("confidence", 0.5) + 
                                                                                rel.get("attributes", {}).get("confidence", 0.5)) / 2)
                    else:
                        relationship_map[key] = rel
                        all_relationships.append(rel)
                        
            except Exception as e:
                logging.warning(f"Error in hybrid extraction with method {method}: {e}")
                continue
        
        # Filter by confidence
        if confidence_threshold > 0:
            filtered_relationships = []
            for rel in all_relationships:
                confidence = rel.get("attributes", {}).get("confidence", 1.0)
                if confidence >= confidence_threshold:
                    filtered_relationships.append(rel)
            all_relationships = filtered_relationships
        
        # Limit results
        all_entities = all_entities[:max_entities]
        all_relationships = all_relationships[:max_relationships]
        
        return {
            "entities": all_entities,
            "relationships": all_relationships,
            "method": f"hybrid_{'+'.join(methods)}",
            "stats": {
                "num_entities": len(all_entities),
                "num_relationships": len(all_relationships)
            }
        }
    
    def build_graph(self, extraction_result: Dict[str, Any]) -> nx.MultiDiGraph:
        """
        ساخت گراف NetworkX از موجودیت‌ها و روابط استخراج شده
        
        Args:
            extraction_result: نتیجه استخراج (از extract_simple, extract_spacy, یا extract_llm)
            
        Returns:
            NetworkX MultiDiGraph
        """
        if not extraction_result:
            raise ValueError("نتیجه استخراج خالی است")
        
        G = nx.MultiDiGraph()
        
        entities = extraction_result.get("entities", [])
        relationships = extraction_result.get("relationships", [])
        
        if not entities:
            raise ValueError("هیچ موجودیتی استخراج نشده است")
        
        if not isinstance(entities, list):
            raise ValueError("موجودیت‌ها باید به صورت لیست باشند")
        
        if not isinstance(relationships, list):
            raise ValueError("روابط باید به صورت لیست باشند")
        
        # Add nodes with validation
        for entity in entities:
            if not isinstance(entity, dict):
                logging.warning(f"Invalid entity format: {entity}")
                continue
            
            entity_id = entity.get("id")
            if not entity_id:
                logging.warning(f"Entity missing ID: {entity}")
                continue
            
            entity_name = entity.get("name", entity_id)
            entity_type = entity.get("type", "Unknown")
            attributes = entity.get("attributes", {})
            
            if not isinstance(attributes, dict):
                attributes = {}
            
            try:
                G.add_node(
                    entity_id,
                    name=entity_name,
                    kind=entity_type,
                    type=entity_type,
                    **attributes
                )
            except Exception as e:
                logging.warning(f"Error adding node {entity_id}: {e}")
                continue
        
        # Validate graph has nodes
        if G.number_of_nodes() == 0:
            raise ValueError("هیچ نودی به گراف اضافه نشد")
        
        # Add edges with validation
        edges_added = 0
        for rel in relationships:
            if not isinstance(rel, dict):
                logging.warning(f"Invalid relationship format: {rel}")
                continue
            
            source = rel.get("source")
            target = rel.get("target")
            metaedge = rel.get("metaedge", "related_to")
            attributes = rel.get("attributes", {})
            
            if not source or not target:
                logging.warning(f"Relationship missing source or target: {rel}")
                continue
            
            if not isinstance(attributes, dict):
                attributes = {}
            
            if G.has_node(source) and G.has_node(target):
                try:
                    # استخراج نام رابطه از rel یا attributes
                    relation_name = rel.get("relation")
                    if not relation_name:
                        # سعی کن از attributes استخراج کن
                        relation_name = attributes.get("description") or attributes.get("verb") or attributes.get("pattern")
                    if not relation_name:
                        # Fallback: استفاده از metaedge
                        relation_name = metaedge
                    
                    # Add edge with both metaedge and relation
                    # Store relation meaning for display
                    relation_meaning = relation_name
                    if metaedge and metaedge != relation_name:
                        # If we have a metaedge code, add it as context
                        relation_meaning = f"{relation_name} ({metaedge})"
                    
                    G.add_edge(
                        source,
                        target,
                        metaedge=metaedge,  # Hetionet code
                        relation=relation_name,  # نام واقعی رابطه از متن
                        relation_meaning=relation_meaning,  # مفهوم رابطه برای نمایش
                        **attributes
                    )
                    edges_added += 1
                except Exception as e:
                    logging.warning(f"Error adding edge {source} -> {target}: {e}")
                    continue
            else:
                logging.debug(f"Skipping edge {source} -> {target}: nodes not in graph")
        
        if edges_added == 0 and len(relationships) > 0:
            logging.warning("هیچ یالی به گراف اضافه نشد")
        
        return G
    
    def save_graph(self, graph: nx.MultiDiGraph, output_dir: str = "uploaded_graphs", 
                   filename: Optional[str] = None) -> str:
        """
        ذخیره گراف به فایل PKL
        
        Args:
            graph: گراف NetworkX
            output_dir: پوشه خروجی
            filename: نام فایل (اختیاری، اگر نباشد با timestamp ساخته می‌شود)
            
        Returns:
            مسیر فایل ذخیره شده
        """
        import pickle
        
        if graph is None:
            raise ValueError("گراف نمی‌تواند None باشد")
        
        if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            raise ValueError("گراف باید از نوع NetworkX باشد")
        
        if graph.number_of_nodes() == 0:
            raise ValueError("گراف خالی نمی‌تواند ذخیره شود")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            raise ValueError(f"خطا در ایجاد پوشه خروجی: {str(e)}")
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_text_graph.pkl"
        
        # Validate filename
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(graph, f)
            logging.info(f"Graph saved to {filepath}")
        except Exception as e:
            raise ValueError(f"خطا در ذخیره گراف: {str(e)}")
        
        return filepath
    
    def _map_spacy_label_to_type(self, label: str) -> str:
        """Map spaCy entity label to Hetionet type"""
        label_mapping = {
            "PERSON": "Gene",
            "ORG": "Gene",
            "GPE": "Anatomy",
            "LOC": "Anatomy",
            "EVENT": "Biological Process",
            "PRODUCT": "Compound",
            "DISEASE": "Disease",
            "GENE_OR_GENE_PRODUCT": "Gene",
            "CELLULAR_COMPONENT": "Cellular Component",
            "SIMPLE_CHEMICAL": "Compound",
            "DISEASE_OR_DISORDER": "Disease",
            "ANATOMICAL_SYSTEM": "Anatomy",
            "BIOLOGICAL_PROCESS": "Biological Process",
            "CELL": "Anatomy",
            "TISSUE": "Anatomy"
        }
        return label_mapping.get(label, "Gene")
    
    def _extract_subject_phrase(self, verb_token, doc) -> Optional[str]:
        """Extract full subject phrase for a verb"""
        for child in verb_token.children:
            if child.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                # Get the full noun phrase
                phrase = self._get_full_noun_phrase(child, doc)
                if phrase:
                    return phrase
        return None
    
    def _extract_object_phrase(self, verb_token, doc) -> Optional[str]:
        """Extract full object phrase for a verb"""
        for child in verb_token.children:
            if child.dep_ in ["dobj", "pobj", "attr", "acomp"]:
                phrase = self._get_full_noun_phrase(child, doc)
                if phrase:
                    return phrase
        return None
    
    def _get_full_noun_phrase(self, token, doc) -> Optional[str]:
        """Get the full noun phrase containing a token"""
        # Check if token is part of a noun chunk
        for chunk in doc.noun_chunks:
            if token.i >= chunk.start and token.i < chunk.end:
                return chunk.text.strip()
        
        # If not in a chunk, try to get the subtree
        subtree = list(token.subtree)
        if len(subtree) > 1:
            # Get text from first to last token in subtree
            start = min(t.i for t in subtree)
            end = max(t.i for t in subtree) + 1
            phrase = doc[start:end].text.strip()
            # Remove leading/trailing punctuation
            phrase = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', phrase)
            return phrase if phrase else None
        
        # Fallback to token text
        if token.pos_ in ["NOUN", "PROPN"]:
            return token.text.strip()
        
        return None
    
    def _find_entity_by_phrase(self, phrase: str, entity_map: Dict[str, str]) -> Optional[str]:
        """Find entity ID by matching phrase to entity names"""
        if not phrase:
            return None
        
        phrase_lower = phrase.lower().strip()
        
        # Exact match
        if phrase in entity_map:
            return entity_map[phrase]
        
        # Try case-insensitive exact match
        for name, eid in entity_map.items():
            if name.lower() == phrase_lower:
                return eid
        
        # Partial match - phrase contains entity name
        for name, eid in entity_map.items():
            if phrase_lower in name.lower() or name.lower() in phrase_lower:
                return eid
        
        # Word-level match - check if key words match
        phrase_words = set(phrase_lower.split())
        for name, eid in entity_map.items():
            name_words = set(name.lower().split())
            # If there's significant overlap
            if len(phrase_words & name_words) > 0 and len(phrase_words & name_words) / max(len(phrase_words), 1) > 0.5:
                return eid
        
        return None
    
    def _map_metaedge_to_readable_name(self, metaedge: str) -> str:
        """
        تبدیل metaedge code به نام معنی‌دار و خوانا
        
        Args:
            metaedge: کد metaedge (مثل GoA, GiG, CtD)
            
        Returns:
            نام معنی‌دار رابطه
        """
        metaedge_to_name = {
            "GoA": "goes to",
            "GiG": "interacts with",
            "GcG": "correlates with",
            "Gr>G": "regulates",
            "CtD": "treats",
            "CuG": "upregulates",
            "CdG": "downregulates",
            "GpBP": "participates in",
            "GpPW": "participates in pathway",
            "GpMF": "has molecular function",
            "GpCC": "located in",
            "DaG": "associates with",
            "DdG": "downregulates",
            "DuG": "upregulates",
        }
        return metaedge_to_name.get(metaedge, "related to")
    
    def _extract_relation_name_from_text(self, verb_token, sent, preposition: Optional[str] = None) -> str:
        """
        استخراج نام رابطه از متن بر اساس فعل و جمله
        
        Args:
            verb_token: توکن فعل از spaCy
            sent: جمله از spaCy
            preposition: حرف اضافه (در صورت وجود)
            
        Returns:
            نام رابطه از متن (مثل "interacts with", "treats", "participates in")
        """
        try:
            verb_text = verb_token.text
            verb_lemma = verb_token.lemma_
            
            # اگر حرف اضافه وجود دارد، آن را اضافه کن
            if preposition:
                # پیدا کردن حرف اضافه در جمله
                prep_text = preposition.lower()
                # ساخت نام رابطه: verb + preposition
                relation_name = f"{verb_lemma} {prep_text}"
            else:
                # فقط فعل
                relation_name = verb_lemma
            
            # تبدیل به فرم خوانا (مثلاً "interact_with" -> "interacts with")
            # برای فعل‌های رایج، نام بهتری بده
            relation_mapping = {
                "interact": "interacts with",
                "regulate": "regulates",
                "control": "controls",
                "treat": "treats",
                "participate": "participates in",
                "associate": "associated with",
                "bind": "binds to",
                "express": "expressed in",
                "localize": "localized in",
                "present": "presents",
                "upregulate": "upregulates",
                "downregulate": "downregulates",
                "inhibit": "inhibits",
                "activate": "activates",
                "cause": "causes",
                "lead": "leads to",
                "partner": "partnered with",
                "collaborate": "collaborates with",
                "develop": "develops",
                "sign": "signed agreement with",
                "expand": "expands",
                "hire": "hired",
                "supply": "supplies",
                "provide": "provides",
                "migrate": "migrates to"
            }
            
            # اگر mapping وجود دارد، از آن استفاده کن
            if verb_lemma in relation_mapping:
                relation_name = relation_mapping[verb_lemma]
            elif preposition:
                # اگر حرف اضافه داریم، verb + prep
                relation_name = f"{verb_lemma} {preposition.lower()}"
            else:
                # فقط فعل
                relation_name = verb_lemma
            
            return relation_name
        except Exception as e:
            logging.warning(f"Error extracting relation name: {e}")
            # Fallback: استفاده از lemma فعل
            return verb_token.lemma_ if hasattr(verb_token, 'lemma_') else str(verb_token)
    
    def _map_verb_to_relation(self, verb: str) -> Optional[str]:
        """Map verb to Hetionet metaedge type"""
        verb_lower = verb.lower()
        
        verb_mapping = {
            "regulate": "Gr>G",
            "regulates": "Gr>G",
            "regulating": "Gr>G",
            "control": "Gr>G",
            "controls": "Gr>G",
            "controlling": "Gr>G",
            "interact": "GiG",
            "interacts": "GiG",
            "interacting": "GiG",
            "interaction": "GiG",
            "bind": "GiG",
            "binds": "GiG",
            "binding": "GiG",
            "treat": "CtD",
            "treats": "CtD",
            "treating": "CtD",
            "treatment": "CtD",
            "cure": "CtD",
            "cures": "CtD",
            "curing": "CtD",
            "participate": "GpBP",
            "participates": "GpBP",
            "participating": "GpBP",
            "participation": "GpBP",
            "involve": "GpBP",
            "involves": "GpBP",
            "involving": "GpBP",
            "associate": "DaG",
            "associates": "DaG",
            "associated": "DaG",
            "association": "DaG",
            "relate": "DaG",
            "relates": "DaG",
            "related": "DaG",
            "relation": "DaG",
            "upregulate": "CuG",
            "upregulates": "CuG",
            "upregulating": "CuG",
            "downregulate": "CdG",
            "downregulates": "CdG",
            "downregulating": "CdG",
            "inhibit": "Gr>G",
            "inhibits": "Gr>G",
            "inhibiting": "Gr>G",
            "activate": "Gr>G",
            "activates": "Gr>G",
            "activating": "Gr>G",
            "express": "GpBP",
            "expresses": "GpBP",
            "expressing": "GpBP",
            "cause": "DaG",
            "causes": "DaG",
            "causing": "DaG",
            "lead": "DaG",
            "leads": "DaG",
            "leading": "DaG",
            # Business/General verbs
            "partner": "GiG",
            "partners": "GiG",
            "partnered": "GiG",
            "partnering": "GiG",
            "collaborate": "GiG",
            "collaborates": "GiG",
            "collaborated": "GiG",
            "collaborating": "GiG",
            "collaboration": "GiG",
            "develop": "GiG",
            "develops": "GiG",
            "developed": "GiG",
            "developing": "GiG",
            "sign": "GiG",
            "signs": "GiG",
            "signed": "GiG",
            "signing": "GiG",
            "agree": "GiG",
            "agrees": "GiG",
            "agreed": "GiG",
            "agreement": "GiG",
            "expand": "GiG",
            "expands": "GiG",
            "expanded": "GiG",
            "expanding": "GiG",
            "hire": "GiG",
            "hires": "GiG",
            "hired": "GiG",
            "hiring": "GiG",
            "supply": "GiG",
            "supplies": "GiG",
            "supplied": "GiG",
            "provide": "GiG",
            "provides": "GiG",
            "provided": "GiG",
            "providing": "GiG",
            "migrate": "GiG",
            "migrates": "GiG",
            "migrated": "GiG",
            "migrating": "GiG"
        }
        
        # Check exact match first
        if verb_lower in verb_mapping:
            return verb_mapping[verb_lower]
        
        # Check if any key is in the verb
        for key, value in verb_mapping.items():
            if key in verb_lower:
                return value
        
        # Default: return None to indicate no specific relationship
        return None
    
    def extract_persian(self, text: str, max_entities: int = 100, max_relationships: int = 200, **kwargs) -> Dict[str, Any]:
        """
        استخراج موجودیت‌ها و روابط از متن فارسی با استفاده از مدل‌های فارسی
        
        Args:
            text: متن ورودی فارسی
            max_entities: حداکثر تعداد موجودیت‌ها
            max_relationships: حداکثر تعداد روابط
            **kwargs: پارامترهای اضافی
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not NEW_MODULES_AVAILABLE:
            raise ValueError("ماژول‌های فارسی در دسترس نیستند. لطفاً requirements.txt را نصب کنید.")
        
        # Normalize text
        normalized_text = text
        if self.persian_normalizer:
            normalized_text = self.persian_normalizer.normalize(text)
        
        # Use Persian spaCy model if available
        nlp_model = self._get_spacy_model("fa")
        
        # Use modular pipeline for Persian
        try:
            pipeline = ModularExtractionPipeline(
                language="fa",
                enable_normalization=True,
                enable_ner=True,
                enable_relation_extraction=True,
                enable_coreference=kwargs.get("enable_coreference", False)
            )
            result = pipeline.process(normalized_text)
            
            # Convert to standard format
            entities = []
            for i, ent in enumerate(result.get("entities", [])):
                entities.append({
                    "id": f"ENTITY_{i}",
                    "name": ent.get("text", "") or ent.get("name", ""),
                    "type": ent.get("label", "Gene"),
                    "attributes": {
                        "score": ent.get("score", 0.5),
                        "confidence": ent.get("confidence", 0.5),
                        "language": "fa"
                    }
                })
            
            relationships = []
            for rel in result.get("relationships", []):
                relationships.append({
                    "source": rel.get("source", ""),
                    "target": rel.get("target", ""),
                    "metaedge": rel.get("metaedge", "GiG"),
                    "relation": rel.get("relation", ""),
                    "attributes": rel.get("attributes", {})
                })
            
            return {
                "entities": entities[:max_entities],
                "relationships": relationships[:max_relationships],
                "method": "persian",
                "stats": {
                    "num_entities": len(entities),
                    "num_relationships": len(relationships)
                }
            }
        except Exception as e:
            logging.warning(f"Persian extraction failed: {e}. Falling back to spaCy.")
            # Fallback to spaCy with Persian model
            if nlp_model:
                return self.extract_spacy(normalized_text, max_entities=max_entities, max_relationships=max_relationships)
            else:
                raise ValueError(f"استخراج فارسی ناموفق بود: {str(e)}")
    
    def extract_span_based(self, text: str, model_type: str = "biobert", max_entities: int = 100, 
                          max_relationships: int = 200, **kwargs) -> Dict[str, Any]:
        """
        استخراج مبتنی بر Span با BioBERT یا SciBERT
        
        Args:
            text: متن ورودی
            model_type: نوع مدل (biobert/scibert/auto)
            max_entities: حداکثر تعداد موجودیت‌ها
            max_relationships: حداکثر تعداد روابط
            **kwargs: پارامترهای اضافی
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not NEW_MODULES_AVAILABLE:
            raise ValueError("ماژول‌های span-based در دسترس نیستند.")
        
        language = self._detect_text_language(text)
        
        # Select extractor
        if model_type == "biobert":
            extractor = BioBERTExtractor()
        elif model_type == "scibert":
            extractor = SciBERTExtractor()
        else:
            extractor = SpanBasedExtractor(language=language)
        
        # Extract entities
        entities_raw = extractor.extract_entities(text)
        
        # Convert to standard format
        entities = []
        entity_map = {}
        for i, ent in enumerate(entities_raw):
            entity_id = f"ENTITY_{i}"
            entity_name = ent.get("text", "")
            entities.append({
                "id": entity_id,
                "name": entity_name,
                "type": ent.get("label", "Gene"),
                "attributes": {
                    "score": ent.get("score", 0.5),
                    "span": ent.get("span", (0, 0)),
                    "extraction_method": "span_based"
                }
            })
            entity_map[entity_name] = entity_id
        
        # Extract relationships with BERT
        relationships = []
        if NEW_MODULES_AVAILABLE:
            try:
                rel_extractor = BERTRelationExtractor(language=language)
                relationships_raw = rel_extractor.extract_relations(text, entities_raw, max_pairs=100)
                
                for rel in relationships_raw:
                    source_name = rel.get("source", "")
                    target_name = rel.get("target", "")
                    source_id = entity_map.get(source_name)
                    target_id = entity_map.get(target_name)
                    
                    if source_id and target_id:
                        relationships.append({
                            "source": source_id,
                            "target": target_id,
                            "metaedge": rel.get("metaedge", "GiG"),
                            "relation": rel.get("relation", ""),
                            "attributes": rel.get("attributes", {})
                        })
            except Exception as e:
                logging.warning(f"BERT relation extraction failed: {e}")
        
        return {
            "entities": entities[:max_entities],
            "relationships": relationships[:max_relationships],
            "method": f"span_based_{model_type}",
            "stats": {
                "num_entities": len(entities),
                "num_relationships": len(relationships)
            }
        }
    
    def extract_with_coreference(self, text: str, base_method: str = "spacy", 
                                 max_entities: int = 100, max_relationships: int = 200, **kwargs) -> Dict[str, Any]:
        """
        استخراج با Coreference Resolution
        
        Args:
            text: متن ورودی
            base_method: روش پایه استخراج
            max_entities: حداکثر تعداد موجودیت‌ها
            max_relationships: حداکثر تعداد روابط
            **kwargs: پارامترهای اضافی
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not NEW_MODULES_AVAILABLE:
            raise ValueError("ماژول coreference resolution در دسترس نیست.")
        
        # Extract with base method
        extraction_result = self.extract(text, method=base_method, max_entities=max_entities, 
                                         max_relationships=max_relationships, **kwargs)
        
        # Apply coreference resolution
        language = self._detect_text_language(text)
        nlp_model = self._get_spacy_model(language)
        
        try:
            resolver = CoreferenceResolver(language=language, spacy_model=nlp_model)
            reference_map = resolver.resolve(text, extraction_result.get("entities", []))
            
            # Merge entities
            merged_entities = resolver.merge_entities(extraction_result.get("entities", []), reference_map)
            
            extraction_result["entities"] = merged_entities
            extraction_result["method"] = f"{base_method}_with_coreference"
            
        except Exception as e:
            logging.warning(f"Coreference resolution failed: {e}")
        
        return extraction_result
    
    def extract_long_text(self, text: str, method: str = "spacy", 
                         chunking_strategy: str = "smart", chunk_overlap: float = 0.2,
                         max_tokens: int = 512, max_entities: int = 100, 
                         max_relationships: int = 200, **kwargs) -> Dict[str, Any]:
        """
        استخراج از متن‌های طولانی با chunking
        
        Args:
            text: متن ورودی
            method: روش استخراج پایه
            chunking_strategy: استراتژی chunking (smart/sliding_window/sentence/paragraph)
            chunk_overlap: نسبت overlap در sliding window
            max_tokens: حداکثر توکن در هر chunk
            max_entities: حداکثر تعداد موجودیت‌ها
            max_relationships: حداکثر تعداد روابط
            **kwargs: پارامترهای اضافی
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not NEW_MODULES_AVAILABLE:
            raise ValueError("ماژول‌های chunking در دسترس نیستند.")
        
        language = self._detect_text_language(text)
        
        # Initialize chunker
        strategy_map = {
            "smart": ChunkingStrategy.SMART,
            "sliding_window": ChunkingStrategy.SLIDING_WINDOW,
            "sentence": ChunkingStrategy.SENTENCE,
            "paragraph": ChunkingStrategy.PARAGRAPH
        }
        
        chunker = SmartChunker(
            strategy=strategy_map.get(chunking_strategy, ChunkingStrategy.SMART),
            max_tokens=max_tokens,
            overlap_ratio=chunk_overlap,
            language=language
        )
        
        # Chunk text
        chunks = chunker.chunk(text)
        
        if not chunks:
            # Fallback to simple extraction
            return self.extract(text, method=method, max_entities=max_entities, 
                              max_relationships=max_relationships, **kwargs)
        
        # Process each chunk
        chunk_results = []
        for chunk_data in chunks:
            chunk_text = chunk_data.get("text", "")
            if not chunk_text:
                continue
            
            try:
                result = self.extract(chunk_text, method=method, max_entities=max_entities, 
                                     max_relationships=max_relationships, **kwargs)
                result["chunk_metadata"] = chunk_data
                chunk_results.append(result)
            except Exception as e:
                logging.warning(f"Error processing chunk: {e}")
                continue
        
        # Merge results hierarchically
        merger = HierarchicalMerger(
            weight_by_frequency=True,
            min_confidence=kwargs.get("min_confidence", 0.5)
        )
        
        merged_result = merger.merge_chunk_results(chunk_results)
        
        return {
            "entities": merged_result.get("entities", [])[:max_entities],
            "relationships": merged_result.get("relationships", [])[:max_relationships],
            "method": f"{method}_long_text_{chunking_strategy}",
            "stats": {
                "num_entities": len(merged_result.get("entities", [])),
                "num_relationships": len(merged_result.get("relationships", [])),
                "num_chunks": len(chunks)
            }
        }
    
    def extract(self, text: str, method: str = "simple", enable_preprocessing: bool = False, 
                language: str = "auto", **kwargs) -> Dict[str, Any]:
        """
        استخراج موجودیت‌ها و روابط با روش انتخابی
        
        نکته مهم در مورد پیش‌پردازش:
        - برای مدل‌های زبانی (LLM): متن اصلی استفاده می‌شود تا معنی کامل حفظ شود
        - برای روش‌های rule-based و spaCy: می‌توان از متن پیش‌پردازش شده استفاده کرد
          تا stop words در گراف ظاهر نشوند
        
        Args:
            text: متن ورودی
            method: روش استخراج (simple, spacy, spacy_svo_enhanced, llm, llm_multipass, hybrid)
            enable_preprocessing: فعال‌سازی پیش‌پردازش (حذف stop words از گراف)
            language: زبان متن برای پیش‌پردازش (auto/fa/en)
            **kwargs: پارامترهای اضافی
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not text or not text.strip():
            raise ValueError("متن ورودی نمی‌تواند خالی باشد")
        
        valid_methods = ["simple", "spacy", "spacy_svo_enhanced", "llm", "llm_multipass", "hybrid", 
                        "persian", "span_based", "with_coreference", "long_text",
                        "joint_er", "autoregressive", "edc", "incremental"]
        if method not in valid_methods:
            raise ValueError(f"روش استخراج نامعتبر: {method}. روش‌های مجاز: {', '.join(valid_methods)}")
        
        # تعیین اینکه آیا این روش از LLM استفاده می‌کند
        llm_methods = ["llm", "llm_multipass"]
        uses_llm = method in llm_methods or (method == "hybrid" and "llm" in kwargs.get("hybrid_methods", []))
        
        # برای روش‌های LLM: متن اصلی را نگه دار (stop words برای معنی مهم هستند)
        # برای روش‌های rule-based/spaCy: می‌توان از متن پیش‌پردازش شده استفاده کرد
        if enable_preprocessing and not uses_llm:
            # پیش‌پردازش فقط برای روش‌های غیر-LLM
            processed_text = self._preprocess_text_for_graph(text, language=language, remove_stop_words=True)
            logging.info(f"Preprocessing applied for method {method}: {len(text)} -> {len(processed_text)} chars")
        else:
            processed_text = text
        
        # برای روش‌های LLM: همیشه از متن اصلی استفاده کن
        text_for_extraction = text if uses_llm else processed_text
        
        try:
            if method == "simple":
                return self.extract_simple(text_for_extraction, **kwargs)
            elif method == "spacy":
                return self.extract_spacy(text_for_extraction, **kwargs)
            elif method == "spacy_svo_enhanced":
                return self.extract_spacy_svo_enhanced(text_for_extraction, **kwargs)
            elif method == "llm":
                # برای LLM: متن اصلی را استفاده کن (حتی اگر preprocessing فعال باشد)
                return self.extract_llm(text, **kwargs)
            elif method == "llm_multipass":
                # برای LLM: متن اصلی را استفاده کن
                return self.extract_llm_multipass(text, **kwargs)
            elif method == "hybrid":
                # برای hybrid: اگر LLM در hybrid_methods است، متن اصلی را به آن بده
                hybrid_methods = kwargs.get("hybrid_methods", kwargs.get("methods", ["spacy", "llm"]))
                kwargs["methods"] = hybrid_methods  # اطمینان از اینکه methods تنظیم شده است
                if "llm" in hybrid_methods or "llm_multipass" in hybrid_methods:
                    # متن اصلی را برای LLM و متن پیش‌پردازش شده را برای بقیه
                    kwargs["original_text"] = text  # برای استفاده در hybrid
                return self.extract_hybrid(text_for_extraction, **kwargs)
            elif method == "persian":
                return self.extract_persian(text_for_extraction, **kwargs)
            elif method == "span_based":
                return self.extract_span_based(text_for_extraction, **kwargs)
            elif method == "with_coreference":
                base_method = kwargs.pop("base_method", "spacy")
                return self.extract_with_coreference(text_for_extraction, base_method=base_method, **kwargs)
            elif method == "long_text":
                return self.extract_long_text(text_for_extraction, **kwargs)
            elif method == "joint_er":
                return self.extract_joint_er(text_for_extraction, **kwargs)
            elif method == "autoregressive":
                # برای autoregressive: متن اصلی را استفاده کن (نیاز به LLM)
                return self.extract_autoregressive(text, **kwargs)
            elif method == "edc":
                # برای EDC: متن اصلی را استفاده کن (نیاز به LLM)
                return self.extract_edc(text, **kwargs)
            elif method == "incremental":
                return self.extract_incremental(text_for_extraction, **kwargs)
        except ValueError as e:
            # Re-raise validation errors
            raise
        except Exception as e:
            logging.error(f"Error in {method} extraction: {e}")
            raise ValueError(f"خطا در استخراج با روش {method}: {str(e)}")
    
    def process_url_to_graph(self, url: str, method: str = "simple",
                             use_wikipedia_extraction: bool = True,
                             save: bool = True, output_dir: str = "uploaded_graphs",
                             enable_entity_resolution: bool = True,
                             enable_relationship_weighting: bool = True,
                             min_relationship_weight: float = 0.0,
                             remove_isolated_nodes: bool = False,
                             enable_preprocessing: bool = False,
                             language: str = "auto",
                             **kwargs) -> Dict[str, Any]:
        """
        پردازش کامل: استخراج متن از URL و ساخت گراف
        
        این متد دو مرحله دارد:
        1. استخراج متن مفید از URL (با حذف محتوای غیرضروری)
        2. تبدیل متن به گراف با روش انتخابی
        
        Args:
            url: URL صفحه وب
            method: روش استخراج گراف
            use_wikipedia_extraction: استفاده از استخراج تخصصی برای ویکی‌پدیا
            save: آیا گراف ذخیره شود؟
            output_dir: پوشه خروجی
            enable_entity_resolution: فعال/غیرفعال کردن entity resolution
            enable_relationship_weighting: فعال/غیرفعال کردن relationship weighting
            min_relationship_weight: حداقل وزن برای نگه داشتن رابطه
            remove_isolated_nodes: حذف نودهای ایزوله
            enable_preprocessing: فعال‌سازی پیش‌پردازش (حذف stop words از گراف)
            language: زبان متن برای پیش‌پردازش (auto/fa/en)
            **kwargs: پارامترهای اضافی برای استخراج
            
        Returns:
            Dictionary containing graph info and filepath
        """
        # Import URL extractors
        try:
            from url_extractor import extract_text_from_url, is_valid_url
            from urllib.parse import quote
            URL_EXTRACTOR_AVAILABLE = True
        except ImportError:
            URL_EXTRACTOR_AVAILABLE = False
            logging.warning("URL extractor not available")
        
        try:
            from wikipedia_extractor import WikipediaExtractor
            WIKIPEDIA_EXTRACTOR_AVAILABLE = True
        except ImportError:
            WIKIPEDIA_EXTRACTOR_AVAILABLE = False
            logging.warning("Wikipedia extractor not available")
        
        # بررسی معتبر بودن URL
        if not URL_EXTRACTOR_AVAILABLE:
            raise ValueError("استخراج از URL در دسترس نیست")
        
        if not is_valid_url(url):
            raise ValueError("URL نامعتبر است")
        
        # مرحله 1: استخراج متن از URL
        text = None
        is_wikipedia = 'wikipedia.org' in url.lower()
        
        if is_wikipedia and WIKIPEDIA_EXTRACTOR_AVAILABLE and use_wikipedia_extraction:
            # استفاده از استخراج تخصصی ویکی‌پدیا
            try:
                wiki_language = 'fa' if 'fa.wikipedia' in url else 'en'
                wiki_extractor = WikipediaExtractor(language=wiki_language)
                wiki_data = wiki_extractor.extract_from_url(url)
                
                if "error" not in wiki_data:
                    text = wiki_data.get("text", "")
                    
                    # اگر متن خالی یا خیلی کوتاه است، از get_full_text استفاده کن
                    if not text or len(text.strip()) < 200:
                        logging.info(f"Text too short ({len(text)} chars), trying get_full_text...")
                        full_text = wiki_extractor.get_full_text(wiki_data.get("title", ""))
                        if full_text and len(full_text.strip()) > len(text.strip()):
                            text = full_text
                            logging.info(f"Using get_full_text: {len(text)} characters")
                    
                    # اگر هنوز متن کافی نداریم، از HTML parsing مستقیم استفاده کن
                    if not text or len(text.strip()) < 200:
                        logging.info("Trying direct HTML extraction...")
                        try:
                            encoded_title = quote(wiki_data.get("title", "").replace(' ', '_'))
                            wiki_url = f"{wiki_extractor.base_url}/wiki/{encoded_title}"
                            direct_text = extract_text_from_url(wiki_url, clean_content=True, max_length=10000)
                            if direct_text and len(direct_text.strip()) > len(text.strip()):
                                text = direct_text
                                logging.info(f"Using direct HTML extraction: {len(text)} characters")
                        except Exception as e:
                            logging.warning(f"Direct HTML extraction failed: {e}")
                    
                    # اعمال محدودیت نهایی طول
                    if text and len(text) > 10000:
                        # قطع کردن هوشمند
                        from url_extractor import _truncate_text_intelligently
                        text = _truncate_text_intelligently(text, 10000)
                        logging.info(f"Final text length limited to {len(text)} characters")
                    
                    if not text or len(text.strip()) < 50:
                        raise ValueError(f"نمی‌توان متن کافی از ویکی‌پدیا استخراج کرد (فقط {len(text)} کاراکتر)")
                    
                    logging.info(f"Wikipedia text extracted successfully: {len(text)} characters")
                else:
                    # Fallback به استخراج عادی
                    logging.warning(f"Wikipedia extraction failed: {wiki_data.get('error')}")
                    text = extract_text_from_url(url, clean_content=True, max_length=10000)
            except Exception as e:
                logging.warning(f"Wikipedia extraction failed: {e}")
                text = extract_text_from_url(url, clean_content=True, max_length=10000)
        else:
            # استخراج عادی از URL با clean_content=True
            text = extract_text_from_url(url, clean_content=True, max_length=10000)
        
        if not text or not text.strip():
            raise ValueError("نمی‌توان متن را از URL استخراج کرد")
        
        logging.info(f"Text extracted from URL ({len(text)} characters), proceeding to graph construction")
        
        # مرحله 2: تبدیل متن به گراف
        return self.process_text_to_graph(
            text=text,
            method=method,
            save=save,
            output_dir=output_dir,
            enable_entity_resolution=enable_entity_resolution,
            enable_relationship_weighting=enable_relationship_weighting,
            min_relationship_weight=min_relationship_weight,
            remove_isolated_nodes=remove_isolated_nodes,
            enable_preprocessing=enable_preprocessing,
            language=language,
            **kwargs
        )
    
    def process_text_to_graph(self, text: str, method: str = "simple", 
                              save: bool = True, output_dir: str = "uploaded_graphs",
                              enable_entity_resolution: bool = True,
                              enable_relationship_weighting: bool = True,
                              min_relationship_weight: float = 0.0,
                              remove_isolated_nodes: bool = False,
                              enable_preprocessing: bool = False,
                              language: str = "auto",
                              **kwargs) -> Dict[str, Any]:
        """
        پردازش کامل: استخراج از متن و ساخت گراف
        
        Args:
            text: متن ورودی
            method: روش استخراج
            save: آیا گراف ذخیره شود؟
            output_dir: پوشه خروجی
            enable_entity_resolution: فعال/غیرفعال کردن entity resolution
            enable_relationship_weighting: فعال/غیرفعال کردن relationship weighting
            min_relationship_weight: حداقل وزن برای نگه داشتن رابطه
            remove_isolated_nodes: حذف نودهای ایزوله
            enable_preprocessing: فعال‌سازی پیش‌پردازش (حذف stop words از گراف)
            language: زبان متن برای پیش‌پردازش (auto/fa/en)
            **kwargs: پارامترهای اضافی برای استخراج
            
        Returns:
            Dictionary containing graph info and filepath
        """
        try:
            # Extract entities and relationships
            extraction_result = self.extract(text, method=method, 
                                            enable_preprocessing=enable_preprocessing,
                                            language=language, **kwargs)
            
            # Validate extraction result
            if not extraction_result:
                raise ValueError("نتیجه استخراج خالی است")
            
            if "entities" not in extraction_result:
                raise ValueError("نتیجه استخراج فاقد موجودیت‌ها است")
            
            if "relationships" not in extraction_result:
                extraction_result["relationships"] = []
            
            # Build graph
            graph = self.build_graph(extraction_result)
            
            # Validate graph
            if graph.number_of_nodes() == 0:
                raise ValueError("گراف ساخته شده خالی است")
            
            # Apply relationship weighting if enabled
            if enable_relationship_weighting:
                graph = self._apply_relationship_weighting(graph)
            
            # Apply entity resolution if enabled
            resolution_summary = None
            if enable_entity_resolution and self.entity_resolution:
                try:
                    graph = self.entity_resolution.resolve_entities_in_graph(graph, dry_run=False)
                    resolution_summary = self.entity_resolution.get_resolution_summary()
                except Exception as e:
                    logging.warning(f"Entity resolution failed: {e}")
            
            # Remove low-weight relationships
            if min_relationship_weight > 0:
                edges_to_remove = []
                for u, v, data in graph.edges(data=True):
                    weight = data.get("weight", 1.0)
                    if weight < min_relationship_weight:
                        edges_to_remove.append((u, v))
                for u, v in edges_to_remove:
                    graph.remove_edge(u, v)
            
            # Remove isolated nodes if requested
            if remove_isolated_nodes:
                isolated = list(nx.isolates(graph))
                graph.remove_nodes_from(isolated)
            
            # Calculate graph statistics
            stats = self._calculate_graph_statistics(graph)
            
            # Save if requested
            filepath = None
            if save:
                try:
                    filepath = self.save_graph(graph, output_dir=output_dir)
                except Exception as e:
                    logging.error(f"Error saving graph: {e}")
                    raise ValueError(f"خطا در ذخیره گراف: {str(e)}")
            
            return {
                "graph": graph,
                "extraction_result": extraction_result,
                "filepath": filepath,
                "stats": {
                    "num_nodes": graph.number_of_nodes(),
                    "num_edges": graph.number_of_edges(),
                    "num_entities": extraction_result.get("stats", {}).get("num_entities", len(extraction_result.get("entities", []))),
                    "num_relationships": extraction_result.get("stats", {}).get("num_relationships", len(extraction_result.get("relationships", []))),
                    **stats
                },
                "resolution_summary": resolution_summary
            }
        except ValueError as e:
            # Re-raise validation errors
            raise
        except Exception as e:
            logging.error(f"Error in process_text_to_graph: {e}")
            raise ValueError(f"خطا در پردازش متن به گراف: {str(e)}")
    
    def _apply_relationship_weighting(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Apply weighting to relationships based on frequency and confidence"""
        # Count relationship frequencies
        relationship_counts = {}
        for u, v, data in graph.edges(data=True):
            metaedge = data.get("metaedge", "GiG")
            key = (u, v, metaedge)
            relationship_counts[key] = relationship_counts.get(key, 0) + 1
        
        # Update edge weights
        for u, v, data in graph.edges(data=True):
            metaedge = data.get("metaedge", "GiG")
            key = (u, v, metaedge)
            
            # Base weight from frequency
            frequency_weight = relationship_counts.get(key, 1)
            
            # Confidence weight
            confidence = data.get("attributes", {}).get("confidence", 0.5)
            
            # Combined weight
            weight = frequency_weight * (0.5 + 0.5 * confidence)
            data["weight"] = weight
            data["frequency"] = frequency_weight
        
        return graph
    
    def _calculate_graph_statistics(self, graph: nx.MultiDiGraph) -> Dict[str, Any]:
        """Calculate detailed graph statistics"""
        stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "node_types": {},
            "edge_types": {},
            "avg_degree": 0.0,
            "density": 0.0
        }
        
        # Node type distribution
        for node, data in graph.nodes(data=True):
            node_type = data.get("type") or data.get("kind", "Unknown")
            stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
        
        # Edge type distribution
        for u, v, data in graph.edges(data=True):
            edge_type = data.get("metaedge") or data.get("relation", "unknown")
            stats["edge_types"][edge_type] = stats["edge_types"].get(edge_type, 0) + 1
        
        # Average degree
        if graph.number_of_nodes() > 0:
            total_degree = sum(dict(graph.degree()).values())
            stats["avg_degree"] = total_degree / graph.number_of_nodes()
        
        # Density
        if graph.number_of_nodes() > 1:
            max_edges = graph.number_of_nodes() * (graph.number_of_nodes() - 1)
            if isinstance(graph, nx.MultiDiGraph):
                max_edges = max_edges * 2  # Directed graph
            stats["density"] = graph.number_of_edges() / max_edges if max_edges > 0 else 0.0
        
        return stats
    
    def extract_joint_er(self, text: str, max_entities: int = 100, max_relationships: int = 200,
                        structure_iterations: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Joint Entity-Relation Extraction با Graph Structure Learning
        
        این روش بر اساس GraphER و روش‌های state-of-the-art است که موجودیت‌ها و روابط را
        به صورت همزمان استخراج می‌کند و ساختار گراف را در حین استخراج بهینه می‌کند.
        
        Args:
            text: متن ورودی
            max_entities: حداکثر تعداد موجودیت‌ها
            max_relationships: حداکثر تعداد روابط
            structure_iterations: تعداد تکرار برای بهینه‌سازی ساختار گراف
            **kwargs: پارامترهای اضافی
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not text or not text.strip():
            raise ValueError("متن ورودی نمی‌تواند خالی باشد")
        
        # مرحله 1: استخراج اولیه موجودیت‌ها و روابط با روش‌های مختلف
        # استفاده از ترکیب spaCy و LLM برای استخراج اولیه
        initial_entities = []
        initial_relationships = []
        
        # استخراج با spaCy برای موجودیت‌ها
        if self.nlp:
            try:
                doc = self.nlp(text)
                entity_map = {}
                for ent in doc.ents:
                    if ent.label_ not in ["CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY", "DATE", "TIME"]:
                        entity_id = f"ENT_{len(initial_entities)}"
                        entity_map[ent.text] = entity_id
                        initial_entities.append({
                            "id": entity_id,
                            "name": ent.text,
                            "type": ent.label_,
                            "attributes": {"start": ent.start_char, "end": ent.end_char}
                        })
            except Exception as e:
                logging.warning(f"spaCy entity extraction failed: {e}")
        
        # استخراج روابط با استفاده از dependency parsing
        if self.nlp and initial_entities:
            try:
                doc = self.nlp(text)
                entity_spans = {ent["name"]: ent for ent in initial_entities}
                
                for sent in doc.sents:
                    # پیدا کردن موجودیت‌ها در جمله
                    sent_entities = []
                    for token in sent:
                        for ent_name, ent_data in entity_spans.items():
                            if token.text in ent_name or ent_name in token.text:
                                sent_entities.append((token.i, ent_data))
                                break
                    
                    # استخراج روابط از dependency tree
                    for token in sent:
                        if token.dep_ in ["nsubj", "dobj", "pobj", "attr", "acomp"]:
                            head_ent = None
                            dep_ent = None
                            
                            # پیدا کردن موجودیت در head
                            for ent_idx, ent_data in sent_entities:
                                if token.head.i == ent_idx or abs(token.head.i - ent_idx) <= 2:
                                    head_ent = ent_data
                                    break
                            
                            # پیدا کردن موجودیت در dependent
                            for ent_idx, ent_data in sent_entities:
                                if token.i == ent_idx or abs(token.i - ent_idx) <= 2:
                                    dep_ent = ent_data
                                    break
                            
                            if head_ent and dep_ent and head_ent["id"] != dep_ent["id"]:
                                relation_name = token.lemma_ if token.lemma_ else token.text.lower()
                                initial_relationships.append({
                                    "source": head_ent["id"],
                                    "target": dep_ent["id"],
                                    "relation": relation_name,
                                    "metaedge": "GiG",  # Default
                                    "attributes": {"dep": token.dep_, "confidence": 0.7}
                                })
            except Exception as e:
                logging.warning(f"Dependency parsing failed: {e}")
        
        # مرحله 2: بهینه‌سازی ساختار گراف با تکرار
        entities = initial_entities[:max_entities]
        relationships = initial_relationships[:max_relationships]
        
        # ساخت گراف موقت برای بهینه‌سازی
        temp_graph = nx.MultiDiGraph()
        for entity in entities:
            temp_graph.add_node(entity["id"], **entity)
        
        for rel in relationships:
            if temp_graph.has_node(rel["source"]) and temp_graph.has_node(rel["target"]):
                temp_graph.add_edge(rel["source"], rel["target"], **rel)
        
        # تکرار برای بهینه‌سازی ساختار
        for iteration in range(structure_iterations):
            # محاسبه اهمیت نودها بر اساس centrality
            try:
                degree_centrality = nx.degree_centrality(temp_graph)
                betweenness_centrality = nx.betweenness_centrality(temp_graph)
                
                # به‌روزرسانی موجودیت‌ها بر اساس اهمیت
                for entity in entities:
                    entity_id = entity["id"]
                    if entity_id in temp_graph:
                        entity["attributes"]["degree_centrality"] = degree_centrality.get(entity_id, 0.0)
                        entity["attributes"]["betweenness_centrality"] = betweenness_centrality.get(entity_id, 0.0)
                
                # حذف روابط ضعیف
                relationships_to_keep = []
                for rel in relationships:
                    source_importance = degree_centrality.get(rel["source"], 0.0)
                    target_importance = degree_centrality.get(rel["target"], 0.0)
                    confidence = rel.get("attributes", {}).get("confidence", 0.5)
                    
                    # نگه داشتن روابط با اهمیت بالا
                    if (source_importance + target_importance) / 2 > 0.1 or confidence > 0.6:
                        relationships_to_keep.append(rel)
                
                relationships = relationships_to_keep[:max_relationships]
                
                # بازسازی گراف
                temp_graph = nx.MultiDiGraph()
                for entity in entities:
                    temp_graph.add_node(entity["id"], **entity)
                for rel in relationships:
                    if temp_graph.has_node(rel["source"]) and temp_graph.has_node(rel["target"]):
                        temp_graph.add_edge(rel["source"], rel["target"], **rel)
                        
            except Exception as e:
                logging.warning(f"Graph optimization iteration {iteration} failed: {e}")
                break
        
        return {
            "entities": entities,
            "relationships": relationships,
            "method": "joint_er",
            "stats": {
                "num_entities": len(entities),
                "num_relationships": len(relationships),
                "structure_iterations": structure_iterations
            }
        }
    
    def extract_autoregressive(self, text: str, model: str = "mistralai/Mistral-7B-Instruct-v0.2",
                              max_entities: int = 100, max_relationships: int = 200,
                              max_generation_length: int = 2048, **kwargs) -> Dict[str, Any]:
        """
        Autoregressive Text-to-Graph با استفاده از Transformer Encoder-Decoder
        
        این روش گراف را به صورت sequential تولید می‌کند و از ساختار autoregressive
        برای تولید موجودیت‌ها و روابط استفاده می‌کند.
        
        Args:
            text: متن ورودی
            model: نام مدل LLM
            max_entities: حداکثر تعداد موجودیت‌ها
            max_relationships: حداکثر تعداد روابط
            max_generation_length: حداکثر طول تولید
            **kwargs: پارامترهای اضافی
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not text or not text.strip():
            raise ValueError("متن ورودی نمی‌تواند خالی باشد")
        
        if not self.hf_client:
            # Fallback به روش LLM عادی
            logging.warning("HuggingFace client not available, falling back to standard LLM extraction")
            return self.extract_llm(text, model=model, max_entities=max_entities,
                                   max_relationships=max_relationships, **kwargs)
        
        # Prompt برای autoregressive generation
        prompt = f"""You are a knowledge graph extraction system. Extract entities and relationships from the text in an autoregressive manner.

Generate the graph step by step:
1. First, identify all entities
2. Then, for each entity, identify its relationships with other entities
3. Format as JSON

Text:
{text}

Generate entities and relationships in JSON format:
{{
  "entities": [
    {{"id": "E1", "name": "entity_name", "type": "EntityType", "attributes": {{}}}}
  ],
  "relationships": [
    {{"source": "E1", "target": "E2", "relation": "relation_name", "metaedge": "GiG", "attributes": {{}}}}
  ]
}}

Response:"""
        
        try:
            response = self.hf_client.text_generation(
                prompt,
                model=model,
                max_new_tokens=max_generation_length,
                temperature=0.3,
                return_full_text=False
            )
            
            # Parse response
            result = self._parse_llm_response(response)
            
            if result and "entities" in result and "relationships" in result:
                entities = result["entities"][:max_entities]
                relationships = result["relationships"][:max_relationships]
                
                return {
                    "entities": entities,
                    "relationships": relationships,
                    "method": "autoregressive",
                    "stats": {
                        "num_entities": len(entities),
                        "num_relationships": len(relationships)
                    }
                }
            else:
                # Fallback
                return self.extract_llm(text, model=model, max_entities=max_entities,
                                      max_relationships=max_relationships, **kwargs)
        except Exception as e:
            logging.error(f"Autoregressive extraction failed: {e}")
            # Fallback
            return self.extract_llm(text, model=model, max_entities=max_entities,
                                  max_relationships=max_relationships, **kwargs)
    
    def extract_edc(self, text: str, max_entities: int = 100, max_relationships: int = 200,
                   use_rag: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Extract-Define-Canonicalize (EDC) Framework
        
        یک روش سه مرحله‌ای برای ساخت گراف دانش:
        1. Extract: استخراج اولیه triplets
        2. Define: تعریف schema با استفاده از LLM
        3. Canonicalize: استانداردسازی روابط
        
        Args:
            text: متن ورودی
            max_entities: حداکثر تعداد موجودیت‌ها
            max_relationships: حداکثر تعداد روابط
            use_rag: استفاده از RAG برای بهبود استخراج
            **kwargs: پارامترهای اضافی
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not text or not text.strip():
            raise ValueError("متن ورودی نمی‌تواند خالی باشد")
        
        # مرحله 1: Extract - استخراج اولیه triplets
        logging.info("EDC Phase 1: Extracting initial triplets")
        initial_result = self.extract_hybrid(
            text,
            max_entities=max_entities * 2,  # بیشتر استخراج کن برای فیلتر کردن
            max_relationships=max_relationships * 2,
            methods=["spacy", "llm"] if self.hf_client else ["spacy"],
            **kwargs
        )
        
        initial_entities = initial_result.get("entities", [])[:max_entities * 2]
        initial_relationships = initial_result.get("relationships", [])[:max_relationships * 2]
        
        # مرحله 2: Define - تعریف schema با LLM
        logging.info("EDC Phase 2: Defining schema")
        if self.hf_client and use_rag:
            # استفاده از LLM برای تعریف schema
            schema_prompt = f"""Analyze the following extracted entities and relationships and define a canonical schema.

Entities: {[e.get('name', '') for e in initial_entities[:20]]}
Relationships: {[r.get('relation', '') for r in initial_relationships[:20]]}

Define canonical entity types and relation types. Return JSON:
{{
  "entity_types": ["Type1", "Type2"],
  "relation_types": ["Relation1", "Relation2"],
  "canonical_mapping": {{
    "old_relation": "canonical_relation"
  }}
}}

Response:"""
            
            try:
                schema_response = self.hf_client.text_generation(
                    schema_prompt,
                    model="mistralai/Mistral-7B-Instruct-v0.2",
                    max_new_tokens=512,
                    temperature=0.2,
                    return_full_text=False
                )
                
                schema_result = self._parse_llm_response(schema_response)
                canonical_mapping = schema_result.get("canonical_mapping", {}) if schema_result else {}
            except Exception as e:
                logging.warning(f"Schema definition failed: {e}")
                canonical_mapping = {}
        else:
            canonical_mapping = {}
        
        # مرحله 3: Canonicalize - استانداردسازی
        logging.info("EDC Phase 3: Canonicalizing entities and relationships")
        
        # استانداردسازی موجودیت‌ها
        canonicalized_entities = []
        entity_name_map = {}  # برای merge کردن موجودیت‌های مشابه
        
        for entity in initial_entities:
            entity_name = entity.get("name", "").strip().lower()
            entity_type = entity.get("type", "Unknown")
            
            # Normalize entity name
            normalized_name = entity_name.title() if entity_name else ""
            
            # Check if similar entity exists
            if normalized_name in entity_name_map:
                # Merge with existing entity
                existing_id = entity_name_map[normalized_name]
                # Update attributes
                for e in canonicalized_entities:
                    if e["id"] == existing_id:
                        # Merge attributes
                        existing_attrs = e.get("attributes", {})
                        new_attrs = entity.get("attributes", {})
                        existing_attrs.update(new_attrs)
                        break
            else:
                # New entity
                entity_id = f"ENT_{len(canonicalized_entities)}"
                entity_name_map[normalized_name] = entity_id
                canonicalized_entities.append({
                    "id": entity_id,
                    "name": normalized_name,
                    "type": entity_type,
                    "attributes": entity.get("attributes", {})
                })
        
        # استانداردسازی روابط
        canonicalized_relationships = []
        relationship_set = set()  # برای جلوگیری از duplicate
        
        for rel in initial_relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            relation = rel.get("relation", "").strip().lower()
            
            # Canonicalize relation name
            canonical_relation = canonical_mapping.get(relation, relation)
            if not canonical_relation:
                canonical_relation = relation
            
            # Normalize relation
            normalized_relation = canonical_relation.title() if canonical_relation else "Related"
            
            # Map source and target to canonical entity IDs
            source_normalized = source.lower().strip()
            target_normalized = target.lower().strip()
            
            source_id = None
            target_id = None
            
            for entity in canonicalized_entities:
                if entity["name"].lower() == source_normalized or source_normalized in entity["name"].lower():
                    source_id = entity["id"]
                if entity["name"].lower() == target_normalized or target_normalized in entity["name"].lower():
                    target_id = entity["id"]
            
            if source_id and target_id and source_id != target_id:
                rel_key = (source_id, target_id, normalized_relation)
                if rel_key not in relationship_set:
                    relationship_set.add(rel_key)
                    canonicalized_relationships.append({
                        "source": source_id,
                        "target": target_id,
                        "relation": normalized_relation,
                        "metaedge": rel.get("metaedge", "GiG"),
                        "attributes": rel.get("attributes", {})
                    })
        
        return {
            "entities": canonicalized_entities[:max_entities],
            "relationships": canonicalized_relationships[:max_relationships],
            "method": "edc",
            "stats": {
                "num_entities": len(canonicalized_entities),
                "num_relationships": len(canonicalized_relationships),
                "initial_entities": len(initial_entities),
                "initial_relationships": len(initial_relationships)
            }
        }
    
    def extract_incremental(self, text: str, chunk_size: int = 500, overlap: int = 100,
                          base_method: str = "spacy", max_entities: int = 100,
                          max_relationships: int = 200, **kwargs) -> Dict[str, Any]:
        """
        Incremental Knowledge Graph Construction
        
        ساخت تدریجی گراف با پردازش متن به صورت chunk و ادغام نتایج.
        این روش برای متن‌های طولانی مناسب است و گراف را به صورت incremental می‌سازد.
        
        Args:
            text: متن ورودی
            chunk_size: اندازه هر chunk
            overlap: overlap بین chunkها
            base_method: روش پایه برای استخراج
            max_entities: حداکثر تعداد موجودیت‌ها
            max_relationships: حداکثر تعداد روابط
            **kwargs: پارامترهای اضافی
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not text or not text.strip():
            raise ValueError("متن ورودی نمی‌تواند خالی باشد")
        
        # تقسیم متن به chunks با overlap
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]
            
            # اضافه کردن context از chunk قبلی
            if start > 0 and overlap > 0:
                overlap_start = max(0, start - overlap)
                context = text[overlap_start:start]
                chunk_text = context + " " + chunk_text
            
            chunks.append({
                "text": chunk_text,
                "start": start,
                "end": end
            })
            
            start = end - overlap if overlap > 0 else end
        
        # پردازش incremental
        all_entities = []
        all_relationships = []
        entity_map = {}  # name -> entity_id
        relationship_set = set()
        
        for i, chunk in enumerate(chunks):
            logging.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            try:
                # استخراج از chunk
                chunk_result = self.extract(
                    chunk["text"],
                    method=base_method,
                    max_entities=max_entities,
                    max_relationships=max_relationships,
                    **kwargs
                )
                
                chunk_entities = chunk_result.get("entities", [])
                chunk_relationships = chunk_result.get("relationships", [])
                
                # ادغام موجودیت‌ها
                for entity in chunk_entities:
                    entity_name = entity.get("name", "").strip().lower()
                    
                    if entity_name in entity_map:
                        # موجودیت تکراری - merge attributes
                        entity_id = entity_map[entity_name]
                        for e in all_entities:
                            if e["id"] == entity_id:
                                # Merge attributes
                                existing_attrs = e.get("attributes", {})
                                new_attrs = entity.get("attributes", {})
                                
                                # Track chunk appearances
                                if "chunks" not in existing_attrs:
                                    existing_attrs["chunks"] = []
                                existing_attrs["chunks"].append(i)
                                existing_attrs.update(new_attrs)
                                break
                    else:
                        # موجودیت جدید
                        entity_id = f"ENT_{len(all_entities)}"
                        entity_map[entity_name] = entity_id
                        entity["id"] = entity_id
                        entity["attributes"]["chunks"] = [i]
                        all_entities.append(entity)
                
                # ادغام روابط
                for rel in chunk_relationships:
                    source_name = rel.get("source", "").strip().lower()
                    target_name = rel.get("target", "").strip().lower()
                    relation = rel.get("relation", "").strip()
                    
                    # Map to entity IDs
                    source_id = entity_map.get(source_name)
                    target_id = entity_map.get(target_name)
                    
                    if source_id and target_id:
                        rel_key = (source_id, target_id, relation)
                        if rel_key not in relationship_set:
                            relationship_set.add(rel_key)
                            rel["source"] = source_id
                            rel["target"] = target_id
                            rel["attributes"]["chunk"] = i
                            all_relationships.append(rel)
                        else:
                            # Update confidence/frequency
                            for r in all_relationships:
                                if (r["source"], r["target"], r["relation"]) == rel_key:
                                    r["attributes"]["frequency"] = r["attributes"].get("frequency", 1) + 1
                                    break
                
            except Exception as e:
                logging.warning(f"Error processing chunk {i+1}: {e}")
                continue
        
        return {
            "entities": all_entities[:max_entities],
            "relationships": all_relationships[:max_relationships],
            "method": f"incremental_{base_method}",
            "stats": {
                "num_entities": len(all_entities),
                "num_relationships": len(all_relationships),
                "num_chunks": len(chunks)
            }
        }


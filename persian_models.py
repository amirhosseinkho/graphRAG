# -*- coding: utf-8 -*-
"""
Persian Models - مدل‌های HuggingFace فارسی برای NER و Relation Extraction
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import torch

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")


class PersianNERModel:
    """مدل NER فارسی با استفاده از ParsBERT"""
    
    def __init__(self, model_name: str = "HooshvareLab/bert-fa-base-uncased"):
        """
        Initialize Persian NER model
        
        Args:
            model_name: نام مدل HuggingFace
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self._load_model()
    
    def _load_model(self):
        """بارگذاری مدل"""
        try:
            # Try to load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # For NER, we'll use a pipeline if available
            # Note: ParsBERT doesn't have a pre-trained NER model, so we'll use it for embeddings
            # and apply a simple NER approach or fine-tune
            try:
                # Try to create NER pipeline (may need fine-tuned model)
                self.ner_pipeline = pipeline(
                    "token-classification",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    aggregation_strategy="simple"
                )
            except Exception:
                # Fallback: use model for embeddings only
                logging.warning(f"Could not load NER pipeline for {self.model_name}. Using embeddings only.")
                self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        except Exception as e:
            logging.error(f"Failed to load Persian NER model: {e}")
            raise
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        استخراج موجودیت‌ها از متن فارسی
        
        Args:
            text: متن ورودی فارسی
            
        Returns:
            لیست موجودیت‌های استخراج شده
        """
        if not self.ner_pipeline:
            # Fallback: simple rule-based extraction for Persian
            return self._simple_persian_ner(text)
        
        try:
            results = self.ner_pipeline(text)
            entities = []
            for result in results:
                entities.append({
                    "text": result.get("word", ""),
                    "label": result.get("entity_group", "UNKNOWN"),
                    "score": result.get("score", 0.0),
                    "start": result.get("start", 0),
                    "end": result.get("end", 0)
                })
            return entities
        except Exception as e:
            logging.warning(f"NER pipeline failed: {e}. Using fallback.")
            return self._simple_persian_ner(text)
    
    def _simple_persian_ner(self, text: str) -> List[Dict[str, Any]]:
        """استخراج ساده موجودیت‌ها با الگوهای فارسی"""
        entities = []
        
        # Patterns for Persian biomedical entities
        # Gene names (usually uppercase Latin letters)
        gene_pattern = r'\b[A-Z][A-Z0-9]+\b'
        genes = re.findall(gene_pattern, text)
        for gene in set(genes):
            entities.append({
                "text": gene,
                "label": "GENE",
                "score": 0.7,
                "start": text.find(gene),
                "end": text.find(gene) + len(gene)
            })
        
        # Persian disease keywords
        disease_keywords = ['سرطان', 'بیماری', 'سندرم', 'اختلال', 'تومور', 'کارسینوما']
        for keyword in disease_keywords:
            if keyword in text:
                start = text.find(keyword)
                entities.append({
                    "text": keyword,
                    "label": "DISEASE",
                    "score": 0.8,
                    "start": start,
                    "end": start + len(keyword)
                })
        
        return entities


class PersianRelationExtractor:
    """استخراج روابط با استفاده از mT5"""
    
    def __init__(self, model_name: str = "persiannlp/mt5-base-parsinlu"):
        """
        Initialize Persian Relation Extractor
        
        Args:
            model_name: نام مدل HuggingFace
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """بارگذاری مدل"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            logging.info(f"Loaded Persian relation extractor: {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to load Persian relation extractor: {e}")
            raise
    
    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        استخراج روابط بین موجودیت‌ها
        
        Args:
            text: متن ورودی
            entities: لیست موجودیت‌های استخراج شده
            
        Returns:
            لیست روابط استخراج شده
        """
        if not entities or len(entities) < 2:
            return []
        
        relations = []
        
        # Create pairs of entities
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities[i+1:], start=i+1):
                # Extract context around entities
                ent1_text = ent1.get("text", "")
                ent2_text = ent2.get("text", "")
                
                # Find sentences containing both entities
                sentences = self._extract_relevant_sentences(text, ent1_text, ent2_text)
                
                for sentence in sentences:
                    # Use mT5 to extract relation
                    relation = self._extract_relation_with_mt5(sentence, ent1_text, ent2_text)
                    if relation:
                        relations.append({
                            "source": ent1.get("text", ""),
                            "target": ent2.get("text", ""),
                            "relation": relation,
                            "sentence": sentence,
                            "confidence": 0.7
                        })
        
        return relations
    
    def _extract_relevant_sentences(self, text: str, ent1: str, ent2: str) -> List[str]:
        """استخراج جملات مرتبط با دو موجودیت"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?؟]\s+', text)
        relevant = []
        
        for sent in sentences:
            if ent1 in sent and ent2 in sent:
                relevant.append(sent.strip())
        
        return relevant
    
    def _extract_relation_with_mt5(self, sentence: str, ent1: str, ent2: str) -> Optional[str]:
        """استخراج رابطه با استفاده از mT5"""
        try:
            # Create prompt for relation extraction
            prompt = f"رابطه بین {ent1} و {ent2} در جمله زیر چیست؟\n{sentence}"
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=50,
                    num_beams=2,
                    early_stopping=True
                )
            
            # Decode
            relation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return relation.strip()
        except Exception as e:
            logging.warning(f"mT5 relation extraction failed: {e}")
            return None

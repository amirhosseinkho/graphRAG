# -*- coding: utf-8 -*-
"""
BERT Relation Extractor - استخراج روابط با استفاده از BERT
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")


class BERTRelationExtractor:
    """استخراج روابط با استفاده از BERT"""
    
    def __init__(self, model_name: str = "bert-base-uncased", language: str = "en"):
        """
        Initialize BERT relation extractor
        
        Args:
            model_name: نام مدل BERT
            language: زبان (en/fa)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        self.model_name = model_name
        self.language = language
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        # Common relation types
        self.relation_types = [
            "REGULATES", "INTERACTS_WITH", "TREATS", "ASSOCIATES_WITH",
            "PARTICIPATES_IN", "CAUSES", "PREVENTS", "INHIBITS", "ACTIVATES"
        ]
    
    def _load_model(self):
        """بارگذاری مدل"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Use base model for embeddings, then classify relations
            self.model = AutoModel.from_pretrained(self.model_name)
            logging.info(f"Loaded BERT model for relation extraction: {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to load BERT model: {e}")
            raise
    
    def extract_relations(self, 
                         text: str, 
                         entities: List[Dict[str, Any]],
                         max_pairs: int = 100) -> List[Dict[str, Any]]:
        """
        استخراج روابط بین موجودیت‌ها
        
        Args:
            text: متن ورودی
            entities: لیست موجودیت‌ها
            max_pairs: حداکثر تعداد جفت‌های موجودیت برای بررسی
            
        Returns:
            لیست روابط استخراج شده
        """
        if not entities or len(entities) < 2:
            return []
        
        relations = []
        
        # Create entity pairs
        pairs = []
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities[i+1:], start=i+1):
                if len(pairs) >= max_pairs:
                    break
                pairs.append((ent1, ent2))
        
        # Extract relations for each pair
        for ent1, ent2 in pairs:
            relation = self._extract_relation_for_pair(text, ent1, ent2)
            if relation:
                relations.append(relation)
        
        return relations
    
    def _extract_relation_for_pair(self, 
                                   text: str, 
                                   ent1: Dict[str, Any], 
                                   ent2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """استخراج رابطه برای یک جفت موجودیت"""
        ent1_text = ent1.get("text", "")
        ent2_text = ent2.get("text", "")
        
        # Find context sentence
        sentence = self._find_context_sentence(text, ent1_text, ent2_text)
        if not sentence:
            return None
        
        # Create input for BERT
        # Format: [CLS] sentence [SEP] entity1 [SEP] entity2 [SEP]
        input_text = f"{sentence} [SEP] {ent1_text} [SEP] {ent2_text}"
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :]
            
            # Classify relation type (simplified - in practice, use a fine-tuned classifier)
            relation_type = self._classify_relation(cls_embedding, sentence, ent1_text, ent2_text)
            
            if relation_type:
                return {
                    "source": ent1_text,
                    "target": ent2_text,
                    "relation": relation_type,
                    "metaedge": self._map_relation_to_metaedge(relation_type),
                    "sentence": sentence,
                    "confidence": 0.7,
                    "attributes": {
                        "extraction_method": "bert",
                        "model": self.model_name
                    }
                }
        except Exception as e:
            logging.warning(f"BERT relation extraction failed: {e}")
        
        return None
    
    def _find_context_sentence(self, text: str, ent1: str, ent2: str) -> Optional[str]:
        """یافتن جمله حاوی هر دو موجودیت"""
        # Simple sentence splitting
        import re
        sentences = re.split(r'[.!?؟]\s+', text)
        
        for sentence in sentences:
            if ent1 in sentence and ent2 in sentence:
                return sentence.strip()
        
        return None
    
    def _classify_relation(self, 
                          embedding: torch.Tensor,
                          sentence: str,
                          ent1: str,
                          ent2: str) -> Optional[str]:
        """طبقه‌بندی نوع رابطه (simplified)"""
        # This is a simplified version. In practice, use a fine-tuned classifier
        sentence_lower = sentence.lower()
        
        # Pattern matching for relation types
        if any(word in sentence_lower for word in ["regulates", "controls", "تنظیم", "کنترل"]):
            return "REGULATES"
        elif any(word in sentence_lower for word in ["interacts", "binds", "تعامل", "اتصال"]):
            return "INTERACTS_WITH"
        elif any(word in sentence_lower for word in ["treats", "cures", "درمان", "معالجه"]):
            return "TREATS"
        elif any(word in sentence_lower for word in ["causes", "leads", "سبب", "موجب"]):
            return "CAUSES"
        elif any(word in sentence_lower for word in ["prevents", "inhibits", "جلوگیری", "مهار"]):
            return "PREVENTS"
        elif any(word in sentence_lower for word in ["activates", "activates", "فعال", "فعالسازی"]):
            return "ACTIVATES"
        
        return "ASSOCIATES_WITH"  # Default
    
    def _map_relation_to_metaedge(self, relation_type: str) -> str:
        """نگاشت نوع رابطه به metaedge Hetionet"""
        mapping = {
            "REGULATES": "Gr>G",
            "INTERACTS_WITH": "GiG",
            "TREATS": "CtD",
            "CAUSES": "DaG",
            "PREVENTS": "CtD",
            "INHIBITS": "GiG",
            "ACTIVATES": "Gr>G",
            "ASSOCIATES_WITH": "DaG",
            "PARTICIPATES_IN": "GpBP"
        }
        return mapping.get(relation_type, "GiG")

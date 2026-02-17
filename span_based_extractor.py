# -*- coding: utf-8 -*-
"""
Span-Based Extractor - استخراج مبتنی بر Span با مدل‌های BioBERT و SciBERT
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import torch

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")


class SpanBasedExtractor:
    """استخراج مبتنی بر Span با مدل‌های زیست‌پزشکی"""
    
    def __init__(self, model_name: Optional[str] = None, language: str = "en"):
        """
        Initialize span-based extractor
        
        Args:
            model_name: نام مدل (None برای انتخاب خودکار)
            language: زبان متن (en/fa)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        self.language = language
        
        # Select model based on language
        if model_name is None:
            if language == "fa":
                model_name = "HooshvareLab/bert-fa-base-uncased"
            else:
                # Try BioBERT first, fallback to SciBERT
                model_name = "dmis-lab/biobert-v1.1"
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self._load_model()
    
    def _load_model(self):
        """بارگذاری مدل"""
        try:
            # Try to load as NER pipeline
            try:
                self.ner_pipeline = pipeline(
                    "token-classification",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    aggregation_strategy="simple"
                )
                logging.info(f"Loaded NER pipeline: {self.model_name}")
            except Exception:
                # Fallback: load tokenizer and model separately
                logging.warning(f"Could not load NER pipeline for {self.model_name}. Loading model directly.")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                try:
                    self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
                except Exception:
                    # Use base model for embeddings
                    from transformers import AutoModel
                    self.model = AutoModel.from_pretrained(self.model_name)
                    logging.info(f"Loaded base model for embeddings: {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to load span-based model: {e}")
            raise
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        استخراج موجودیت‌ها با استفاده از span-based approach
        
        Args:
            text: متن ورودی
            
        Returns:
            لیست موجودیت‌های استخراج شده با spans
        """
        if self.ner_pipeline:
            return self._extract_with_pipeline(text)
        else:
            return self._extract_with_model(text)
    
    def _extract_with_pipeline(self, text: str) -> List[Dict[str, Any]]:
        """استخراج با استفاده از pipeline"""
        try:
            results = self.ner_pipeline(text)
            entities = []
            for result in results:
                entities.append({
                    "text": result.get("word", ""),
                    "label": result.get("entity_group", "UNKNOWN"),
                    "score": result.get("score", 0.0),
                    "start": result.get("start", 0),
                    "end": result.get("end", 0),
                    "span": (result.get("start", 0), result.get("end", 0))
                })
            return entities
        except Exception as e:
            logging.warning(f"Pipeline extraction failed: {e}")
            return []
    
    def _extract_with_model(self, text: str) -> List[Dict[str, Any]]:
        """استخراج با استفاده از مدل مستقیم"""
        if not self.tokenizer or not self.model:
            return []
        
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_labels = torch.argmax(predictions, dim=-1)
            
            # Convert to entities
            entities = []
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            current_entity = None
            for i, (token, label_id) in enumerate(zip(tokens, predicted_labels[0])):
                label = self.model.config.id2label.get(label_id.item(), "O")
                
                if label.startswith("B-"):
                    # Start of new entity
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "text": token.replace("##", ""),
                        "label": label[2:],
                        "score": predictions[0][i][label_id].item(),
                        "start": i,
                        "end": i + 1,
                        "span": (i, i + 1)
                    }
                elif label.startswith("I-") and current_entity:
                    # Continue entity
                    current_entity["text"] += token.replace("##", "")
                    current_entity["end"] = i + 1
                    current_entity["span"] = (current_entity["start"], i + 1)
                else:
                    # End of entity
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            
            if current_entity:
                entities.append(current_entity)
            
            return entities
        except Exception as e:
            logging.warning(f"Model extraction failed: {e}")
            return []


class BioBERTExtractor(SpanBasedExtractor):
    """استخراج با BioBERT"""
    
    def __init__(self):
        super().__init__(model_name="dmis-lab/biobert-v1.1", language="en")


class SciBERTExtractor(SpanBasedExtractor):
    """استخراج با SciBERT"""
    
    def __init__(self):
        super().__init__(model_name="allenai/scibert_scivocab_uncased", language="en")

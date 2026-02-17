# -*- coding: utf-8 -*-
"""
Modular Pipeline - Pipeline ماژولار قابل سفارشی‌سازی
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

# Import modules
try:
    from persian_normalizer import PersianNormalizer, detect_language
    from span_based_extractor import SpanBasedExtractor
    from bert_relation_extractor import BERTRelationExtractor
    from coreference_resolver import CoreferenceResolver
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    logging.warning(f"Some modules not available: {e}")


class PipelineStage(Enum):
    """مراحل pipeline"""
    NORMALIZATION = "normalization"
    NER = "ner"
    RELATION_EXTRACTION = "relation_extraction"
    COREFERENCE_RESOLUTION = "coreference_resolution"


class ModularExtractionPipeline:
    """Pipeline ماژولار برای استخراج"""
    
    def __init__(self,
                 language: str = "auto",
                 enable_normalization: bool = True,
                 enable_ner: bool = True,
                 enable_relation_extraction: bool = True,
                 enable_coreference: bool = False,
                 ner_model: Optional[str] = None,
                 relation_model: Optional[str] = None):
        """
        Initialize modular pipeline
        
        Args:
            language: زبان (auto/fa/en)
            enable_normalization: فعال کردن نرمال‌سازی
            enable_ner: فعال کردن NER
            enable_relation_extraction: فعال کردن استخراج روابط
            enable_coreference: فعال کردن coreference resolution
            ner_model: مدل NER (None برای انتخاب خودکار)
            relation_model: مدل Relation Extraction (None برای انتخاب خودکار)
        """
        self.language = language
        self.enable_normalization = enable_normalization
        self.enable_ner = enable_ner
        self.enable_relation_extraction = enable_relation_extraction
        self.enable_coreference = enable_coreference
        
        # Initialize components
        self.normalizer = None
        if enable_normalization and MODULES_AVAILABLE:
            try:
                self.normalizer = PersianNormalizer(enable_spell_check=False)
            except Exception as e:
                logging.warning(f"Failed to initialize normalizer: {e}")
        
        self.ner_extractor = None
        if enable_ner and MODULES_AVAILABLE:
            try:
                if language == "fa" or language == "auto":
                    from persian_models import PersianNERModel
                    self.ner_extractor = PersianNERModel()
                else:
                    self.ner_extractor = SpanBasedExtractor(model_name=ner_model, language=language)
            except Exception as e:
                logging.warning(f"Failed to initialize NER extractor: {e}")
        
        self.relation_extractor = None
        if enable_relation_extraction and MODULES_AVAILABLE:
            try:
                if language == "fa" or language == "auto":
                    from persian_models import PersianRelationExtractor
                    self.relation_extractor = PersianRelationExtractor()
                else:
                    self.relation_extractor = BERTRelationExtractor(
                        model_name=relation_model or "bert-base-uncased",
                        language=language
                    )
            except Exception as e:
                logging.warning(f"Failed to initialize relation extractor: {e}")
        
        self.coreference_resolver = None
        if enable_coreference and MODULES_AVAILABLE:
            try:
                self.coreference_resolver = CoreferenceResolver(language=language)
            except Exception as e:
                logging.warning(f"Failed to initialize coreference resolver: {e}")
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        پردازش متن با pipeline
        
        Args:
            text: متن ورودی
            
        Returns:
            Dictionary حاوی موجودیت‌ها و روابط استخراج شده
        """
        # Detect language if auto
        detected_language = self.language
        if self.language == "auto" and MODULES_AVAILABLE:
            detected_language = detect_language(text)
        
        # Stage 1: Normalization
        normalized_text = text
        if self.enable_normalization and self.normalizer:
            try:
                normalized_text = self.normalizer.normalize(text)
            except Exception as e:
                logging.warning(f"Normalization failed: {e}")
        
        # Stage 2: NER
        entities = []
        if self.enable_ner and self.ner_extractor:
            try:
                entities = self.ner_extractor.extract_entities(normalized_text)
            except Exception as e:
                logging.warning(f"NER extraction failed: {e}")
        
        # Stage 3: Relation Extraction
        relationships = []
        if self.enable_relation_extraction and self.relation_extractor and entities:
            try:
                if hasattr(self.relation_extractor, 'extract_relations'):
                    relationships = self.relation_extractor.extract_relations(normalized_text, entities)
                else:
                    # Fallback for different API
                    relationships = []
            except Exception as e:
                logging.warning(f"Relation extraction failed: {e}")
        
        # Stage 4: Coreference Resolution
        if self.enable_coreference and self.coreference_resolver:
            try:
                reference_map = self.coreference_resolver.resolve(normalized_text, entities)
                entities = self.coreference_resolver.merge_entities(entities, reference_map)
            except Exception as e:
                logging.warning(f"Coreference resolution failed: {e}")
        
        return {
            "entities": entities,
            "relationships": relationships,
            "language": detected_language,
            "normalized_text": normalized_text,
            "stats": {
                "num_entities": len(entities),
                "num_relationships": len(relationships)
            }
    }

# -*- coding: utf-8 -*-
"""
Coreference Resolver - حل ارجاعات برای ادغام موجودیت‌های مشابه
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import re

# Try to import neuralcoref for English
try:
    import neuralcoref
    NEURALCOREF_AVAILABLE = True
except ImportError:
    NEURALCOREF_AVAILABLE = False
    logging.warning("neuralcoref not available for English. Install with: pip install neuralcoref")

# Try to import spacy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class CoreferenceResolver:
    """حل ارجاعات برای ادغام موجودیت‌های مشابه"""
    
    def __init__(self, language: str = "en", spacy_model=None):
        """
        Initialize coreference resolver
        
        Args:
            language: زبان (en/fa)
            spacy_model: مدل spaCy (اختیاری)
        """
        self.language = language
        self.nlp = spacy_model
        
        # Persian coreference patterns
        self.persian_references = {
            "این": "this",
            "آن": "that",
            "این ژن": "this gene",
            "آن پروتئین": "that protein",
            "این بیماری": "this disease",
            "آن دارو": "that drug"
        }
        
        # English coreference patterns
        self.english_references = {
            "this": True,
            "that": True,
            "these": True,
            "those": True,
            "it": True,
            "they": True,
            "he": True,
            "she": True
        }
    
    def resolve(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        حل ارجاعات و بازگشت نگاشت موجودیت‌های ارجاعی به موجودیت‌های اصلی
        
        Args:
            text: متن ورودی
            entities: لیست موجودیت‌های استخراج شده
            
        Returns:
            Dictionary mapping reference entities to canonical entities
        """
        if self.language == "en" and NEURALCOREF_AVAILABLE and self.nlp:
            return self._resolve_with_neuralcoref(text, entities)
        else:
            return self._resolve_with_patterns(text, entities)
    
    def _resolve_with_neuralcoref(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """حل ارجاعات با neuralcoref برای انگلیسی"""
        try:
            # Add neuralcoref to pipeline if not already added
            if "neuralcoref" not in self.nlp.pipe_names:
                neuralcoref.add_to_pipe(self.nlp)
            
            doc = self.nlp(text)
            
            # Get coreference clusters
            clusters = doc._.coref_clusters
            
            # Create mapping
            reference_map = {}
            
            for cluster in clusters:
                # Get main mention (usually the first one)
                main_mention = cluster.main.text
                
                # Map all mentions in cluster to main mention
                for mention in cluster.mentions:
                    if mention.text != main_mention:
                        reference_map[mention.text] = main_mention
            
            return reference_map
        except Exception as e:
            logging.warning(f"neuralcoref resolution failed: {e}")
            return self._resolve_with_patterns(text, entities)
    
    def _resolve_with_patterns(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """حل ارجاعات با الگوهای ساده"""
        reference_map = {}
        
        # Find references in text
        if self.language == "fa":
            # Persian patterns
            patterns = [
                (r'این\s+(\w+)', "this"),
                (r'آن\s+(\w+)', "that"),
                (r'این\s+ژن', "this gene"),
                (r'آن\s+پروتئین', "that protein"),
            ]
        else:
            # English patterns
            patterns = [
                (r'this\s+(\w+)', "this"),
                (r'that\s+(\w+)', "that"),
                (r'the\s+(\w+)', "the"),
            ]
        
        # Find references and map to nearby entities
        for pattern, ref_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ref_text = match.group(0)
                ref_start = match.start()
                
                # Find nearest entity
                nearest_entity = self._find_nearest_entity(ref_start, entities, text)
                if nearest_entity:
                    reference_map[ref_text] = nearest_entity.get("text", "")
        
        return reference_map
    
    def _find_nearest_entity(self, position: int, entities: List[Dict[str, Any]], text: str) -> Optional[Dict[str, Any]]:
        """یافتن نزدیک‌ترین موجودیت به موقعیت"""
        if not entities:
            return None
        
        min_distance = float('inf')
        nearest = None
        
        for entity in entities:
            entity_start = entity.get("start", 0)
            entity_end = entity.get("end", len(text))
            
            # Calculate distance
            if position < entity_start:
                distance = entity_start - position
            elif position > entity_end:
                distance = position - entity_end
            else:
                distance = 0  # Inside entity
            
            if distance < min_distance:
                min_distance = distance
                nearest = entity
        
        # Only return if within reasonable distance (e.g., 200 characters)
        if min_distance < 200:
            return nearest
        
        return None
    
    def merge_entities(self, entities: List[Dict[str, Any]], reference_map: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        ادغام موجودیت‌های ارجاعی با موجودیت‌های اصلی
        
        Args:
            entities: لیست موجودیت‌ها
            reference_map: نگاشت ارجاعات
            
        Returns:
            لیست موجودیت‌های ادغام شده
        """
        if not reference_map:
            return entities
        
        # Create entity map by text
        entity_map = {ent.get("text", ""): ent for ent in entities}
        
        # Merge references
        merged_entities = []
        processed_references = set()
        
        for entity in entities:
            entity_text = entity.get("text", "")
            
            # Check if this entity is a reference
            if entity_text in reference_map:
                canonical_text = reference_map[entity_text]
                
                # Merge with canonical entity
                if canonical_text in entity_map:
                    canonical_entity = entity_map[canonical_text]
                    # Update canonical entity with reference info
                    if "references" not in canonical_entity:
                        canonical_entity["references"] = []
                    canonical_entity["references"].append(entity_text)
                    processed_references.add(entity_text)
                    continue
            
            # Add non-reference entities
            if entity_text not in processed_references:
                merged_entities.append(entity)
        
        return merged_entities

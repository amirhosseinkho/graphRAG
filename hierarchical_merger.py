# -*- coding: utf-8 -*-
"""
Hierarchical Merger - ادغام سلسله‌مراتبی نتایج chunkها
"""

import logging
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict


class HierarchicalMerger:
    """ادغام سلسله‌مراتبی نتایج chunkها"""
    
    def __init__(self, 
                 weight_by_frequency: bool = True,
                 min_confidence: float = 0.5,
                 similarity_threshold: float = 0.8):
        """
        Initialize hierarchical merger
        
        Args:
            weight_by_frequency: وزن‌دهی بر اساس تکرار
            min_confidence: حداقل confidence
            similarity_threshold: آستانه شباهت برای ادغام موجودیت‌های مشابه
        """
        self.weight_by_frequency = weight_by_frequency
        self.min_confidence = min_confidence
        self.similarity_threshold = similarity_threshold
    
    def merge_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        ادغام نتایج چندین chunk
        
        Args:
            chunk_results: لیست نتایج استخراج از chunkها
            
        Returns:
            Dictionary حاوی موجودیت‌ها و روابط ادغام شده
        """
        if not chunk_results:
            return {"entities": [], "relationships": []}
        
        # Merge entities
        merged_entities = self._merge_entities([r.get("entities", []) for r in chunk_results])
        
        # Merge relationships
        merged_relationships = self._merge_relationships([r.get("relationships", []) for r in chunk_results])
        
        return {
            "entities": merged_entities,
            "relationships": merged_relationships,
            "stats": {
                "num_chunks": len(chunk_results),
                "num_entities": len(merged_entities),
                "num_relationships": len(merged_relationships)
            }
        }
    
    def _merge_entities(self, entity_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """ادغام موجودیت‌ها از چندین chunk"""
        # Count frequency and collect entities
        entity_map = {}  # text -> entity with metadata
        entity_frequency = defaultdict(int)
        entity_confidence_sum = defaultdict(float)
        
        for entity_list in entity_lists:
            for entity in entity_list:
                entity_text = entity.get("text", "") or entity.get("name", "")
                if not entity_text:
                    continue
                
                # Normalize text for comparison
                normalized_text = self._normalize_text(entity_text)
                
                # Update frequency
                entity_frequency[normalized_text] += 1
                
                # Sum confidence scores
                confidence = entity.get("score", 0.5) or entity.get("confidence", 0.5)
                entity_confidence_sum[normalized_text] += confidence
                
                # Store entity (keep the one with highest confidence)
                if normalized_text not in entity_map:
                    entity_map[normalized_text] = entity.copy()
                else:
                    existing_conf = entity_map[normalized_text].get("score", 0.5) or entity_map[normalized_text].get("confidence", 0.5)
                    if confidence > existing_conf:
                        entity_map[normalized_text] = entity.copy()
        
        # Create merged entities with weights
        merged = []
        for normalized_text, entity in entity_map.items():
            frequency = entity_frequency[normalized_text]
            avg_confidence = entity_confidence_sum[normalized_text] / frequency if frequency > 0 else 0.5
            
            # Filter by minimum confidence
            if avg_confidence < self.min_confidence:
                continue
            
            # Add frequency and weight
            merged_entity = entity.copy()
            merged_entity["frequency"] = frequency
            merged_entity["weight"] = frequency if self.weight_by_frequency else 1.0
            merged_entity["confidence"] = avg_confidence
            merged_entity["score"] = avg_confidence  # For compatibility
            
            merged.append(merged_entity)
        
        # Sort by weight/frequency
        merged.sort(key=lambda x: x.get("weight", 0) * x.get("confidence", 0), reverse=True)
        
        return merged
    
    def _merge_relationships(self, relationship_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """ادغام روابط از چندین chunk"""
        # Create relationship key: (source, target, relation_type)
        rel_map = {}
        rel_frequency = defaultdict(int)
        rel_confidence_sum = defaultdict(float)
        
        for rel_list in relationship_lists:
            for rel in rel_list:
                source = rel.get("source", "")
                target = rel.get("target", "")
                relation = rel.get("relation", "") or rel.get("metaedge", "")
                
                if not source or not target:
                    continue
                
                # Normalize for comparison
                normalized_source = self._normalize_text(source)
                normalized_target = self._normalize_text(target)
                normalized_relation = self._normalize_text(relation)
                
                key = (normalized_source, normalized_target, normalized_relation)
                
                # Update frequency
                rel_frequency[key] += 1
                
                # Sum confidence
                confidence = rel.get("confidence", 0.5) or rel.get("score", 0.5)
                rel_confidence_sum[key] += confidence
                
                # Store relationship
                if key not in rel_map:
                    rel_map[key] = rel.copy()
                else:
                    # Update if higher confidence
                    existing_conf = rel_map[key].get("confidence", 0.5) or rel_map[key].get("score", 0.5)
                    if confidence > existing_conf:
                        rel_map[key] = rel.copy()
        
        # Create merged relationships
        merged = []
        for key, rel in rel_map.items():
            frequency = rel_frequency[key]
            avg_confidence = rel_confidence_sum[key] / frequency if frequency > 0 else 0.5
            
            # Filter by minimum confidence
            if avg_confidence < self.min_confidence:
                continue
            
            # Add frequency and weight
            merged_rel = rel.copy()
            merged_rel["frequency"] = frequency
            merged_rel["weight"] = frequency if self.weight_by_frequency else 1.0
            merged_rel["confidence"] = avg_confidence
            
            # Update attributes
            if "attributes" not in merged_rel:
                merged_rel["attributes"] = {}
            merged_rel["attributes"]["frequency"] = frequency
            merged_rel["attributes"]["weight"] = merged_rel["weight"]
            
            merged.append(merged_rel)
        
        # Sort by weight
        merged.sort(key=lambda x: x.get("weight", 0) * x.get("confidence", 0), reverse=True)
        
        return merged
    
    def _normalize_text(self, text: str) -> str:
        """نرمال‌سازی متن برای مقایسه"""
        if not text:
            return ""
        
        # Lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _similarity(self, text1: str, text2: str) -> float:
        """محاسبه شباهت بین دو متن (Jaccard similarity)"""
        words1 = set(self._normalize_text(text1).split())
        words2 = set(self._normalize_text(text2).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

# -*- coding: utf-8 -*-
"""
Smart Chunker - تقسیم هوشمند متن برای پردازش متن‌های طولانی
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# Try to import hazm for Persian sentence tokenization
try:
    from hazm import sent_tokenize, word_tokenize
    HAZM_AVAILABLE = True
except ImportError:
    HAZM_AVAILABLE = False

# Try to import spacy for sentence segmentation
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class ChunkingStrategy(Enum):
    """استراتژی‌های chunking"""
    SENTENCE = "sentence"  # تقسیم بر اساس جملات
    PARAGRAPH = "paragraph"  # تقسیم بر اساس پاراگراف
    SMART = "smart"  # ترکیب هوشمند
    SLIDING_WINDOW = "sliding_window"  # Sliding window با overlap


class SmartChunker:
    """کلاس برای تقسیم هوشمند متن"""
    
    def __init__(self, 
                 strategy: ChunkingStrategy = ChunkingStrategy.SMART,
                 max_tokens: int = 512,
                 overlap_ratio: float = 0.2,
                 language: str = "auto"):
        """
        Initialize smart chunker
        
        Args:
            strategy: استراتژی chunking
            max_tokens: حداکثر تعداد توکن در هر chunk
            overlap_ratio: نسبت overlap در sliding window (0.0 تا 1.0)
            language: زبان متن (auto/fa/en)
        """
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio
        self.language = language
        
        # Initialize spaCy for sentence segmentation if available
        self.nlp = None
        if SPACY_AVAILABLE and language != "fa":
            try:
                # Try to load appropriate model
                if language == "en":
                    self.nlp = spacy.load("en_core_web_sm")
                else:
                    # Try English as default
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                    except:
                        pass
            except Exception as e:
                logging.warning(f"Failed to load spaCy model: {e}")
    
    def _detect_language(self, text: str) -> str:
        """تشخیص زبان متن"""
        if self.language != "auto":
            return self.language
        
        # Simple detection based on Persian characters
        persian_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = persian_chars + latin_chars
        
        if total_chars == 0:
            return "en"
        
        if persian_chars / total_chars > 0.5:
            return "fa"
        return "en"
    
    def _split_sentences(self, text: str, lang: str) -> List[str]:
        """تقسیم متن به جملات"""
        if lang == "fa" and HAZM_AVAILABLE:
            try:
                return sent_tokenize(text)
            except Exception as e:
                logging.warning(f"hazm sentence tokenization failed: {e}")
        
        # Fallback: use regex or spaCy
        if self.nlp:
            try:
                doc = self.nlp(text)
                return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except Exception as e:
                logging.warning(f"spaCy sentence segmentation failed: {e}")
        
        # Simple regex-based sentence splitting
        # Persian sentence endings: . ! ? ؟ . 
        if lang == "fa":
            sentences = re.split(r'[.!?؟]\s+', text)
        else:
            sentences = re.split(r'[.!?]\s+', text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """تقسیم متن به پاراگراف‌ها"""
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _estimate_tokens(self, text: str) -> int:
        """تخمین تعداد توکن (تقریبی: 1 token ≈ 0.75 word)"""
        words = len(text.split())
        return int(words * 1.33)  # Rough estimate
    
    def chunk_by_sentence(self, text: str) -> List[Dict[str, Any]]:
        """تقسیم بر اساس جملات"""
        lang = self._detect_language(text)
        sentences = self._split_sentences(text, lang)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)
            
            if current_tokens + sent_tokens > self.max_tokens and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start_char": text.find(chunk_text),
                    "end_char": text.find(chunk_text) + len(chunk_text),
                    "tokens": current_tokens,
                    "strategy": "sentence"
                })
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(sentence)
            current_tokens += sent_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "start_char": text.find(chunk_text),
                "end_char": text.find(chunk_text) + len(chunk_text),
                "tokens": current_tokens,
                "strategy": "sentence"
            })
        
        return chunks
    
    def chunk_by_paragraph(self, text: str) -> List[Dict[str, Any]]:
        """تقسیم بر اساس پاراگراف"""
        paragraphs = self._split_paragraphs(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            
            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start_char": text.find(chunk_text),
                    "end_char": text.find(chunk_text) + len(chunk_text),
                    "tokens": current_tokens,
                    "strategy": "paragraph"
                })
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(para)
            current_tokens += para_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "start_char": text.find(chunk_text),
                "end_char": text.find(chunk_text) + len(chunk_text),
                "tokens": current_tokens,
                "strategy": "paragraph"
            })
        
        return chunks
    
    def chunk_smart(self, text: str) -> List[Dict[str, Any]]:
        """تقسیم هوشمند: ترکیب پاراگراف و جمله"""
        lang = self._detect_language(text)
        paragraphs = self._split_paragraphs(text)
        
        chunks = []
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            
            if para_tokens <= self.max_tokens:
                # Paragraph fits in one chunk
                chunks.append({
                    "text": para,
                    "start_char": text.find(para),
                    "end_char": text.find(para) + len(para),
                    "tokens": para_tokens,
                    "strategy": "smart_paragraph"
                })
            else:
                # Split paragraph into sentences
                sentences = self._split_sentences(para, lang)
                current_chunk = []
                current_tokens = 0
                
                for sentence in sentences:
                    sent_tokens = self._estimate_tokens(sentence)
                    
                    if current_tokens + sent_tokens > self.max_tokens and current_chunk:
                        chunk_text = " ".join(current_chunk)
                        chunks.append({
                            "text": chunk_text,
                            "start_char": text.find(chunk_text, text.find(para)),
                            "end_char": text.find(chunk_text, text.find(para)) + len(chunk_text),
                            "tokens": current_tokens,
                            "strategy": "smart_sentence"
                        })
                        current_chunk = []
                        current_tokens = 0
                    
                    current_chunk.append(sentence)
                    current_tokens += sent_tokens
                
                # Add remaining chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "start_char": text.find(chunk_text, text.find(para)),
                        "end_char": text.find(chunk_text, text.find(para)) + len(chunk_text),
                        "tokens": current_tokens,
                        "strategy": "smart_sentence"
                    })
        
        return chunks
    
    def chunk_sliding_window(self, text: str) -> List[Dict[str, Any]]:
        """تقسیم با sliding window و overlap"""
        lang = self._detect_language(text)
        sentences = self._split_sentences(text, lang)
        
        if not sentences:
            return [{"text": text, "start_char": 0, "end_char": len(text), "tokens": self._estimate_tokens(text), "strategy": "sliding_window"}]
        
        chunks = []
        overlap_tokens = int(self.max_tokens * self.overlap_ratio)
        
        i = 0
        while i < len(sentences):
            current_chunk = []
            current_tokens = 0
            
            # Start from overlap point if not first chunk
            start_idx = i
            if i > 0 and chunks:
                # Include overlap from previous chunk
                overlap_sentences = []
                overlap_count = 0
                for j in range(len(chunks[-1]["sentences"]) - 1, -1, -1):
                    if overlap_count >= overlap_tokens:
                        break
                    overlap_sentences.insert(0, chunks[-1]["sentences"][j])
                    overlap_count += self._estimate_tokens(chunks[-1]["sentences"][j])
                
                current_chunk.extend(overlap_sentences)
                current_tokens = overlap_count
            
            # Add sentences until max_tokens
            while i < len(sentences) and current_tokens + self._estimate_tokens(sentences[i]) <= self.max_tokens:
                current_chunk.append(sentences[i])
                current_tokens += self._estimate_tokens(sentences[i])
                i += 1
            
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start_char": text.find(chunk_text),
                    "end_char": text.find(chunk_text) + len(chunk_text),
                    "tokens": current_tokens,
                    "strategy": "sliding_window",
                    "sentences": current_chunk.copy(),
                    "overlap": i > 0
                })
            
            # Prevent infinite loop
            if i == start_idx:
                i += 1
        
        return chunks
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        تقسیم متن به chunkها بر اساس استراتژی انتخاب شده
        
        Args:
            text: متن ورودی
            
        Returns:
            لیست chunkها با metadata
        """
        if not text or not text.strip():
            return []
        
        if self.strategy == ChunkingStrategy.SENTENCE:
            return self.chunk_by_sentence(text)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self.chunk_by_paragraph(text)
        elif self.strategy == ChunkingStrategy.SLIDING_WINDOW:
            return self.chunk_sliding_window(text)
        else:  # SMART
            return self.chunk_smart(text)


class SlidingWindowProcessor:
    """پردازشگر با sliding window برای حفظ context"""
    
    def __init__(self, chunker: SmartChunker):
        """
        Initialize sliding window processor
        
        Args:
            chunker: SmartChunker instance
        """
        self.chunker = chunker
    
    def process(self, text: str, extractor_func) -> List[Dict[str, Any]]:
        """
        پردازش متن با sliding window
        
        Args:
            text: متن ورودی
            extractor_func: تابع استخراج که روی هر chunk اجرا می‌شود
            
        Returns:
            لیست نتایج استخراج از تمام chunkها
        """
        chunks = self.chunker.chunk_sliding_window(text)
        results = []
        
        for i, chunk_data in enumerate(chunks):
            try:
                result = extractor_func(chunk_data["text"])
                result["chunk_index"] = i
                result["chunk_metadata"] = chunk_data
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing chunk {i}: {e}")
                continue
        
        return results

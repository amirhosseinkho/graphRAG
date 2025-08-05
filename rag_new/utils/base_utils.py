# -*- coding: utf-8 -*-
"""
Base Utils - توابع پایه RAG
"""

import re
import tiktoken
from typing import Union, List

def rmSpace(text: str) -> str:
    """حذف فاصله‌های اضافی از متن"""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip())

def get_float(value: Union[str, float, int]) -> float:
    """تبدیل مقدار به عدد اعشاری"""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """محاسبه تعداد توکن‌ها در متن"""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))
    except Exception:
        # در صورت عدم دسترسی به tiktoken، تخمین ساده
        return len(string.split()) * 1.3

def clean_text(text: str) -> str:
    """پاکسازی متن"""
    if not text:
        return ""
    
    # حذف کاراکترهای خاص
    text = re.sub(r'[^\w\s\-.,!?;:()]', '', text)
    
    # حذف فاصله‌های اضافی
    text = rmSpace(text)
    
    return text

def split_text(text: str, max_tokens: int = 1000) -> List[str]:
    """تقسیم متن به بخش‌های کوچکتر"""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for word in words:
        word_tokens = num_tokens_from_string(word)
        if current_tokens + word_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_tokens += word_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def calculate_similarity(text1: str, text2: str) -> float:
    """محاسبه شباهت بین دو متن"""
    if not text1 or not text2:
        return 0.0
    
    # تبدیل به مجموعه کلمات
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    # محاسبه Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """استخراج کلمات کلیدی از متن"""
    if not text:
        return []
    
    # حذف کلمات توقف
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    # تقسیم به کلمات
    words = re.findall(r'\b\w+\b', text.lower())
    
    # حذف کلمات توقف و کلمات کوتاه
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # شمارش فراوانی
    word_count = {}
    for word in keywords:
        word_count[word] = word_count.get(word, 0) + 1
    
    # مرتب‌سازی بر اساس فراوانی
    sorted_keywords = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, count in sorted_keywords[:max_keywords]]

def normalize_text(text: str) -> str:
    """نرمال‌سازی متن"""
    if not text:
        return ""
    
    # تبدیل به حروف کوچک
    text = text.lower()
    
    # حذف کاراکترهای خاص
    text = re.sub(r'[^\w\s]', '', text)
    
    # حذف فاصله‌های اضافی
    text = rmSpace(text)
    
    return text

def truncate_text(text: str, max_length: int = 1000) -> str:
    """کوتاه کردن متن"""
    if not text or len(text) <= max_length:
        return text
    
    # کوتاه کردن تا آخرین کلمه کامل
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        return truncated[:last_space] + "..."
    
    return truncated + "..."

def merge_texts(texts: List[str], separator: str = " ") -> str:
    """ادغام چندین متن"""
    if not texts:
        return ""
    
    return separator.join(text for text in texts if text)

def remove_duplicates(texts: List[str]) -> List[str]:
    """حذف متن‌های تکراری"""
    seen = set()
    unique_texts = []
    
    for text in texts:
        normalized = normalize_text(text)
        if normalized not in seen:
            seen.add(normalized)
            unique_texts.append(text)
    
    return unique_texts 
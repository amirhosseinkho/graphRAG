# -*- coding: utf-8 -*-
"""
RAG Settings - تنظیمات RAG
"""

import os
from typing import Dict, Any

# فیلدهای پیش‌فرض
TAG_FLD = "tag_fld"
PAGERANK_FLD = "pagerank_flt"

# تنظیمات پیش‌فرض
DEFAULT_SETTINGS = {
    "max_concurrent_chats": 10,
    "llm_timeout_seconds": 600,
    "llm_max_retries": 5,
    "llm_base_delay": 2.0,
    "max_token_length": 8192,
    "similarity_threshold": 0.3,
    "vector_similarity_weight": 0.7,
    "text_similarity_weight": 0.3,
    "max_entities": 6,
    "max_relations": 6,
    "max_communities": 1,
    "cache_expiration": 3600,
    "default_model": "gpt-3.5-turbo",
    "default_embedding_model": "text-embedding-ada-002"
}

# تنظیمات مدل‌های مختلف
MODEL_SETTINGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "models": {
            "gpt-4o": {"max_tokens": 4096, "temperature": 0.7},
            "gpt-4o-mini": {"max_tokens": 4096, "temperature": 0.7},
            "gpt-4-turbo": {"max_tokens": 4096, "temperature": 0.7},
            "gpt-4": {"max_tokens": 4096, "temperature": 0.7},
            "gpt-3.5-turbo": {"max_tokens": 4096, "temperature": 0.7},
            "gpt-3.5-turbo-16k": {"max_tokens": 16384, "temperature": 0.7}
        }
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/",
        "models": {
            "claude-3-5-sonnet": {"max_tokens": 4096, "temperature": 0.7},
            "claude-3-5-haiku": {"max_tokens": 4096, "temperature": 0.7},
            "claude-3-opus": {"max_tokens": 4096, "temperature": 0.7},
            "claude-3-sonnet": {"max_tokens": 4096, "temperature": 0.7},
            "claude-3-haiku": {"max_tokens": 4096, "temperature": 0.7}
        }
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "models": {
            "gemini-1.5-pro": {"max_tokens": 4096, "temperature": 0.7},
            "gemini-1.5-flash": {"max_tokens": 4096, "temperature": 0.7},
            "gemini-1.0-pro": {"max_tokens": 4096, "temperature": 0.7},
            "gemini-1.0-flash": {"max_tokens": 4096, "temperature": 0.7}
        }
    }
}

# تنظیمات embedding
EMBEDDING_SETTINGS = {
    "openai": {
        "text-embedding-ada-002": {"dimensions": 1536},
        "text-embedding-3-small": {"dimensions": 1536},
        "text-embedding-3-large": {"dimensions": 3072}
    },
    "sentence_transformers": {
        "all-MiniLM-L6-v2": {"dimensions": 384},
        "paraphrase-multilingual-MiniLM-L12-v2": {"dimensions": 384},
        "all-mpnet-base-v2": {"dimensions": 768}
    }
}

# تنظیمات Redis
REDIS_SETTINGS = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "db": int(os.getenv("REDIS_DB", 0)),
    "password": os.getenv("REDIS_PASSWORD", None),
    "decode_responses": True
}

# تنظیمات Elasticsearch
ES_SETTINGS = {
    "hosts": os.getenv("ES_HOSTS", "localhost:9200").split(","),
    "username": os.getenv("ES_USERNAME", None),
    "password": os.getenv("ES_PASSWORD", None),
    "timeout": int(os.getenv("ES_TIMEOUT", 30))
}

# تنظیمات پایگاه داده
DB_SETTINGS = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "ragflow"),
    "username": os.getenv("DB_USERNAME", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password")
}

def get_setting(key: str, default: Any = None) -> Any:
    """دریافت تنظیم از متغیرهای محیطی یا مقادیر پیش‌فرض"""
    return os.getenv(key, DEFAULT_SETTINGS.get(key, default))

def get_model_setting(provider: str, model: str, setting: str) -> Any:
    """دریافت تنظیم مدل خاص"""
    return MODEL_SETTINGS.get(provider, {}).get("models", {}).get(model, {}).get(setting)

def get_embedding_setting(provider: str, model: str, setting: str) -> Any:
    """دریافت تنظیم embedding خاص"""
    return EMBEDDING_SETTINGS.get(provider, {}).get(model, {}).get(setting) 
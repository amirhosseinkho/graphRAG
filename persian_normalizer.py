# -*- coding: utf-8 -*-
"""
Persian Text Normalizer - نرمال‌سازی و اصلاح املای فارسی
"""

import re
import logging
from typing import Optional

# Try to import hazm for Persian processing
try:
    from hazm import Normalizer, word_tokenize, sent_tokenize
    HAZM_AVAILABLE = True
except ImportError:
    HAZM_AVAILABLE = False
    logging.warning("hazm not available. Install with: pip install hazm")

# Try to import Persian spell checker
try:
    from faspellchecker import SpellChecker as PersianSpellChecker
    SPELL_CHECKER_AVAILABLE = True
    VIRASTAR_AVAILABLE = False
except ImportError:
    try:
        from persian_spell_checker import PersianSpellChecker
        SPELL_CHECKER_AVAILABLE = True
        VIRASTAR_AVAILABLE = False
    except ImportError:
        try:
            from virastar import PersianEditor
            SPELL_CHECKER_AVAILABLE = True
            VIRASTAR_AVAILABLE = True
        except ImportError:
            SPELL_CHECKER_AVAILABLE = False
            VIRASTAR_AVAILABLE = False
            logging.warning("Persian spell checker not available. Spell checking will be disabled. For spell checking, you can install: pip install virastar")


class PersianNormalizer:
    """کلاس برای نرمال‌سازی متن فارسی"""
    
    def __init__(self, enable_spell_check: bool = False):
        """
        Initialize Persian normalizer
        
        Args:
            enable_spell_check: فعال کردن اصلاح املای خودکار
        """
        self.enable_spell_check = enable_spell_check
        
        # Initialize hazm normalizer if available
        self.hazm_normalizer = None
        if HAZM_AVAILABLE:
            try:
                self.hazm_normalizer = Normalizer()
            except Exception as e:
                logging.warning(f"Failed to initialize hazm Normalizer: {e}")
        
        # Initialize spell checker if enabled
        self.spell_checker = None
        if enable_spell_check:
            if SPELL_CHECKER_AVAILABLE:
                try:
                    if VIRASTAR_AVAILABLE:
                        self.spell_checker = PersianEditor()
                    else:
                        self.spell_checker = PersianSpellChecker()
                except Exception as e:
                    logging.warning(f"Failed to initialize spell checker: {e}")
                    self.spell_checker = None
    
    def normalize(self, text: str) -> str:
        """
        نرمال‌سازی متن فارسی
        
        Args:
            text: متن ورودی
            
        Returns:
            متن نرمال‌سازی شده
        """
        if not text:
            return ""
        
        # Use hazm normalizer if available
        if self.hazm_normalizer:
            try:
                text = self.hazm_normalizer.normalize(text)
            except Exception as e:
                logging.warning(f"hazm normalization failed: {e}")
        
        # Convert different forms of ی/ک
        text = text.replace("ي", "ی")  # Arabic yeh to Persian yeh
        text = text.replace("ك", "ک")  # Arabic kaf to Persian kaf
        text = text.replace("ة", "ه")  # Arabic teh marbuta to heh
        
        # Normalize half-space (ZWNJ)
        text = text.replace("\u200c", " ")  # Zero-width non-joiner to space
        text = text.replace("\u200d", "")   # Zero-width joiner removal
        
        # Normalize Persian/Arabic numbers to Persian
        persian_digits = "۰۱۲۳۴۵۶۷۸۹"
        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        english_digits = "0123456789"
        
        # Convert Arabic digits to Persian
        for i, arabic_digit in enumerate(arabic_digits):
            text = text.replace(arabic_digit, persian_digits[i])
        
        # Optionally convert English digits to Persian (commented out by default)
        # for i, eng_digit in enumerate(english_digits):
        #     text = text.replace(eng_digit, persian_digits[i])
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def correct_spelling(self, text: str) -> str:
        """
        اصلاح املای فارسی
        
        Args:
            text: متن ورودی
            
        Returns:
            متن با املای اصلاح شده
        """
        if not text or not self.enable_spell_check or not self.spell_checker:
            return text
        
        try:
            if VIRASTAR_AVAILABLE and isinstance(self.spell_checker, PersianEditor):
                # virastar usage
                return self.spell_checker.edit(text)
            elif isinstance(self.spell_checker, PersianSpellChecker):
                # fa-spellchecker or persian-spell-checker usage
                words = word_tokenize(text) if HAZM_AVAILABLE else text.split()
                corrected_words = []
                for word in words:
                    # fa-spellchecker uses correction() method
                    if hasattr(self.spell_checker, 'correction'):
                        corrected = self.spell_checker.correction(word)
                    else:
                        corrected = self.spell_checker.correct(word)
                    corrected_words.append(corrected if corrected else word)
                return " ".join(corrected_words)
        except Exception as e:
            logging.warning(f"Spell checking failed: {e}")
            return text
        
        return text
    
    def normalize_and_correct(self, text: str) -> str:
        """
        نرمال‌سازی و اصلاح املای متن
        
        Args:
            text: متن ورودی
            
        Returns:
            متن نرمال‌سازی و اصلاح شده
        """
        normalized = self.normalize(text)
        if self.enable_spell_check:
            normalized = self.correct_spelling(normalized)
        return normalized


def detect_language(text: str) -> str:
    """
    تشخیص زبان متن (فارسی یا انگلیسی)
    
    Args:
        text: متن ورودی
        
    Returns:
        'fa' برای فارسی، 'en' برای انگلیسی، 'mixed' برای ترکیبی
    """
    if not text or not text.strip():
        return 'en'  # Default to English
    
    # Count Persian characters (Unicode range: \u0600-\u06FF)
    persian_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    # Count English/Latin characters
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    # Total characters (excluding spaces and punctuation)
    total_chars = len(re.findall(r'[\u0600-\u06FFa-zA-Z]', text))
    
    if total_chars == 0:
        return 'en'  # Default if no characters detected
    
    persian_ratio = persian_chars / total_chars if total_chars > 0 else 0
    
    # If more than 50% Persian characters, consider it Persian
    if persian_ratio > 0.5:
        return 'fa'
    elif persian_ratio > 0.1:
        return 'mixed'
    else:
        return 'en'


def is_persian(text: str) -> bool:
    """
    بررسی اینکه آیا متن فارسی است یا نه
    
    Args:
        text: متن ورودی
        
    Returns:
        True اگر فارسی باشد، False در غیر این صورت
    """
    lang = detect_language(text)
    return lang == 'fa' or lang == 'mixed'

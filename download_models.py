# -*- coding: utf-8 -*-
"""
Download Models Script - اسکریپت دانلود خودکار مدل‌ها
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_command(command, description):
    """اجرای دستور و نمایش نتیجه"""
    logging.info(f"در حال {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logging.info(f"✓ {description} موفق بود")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ خطا در {description}: {e.stderr}")
        return False


def download_spacy_models():
    """دانلود مدل‌های spaCy"""
    models = [
        ("en_core_web_sm", "مدل spaCy انگلیسی"),
        ("fa_core_news_sm", "مدل spaCy فارسی")
    ]
    
    for model_name, description in models:
        command = f"{sys.executable} -m spacy download {model_name}"
        run_command(command, f"دانلود {description} ({model_name})")


def download_huggingface_models():
    """دانلود مدل‌های HuggingFace (فقط دانلود tokenizer و config، مدل‌ها lazy load می‌شوند)"""
    models = [
        ("HooshvareLab/bert-fa-base-uncased", "ParsBERT"),
        ("persiannlp/mt5-base-parsinlu", "mT5 فارسی"),
        ("dmis-lab/biobert-v1.1", "BioBERT"),
        ("allenai/scibert_scivocab_uncased", "SciBERT"),
        ("bert-base-uncased", "BERT پایه")
    ]
    
    logging.info("مدل‌های HuggingFace به صورت lazy loading بارگذاری می‌شوند.")
    logging.info("مدل‌های زیر در اولین استفاده دانلود خواهند شد:")
    for model_name, description in models:
        logging.info(f"  - {model_name} ({description})")


def main():
    """تابع اصلی"""
    logging.info("=" * 60)
    logging.info("اسکریپت دانلود مدل‌ها")
    logging.info("=" * 60)
    
    # Check if spacy is installed
    try:
        import spacy
        logging.info("✓ spaCy نصب شده است")
    except ImportError:
        logging.error("✗ spaCy نصب نشده است. لطفاً ابتدا requirements.txt را نصب کنید.")
        return
    
    # Check if transformers is installed
    try:
        import transformers
        logging.info("✓ transformers نصب شده است")
    except ImportError:
        logging.error("✗ transformers نصب نشده است. لطفاً ابتدا requirements.txt را نصب کنید.")
        return
    
    # Download spaCy models
    logging.info("\n" + "=" * 60)
    logging.info("دانلود مدل‌های spaCy")
    logging.info("=" * 60)
    download_spacy_models()
    
    # Info about HuggingFace models
    logging.info("\n" + "=" * 60)
    logging.info("مدل‌های HuggingFace")
    logging.info("=" * 60)
    download_huggingface_models()
    
    logging.info("\n" + "=" * 60)
    logging.info("دانلود مدل‌ها تکمیل شد!")
    logging.info("=" * 60)
    logging.info("\nنکته: مدل‌های HuggingFace در اولین استفاده به صورت خودکار دانلود می‌شوند.")
    logging.info("برای استفاده از مدل‌های HuggingFace، نیاز به توکن HuggingFace دارید.")
    logging.info("توکن خود را از https://huggingface.co/settings/tokens دریافت کنید.")


if __name__ == "__main__":
    main()

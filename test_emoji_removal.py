#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست حذف ایموجی‌ها از متن زمینه
"""

import re
import sys
import os

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_context_generator import remove_emojis
from graphrag_service import remove_emojis as remove_emojis_service

def test_emoji_removal():
    """تست تابع حذف ایموجی‌ها"""
    print("تست حذف ایموجی‌ها از متن زمینه")
    print("=" * 50)
    
    # نمونه‌های متن با ایموجی
    test_texts = [
        "🧬 **متن زمینه هوشمند برای سوال:** What is TP53?",
        "📋 **نودهای کلیدی (با اطلاعات معنادار):**",
        "🔗 **روابط معنادار:**",
        "🔬 **تحلیل زیستی و استنتاجات:**",
        "🧠 **استنتاجات زیستی:**",
        "🏥 **اهمیت بالینی:**",
        "💊 **روابط درمانی:**",
        "📊 **آمار بازیابی:**",
        "🏷️ **تحلیل نوع‌شناسی نودها:**",
        "🔄 **تحلیل مسیر زیستی برای:** How does TP53 work?",
        "⚙️ **Related Biological Processes:**",
        "🛤️ **Related Pathways:**",
        "🔍 **Key Results:**",
        "📌 **Instructions:** Analyze biological relevance."
    ]
    
    print("نمونه‌های متن با ایموجی:")
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")
    
    print("\nنتایج حذف ایموجی‌ها:")
    for i, text in enumerate(test_texts, 1):
        cleaned_text = remove_emojis(text)
        print(f"{i}. {cleaned_text}")
    
    # تست تابع از graphrag_service
    print("\nتست تابع از graphrag_service:")
    for i, text in enumerate(test_texts[:5], 1):
        cleaned_text = remove_emojis_service(text)
        print(f"{i}. {cleaned_text}")
    
    print("\nتست موفقیت‌آمیز!")

if __name__ == "__main__":
    test_emoji_removal() 
# -*- coding: utf-8 -*-
"""
URL Extractor - استخراج متن از URL
"""

import logging
import re
from typing import Optional
from urllib.parse import urlparse

# Try to import requests
try:
    import requests
    from requests.exceptions import RequestException, Timeout
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests not available. Install with: pip install requests")

# Try to import BeautifulSoup
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    BeautifulSoup = None  # برای جلوگیری از خطا در type hints
    logging.warning("beautifulsoup4 not available. Install with: pip install beautifulsoup4")


def is_valid_url(url: str) -> bool:
    """
    بررسی معتبر بودن URL
    
    Args:
        url: URL برای بررسی
        
    Returns:
        True اگر URL معتبر باشد
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except Exception:
        return False


def extract_text_from_url(url: str, timeout: int = 30, clean_content: bool = True, 
                          max_length: int = 10000) -> Optional[str]:
    """
    استخراج متن مفید از URL با حذف محتوای غیرضروری
    
    Args:
        url: URL صفحه وب
        timeout: زمان انتظار (ثانیه)
        clean_content: آیا محتوای غیرضروری حذف شود؟ (navigation، footer، sidebar، ads)
        max_length: حداکثر طول متن استخراج شده (پیش‌فرض: 50000 کاراکتر)
        
    Returns:
        متن استخراج شده یا None در صورت خطا
    """
    if not REQUESTS_AVAILABLE:
        logging.error("requests library not available")
        return None
    
    if not is_valid_url(url):
        logging.error(f"Invalid URL: {url}")
        return None
    
    try:
        # درخواست HTTP
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        
        # استخراج متن با BeautifulSoup
        if BEAUTIFULSOUP_AVAILABLE and BeautifulSoup is not None:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            if clean_content:
                # حذف محتوای غیرضروری
                text = _extract_clean_content(soup)
            else:
                # روش ساده (بدون پاکسازی پیشرفته)
                for script in soup(["script", "style", "meta", "link", "noscript"]):
                    script.decompose()
                text = soup.get_text()
            
            # پاکسازی نهایی متن
            text = _clean_text(text)
            
            # اعمال محدودیت طول
            if max_length > 0 and len(text) > max_length:
                text = _truncate_text_intelligently(text, max_length)
                logging.info(f"Text truncated from {len(text) + (len(text) - max_length)} to {len(text)} characters")
            
            return text
        else:
            # Fallback: استفاده از regex ساده
            text = response.text
            # حذف تگ‌های HTML
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', '', text)
            # حذف فاصله‌های اضافی
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
            
    except Timeout:
        logging.error(f"Timeout while fetching URL: {url}")
        return None
    except RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error extracting text from URL {url}: {e}")
        return None


def _extract_clean_content(soup: BeautifulSoup) -> str:
    """
    استخراج محتوای مفید از HTML با حذف navigation، footer، sidebar، ads
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        متن پاک شده
    """
    if soup is None:
        return ""
    
    # حذف تگ‌های غیرضروری
    unwanted_tags = ["script", "style", "meta", "link", "noscript", "iframe", "embed", "object"]
    for tag in unwanted_tags:
        for element in soup.find_all(tag):
            element.decompose()
    
    # حذف navigation
    for nav in soup.find_all(["nav", "header", "footer"]):
        nav.decompose()
    
    # حذف sidebar و aside
    for aside in soup.find_all(["aside", "sidebar"]):
        aside.decompose()
    
    # حذف advertising و promotional content
    ad_classes = ["ad", "advertisement", "ads", "promo", "promotion", "sponsor", "sponsored"]
    for class_name in ad_classes:
        for element in soup.find_all(class_=re.compile(class_name, re.I)):
            element.decompose()
    
    # حذف cookie banners و popups
    popup_classes = ["cookie", "popup", "modal", "overlay", "banner"]
    for class_name in popup_classes:
        for element in soup.find_all(class_=re.compile(class_name, re.I)):
            element.decompose()
    
    # پیدا کردن محتوای اصلی
    # اولویت 1: تگ‌های semantic HTML5
    main_content = None
    for tag_name in ["main", "article", "section"]:
        main_content = soup.find(tag_name)
        if main_content:
            break
    
    # اولویت 2: div با کلاس‌های محتوا
    if not main_content:
        content_classes = ["content", "main-content", "post-content", "entry-content", "article-content", "body-content"]
        for class_name in content_classes:
            main_content = soup.find("div", class_=re.compile(class_name, re.I))
            if main_content:
                break
    
    # اولویت 3: body (اگر محتوای اصلی پیدا نشد)
    if not main_content:
        main_content = soup.find("body")
    
    if main_content:
        # استخراج پاراگراف‌ها و headings
        text_parts = []
        
        # استخراج headings
        for heading in main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            heading_text = heading.get_text(strip=True)
            if heading_text and len(heading_text) > 2:
                text_parts.append(f"\n{heading_text}\n")
        
        # استخراج پاراگراف‌ها
        for p in main_content.find_all("p"):
            p_text = p.get_text(strip=True)
            # فیلتر کردن پاراگراف‌های کوتاه و غیرمفید
            if p_text and len(p_text) > 30:  # حداقل 30 کاراکتر
                # حذف پاراگراف‌هایی که فقط لینک هستند
                links_in_p = p.find_all("a")
                if len(links_in_p) < len(p_text) / 10:  # کمتر از 10% لینک
                    text_parts.append(p_text)
        
        # استخراج لیست‌ها
        for ul in main_content.find_all(["ul", "ol"]):
            list_items = []
            for li in ul.find_all("li", recursive=False):
                li_text = li.get_text(strip=True)
                if li_text and len(li_text) > 10:
                    list_items.append(f"• {li_text}")
            if list_items:
                text_parts.append("\n".join(list_items))
        
        return "\n\n".join(text_parts)
    
    # Fallback: استخراج همه متن
    return soup.get_text()


def _clean_text(text: str) -> str:
    """
    پاکسازی نهایی متن
    
    Args:
        text: متن خام
        
    Returns:
        متن پاک شده
    """
    if not text:
        return ""
    
    # حذف خطوط خالی متوالی
    lines = []
    prev_empty = False
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
            prev_empty = False
        elif not prev_empty:
            lines.append("")  # یک خط خالی
            prev_empty = True
    
    text = "\n".join(lines)
    
    # حذف فاصله‌های اضافی
    text = re.sub(r'[ \t]+', ' ', text)
    
    # حذف خطوط خالی بیش از 2 خط
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # حذف کاراکترهای کنترل
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()


def _truncate_text_intelligently(text: str, max_length: int) -> str:
    """
    قطع کردن هوشمند متن در مرزهای منطقی (جمله، پاراگراف)
    
    Args:
        text: متن کامل
        max_length: حداکثر طول مورد نظر
        
    Returns:
        متن قطع شده در مرز منطقی
    """
    if len(text) <= max_length:
        return text
    
    # تلاش برای قطع در پایان پاراگراف
    truncated = text[:max_length]
    
    # پیدا کردن آخرین پاراگراف کامل
    last_paragraph_end = truncated.rfind('\n\n')
    if last_paragraph_end > max_length * 0.8:  # اگر بیش از 80% متن را حفظ کنیم
        truncated = truncated[:last_paragraph_end]
    else:
        # پیدا کردن آخرین جمله کامل
        sentence_endings = ['. ', '.\n', '! ', '!\n', '? ', '?\n']
        last_sentence_end = -1
        for ending in sentence_endings:
            pos = truncated.rfind(ending)
            if pos > last_sentence_end and pos > max_length * 0.7:
                last_sentence_end = pos + len(ending)
        
        if last_sentence_end > 0:
            truncated = truncated[:last_sentence_end]
        else:
            # اگر هیچ مرز منطقی پیدا نشد، از کلمه آخر استفاده کن
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.9:
                truncated = truncated[:last_space]
            else:
                # آخرین راه حل: قطع مستقیم
                truncated = truncated[:max_length]
    
    # اضافه کردن نشانگر قطع شدن
    if len(text) > len(truncated):
        truncated += "\n\n[... متن ادامه دارد ...]"
    
    return truncated

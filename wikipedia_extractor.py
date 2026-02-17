# -*- coding: utf-8 -*-
"""
Wikipedia Extractor - استخراج تخصصی از ویکی‌پدیا
"""

import logging
import re
from typing import Dict, List, Any, Optional
from urllib.parse import quote, unquote

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests not available")

# Try to import BeautifulSoup
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    BeautifulSoup = None  # برای جلوگیری از خطا در type hints


class WikipediaExtractor:
    """استخراج تخصصی از ویکی‌پدیا با استفاده از API و parsing HTML"""
    
    def __init__(self, language: str = "fa"):
        """
        Initialize Wikipedia extractor
        
        Args:
            language: زبان ویکی‌پدیا (fa/en/...)
        """
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org"
        self.api_url = f"{self.base_url}/api/rest_v1"
    
    def extract_from_url(self, url: str) -> Dict[str, Any]:
        """
        استخراج اطلاعات از URL ویکی‌پدیا
        
        Args:
            url: URL صفحه ویکی‌پدیا
            
        Returns:
            Dictionary حاوی اطلاعات استخراج شده
        """
        # استخراج عنوان از URL
        title = self._extract_title_from_url(url)
        if not title:
            return {"error": "نمی‌توان عنوان را از URL استخراج کرد"}
        
        return self.extract_from_title(title)
    
    def extract_from_title(self, title: str) -> Dict[str, Any]:
        """
        استخراج اطلاعات از عنوان صفحه ویکی‌پدیا
        
        Args:
            title: عنوان صفحه ویکی‌پدیا
            
        Returns:
            Dictionary حاوی اطلاعات استخراج شده
        """
        result = {
            "title": title,
            "text": "",
            "infobox": {},
            "categories": [],
            "links": [],
            "sections": {},
            "entities": [],
            "relationships": []
        }
        
        # اولویت 1: استخراج با HTML parsing (برای متن کامل)
        html_result = self._extract_via_html(title)
        if html_result:
            # استخراج متن از HTML
            if "sections" in html_result:
                # استفاده از sections برای ساخت متن کامل
                sections = html_result["sections"]
                text_parts = []
                
                # اضافه کردن عنوان
                text_parts.append(f"# {title}")
                
                # اضافه کردن همه sections
                for section_name, section_content in sections.items():
                    if section_content and len(section_content.strip()) > 20:
                        text_parts.append(f"\n## {section_name}\n{section_content}")
                
                if text_parts:
                    result["text"] = "\n\n".join(text_parts)
                    result["sections"] = sections
            
            # اضافه کردن سایر اطلاعات
            if "infobox" in html_result:
                result["infobox"] = html_result["infobox"]
            if "categories" in html_result:
                result["categories"] = html_result["categories"]
            if "links" in html_result:
                result["links"] = html_result["links"]
        
        # اولویت 2: اگر متن از HTML کافی نبود، از API استفاده کن
        if not result.get("text") or len(result.get("text", "").strip()) < 200:
            api_result = self._extract_via_api(title)
            if api_result and api_result.get("text"):
                api_text = api_result.get("text", "")
                if len(api_text.strip()) > len(result.get("text", "").strip()):
                    result["text"] = api_text
                    # حفظ سایر اطلاعات از API
                    if "description" in api_result:
                        result["description"] = api_result["description"]
        
        # اگر هنوز متن کافی نداریم، از HTML parsing مستقیم استفاده کن
        if not result.get("text") or len(result.get("text", "").strip()) < 200:
            try:
                encoded_title = quote(title.replace(' ', '_'))
                url = f"{self.base_url}/wiki/{encoded_title}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    direct_text = self._extract_text_from_wikipedia_html(soup)
                    if direct_text and len(direct_text.strip()) > len(result.get("text", "").strip()):
                        result["text"] = direct_text
            except Exception as e:
                logging.warning(f"Direct HTML extraction failed: {e}")
        
        # استخراج موجودیت‌ها و روابط
        result["entities"], result["relationships"] = self._extract_entities_and_relations(result)
        
        return result
    
    def _extract_title_from_url(self, url: str) -> Optional[str]:
        """استخراج عنوان از URL"""
        # Pattern: https://fa.wikipedia.org/wiki/Title
        match = re.search(r'/wiki/([^?#]+)', url)
        if match:
            return unquote(match.group(1)).replace('_', ' ')
        return None
    
    def _extract_via_api(self, title: str) -> Optional[Dict[str, Any]]:
        """استخراج با استفاده از Wikipedia REST API"""
        if not REQUESTS_AVAILABLE:
            return None
        
        try:
            encoded_title = quote(title.replace(' ', '_'))
            
            # روش 1: استفاده از TextExtracts API با exlimit برای دریافت متن کامل
            try:
                # استفاده از MediaWiki API برای استخراج متن خالص
                api_url = f"{self.base_url}/w/api.php"
                params = {
                    "action": "query",
                    "format": "json",
                    "titles": title,
                    "prop": "extracts",
                    "exintro": False,  # دریافت متن کامل، نه فقط مقدمه
                    "explaintext": True,  # متن خالص بدون HTML
                    "exsectionformat": "plain",
                    "exlimit": 1,  # دریافت یک صفحه
                    "exchars": 100000  # حداکثر 100000 کاراکتر (افزایش از پیش‌فرض)
                }
                
                response = requests.get(api_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    pages = data.get("query", {}).get("pages", {})
                    for page_id, page_data in pages.items():
                        extract = page_data.get("extract", "")
                        if extract and len(extract.strip()) > 50:  # بررسی اینکه متن خالی نباشد
                            # پاکسازی متن با محدودیت طول
                            extract = self._clean_wikipedia_text(extract, max_length=10000)
                            logging.info(f"Extracted {len(extract)} characters from Wikipedia API")
                            return {
                                "text": extract,
                                "description": page_data.get("description", ""),
                                "pageid": page_id
                            }
            except Exception as e:
                logging.warning(f"TextExtracts API failed: {e}")
            
            # روش 2: استفاده از REST API برای HTML (اولویت برای متن کامل)
            try:
                url = f"{self.api_url}/page/html/{encoded_title}"
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    html_content = response.text
                    
                    # استخراج متن از HTML
                    if BEAUTIFULSOUP_AVAILABLE and BeautifulSoup is not None:
                        try:
                            soup = BeautifulSoup(html_content, 'html.parser')
                            text = self._extract_text_from_wikipedia_html(soup)
                            
                            if text and len(text.strip()) > 100:  # بررسی اینکه متن کافی است
                                # اعمال محدودیت طول
                                text = self._clean_wikipedia_text(text, max_length=10000)
                                logging.info(f"Extracted {len(text)} characters from Wikipedia HTML")
                                return {
                                    "text": text,
                                    "html": html_content
                                }
                        except Exception as e:
                            logging.warning(f"BeautifulSoup parsing failed: {e}")
            except Exception as e:
                logging.warning(f"REST API HTML extraction failed: {e}")
            
            # روش 3: استفاده از summary API (fallback)
            try:
                url = f"{self.api_url}/page/summary/{encoded_title}"
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    extract = data.get("extract", "")
                    if extract:
                        extract = self._clean_wikipedia_text(extract, max_length=10000)
                        return {
                            "text": extract,
                            "description": data.get("description", ""),
                            "thumbnail": data.get("thumbnail", {}).get("source", "")
                        }
            except Exception as e:
                logging.warning(f"Summary API failed: {e}")
                
        except Exception as e:
            logging.warning(f"Wikipedia API extraction failed: {e}")
        
        return None
    
    def _clean_wikipedia_text(self, text: str, max_length: int = 10000) -> str:
        """پاکسازی متن ویکی‌پدیا"""
        if not text:
            return ""
        
        # حذف reference markers مثل [1], [2]
        text = re.sub(r'\[\d+\]', '', text)
        
        # حذف citation templates
        text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
        
        # حذف خطوط خالی متوالی
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # حذف فاصله‌های اضافی
        text = re.sub(r'[ \t]+', ' ', text)
        
        text = text.strip()
        
        # اعمال محدودیت طول
        if max_length > 0 and len(text) > max_length:
            text = self._truncate_text_intelligently(text, max_length)
            logging.info(f"Wikipedia text truncated to {len(text)} characters")
        
        return text
    
    def _truncate_text_intelligently(self, text: str, max_length: int) -> str:
        """قطع کردن هوشمند متن در مرزهای منطقی"""
        if len(text) <= max_length:
            return text
        
        # تلاش برای قطع در پایان section
        truncated = text[:max_length]
        
        # پیدا کردن آخرین section (##)
        last_section_end = truncated.rfind('\n##')
        if last_section_end > max_length * 0.7:
            # پیدا کردن پایان section قبلی
            section_start = truncated.rfind('\n##', 0, last_section_end)
            if section_start > 0:
                # پیدا کردن پایان پاراگراف بعد از section قبلی
                next_paragraph_end = truncated.find('\n\n', section_start)
                if next_paragraph_end > 0 and next_paragraph_end < max_length:
                    truncated = truncated[:next_paragraph_end]
                else:
                    truncated = truncated[:last_section_end]
            else:
                truncated = truncated[:last_section_end]
        else:
            # پیدا کردن آخرین پاراگراف کامل
            last_paragraph_end = truncated.rfind('\n\n')
            if last_paragraph_end > max_length * 0.8:
                truncated = truncated[:last_paragraph_end]
            else:
                # پیدا کردن آخرین جمله کامل
                sentence_endings = ['. ', '.\n', '! ', '!\n', '? ', '?\n', '.\n\n']
                last_sentence_end = -1
                for ending in sentence_endings:
                    pos = truncated.rfind(ending)
                    if pos > last_sentence_end and pos > max_length * 0.7:
                        last_sentence_end = pos + len(ending)
                
                if last_sentence_end > 0:
                    truncated = truncated[:last_sentence_end]
                else:
                    # آخرین راه حل: قطع در کلمه
                    last_space = truncated.rfind(' ')
                    if last_space > max_length * 0.9:
                        truncated = truncated[:last_space]
                    else:
                        truncated = truncated[:max_length]
        
        # اضافه کردن نشانگر
        if len(text) > len(truncated):
            truncated += "\n\n[... متن ادامه دارد ...]"
        
        return truncated
    
    def _extract_via_html(self, title: str) -> Optional[Dict[str, Any]]:
        """استخراج با parsing HTML صفحه"""
        if not REQUESTS_AVAILABLE or not BEAUTIFULSOUP_AVAILABLE:
            return None
        
        if BeautifulSoup is None:
            return None
        
        try:
            encoded_title = quote(title.replace(' ', '_'))
            url = f"{self.base_url}/wiki/{encoded_title}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            result = {}
            
            # استخراج Infobox
            infobox = self._extract_infobox(soup)
            if infobox:
                result["infobox"] = infobox
            
            # استخراج Categories
            categories = self._extract_categories(soup)
            if categories:
                result["categories"] = categories
            
            # استخراج لینک‌های داخلی
            links = self._extract_internal_links(soup)
            if links:
                result["links"] = links
            
            # استخراج بخش‌ها
            sections = self._extract_sections(soup)
            if sections:
                result["sections"] = sections
            
            return result
        except Exception as e:
            logging.warning(f"Wikipedia HTML extraction failed: {e}")
        
        return None
    
    def _extract_text_from_wikipedia_html(self, soup) -> str:
        """استخراج متن مفید از HTML ویکی‌پدیا"""
        if soup is None:
            return ""
        
        # حذف تگ‌های غیرضروری
        unwanted_tags = ["script", "style", "nav", "header", "footer", "aside", 
                        "table", "infobox", "mw-editsection", "reference", 
                        "mw-references-wrap", "thumb", "gallery", "coordinates"]
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # حذف کلاس‌های غیرضروری
        unwanted_classes = ["navbox", "vertical-navbox", "infobox", "metadata",
                          "mw-jump-link", "mw-editsection", "reference",
                          "hatnote", "dablink", "ambox"]
        for class_name in unwanted_classes:
            for element in soup.find_all(class_=re.compile(class_name, re.I)):
                element.decompose()
        
        # استخراج محتوای اصلی
        content_div = soup.find("div", {"class": "mw-parser-output"})
        if not content_div:
            content_div = soup.find("div", {"id": "content"})
        
        if content_div:
            text_parts = []
            
            # استخراج headings
            for heading in content_div.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
                heading_text = heading.get_text(strip=True)
                # حذف edit links از headings
                heading_text = re.sub(r'\[edit\]', '', heading_text, flags=re.IGNORECASE)
                if heading_text and len(heading_text) > 2:
                    text_parts.append(f"\n{heading_text}\n")
            
            # استخراج پاراگراف‌ها
            paragraphs = content_div.find_all("p")
            for p in paragraphs:
                # حذف reference links از پاراگراف
                for ref in p.find_all(["sup", "span"], class_=re.compile("reference", re.I)):
                    ref.decompose()
                
                text = p.get_text(strip=True)
                # حذف reference markers
                text = re.sub(r'\[\d+\]', '', text)
                text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
                
                # فقط پاراگراف‌های معنی‌دار (حداقل 30 کاراکتر)
                if text and len(text) > 30:
                    # بررسی اینکه پاراگراف فقط لینک نیست
                    links = p.find_all("a")
                    if len(links) < len(text) / 5:  # کمتر از 20% لینک
                        text_parts.append(text)
            
            # استخراج لیست‌ها
            for ul in content_div.find_all(["ul", "ol"]):
                list_items = []
                for li in ul.find_all("li", recursive=False):
                    # حذف reference links
                    for ref in li.find_all(["sup", "span"], class_=re.compile("reference", re.I)):
                        ref.decompose()
                    li_text = li.get_text(strip=True)
                    li_text = re.sub(r'\[\d+\]', '', li_text)
                    if li_text and len(li_text) > 15:
                        list_items.append(f"• {li_text}")
                if list_items:
                    text_parts.append("\n".join(list_items))
            
            result = "\n\n".join(text_parts)
            return self._clean_wikipedia_text(result, max_length=10000)
        
        # Fallback: استخراج همه متن
        return self._clean_wikipedia_text(soup.get_text())
    
    def _extract_infobox(self, soup) -> Dict[str, Any]:
        """استخراج Infobox"""
        infobox = {}
        
        if soup is None:
            return infobox
        
        # جستجوی infobox
        infobox_div = soup.find("table", {"class": "infobox"})
        if not infobox_div:
            infobox_div = soup.find("div", {"class": "infobox"})
        
        if infobox_div:
            rows = infobox_div.find_all("tr")
            for row in rows:
                th = row.find("th")
                td = row.find("td")
                if th and td:
                    key = th.get_text(strip=True)
                    value = td.get_text(strip=True)
                    if key and value:
                        infobox[key] = value
        
        return infobox
    
    def _extract_categories(self, soup) -> List[str]:
        """استخراج دسته‌بندی‌ها"""
        categories = []
        
        if soup is None:
            return categories
        
        # جستجوی لینک‌های دسته‌بندی
        cat_links = soup.find_all("a", href=re.compile(r'/wiki/Category:'))
        for link in cat_links:
            cat_name = link.get_text(strip=True)
            if cat_name:
                categories.append(cat_name)
        
        return categories
    
    def _extract_internal_links(self, soup) -> List[str]:
        """استخراج لینک‌های داخلی ویکی‌پدیا"""
        links = []
        
        if soup is None:
            return links
        
        # جستجوی لینک‌های داخلی در محتوای اصلی
        content_div = soup.find("div", {"class": "mw-parser-output"})
        if content_div:
            internal_links = content_div.find_all("a", href=re.compile(r'/wiki/[^:]+$'))
        else:
            internal_links = soup.find_all("a", href=re.compile(r'/wiki/[^:]+$'))
        
        seen = set()
        
        for link in internal_links:
            href = link.get("href", "")
            if href.startswith("/wiki/"):
                title = unquote(href[6:]).replace('_', ' ')
                # فیلتر کردن لینک‌های غیرضروری
                if (title and title not in seen and 
                    not title.startswith("Category:") and
                    not title.startswith("File:") and
                    not title.startswith("Image:") and
                    not title.startswith("Template:") and
                    not title.startswith("Help:") and
                    not title.startswith("Special:")):
                    links.append(title)
                    seen.add(title)
        
        return links[:100]  # محدود کردن به 100 لینک اول
    
    def _extract_sections(self, soup) -> Dict[str, str]:
        """استخراج بخش‌های صفحه با متن کامل"""
        sections = {}
        
        if soup is None:
            return sections
        
        # پیدا کردن محتوای اصلی
        content_div = soup.find("div", {"class": "mw-parser-output"})
        if not content_div:
            return sections
        
        # حذف reference links و غیره
        for ref in content_div.find_all(["sup", "span"], class_=re.compile("reference", re.I)):
            ref.decompose()
        
        # جستجوی heading ها
        headings = content_div.find_all(["h1", "h2", "h3", "h4", "h5"])
        current_section = "مقدمه"
        current_content = []
        
        # استخراج محتوای قبل از اولین heading
        first_heading = headings[0] if headings else None
        if first_heading:
            for element in content_div.children:
                if element == first_heading:
                    break
                if hasattr(element, 'name') and element.name == "p":
                    text = element.get_text(strip=True)
                    text = re.sub(r'\[\d+\]', '', text)  # حذف reference markers
                    if text and len(text) > 20:
                        current_content.append(text)
                elif hasattr(element, 'name') and element.name in ["ul", "ol"]:
                    list_items = []
                    for li in element.find_all("li", recursive=False):
                        li_text = li.get_text(strip=True)
                        li_text = re.sub(r'\[\d+\]', '', li_text)
                        if li_text and len(li_text) > 10:
                            list_items.append(f"• {li_text}")
                    if list_items:
                        current_content.append("\n".join(list_items))
        
        if current_content:
            sections[current_section] = "\n".join(current_content)
        
        # پردازش هر heading و محتوای آن
        for i, heading in enumerate(headings):
            heading_text = heading.get_text(strip=True)
            # حذف edit links
            heading_text = re.sub(r'\[edit\]', '', heading_text, flags=re.IGNORECASE)
            
            if not heading_text or len(heading_text) < 2:
                continue
            
            # شروع بخش جدید
            current_section = heading_text
            current_content = []
            
            # پیدا کردن محتوای این بخش (تا heading بعدی)
            next_heading = headings[i + 1] if i + 1 < len(headings) else None
            
            # جمع‌آوری محتوای بخش
            current = heading.next_sibling
            while current:
                if current == next_heading:
                    break
                
                if hasattr(current, 'name'):
                    if current.name == "p":
                        text = current.get_text(strip=True)
                        text = re.sub(r'\[\d+\]', '', text)
                        if text and len(text) > 20:
                            current_content.append(text)
                    elif current.name in ["ul", "ol"]:
                        list_items = []
                        for li in current.find_all("li", recursive=False):
                            li_text = li.get_text(strip=True)
                            li_text = re.sub(r'\[\d+\]', '', li_text)
                            if li_text and len(li_text) > 10:
                                list_items.append(f"• {li_text}")
                        if list_items:
                            current_content.append("\n".join(list_items))
                    elif current.name in ["h1", "h2", "h3", "h4", "h5"]:
                        # اگر به heading دیگری رسیدیم، متوقف شو
                        break
                
                current = current.next_sibling
            
            # ذخیره بخش
            if current_content:
                sections[current_section] = "\n".join(current_content)
        
        return sections
    
    def _extract_entities_and_relations(self, data: Dict[str, Any]) -> tuple[List[Dict], List[Dict]]:
        """استخراج موجودیت‌ها و روابط از داده‌های ویکی‌پدیا"""
        entities = []
        relationships = []
        
        title = data.get("title", "")
        infobox = data.get("infobox", {})
        categories = data.get("categories", [])
        links = data.get("links", [])
        
        # موجودیت اصلی (عنوان صفحه)
        if title:
            entities.append({
                "id": f"ENTITY_0",
                "name": title,
                "type": "Concept",
                "attributes": {
                    "source": "wikipedia_title",
                    "wikipedia_url": f"{self.base_url}/wiki/{quote(title.replace(' ', '_'))}"
                }
            })
        
        # موجودیت‌ها از Infobox
        for key, value in infobox.items():
            if value and len(value) < 100:  # فقط مقادیر کوتاه
                entities.append({
                    "id": f"ENTITY_{len(entities)}",
                    "name": value,
                    "type": self._infer_entity_type(key),
                    "attributes": {
                        "source": "wikipedia_infobox",
                        "property": key
                    }
                })
                
                # رابطه بین موجودیت اصلی و موجودیت infobox
                if title:
                    relationships.append({
                        "source": "ENTITY_0",
                        "target": f"ENTITY_{len(entities)-1}",
                        "metaedge": "RELATED_TO",
                        "relation": key,
                        "attributes": {
                            "source": "wikipedia_infobox"
                        }
                    })
        
        # موجودیت‌ها از Categories
        for cat in categories[:10]:  # محدود کردن
            entities.append({
                "id": f"ENTITY_{len(entities)}",
                "name": cat,
                "type": "Category",
                "attributes": {
                    "source": "wikipedia_category"
                }
            })
            
            # رابطه با موجودیت اصلی
            if title:
                relationships.append({
                    "source": "ENTITY_0",
                    "target": f"ENTITY_{len(entities)-1}",
                    "metaedge": "BELONGS_TO",
                    "relation": "belongs to category",
                    "attributes": {
                        "source": "wikipedia_category"
                    }
                })
        
        # موجودیت‌ها از لینک‌های داخلی (اولویت به لینک‌های مهم)
        important_links = links[:30]  # محدود کردن به 30 لینک اول
        
        for link in important_links:
            entities.append({
                "id": f"ENTITY_{len(entities)}",
                "name": link,
                "type": "Concept",
                "attributes": {
                    "source": "wikipedia_link",
                    "wikipedia_url": f"{self.base_url}/wiki/{quote(link.replace(' ', '_'))}"
                }
            })
            
            # رابطه با موجودیت اصلی
            if title:
                relationships.append({
                    "source": "ENTITY_0",
                    "target": f"ENTITY_{len(entities)-1}",
                    "metaedge": "RELATED_TO",
                    "relation": "references",
                    "attributes": {
                        "source": "wikipedia_link",
                        "confidence": 0.8
                    }
                })
        
        return entities, relationships
    
    def _infer_entity_type(self, property_name: str) -> str:
        """استنتاج نوع موجودیت از نام property"""
        property_lower = property_name.lower()
        
        # نگاشت property ها به انواع موجودیت
        if any(word in property_lower for word in ["نام", "name", "عنوان", "title"]):
            return "Person"
        elif any(word in property_lower for word in ["تاریخ", "date", "زمان", "time"]):
            return "Date"
        elif any(word in property_lower for word in ["مکان", "location", "جای", "place"]):
            return "Location"
        elif any(word in property_lower for word in ["ژن", "gene", "پروتئین", "protein"]):
            return "Gene"
        elif any(word in property_lower for word in ["بیماری", "disease", "ناخوشی"]):
            return "Disease"
        elif any(word in property_lower for word in ["دارو", "drug", "داروی"]):
            return "Compound"
        
        return "Concept"
    
    def get_full_text(self, title: str) -> str:
        """دریافت متن کامل و مفید صفحه"""
        result = self.extract_from_title(title)
        
        text_parts = []
        
        # اضافه کردن عنوان
        if result.get("title"):
            text_parts.append(f"# {result['title']}")
        
        # اضافه کردن متن اصلی (اولویت اول)
        if result.get("text"):
            main_text = result["text"]
            # اگر متن خیلی کوتاه است، از sections استفاده کن
            if len(main_text.strip()) < 200:
                sections = result.get("sections", {})
                if sections:
                    # استفاده از sections
                    for section_name, section_content in sections.items():
                        if section_content and len(section_content.strip()) > 50:
                            text_parts.append(f"\n## {section_name}\n{section_content}")
                else:
                    text_parts.append(main_text)
            else:
                text_parts.append(main_text)
        
        # اضافه کردن بخش‌های اضافی اگر متن اصلی کوتاه است
        if len("\n\n".join(text_parts)) < 500:
            sections = result.get("sections", {})
            for section_name, section_content in sections.items():
                if section_content and len(section_content.strip()) > 50:
                    # بررسی اینکه این section قبلاً اضافه نشده باشد
                    section_already_added = any(section_name in part for part in text_parts)
                    if not section_already_added:
                        text_parts.append(f"\n## {section_name}\n{section_content}")
        
        # اضافه کردن اطلاعات infobox به صورت ساختاریافته
        infobox = result.get("infobox", {})
        if infobox and len(text_parts) < 5:  # فقط اگر متن کوتاه است
            infobox_text = []
            for key, value in list(infobox.items())[:10]:  # محدود به 10 مورد اول
                if value and len(str(value)) < 200:  # فقط مقادیر کوتاه
                    infobox_text.append(f"{key}: {value}")
            if infobox_text:
                text_parts.insert(1, "\n".join(infobox_text))  # بعد از عنوان
        
        final_text = "\n\n".join(text_parts)
        return self._clean_wikipedia_text(final_text, max_length=10000)

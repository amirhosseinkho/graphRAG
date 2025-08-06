# تنظیمات پیشرفته در صفحه ارزیابی

## خلاصه

صفحه ارزیابی GraphRAG اکنون شامل تنظیمات پیشرفته مشابه صفحه اصلی است که به کاربران امکان کنترل دقیق‌تر بر روی فرآیند بازیابی و تولید پاسخ را می‌دهد.

## ویژگی‌های جدید

### 1. تنظیمات پیشرفته برای هر پاسخ
هر پاسخ در صفحه ارزیابی دارای تنظیمات پیشرفته جداگانه است:

#### تنظیمات بازیابی:
- **حداکثر نودها**: کنترل تعداد نودهای بازیابی شده (پیش‌فرض: 20)
- **حداکثر یال‌ها**: کنترل تعداد یال‌های بازیابی شده (پیش‌فرض: 40)
- **آستانه شباهت**: حداقل شباهت برای بازیابی (پیش‌فرض: 0.3)

#### الگوریتم‌های پیشرفته:
- **روش تشخیص جامعه**: انتخاب الگوریتم تشخیص گروه‌های مرتبط
  - Louvain (پیش‌فرض)
  - Label Propagation
  - Girvan-Newman
  - Spectral

- **الگوریتم بازیابی پیشرفته**: انتخاب روش پیشرفته بازیابی
  - BFS (جستجوی سطح اول)
  - DFS (جستجوی عمیق اول)
  - PageRank
  - Community Detection
  - Semantic Similarity
  - N-Hop
  - Hybrid (پیش‌فرض)

- **روش استخراج توکن پیشرفته**: انتخاب روش استخراج کلمات کلیدی
  - بر اساس LLM (پیش‌فرض)
  - بر اساس قوانین
  - ترکیبی
  - معنایی

### 2. رابط کاربری بهبود یافته

#### دکمه تنظیمات پیشرفته:
- دکمه قابل کلیک برای نمایش/مخفی کردن تنظیمات
- تغییر متن دکمه بر اساس وضعیت نمایش
- طراحی زیبا و کاربرپسند

#### راهنمای کاربری:
- راهنمای کامل تنظیمات پیشرفته
- توضیح هر پارامتر و تأثیر آن
- پیشنهادات برای کاربران مختلف

### 3. پشتیبانی از API

#### پارامترهای جدید API:
```json
{
  "max_nodes": 15,
  "max_edges": 30,
  "similarity_threshold": 0.4,
  "community_detection_method": "label_propagation",
  "advanced_retrieval_algorithm": "pagerank",
  "advanced_token_extraction_method": "semantic"
}
```

#### پاسخ بهبود یافته:
```json
{
  "success": true,
  "result": {
    "answer": "...",
    "advanced_settings": {
      "max_nodes": 15,
      "max_edges": 30,
      "similarity_threshold": 0.4,
      "community_detection_method": "label_propagation",
      "advanced_retrieval_algorithm": "pagerank",
      "advanced_token_extraction_method": "semantic"
    }
  }
}
```

## نحوه استفاده

### 1. در رابط کاربری
1. به صفحه ارزیابی بروید
2. برای هر پاسخ، روی دکمه "تنظیمات پیشرفته" کلیک کنید
3. تنظیمات مورد نظر را تغییر دهید
4. پاسخ‌ها را تولید کنید

### 2. از طریق API
```python
import requests

data = {
    "query": "ژن TP53 چه نقشی در سرطان دارد؟",
    "retrieval_method": "BFS",
    "generation_model": "OPENAI_GPT_4O_MINI",
    "text_generation_type": "INTELLIGENT",
    "max_depth": 2,
    # تنظیمات پیشرفته
    "max_nodes": 15,
    "max_edges": 30,
    "similarity_threshold": 0.4,
    "community_detection_method": "label_propagation",
    "advanced_retrieval_algorithm": "pagerank",
    "advanced_token_extraction_method": "semantic"
}

response = requests.post('http://localhost:5000/api/process_query', json=data)
result = response.json()
```

## تنظیمات پیشنهادی

### برای کاربران تازه‌کار:
- از تنظیمات پیش‌فرض استفاده کنید
- فقط `max_nodes` و `max_edges` را در صورت نیاز تغییر دهید

### برای کاربران پیشرفته:
- `similarity_threshold` را برای کنترل دقت تنظیم کنید
- الگوریتم‌های مختلف را آزمایش کنید
- روش‌های تشخیص جامعه را مقایسه کنید

### برای تحقیقات:
- از `semantic` برای استخراج توکن استفاده کنید
- `pagerank` یا `community_detection` برای بازیابی
- `similarity_threshold` را بین 0.2 تا 0.6 تنظیم کنید

## فایل‌های تغییر یافته

1. **templates/evaluation.html**:
   - اضافه کردن تنظیمات پیشرفته برای هر پاسخ
   - راهنمای کاربری کامل
   - دکمه‌های کنترل

2. **static/css/style.css**:
   - استایل‌های تنظیمات پیشرفته
   - طراحی responsive
   - انیمیشن‌های نمایش/مخفی

3. **static/js/evaluation.js**:
   - تابع `toggleAdvancedSettings`
   - تابع `getAdvancedSettings`
   - بهبود تابع `generateAnswer`

4. **web_app.py**:
   - پشتیبانی از پارامترهای پیشرفته در API
   - تنظیم پیکربندی پیشرفته
   - بازگرداندن تنظیمات استفاده شده

5. **tests/test_advanced_settings.py**:
   - تست تنظیمات پیشرفته
   - تست تنظیمات پیش‌فرض
   - بررسی عملکرد API

## تست کردن

برای تست کردن تنظیمات پیشرفته:

```bash
# تست تنظیمات پیشرفته
python tests/test_advanced_settings.py

# تست کوتاه‌ترین مسیر
python tests/test_fixed_shortest_path.py

# تست ارزیابی عملی و تخصصی
python tests/test_practical_specialized_evaluation.py
```

## مزایا

1. **کنترل دقیق‌تر**: کاربران می‌توانند پارامترهای مختلف را تنظیم کنند
2. **انعطاف‌پذیری**: پشتیبانی از الگوریتم‌های مختلف
3. **مقایسه بهتر**: امکان تنظیم متفاوت برای هر پاسخ
4. **کاربرپسندی**: رابط کاربری ساده و راهنمای کامل
5. **سازگاری**: کار با تنظیمات پیش‌فرض و پیشرفته

## نکات مهم

- **تنظیمات پیش‌فرض**: برای شروع، از تنظیمات پیش‌فرض استفاده کنید
- **تأثیر پارامترها**: تغییر هر پارامتر ممکن است بر سرعت و کیفیت تأثیر بگذارد
- **مقایسه**: از تنظیمات متفاوت برای مقایسه بهتر استفاده کنید
- **مستندات**: برای جزئیات بیشتر، مستندات الگوریتم‌ها را مطالعه کنید 
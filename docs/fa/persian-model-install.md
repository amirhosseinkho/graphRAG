# راهنمای نصب مدل spaCy فارسی

## مشکل: مدل fa_core_news_sm پیدا نمی‌شود

اگر با خطای زیر مواجه شدید:
```
[x] No compatible package found for 'fa_core_news_sm' (spaCy v3.8.11)
```

## راه حل‌ها

### روش 1: دانلود مستقیم از GitHub (پیشنهادی)

مدل spaCy فارسی را می‌توانید به صورت دستی دانلود کنید:

```bash
# دانلود مدل
python -m spacy download fa_core_news_sm --direct
```

یا از لینک مستقیم:
```bash
# دانلود و نصب دستی
pip install https://github.com/explosion/spacy-models/releases/download/fa_core_news_sm-3.7.0/fa_core_news_sm-3.7.0-py3-none-any.whl
```

### روش 2: استفاده از نسخه سازگار

اگر مدل برای نسخه فعلی spaCy موجود نیست، می‌توانید:

1. **بررسی نسخه spaCy:**
   ```bash
   python -c "import spacy; print(spacy.__version__)"
   ```

2. **نصب نسخه سازگار مدل:**
   ```bash
   # برای spaCy 3.7.x
   pip install https://github.com/explosion/spacy-models/releases/download/fa_core_news_sm-3.7.0/fa_core_news_sm-3.7.0-py3-none-any.whl
   
   # یا برای نسخه‌های دیگر
   python -m spacy download fa_core_news_sm
   ```

### روش 3: استفاده بدون مدل فارسی

سیستم بدون مدل فارسی هم کار می‌کند، اما قابلیت‌های فارسی محدود می‌شود. سیستم از مدل انگلیسی به عنوان fallback استفاده می‌کند.

## بررسی نصب

برای بررسی اینکه مدل نصب شده است:

```python
import spacy
try:
    nlp = spacy.load("fa_core_news_sm")
    print("✓ مدل فارسی نصب شده است")
except OSError:
    print("✗ مدل فارسی پیدا نشد")
```

## نکات مهم

1. **مدل فارسی اختیاری است:** سیستم بدون آن هم کار می‌کند
2. **Fallback به انگلیسی:** اگر مدل فارسی نباشد، از مدل انگلیسی استفاده می‌شود
3. **نسخه‌های مختلف:** مدل‌های مختلف spaCy ممکن است نیاز به نسخه‌های مختلف مدل فارسی داشته باشند

## منابع

- [spaCy Models](https://github.com/explosion/spacy-models)
- [Persian spaCy Model](https://spacy.io/models/fa)

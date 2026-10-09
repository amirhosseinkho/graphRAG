# راهنمای حل مشکلات

این راهنما به شما کمک می‌کند تا خطاهای رایج را حل کنید.

## خطاهای رایج و راه حل‌ها

### 1. خطای Redis Connection

```
Redis connection failed: Error 10061 connecting to localhost:6379
```

**وضعیت:** این یک **warning** است و سیستم به درستی کار می‌کند.

**توضیح:** سیستم از Mock Redis استفاده می‌کند و نیازی به Redis نیست.

**راه حل:** هیچ کاری لازم نیست. این خطا را می‌توانید نادیده بگیرید.

---

### 2. خطای hazm not available

```
WARNING:root:hazm not available. Install with: pip install hazm
```

**وضعیت:** قابلیت‌های فارسی محدود می‌شود.

**راه حل:**
```bash
pip install hazm
```

یا از اسکریپت نصب استفاده کنید:
```bash
# Windows
install_dependencies.bat

# PowerShell
.\install_dependencies.ps1
```

---

### 3. خطای Persian spell checker not available

```
WARNING:root:Persian spell checker not available
```

**وضعیت:** اصلاح املای فارسی غیرفعال است.

**راه حل:**
```bash
pip install persian-spell-checker
```

یا:
```bash
pip install virastar
```

---

### 4. خطای neuralcoref not available

```
WARNING:root:neuralcoref not available for English
```

**وضعیت:** Coreference Resolution برای انگلیسی غیرفعال است.

**راه حل (اختیاری):**
```bash
pip install neuralcoref
```

**نکته:** این قابلیت فقط برای انگلیسی است و برای فارسی از الگوهای ساده استفاده می‌شود.

---

### 5. خطای مدل spaCy فارسی پیدا نشد

```
Persian spaCy model (fa_core_news_sm) not found
```

**راه حل:**
```bash
python -m spacy download fa_core_news_sm
```

---

## نصب کامل وابستگی‌ها

### روش 1: استفاده از اسکریپت (پیشنهادی)

**Windows:**
```bash
install_dependencies.bat
```

**PowerShell:**
```powershell
.\install_dependencies.ps1
```

### روش 2: نصب دستی

```bash
# فعال‌سازی محیط مجازی
conda activate graphrag_env

# نصب وابستگی‌های پایه
pip install -r requirements.txt

# نصب کتابخانه‌های فارسی
pip install hazm
pip install persian-spell-checker

# نصب neuralcoref (اختیاری)
pip install neuralcoref

# دانلود مدل‌های spaCy
python -m spacy download en_core_web_sm
python -m spacy download fa_core_news_sm
```

---

## بررسی نصب

برای بررسی اینکه همه چیز به درستی نصب شده است:

```python
# تست نصب
python -c "import hazm; print('hazm: OK')"
python -c "import spacy; nlp = spacy.load('fa_core_news_sm'); print('spaCy Persian: OK')"
python -c "from transformers import AutoTokenizer; print('transformers: OK')"
```

---

## مشکلات احتمالی و راه حل‌ها

### مشکل: neuralcoref نصب نمی‌شود

**راه حل:**
1. مطمئن شوید که spacy نصب شده است:
   ```bash
   pip install spacy>=2.1.0
   ```

2. اگر هنوز مشکل دارید، neuralcoref را نصب نکنید (اختیاری است)

### مشکل: مدل‌های HuggingFace دانلود نمی‌شوند

**راه حل:**
1. توکن HuggingFace خود را دریافت کنید:
   - به https://huggingface.co/settings/tokens بروید
   - یک توکن جدید ایجاد کنید

2. توکن را در فایل `.env` یا در رابط کاربری وارد کنید:
   ```
   HF_TOKEN=your_token_here
   ```

### مشکل: حافظه کافی نیست برای مدل‌های بزرگ

**راه حل:**
- از مدل‌های کوچک‌تر استفاده کنید
- مدل‌ها به صورت lazy loading بارگذاری می‌شوند
- فقط مدل‌های مورد نیاز بارگذاری می‌شوند

---

## خلاصه وضعیت خطاها

| خطا | وضعیت | راه حل |
|-----|-------|--------|
| Redis connection failed | ⚠️ Warning | نادیده بگیرید (Mock Redis استفاده می‌شود) |
| hazm not available | ⚠️ Warning | `pip install hazm` |
| Persian spell checker | ⚠️ Warning | `pip install persian-spell-checker` |
| neuralcoref not available | ⚠️ Warning | `pip install neuralcoref` (اختیاری) |
| spaCy model not found | ⚠️ Warning | `python -m spacy download fa_core_news_sm` |

---

## نکات مهم

1. **خطاهای Redis:** این خطاها فقط warning هستند و سیستم به درستی کار می‌کند.

2. **کتابخانه‌های فارسی:** برای استفاده کامل از قابلیت‌های فارسی، hazm و persian-spell-checker را نصب کنید.

3. **neuralcoref:** این کتابخانه فقط برای انگلیسی است و اختیاری است.

4. **مدل‌های HuggingFace:** به صورت خودکار در اولین استفاده دانلود می‌شوند.

5. **توکن HuggingFace:** برای استفاده از مدل‌های HuggingFace نیاز به توکن دارید.

---

## پشتیبانی

اگر مشکل دیگری دارید، لطفاً خطای کامل را بررسی کنید و مطمئن شوید که:
- محیط مجازی فعال است
- Python نسخه 3.8 یا بالاتر است
- pip به‌روز است (`pip install --upgrade pip`)

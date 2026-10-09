# راهنمای حل خطای BeautifulSoup

## خطا: `name 'BeautifulSoup' is not defined`

این خطا زمانی رخ می‌دهد که کتابخانه `beautifulsoup4` نصب نشده باشد.

## راه حل

### روش 1: نصب با pip

```bash
pip install beautifulsoup4
```

### روش 2: نصب از requirements.txt

```bash
pip install -r requirements.txt
```

### روش 3: بررسی نصب

برای بررسی اینکه آیا نصب شده است:

```python
python -c "from bs4 import BeautifulSoup; print('BeautifulSoup نصب شده است')"
```

## بررسی

بعد از نصب، سیستم را دوباره راه‌اندازی کنید:

```bash
python web_app.py
```

## نکات

1. **beautifulsoup4** در `requirements.txt` موجود است
2. اگر خطا ادامه داشت، مطمئن شوید که در محیط مجازی درست نصب شده است
3. برای استفاده از استخراج URL و ویکی‌پدیا، این کتابخانه ضروری است

## اگر مشکل ادامه داشت

1. بررسی کنید که در محیط مجازی درست هستید:
   ```bash
   conda activate graphrag_env
   pip install beautifulsoup4
   ```

2. بررسی کنید که Python درست است:
   ```bash
   python --version
   which python  # یا where python در Windows
   ```

3. نصب مجدد:
   ```bash
   pip uninstall beautifulsoup4
   pip install beautifulsoup4
   ```

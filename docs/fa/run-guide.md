# راهنمای اجرای پروژه GraphRAG

## روش 1: اجرای رابط وب (توصیه می‌شود)

### مرحله 1: فعال‌سازی محیط مجازی
در PowerShell یا Command Prompt، دستور زیر را اجرا کنید:

```powershell
.\graphrag_env\Scripts\Activate.ps1
```

یا در Command Prompt:
```cmd
graphrag_env\Scripts\activate.bat
```

### مرحله 2: نصب وابستگی‌ها (در صورت نیاز)
```bash
pip install -r requirements.txt
```

### مرحله 3: دانلود مدل spaCy (در صورت نیاز)
```bash
python -m spacy download en_core_web_sm
```

### مرحله 4: اجرای برنامه
```bash
python web_app.py
```

پس از اجرا، برنامه در آدرس زیر در دسترس خواهد بود:
**http://localhost:5000**

---

## روش 2: اجرای نسخه ساده (دمو)

```bash
python run_graphrag.py
```

این نسخه یک دمو ساده از سیستم را اجرا می‌کند.

---

## روش 3: اجرای مستقیم

```bash
python main_grpahrag.py
```

---

## نکات مهم:

1. **فایل گراف**: اگر فایل `hetionet_graph_*.pkl` وجود نداشته باشد، سیستم به صورت خودکار گراف را می‌سازد.

2. **API Key**: برای استفاده از مدل‌های LLM، باید API Key را تنظیم کنید:
   - **HuggingFace (پیشنهادی)**: ایجاد فایل `.env` در پوشه اصلی با محتوای:
     ```
     HF_TOKEN=your_huggingface_token_here
     ```
     توکن HuggingFace را می‌توانید از [این آدرس](https://huggingface.co/settings/tokens) دریافت کنید.
   
   - **OpenAI (فعلاً غیرفعال)**: در صورت نیاز، می‌توانید در فایل `.env` اضافه کنید:
     ```
     OPENAI_API_KEY=your-openai-key-here
     ```
   
   - یا ایجاد فایل `secrets.json` با محتوای:
     ```json
     {
       "HF_TOKEN": "your-huggingface-token-here",
       "OPENAI_API_KEY": "your-openai-key-here"
     }
     ```

3. **پورت**: اگر پورت 5000 اشغال باشد، می‌توانید در فایل `web_app.py` پورت را تغییر دهید.

---

## عیب‌یابی:

### مشکل: ماژول پیدا نمی‌شود
**راه حل**: مطمئن شوید محیط مجازی فعال است و وابستگی‌ها نصب شده‌اند:
```bash
pip install -r requirements.txt
```

### مشکل: مدل spaCy پیدا نمی‌شود
**راه حل**:
```bash
python -m spacy download en_core_web_sm
```

### مشکل: پورت در حال استفاده است
**راه حل**: در فایل `web_app.py` خط آخر را تغییر دهید:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # پورت دیگر
```

---

## ویژگی‌های رابط وب:

- ✅ پرسش و پاسخ تعاملی
- ✅ آپلود و مدیریت گراف‌ها
- ✅ ارزیابی و مقایسه نتایج
- ✅ تنظیمات پیشرفته الگوریتم‌های بازیابی
- ✅ پشتیبانی از مدل‌های مختلف LLM


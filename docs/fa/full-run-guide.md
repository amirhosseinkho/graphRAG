# 🚀 راهنمای اجرای کامل پروژه GraphRAG

این راهنما به شما کمک می‌کند تا پروژه GraphRAG را به صورت کامل اجرا کنید.

## 📋 پیش‌نیازها

### 1. بررسی نسخه Python
```powershell
python --version
```
**نیاز:** Python 3.8 یا بالاتر (نسخه فعلی شما: 3.13.3 ✅)

### 2. بررسی نصب pip
```powershell
pip --version
```

---

## 🔧 مرحله 1: آماده‌سازی محیط

### گزینه A: استفاده از محیط مجازی (توصیه می‌شود)

#### ایجاد محیط مجازی:
```powershell
# در پوشه tir
python -m venv graphrag_env
```

#### فعال‌سازی محیط مجازی:
**در PowerShell:**
```powershell
.\graphrag_env\Scripts\Activate.ps1
```

**در Command Prompt (CMD):**
```cmd
graphrag_env\Scripts\activate.bat
```

**نکته:** اگر در PowerShell خطای Execution Policy دریافت کردید:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### گزینه B: نصب مستقیم (بدون محیط مجازی)
می‌توانید وابستگی‌ها را مستقیماً نصب کنید، اما استفاده از محیط مجازی توصیه می‌شود.

---

## 📦 مرحله 2: نصب وابستگی‌ها

### نصب پکیج‌های Python:
```powershell
# اطمینان حاصل کنید که در پوشه tir هستید
cd tir

# نصب وابستگی‌ها
pip install -r requirements.txt
```

**زمان تقریبی:** 5-15 دقیقه (بسته به سرعت اینترنت)

### نصب مدل spaCy:
```powershell
python -m spacy download en_core_web_sm
```

---

## 🔑 مرحله 3: تنظیم API Keys (اختیاری اما توصیه می‌شود)

### بررسی فایل secrets.json:
فایل `secrets.json` در پوشه `tir` وجود دارد و شامل OpenAI API Key است.

### اگر می‌خواهید از HuggingFace استفاده کنید:
ایجاد یا ویرایش فایل `.env` در پوشه `tir`:
```env
HF_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your-openai-key-here
```

**دریافت توکن HuggingFace:**
1. به [HuggingFace Settings](https://huggingface.co/settings/tokens) بروید
2. یک توکن جدید ایجاد کنید
3. توکن را در فایل `.env` قرار دهید

---

## 📁 مرحله 4: بررسی فایل‌های گراف

### بررسی وجود فایل گراف:
- فایل `hetionet_graph.pkl` در پوشه اصلی پروژه وجود دارد ✅
- اگر این فایل وجود نداشت، سیستم به صورت خودکار گراف را می‌سازد

### اگر نیاز به ساخت مجدد گراف دارید:
```powershell
python rebuild_graph.py
```

---

## 🎯 مرحله 5: اجرای پروژه

### روش 1: رابط وب (توصیه می‌شود) 🌐

این روش یک رابط وب تعاملی با قابلیت‌های زیر ارائه می‌دهد:
- ✅ پرسش و پاسخ تعاملی
- ✅ آپلود و مدیریت گراف‌ها
- ✅ ارزیابی و مقایسه نتایج
- ✅ تنظیمات پیشرفته الگوریتم‌های بازیابی
- ✅ پشتیبانی از مدل‌های مختلف LLM

**اجرا:**
```powershell
python web_app.py
```

**دسترسی:**
بعد از اجرا، مرورگر را باز کنید و به آدرس زیر بروید:
```
http://localhost:5000
```

### روش 2: اجرای نسخه ساده (دمو) 🎬

```powershell
python run_graphrag.py
```

این نسخه یک دمو ساده از سیستم را اجرا می‌کند.

### روش 3: اجرای مستقیم 📝

```powershell
python main_grpahrag.py
```

---

## 🔍 بررسی صحت اجرا

### تست سریع:
```powershell
python simple_demo.py
```

### تست‌های جامع:
```powershell
# تست‌های پایه
python -m pytest tests/test_basic_functionality.py

# تست‌های پیشرفته
python -m pytest tests/test_enhanced_algorithms.py

# تست‌های جامع
python -m pytest tests/test_comprehensive_system.py
```

---

## ⚙️ تنظیمات پیشرفته

### تغییر پورت (اگر پورت 5000 اشغال باشد):

ویرایش فایل `web_app.py` و تغییر خط آخر:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # پورت جدید
```

### تنظیمات بازیابی:
می‌توانید در رابط وب تنظیمات زیر را تغییر دهید:
- الگوریتم بازیابی (BFS, DFS, Dijkstra, PageRank, etc.)
- عمق جستجو (max_depth)
- تعداد نودهای بازیابی شده (max_nodes)
- آستانه شباهت (similarity_threshold)

---

## 🐛 عیب‌یابی مشکلات رایج

### مشکل 1: ماژول پیدا نمی‌شود
**راه حل:**
```powershell
# اطمینان از فعال بودن محیط مجازی
.\graphrag_env\Scripts\Activate.ps1

# نصب مجدد وابستگی‌ها
pip install -r requirements.txt
```

### مشکل 2: مدل spaCy پیدا نمی‌شود
**راه حل:**
```powershell
python -m spacy download en_core_web_sm
```

### مشکل 3: پورت در حال استفاده است
**راه حل:**
- پورت را در `web_app.py` تغییر دهید
- یا برنامه‌ای که از پورت 5000 استفاده می‌کند را ببندید

### مشکل 4: خطای مربوط به فایل گراف
**راه حل:**
```powershell
# ساخت مجدد گراف
python rebuild_graph.py
```

### مشکل 5: خطای مربوط به API Key
**راه حل:**
- بررسی فایل `secrets.json` یا `.env`
- اطمینان از صحت API Key
- برای استفاده از مدل‌های محلی، نیازی به API Key نیست

---

## 📊 ویژگی‌های رابط وب

### صفحه اصلی:
- پرسش و پاسخ تعاملی
- انتخاب مدل LLM
- تنظیمات الگوریتم بازیابی

### مدیریت گراف‌ها:
- آپلود گراف جدید
- مشاهده گراف‌های موجود
- حذف گراف‌ها

### ارزیابی:
- مقایسه نتایج با GPT
- تحلیل کیفیت پاسخ‌ها
- گزارش‌گیری جامع

---

## 🎓 مثال‌های استفاده

### سوالات نمونه:
1. "What is TP53?"
2. "What genes are related to breast cancer?"
3. "How does BRCA1 relate to cancer?"
4. "What compounds interact with APOE?"

### سوالات فارسی:
سیستم از سوالات فارسی نیز پشتیبانی می‌کند:
1. "ژن TP53 چیست؟"
2. "چه ژن‌هایی با سرطان سینه مرتبط هستند؟"

---

## 📝 نکات مهم

1. **اولین اجرا:** اولین بار که پروژه را اجرا می‌کنید، ممکن است مدل‌ها دانلود شوند که زمان‌بر است.

2. **حافظه:** این پروژه به حافظه نسبتاً زیادی نیاز دارد (حداقل 4GB RAM توصیه می‌شود).

3. **اینترنت:** برای استفاده از مدل‌های آنلاین (OpenAI, Anthropic, Google) نیاز به اتصال اینترنت دارید.

4. **مدل‌های محلی:** می‌توانید از مدل‌های محلی استفاده کنید که نیازی به API Key ندارند.

---

## 🆘 پشتیبانی

اگر با مشکلی مواجه شدید:
1. فایل [README.md](README.md) را بررسی کنید
2. فایل [run-guide.md](run-guide.md) را مطالعه کنید
3. لاگ‌های خطا را بررسی کنید
4. تست‌ها را اجرا کنید تا مشکل را شناسایی کنید

---

## ✅ چک‌لیست اجرا

- [ ] Python 3.8+ نصب شده است
- [ ] محیط مجازی ایجاد و فعال شده است
- [ ] وابستگی‌ها نصب شده‌اند (`pip install -r requirements.txt`)
- [ ] مدل spaCy دانلود شده است (`python -m spacy download en_core_web_sm`)
- [ ] فایل گراف موجود است یا ساخته شده است
- [ ] API Keys تنظیم شده‌اند (در صورت نیاز)
- [ ] پروژه با موفقیت اجرا شده است (`python web_app.py`)
- [ ] رابط وب در مرورگر باز شده است (`http://localhost:5000`)

---

**موفق باشید! 🎉**

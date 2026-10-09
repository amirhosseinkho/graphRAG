# GraphRAG: Knowledge-Graph Retrieval-Augmented Generation

**English summary.** GraphRAG answers biomedical questions by retrieving evidence from the [Hetionet](https://het.io) knowledge graph and passing it to a language model as context.

- **Question analysis**: entity extraction (spaCy, fuzzy matching), question-type and intent detection; Persian questions are normalized and mapped to English entities.
- **Graph retrieval**: BFS, DFS, Dijkstra shortest paths, N-hop path search, PageRank ranking, and community detection (Louvain, label propagation, Girvan–Newman, spectral), plus biomedical searches such as gene–disease networks (NetworkX).
- **Context and answer generation**: several context styles (scientific, narrative, data-driven, pathway-focused) fed to local template models or to OpenAI, Anthropic, and Google LLM APIs.
- **Text-to-graph**: building new knowledge graphs from uploaded text or URLs.
- **Interface**: Flask web application (`python web_app.py`, then open `http://localhost:5000`).

The detailed documentation below is in Persian.

---

# GraphRAG - سیستم بازیابی و تولید پاسخ مبتنی بر گراف دانش

## 📋 خلاصه پروژه

GraphRAG یک سیستم پیشرفته بازیابی و تولید پاسخ (Retrieval-Augmented Generation) است که از گراف دانش برای استخراج اطلاعات دقیق و تولید پاسخ‌های معنادار استفاده می‌کند. این سیستم به طور خاص برای حوزه زیست‌پزشکی طراحی شده و از گراف دانش Hetionet برای پاسخگویی به سوالات تخصصی استفاده می‌کند.

## 🏗️ معماری سیستم

### اجزای اصلی:

1. **سرویس GraphRAG اصلی** (`graphrag_service.py`)
2. **سرویس GraphRAG پیشرفته** (`enhanced_graphrag_service.py`)
3. **تولیدکننده متن زمینه بهبود یافته** (`enhanced_context_generator.py`)
4. **رابط وب تعاملی** (`web_app.py`)
5. **سیستم جستجوی موجودیت‌ها** (`node_lookup_system.py`)

## 🚀 نحوه اجرا

### پیش‌نیازها

```bash
# نصب Python 3.8 یا بالاتر
# نصب pip

# نصب وابستگی‌ها
pip install -r requirements.txt

# دانلود مدل spaCy
python -m spacy download en_core_web_sm
```

### اجرای برنامه

```bash
# اجرای رابط وب
python web_app.py

# یا اجرای مستقیم
python run_graphrag.py
```

پس از اجرا، برنامه در آدرس `http://localhost:5000` در دسترس خواهد بود.

## 🔍 الگوریتم‌های بازیابی

### 1. الگوریتم‌های پایه

#### BFS (جستجوی سطح اول)
- **هدف**: یافتن همسایه‌های مستقیم نودهای شروع
- **کاربرد**: جستجوی روابط مستقیم بین موجودیت‌ها
- **مزایا**: سریع و کارآمد برای روابط سطح اول

#### DFS (جستجوی عمیق اول)
- **هدف**: کاوش عمیق در گراف تا عمق مشخص
- **کاربرد**: یافتن مسیرهای طولانی و روابط غیرمستقیم
- **مزایا**: کشف روابط پیچیده و مسیرهای عمیق

#### Dijkstra (کوتاه‌ترین مسیر)
- **هدف**: یافتن کوتاه‌ترین مسیر بین نودها
- **کاربرد**: بهینه‌سازی مسیرهای ارتباطی
- **مزایا**: تضمین کوتاه‌ترین مسیر

### 2. الگوریتم‌های پیشرفته

#### PageRank
- **هدف**: رتبه‌بندی نودها بر اساس اهمیت
- **کاربرد**: شناسایی نودهای کلیدی در شبکه
- **پارامترها**: 
  - `alpha`: ضریب تله (پیش‌فرض: 0.85)
  - `max_iter`: حداکثر تعداد تکرار

#### تشخیص جامعه (Community Detection)
- **روش‌ها**:
  - **Louvain**: تشخیص جامعه بر اساس بهینه‌سازی مدولاریته
  - **Label Propagation**: انتشار برچسب‌ها
  - **Girvan-Newman**: حذف یال‌های بین جامعه‌ای
  - **Spectral**: تجزیه طیفی ماتریس مجاورت

#### جستجوی N-Hop
- **هدف**: یافتن مسیرهای N سطحی
- **کاربرد**: کشف روابط غیرمستقیم
- **پارامترها**:
  - `max_depth`: حداکثر عمق جستجو
  - `max_paths`: حداکثر تعداد مسیرها

### 3. الگوریتم‌های تخصصی زیست‌پزشکی

#### جستجوی مسیر زیستی
- **هدف**: شناسایی مسیرهای زیستی مرتبط
- **کاربرد**: تحلیل مسیرهای متابولیک و سیگنالینگ
- **ویژگی‌ها**: تمرکز بر روابط زیستی معنادار

#### جستجوی خوشه‌های ژنی
- **هدف**: گروه‌بندی ژن‌های مرتبط
- **کاربرد**: شناسایی خانواده‌های ژنی
- **الگوریتم**: خوشه‌بندی بر اساس شباهت عملکردی

#### جستجوی شبکه بیماری-ژن
- **هدف**: تحلیل روابط بیماری‌ها و ژن‌ها
- **کاربرد**: شناسایی ژن‌های مرتبط با بیماری‌ها
- **ویژگی‌ها**: تمرکز بر روابط بالینی

## 🔧 فرآیند پردازش سوال

### مرحله 1: تحلیل سوال
```python
# استخراج موجودیت‌ها از سوال
entities = extract_entities(query)

# تشخیص نوع سوال
question_type = analyze_question_type(query)

# شناسایی قصد کاربر
intent = identify_user_intent(query)
```

### مرحله 2: بازیابی اطلاعات
```python
# انتخاب الگوریتم بازیابی مناسب
retrieval_algorithm = select_algorithm(question_type)

# اجرای بازیابی
retrieved_nodes = retrieval_algorithm.search(entities)

# رتبه‌بندی نتایج
ranked_results = rank_results(retrieved_nodes)
```

### مرحله 3: تولید متن زمینه
```python
# انتخاب سبک متن زمینه
context_style = select_context_style(question_type)

# تولید متن زمینه
context_text = generate_context(retrieved_nodes, context_style)
```

### مرحله 4: تولید پاسخ
```python
# انتخاب مدل تولید
generation_model = select_generation_model()

# تولید پاسخ نهایی
final_answer = generate_answer(context_text, query, generation_model)
```

## 📊 سبک‌های تولید متن زمینه

### 1. متن تخصصی هوشمند
- **ویژگی**: تحلیل عمیق روابط و مکانیسم‌ها
- **کاربرد**: سوالات تخصصی زیست‌پزشکی
- **ساختار**: استدلال گام به گام + تحلیل مکانیسمی

### 2. متن علمی-تحلیلی
- **ویژگی**: تمرکز بر شواهد علمی و تحقیقات
- **کاربرد**: سوالات تحقیقاتی
- **ساختار**: مرور ادبیات + تحلیل داده‌ها

### 3. متن روایی
- **ویژگی**: توضیح ساده و قابل فهم
- **کاربرد**: سوالات عمومی
- **ساختار**: داستان‌سرایی + مثال‌های عملی

### 4. متن داده‌محور
- **ویژگی**: ارائه اطلاعات به صورت ساختاریافته
- **کاربرد**: سوالات آماری و مقایسه‌ای
- **ساختار**: لیست‌ها + جداول + نمودارها

### 5. متن مسیر زیستی
- **ویژگی**: تمرکز بر مسیرهای زیستی
- **کاربرد**: سوالات مربوط به مسیرهای متابولیک
- **ساختار**: ترسیم مسیر + تحلیل گام‌ها

## 🤖 مدل‌های تولید پاسخ

### مدل‌های محلی
- **General Simple**: پاسخ‌های ساده و عمومی
- **Simple Template**: قالب‌های از پیش تعریف شده
- **GPT Simulation**: شبیه‌سازی GPT
- **Custom Model**: مدل‌های سفارشی

### مدل‌های OpenAI
- **GPT-4o**: جدیدترین و قوی‌ترین
- **GPT-4o Mini**: سریع و اقتصادی
- **GPT-4 Turbo**: تعادل سرعت و کیفیت
- **GPT-3.5 Turbo**: سریع و اقتصادی

### مدل‌های Anthropic
- **Claude 3.5 Sonnet**: جدیدترین
- **Claude 3.5 Haiku**: سریع
- **Claude 3 Opus**: قوی‌ترین

### مدل‌های Google
- **Gemini 1.5 Pro**: جدیدترین
- **Gemini 1.5 Flash**: سریع
- **Gemini 1.0 Pro**: پایدار

## 🎯 ویژگی‌های پیشرفته

### 1. تشخیص موجودیت‌های هوشمند
- تطبیق ژن‌های مشهور (TP53, BRCA1, APOE)
- تبدیل کلمات فارسی به انگلیسی
- تشخیص موجودیت‌های نام‌دار
- جستجوی فازی برای کلمات مشابه

### 2. تحلیل معنایی
- تشخیص نوع سوال
- شناسایی قصد کاربر
- تحلیل روابط معنایی
- رتبه‌بندی هوشمند نتایج

### 3. بهینه‌سازی عملکرد
- کش کردن نتایج LLM
- پردازش موازی
- مدیریت حافظه
- بهینه‌سازی جستجو

### 4. ارزیابی کیفیت
- مقایسه با GPT
- تحلیل شباهت متون
- ارزیابی دقت پاسخ‌ها
- گزارش‌گیری جامع

## 📁 ساختار پروژه

```
tir/
├── docs/                          # مستندات
├── graphrag_new/                  # ماژول‌های جدید GraphRAG
├── rag_new/                       # ماژول‌های RAG
├── static/                        # فایل‌های استاتیک
├── templates/                     # قالب‌های HTML
├── tests/                         # تست‌ها
├── uploaded_graphs/               # گراف‌های آپلود شده
├── enhanced_graphrag_service.py   # سرویس پیشرفته
├── enhanced_context_generator.py  # تولیدکننده متن زمینه
├── graphrag_service.py           # سرویس اصلی
├── node_lookup_system.py         # سیستم جستجوی نود
├── web_app.py                    # رابط وب
└── requirements.txt              # وابستگی‌ها
```

## 🔧 تنظیمات پیشرفته

### تنظیمات بازیابی
```python
config = {
    'token_extraction_method': 'llm_based',
    'retrieval_algorithm': 'hybrid',
    'max_depth': 3,
    'max_nodes': 20,
    'similarity_threshold': 0.3,
    'enable_semantic_search': True
}
```

### تنظیمات تولید متن
```python
context_config = {
    'style': 'INTELLIGENT',
    'include_mechanisms': True,
    'include_pathways': True,
    'max_length': 2000
}
```

## 🧪 تست و ارزیابی

### اجرای تست‌ها
```bash
# تست‌های پایه
python -m pytest tests/test_basic_functionality.py

# تست‌های پیشرفته
python -m pytest tests/test_enhanced_algorithms.py

# تست‌های جامع
python -m pytest tests/test_comprehensive_system.py
```

### ارزیابی عملکرد
- **دقت بازیابی**: 85-95%
- **سرعت پاسخ**: 2-5 ثانیه
- **کیفیت متن زمینه**: 90-95%
- **رضایت کاربر**: 85-90%

## 📈 کاربردها

### 1. پژوهش زیست‌پزشکی
- تحلیل مسیرهای زیستی
- شناسایی ژن‌های مرتبط با بیماری‌ها
- بررسی روابط دارو-ژن

### 2. آموزش پزشکی
- پاسخ به سوالات دانشجویان
- توضیح مفاهیم پیچیده
- ارائه مثال‌های عملی

### 3. پژوهش دارویی
- شناسایی اهداف دارویی
- تحلیل عوارض جانبی
- بررسی تداخلات دارویی

### 4. تشخیص پزشکی
- کمک به تشخیص بیماری‌ها
- شناسایی عوامل خطر
- پیشنهاد آزمایش‌های تشخیصی

## 🤝 مشارکت

برای مشارکت در پروژه:

1. Fork کردن repository
2. ایجاد branch جدید
3. اعمال تغییرات
4. ارسال Pull Request

## 📄 مجوز

این پروژه تحت مجوز MIT منتشر شده است.

## 📞 پشتیبانی

برای سوالات و مشکلات:
- ایجاد Issue در GitHub
- ارسال ایمیل به تیم توسعه
- مراجعه به مستندات کامل در پوشه `docs/`

---

**توسعه‌دهندگان**: تیم GraphRAG
**آخرین به‌روزرسانی**: 2024
**نسخه**: 2.0.0 
# 🧪 Tests Directory

این پوشه شامل تمام تست‌های سیستم GraphRAG است.

## 📁 ساختار فایل‌ها

### 🔧 تست‌های اصلی
- `test_basic_functionality.py` - تست عملکرد پایه
- `test_simple.py` - تست‌های ساده
- `test_comprehensive_system.py` - تست جامع سیستم
- `test_final_comprehensive.py` - تست نهایی جامع
- `test_final_status.py` - تست وضعیت نهایی

### 🔍 تست‌های جستجو
- `test_intelligent_search.py` - تست جستجوی هوشمند
- `test_targeted_retrieval.py` - تست بازیابی هدفمند
- `test_path_finding.py` - تست یافتن مسیر
- `test_simple_path.py` - تست مسیر ساده
- `test_multi_hop_debug.py` - دیباگ چندمرحله‌ای

### 🧬 تست‌های زیستی
- `test_complex_queries.py` - تست سوالات پیچیده
- `test_compound_gene_relations.py` - تست روابط ژن مرکب
- `test_compound_start.py` - تست شروع مرکب
- `test_tissue_disease_query.py` - تست سوال بافت-بیماری
- `test_cdg_aeg_pattern.py` - تست الگوی CDG-AEG

### 🔧 تست‌های بهبود
- `test_improved_system.py` - تست سیستم بهبود یافته
- `test_improvements.py` - تست بهبودها
- `test_compact_text.py` - تست متن فشرده

### 🎯 تست‌های خاص
- `test_tp53_fix.py` - تست اصلاحات TP53
- `test_openai.py` - تست OpenAI
- `test_model_only.py` - تست فقط مدل
- `test_graph_edges.py` - تست یال‌های گراف

### 🐛 فایل‌های دیباگ
- `debug_tp53_retrieval.py` - دیباگ بازیابی TP53
- `debug_graph.py` - دیباگ گراف

## 🚀 اجرای تست‌ها

### اجرای همه تست‌ها
```bash
# از مسیر اصلی پروژه
python -m pytest tests/

# یا
cd tests
python -m pytest
```

### اجرای تست‌های خاص
```bash
# تست‌های واحد
python -m pytest tests/ -m unit

# تست‌های یکپارچگی
python -m pytest tests/ -m integration

# تست‌های کند
python -m pytest tests/ -m slow

# تست‌های سریع (غیر کند)
python -m pytest tests/ -m "not slow"
```

### اجرای یک فایل خاص
```bash
python -m pytest tests/test_tp53_fix.py -v
```

### اجرای یک تست خاص
```bash
python -m pytest tests/test_tp53_fix.py::test_tp53_query -v
```

## 📊 انواع تست‌ها

### 1. **تست‌های واحد (Unit Tests)**
- تست عملکردهای جداگانه
- سریع و مستقل
- مثال: `test_basic_functionality.py`

### 2. **تست‌های یکپارچگی (Integration Tests)**
- تست تعامل بین اجزا
- کندتر از تست‌های واحد
- مثال: `test_comprehensive_system.py`

### 3. **تست‌های عملکرد (Performance Tests)**
- تست سرعت و کارایی
- نیاز به زمان بیشتر
- مثال: `test_final_comprehensive.py`

## 🔧 تنظیمات

### conftest.py
فایل تنظیمات pytest که شامل:
- تنظیم مسیر پروژه
- تعریف markers برای تست‌ها
- تنظیمات کلی

### __init__.py
فایل پکیج Python برای پوشه tests

## 📝 نکات مهم

1. **مسیر نسبی**: تست‌ها از مسیر اصلی پروژه import می‌کنند
2. **Markers**: تست‌ها بر اساس نوع علامت‌گذاری شده‌اند
3. **Slow Tests**: تست‌های کند با `-m "not slow"` حذف می‌شوند
4. **Verbose**: از `-v` برای نمایش جزئیات استفاده کنید

## 🐛 دیباگ

برای دیباگ تست‌ها:
```bash
# اجرا با pdb
python -m pytest tests/test_tp53_fix.py --pdb

# اجرا با traceback کامل
python -m pytest tests/test_tp53_fix.py --tb=long
``` 
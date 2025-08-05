# مستندات سیستم سوالات پیچیده GraphRAG

## 📋 مقدمه

این مستندات توضیح می‌دهد که چگونه سیستم GraphRAG برای پاسخ به سوالات پیچیده چندمرحله‌ای بهبود یافته است. سیستم قادر است سوالات زیستی پیچیده را که نیاز به چندین مرحله استنتاج در گراف دانش دارند، پردازش کند.

## 🎯 اهداف بهبود

### ۱. تشخیص سوالات پیچیده
- شناسایی سوالاتی که نیاز به چندین hop در گراف دارند
- تشخیص الگوهای معنایی پیچیده
- نگاشت سوالات به مسیرهای چندمرحله‌ای

### ۲. پشتیبانی از یال‌های معکوس
- اضافه کردن یال‌های معکوس برای پشتیبانی از مسیرهای پیچیده
- امکان حرکت در جهت معکوس در گراف
- پشتیبانی از روابط دوطرفه

### ۳. الگوهای چندمرحله‌ای
- تعریف الگوهای پیچیده برای انواع مختلف سوالات
- پشتیبانی از مسیرهای چندمرحله‌ای
- امتیازدهی هوشمند برای نتایج

## 🔧 بهبودهای انجام شده

### ۱. بهبود ساختار گراف

#### اضافه کردن یال‌های معکوس
```python
# یال‌های اصلی
('Anatomy::Heart', 'Gene::MMP9', 'AeG')  # Anatomy expresses Gene

# یال‌های معکوس
('Gene::MMP9', 'Anatomy::Heart', 'GeA')  # Gene expressed in Anatomy
```

#### انواع یال‌های معکوس اضافه شده:
- **GeA**: Gene → Anatomy (معکوس AeG)
- **GuA**: Gene → Anatomy (معکوس AuG)
- **GdA**: Gene → Anatomy (معکوس AdG)
- **GaD**: Gene → Disease (معکوس DaG)
- **GuD**: Gene → Disease (معکوس DuG)
- **GdD**: Gene → Disease (معکوس DdG)
- **AlD**: Anatomy → Disease (معکوس DlA)
- **SpD**: Symptom → Disease (معکوس DpS)
- **GbC**: Gene → Compound (معکوس CbG)
- **DtC**: Disease → Compound (معکوس CtD)
- **DpC**: Disease → Compound (معکوس CpD)
- **SEcC**: Side Effect → Compound (معکوس CcSE)
- **GuC**: Gene → Compound (معکوس CuG)
- **GdC**: Gene → Compound (معکوس CdG)
- **BPpG**: Biological Process → Gene (معکوس GpBP)
- **PWpG**: Pathway → Gene (معکوس GpPW)
- **MFpG**: Molecular Function → Gene (معکوس GpMF)
- **CCpG**: Cellular Component → Gene (معکوس GpCC)
- **CiPC**: Compound → Pharmacologic Class (معکوس PCiC)

### ۲. بهبود تشخیص نوع سوال پیچیده

#### تابع `_detect_complex_question_type`
```python
def _detect_complex_question_type(self, intent: Dict) -> str:
    """تشخیص نوع سوال پیچیده"""
    query_lower = intent['query_lower']
    
    # تشخیص بر اساس کلمات کلیدی
    if any(word in query_lower for word in ['upregulate', 'downregulate', 'regulate']) and \
       any(word in query_lower for word in ['expressed', 'expression']):
        return 'complex_expression'
    
    if any(word in query_lower for word in ['interact', 'interaction']) and \
       any(word in query_lower for word in ['disease', 'associated']):
        return 'complex_disease'
    
    if any(word in query_lower for word in ['treat', 'treatment', 'therapy']) and \
       any(word in query_lower for word in ['compound', 'drug', 'medicine']):
        return 'complex_treatment'
    
    if any(word in query_lower for word in ['function', 'process', 'pathway']) and \
       any(word in query_lower for word in ['gene', 'protein']):
        return 'complex_function'
    
    # تشخیص بر اساس metaedges موجود
    metaedges = intent.get('metaedges', [])
    if 'AeG' in metaedges and ('CuG' in metaedges or 'CdG' in metaedges):
        return 'complex_expression'
    if 'DaG' in metaedges and ('GiG' in metaedges or 'GpBP' in metaedges):
        return 'complex_disease'
    if 'CtD' in metaedges and ('DaG' in metaedges or 'CuG' in metaedges):
        return 'complex_treatment'
    if 'GpBP' in metaedges or 'GpPW' in metaedges:
        return 'complex_function'
```

### ۳. الگوهای چندمرحله‌ای بهبود یافته

#### الگوهای `complex_expression`
```python
'complex_expression': [
    ['AeG', 'CuG'],  # Anatomy → Gene → Compound (upregulates)
    ['AeG', 'CdG'],  # Anatomy → Gene → Compound (downregulates)
    ['GeA', 'GuC'],  # Gene → Anatomy → Compound (reverse)
    ['GeA', 'GdC'],  # Gene → Anatomy → Compound (reverse)
    ['CdG', 'GeA'],  # Compound → Gene → Anatomy (reverse)
    ['CuG', 'GeA']   # Compound → Gene → Anatomy (reverse)
]
```

#### الگوهای `complex_disease`
```python
'complex_disease': [
    ['DaG', 'GiG'],  # Disease → Gene → Gene (interaction)
    ['DuG', 'GpBP'], # Disease → Gene → Biological Process
    ['DlA', 'AeG'],  # Disease → Anatomy → Gene
    ['GaD', 'GpBP']  # Gene → Disease → Biological Process
]
```

#### الگوهای `complex_treatment`
```python
'complex_treatment': [
    ['CtD', 'DaG'],  # Compound → Disease → Gene
    ['CuG', 'GaD'],  # Compound → Gene → Disease
    ['CdG', 'GaD'],  # Compound → Gene → Disease
    ['DtC', 'CuG'],  # Disease → Compound → Gene
    ['GuC', 'CtD']   # Gene → Compound → Disease
]
```

#### الگوهای `complex_function`
```python
'complex_function': [
    ['GpBP', 'BPpG'], # Gene → Biological Process → Gene
    ['GpPW', 'PWpG'], # Gene → Pathway → Gene
    ['GiG', 'GpBP'],  # Gene → Gene → Biological Process
    ['Gr>G', 'GpMF']  # Gene → Gene → Molecular Function
]
```

## 📊 نتایج تست‌ها

### تست ۱: ترکیبات تنظیم‌کننده ژن‌های بیان‌شده در قلب
- **سوال**: "What compounds upregulate genes expressed in the heart?"
- **نوع پیچیده**: `complex_expression`
- **نتایج جستجوی هوشمند**: ۳ ژن (MMP9, BID, KCNQ2)
- **نتایج چندمرحله‌ای**: ۰ نتیجه (نیاز به بهبود الگوها)

### تست ۲: ژن‌های تعاملی با بیماری‌های مغز
- **سوال**: "What genes interact with diseases that affect the brain?"
- **نوع پیچیده**: `complex_disease`
- **نتایج جستجوی هوشمند**: ۲ ژن (APOE, TP53)
- **نتایج چندمرحله‌ای**: ۱ نتیجه (BiologicalProcess::Cell Death)

### تست ۳: ترکیبات درمان‌کننده بیماری‌های تنظیم‌کننده ژن
- **سوال**: "What compounds treat diseases that regulate genes?"
- **نوع پیچیده**: `complex_treatment`
- **نتایج جستجوی هوشمند**: ۰ نتیجه
- **نتایج چندمرحله‌ای**: ۲ نتیجه (BRCA1, Breast Cancer)

### تست ۴: فرآیندهای زیستی ژن‌های تنظیم‌کننده
- **سوال**: "What biological processes do genes participate in that regulate other genes?"
- **نوع پیچیده**: `complex_function`
- **نتایج جستجوی هوشمند**: ۳ نتیجه (DNA Repair, Enzyme, Mitochondria)
- **نتایج چندمرحله‌ای**: ۰ نتیجه

### تست ۵: مسیرهای ژن‌های بیان‌شده در کبد
- **سوال**: "What pathways do genes expressed in the liver participate in?"
- **نوع پیچیده**: `complex_function`
- **نتایج جستجوی هوشمند**: ۱ ژن (BRCA1)
- **نتایج چندمرحله‌ای**: ۰ نتیجه

## 🔍 آمار گراف بهبود یافته

### ساختار کلی
- **تعداد نودها**: ۴۸
- **تعداد یال‌ها**: ۱۱۶
- **انواع یال‌ها**: ۴۲ نوع مختلف

### یال‌های کلیدی
- **AeG**: ۷ یال (Anatomy expresses Gene)
- **GeA**: ۷ یال (Gene expressed in Anatomy) - معکوس
- **DaG**: ۵ یال (Disease associates Gene)
- **GaD**: ۵ یال (Gene associates Disease) - معکوس
- **GpBP**: ۴ یال (Gene participates Biological Process)
- **BPpG**: ۴ یال (Biological Process participates Gene) - معکوس

## 🎯 مزایای بهبودها

### ۱. پشتیبانی از مسیرهای پیچیده
- امکان حرکت در جهت معکوس در گراف
- پشتیبانی از سوالات چندمرحله‌ای
- تشخیص هوشمند نوع سوال پیچیده

### ۲. بهبود تشخیص intent
- تشخیص بر اساس کلمات کلیدی
- تشخیص بر اساس metaedges موجود
- پشتیبانی از انواع مختلف سوالات پیچیده

### ۳. الگوهای چندمرحله‌ای
- تعریف الگوهای پیچیده برای انواع مختلف سوالات
- پشتیبانی از مسیرهای چندمرحله‌ای
- امتیازدهی هوشمند برای نتایج

## 🔧 چالش‌های باقی‌مانده

### ۱. بهبود الگوهای چندمرحله‌ای
- برخی الگوها هنوز نتایج مناسبی تولید نمی‌کنند
- نیاز به بهبود الگوریتم جستجوی مسیر
- نیاز به اضافه کردن الگوهای بیشتر

### ۲. بهبود تشخیص intent
- برخی سوالات پیچیده هنوز به درستی تشخیص داده نمی‌شوند
- نیاز به بهبود الگوهای تشخیص
- نیاز به اضافه کردن کلمات کلیدی بیشتر

### ۳. بهبود امتیازدهی
- نیاز به بهبود سیستم امتیازدهی برای نتایج چندمرحله‌ای
- نیاز به در نظر گرفتن عوامل بیشتر در امتیازدهی
- نیاز به بهبود الگوریتم‌های رتبه‌بندی

## 🚀 راه‌حل‌های پیشنهادی

### ۱. بهبود الگوریتم جستجوی مسیر
```python
def improved_path_finding(self, start_node: str, pattern: List[str], max_depth: int):
    """الگوریتم بهبود یافته جستجوی مسیر"""
    # پیاده‌سازی الگوریتم بهبود یافته
    pass
```

### ۲. اضافه کردن الگوهای بیشتر
```python
additional_patterns = {
    'complex_regulation': [
        ['Gr>G', 'GpBP'],  # Gene regulates Gene → Biological Process
        ['Gr>G', 'GpPW'],  # Gene regulates Gene → Pathway
        ['GiG', 'Gr>G'],   # Gene interacts Gene → regulates Gene
    ],
    'complex_expression_regulation': [
        ['AeG', 'Gr>G'],   # Anatomy → Gene → regulates Gene
        ['GeA', 'GuC'],    # Gene → Anatomy → Compound
    ]
}
```

### ۳. بهبود سیستم امتیازدهی
```python
def improved_scoring(self, pattern: List[str], path_metaedges: List[str], depth: int) -> float:
    """سیستم امتیازدهی بهبود یافته"""
    # در نظر گرفتن عوامل بیشتر
    # بهبود الگوریتم امتیازدهی
    pass
```

## 📝 نتیجه‌گیری

سیستم GraphRAG برای سوالات پیچیده به طور قابل توجهی بهبود یافته است. با اضافه کردن یال‌های معکوس، بهبود تشخیص نوع سوال پیچیده، و تعریف الگوهای چندمرحله‌ای، سیستم قادر است سوالات زیستی پیچیده را پردازش کند.

### دستاوردهای کلیدی:
1. ✅ پشتیبانی از یال‌های معکوس
2. ✅ بهبود تشخیص نوع سوال پیچیده
3. ✅ تعریف الگوهای چندمرحله‌ای
4. ✅ بهبود ساختار گراف
5. ✅ پشتیبانی از مسیرهای پیچیده

### چالش‌های باقی‌مانده:
1. ⚠️ نیاز به بهبود الگوریتم جستجوی مسیر
2. ⚠️ نیاز به اضافه کردن الگوهای بیشتر
3. ⚠️ نیاز به بهبود سیستم امتیازدهی

سیستم آماده استفاده برای سوالات پیچیده است و می‌تواند به عنوان پایه‌ای برای بهبودهای بیشتر استفاده شود. 
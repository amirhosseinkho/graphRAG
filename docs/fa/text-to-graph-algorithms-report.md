# گزارش کامل الگوریتم‌های تبدیل متن به گراف

## فهرست مطالب
1. [مقدمه](#مقدمه)
2. [الگوریتم‌های پایه](#الگوریتم‌های-پایه)
3. [الگوریتم‌های مبتنی بر NLP](#الگوریتم‌های-مبتنی-بر-nlp)
4. [الگوریتم‌های مبتنی بر LLM](#الگوریتم‌های-مبتنی-بر-llm)
5. [الگوریتم‌های ترکیبی](#الگوریتم‌های-ترکیبی)
6. [الگوریتم‌های تخصصی](#الگوریتم‌های-تخصصی)
7. [الگوریتم‌های پیشرفته](#الگوریتم‌های-پیشرفته)
8. [مقایسه الگوریتم‌ها](#مقایسه-الگوریتم‌ها)
9. [نتیجه‌گیری](#نتیجه‌گیری)

---

## مقدمه

سیستم تبدیل متن به گراف از **14 الگوریتم مختلف** برای استخراج موجودیت‌ها و روابط از متن استفاده می‌کند. هر الگوریتم برای کاربردهای خاصی طراحی شده است و مزایا و معایب خود را دارد.

### ساختار کلی

هر الگوریتم سه مرحله اصلی دارد:
1. **استخراج موجودیت‌ها (Entity Extraction)**
2. **استخراج روابط (Relationship Extraction)**
3. **ساخت گراف (Graph Construction)**

---

## الگوریتم‌های پایه

### 1. Simple (Rule-based)

#### توضیحات
الگوریتم ساده‌ترین روش استخراج است که بر اساس الگوهای از پیش تعریف شده (Regular Expressions) کار می‌کند.

#### نحوه کار

**مرحله 1: استخراج موجودیت‌ها**

```python
# الگوهای موجودیت
gene_pattern = r'\b[A-Z][A-Z0-9]+\b'  # ژن‌ها: TP53, BRCA1
disease_keywords = ['cancer', 'disease', 'syndrome', ...]
compound_pattern = r'\b[a-z]+(?:ib|mab|zumab|umab|tinib|olol|pril|sartan)\b'
```

**الگوریتم:**
1. جستجوی الگوهای ژن در متن (مثلاً `TP53`, `BRCA1`)
2. جستجوی کلمات کلیدی بیماری (مثلاً `breast cancer`)
3. جستجوی الگوهای دارو (مثلاً `trastuzumab`)

**مرحله 2: استخراج روابط**

```python
relationship_patterns = [
    (r'(\w+)\s+(?:regulates?|controls?)\s+(\w+)', 'Gr>G'),
    (r'(\w+)\s+(?:interacts?\s+with|binds?\s+to)\s+(\w+)', 'GiG'),
    (r'(\w+)\s+(?:treats?|cures?)\s+([\w\s]+)', 'CtD'),
    ...
]
```

**الگوریتم:**
1. جستجوی الگوهای رابطه در متن
2. تطبیق موجودیت‌های source و target
3. ایجاد رابطه با metaedge مناسب

#### مزایا
- ✅ **سرعت بسیار بالا**: پردازش فوری
- ✅ **بدون نیاز به مدل**: نیازی به نصب مدل‌های بزرگ ندارد
- ✅ **قابل پیش‌بینی**: نتایج همیشه یکسان
- ✅ **مناسب برای متن‌های ساختاریافته**

#### معایب
- ❌ **دقت پایین**: فقط الگوهای از پیش تعریف شده را پیدا می‌کند
- ❌ **عدم درک معنی**: نمی‌تواند روابط پیچیده را درک کند
- ❌ **نیاز به الگوهای دستی**: برای هر نوع رابطه باید الگو تعریف شود

#### مثال

**ورودی:**
```
TP53 regulates CDKN1A. Trastuzumab treats breast cancer.
```

**خروجی:**
- موجودیت‌ها: `TP53`, `CDKN1A`, `Trastuzumab`, `breast cancer`
- روابط: `TP53 --[regulates]--> CDKN1A`, `Trastuzumab --[treats]--> breast cancer`

#### پیچیدگی زمانی
- **زمان:** O(n) که n طول متن است
- **فضا:** O(m) که m تعداد موجودیت‌ها و روابط است

---

## الگوریتم‌های مبتنی بر NLP

### 2. spaCy

#### توضیحات
استفاده از کتابخانه spaCy برای پردازش زبان طبیعی و استخراج موجودیت‌ها و روابط.

#### نحوه کار

**مرحله 1: پردازش متن با spaCy**

```python
doc = self.nlp(text)  # پردازش کامل متن
```

**مرحله 2: استخراج موجودیت‌های نام‌گذاری شده (NER)**

```python
for ent in doc.ents:
    entity_type = self._map_spacy_label_to_type(ent.label_)
    # موجودیت‌ها: PERSON, ORG, GPE, ...
```

**مرحله 3: استخراج Noun Chunks**

```python
for chunk in doc.noun_chunks:
    # استخراج عبارات اسمی به عنوان موجودیت‌های بالقوه
```

**مرحله 4: استخراج روابط با Dependency Parsing**

سه روش برای استخراج روابط:

**روش 1: الگوهای فعل (Verb Patterns)**
```python
for token in sent:
    if token.pos_ == "VERB":
        subject_phrase = self._extract_subject_phrase(token, doc)
        object_phrase = self._extract_object_phrase(token, doc)
        # ایجاد رابطه: subject --[verb]--> object
```

**روش 2: الگوهای حرف اضافه (Preposition-based)**
```python
if token.pos_ == "ADP" and token.head.pos_ == "VERB":
    # استخراج روابط از ساختارهای "X with Y", "X in Y"
```

**روش 3: هم‌رخداد موجودیت‌ها (Entity Co-occurrence)**
```python
# اگر دو موجودیت در یک جمله با یک فعل action باشند
if len(entities_in_sentence) >= 2:
    # ایجاد رابطه بین آن‌ها
```

#### مزایا
- ✅ **دقت خوب**: استفاده از مدل‌های آموزش دیده
- ✅ **سرعت مناسب**: سریع‌تر از LLM
- ✅ **استخراج دقیق روابط**: استفاده از dependency parsing
- ✅ **پشتیبانی از چندین زبان**: مدل‌های مختلف برای زبان‌های مختلف

#### معایب
- ❌ **نیاز به مدل**: باید مدل spaCy نصب شود
- ❌ **محدودیت در روابط پیچیده**: فقط روابط ساده را می‌تواند استخراج کند
- ❌ **وابستگی به کیفیت مدل**: کیفیت به مدل استفاده شده بستگی دارد

#### مثال

**ورودی:**
```
TP53 participates in apoptosis and interacts with BRCA1. 
The protein regulates cell cycle progression.
```

**خروجی:**
- موجودیت‌ها: `TP53`, `apoptosis`, `BRCA1`, `protein`, `cell cycle progression`
- روابط:
  - `TP53 --[participates in]--> apoptosis`
  - `TP53 --[interacts with]--> BRCA1`
  - `protein --[regulates]--> cell cycle progression`

#### پیچیدگی زمانی
- **زمان:** O(n) برای پردازش متن + O(m²) برای استخراج روابط
- **فضا:** O(n) برای نگهداری doc

---

### 3. spaCy SVO Enhanced

#### توضیحات
نسخه پیشرفته spaCy با تمرکز بر استخراج دقیق روابط Subject-Verb-Object (SVO).

#### نحوه کار

**تفاوت با spaCy عادی:**

1. **استخراج دقیق‌تر SVO:**
```python
for token in doc:
    if token.pos_ != "VERB":
        continue
    
    subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
    objects = [c for c in token.children if c.dep_ in ("dobj", "obj", "attr", "dative", "oprd")]
    
    # استخراج prepositional objects
    for prep in [c for c in token.children if c.dep_ == "prep"]:
        for pobj in prep.children:
            if pobj.dep_ == "pobj":
                objects.append(pobj)
```

2. **استفاده از Entity Spans:**
```python
def pick_entity_for_span(ents_by_char, start, end):
    # پیدا کردن موجودیت برای یک span خاص
    for s, e, text in ents_by_char:
        if not (end <= s or start >= e):  # overlap
            return text
```

3. **استخراج دقیق‌تر نام رابطه:**
```python
relation_name = self._extract_relation_name_from_text(token, sent)
```

#### مزایا
- ✅ **دقت بالاتر در روابط**: استخراج دقیق‌تر SVO
- ✅ **پشتیبانی از روابط پیچیده**: شامل prepositional objects
- ✅ **استفاده از Entity Spans**: تطبیق بهتر موجودیت‌ها

#### معایب
- ❌ **پیچیدگی بیشتر**: کد پیچیده‌تر از spaCy عادی
- ❌ **زمان پردازش بیشتر**: نیاز به پردازش بیشتر

#### مثال

**ورودی:**
```
The company signed an agreement with the government.
```

**خروجی:**
- موجودیت‌ها: `company`, `agreement`, `government`
- روابط: `company --[signed]--> agreement`, `company --[signed with]--> government`

---

## الگوریتم‌های مبتنی بر LLM

### 4. LLM (Single-pass)

#### توضیحات
استفاده از مدل‌های زبانی بزرگ (LLM) برای استخراج موجودیت‌ها و روابط در یک پاس.

#### نحوه کار

**مرحله 1: آماده‌سازی Prompt**

```python
prompt_template = """
Extract entities and relationships from the following text as JSON.
Format:
{
  "entities": [
    {"id": "unique_id", "name": "entity_name", "type": "Gene|Disease|...", "attributes": {}}
  ],
  "relationships": [
    {"source": "source_entity_id", "target": "target_entity_id", 
     "metaedge": "GcG|GiG|...", "relation": "actual_relation_name_from_text", 
     "attributes": {}}
  ]
}

CRITICAL INSTRUCTIONS FOR RELATIONSHIPS:
1. The "relation" field MUST contain the actual verb/phrase from the text.
2. DO NOT use metaedge codes as the relation name.
3. Use natural, descriptive phrases.

Text:
{text}
Response:
"""
```

**مرحله 2: فراخوانی LLM**

```python
response = self.hf_client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.2,
    max_tokens=4096
)
```

**مرحله 3: Parsing پاسخ**

```python
response_text = response.choices[0].message.content.strip()
result = self._parse_llm_response(response_text)
```

**مرحله 4: پردازش و فیلتر کردن**

```python
# فیلتر بر اساس confidence threshold
if confidence_threshold > 0:
    filtered_relationships = [
        rel for rel in relationships 
        if rel.get("attributes", {}).get("confidence", 1.0) >= confidence_threshold
    ]
```

#### مزایا
- ✅ **دقت بسیار بالا**: استفاده از مدل‌های پیشرفته
- ✅ **درک معنی**: می‌تواند روابط پیچیده را درک کند
- ✅ **انعطاف‌پذیری**: نیازی به الگوهای از پیش تعریف شده ندارد
- ✅ **پشتیبانی از چندین زبان**: مدل‌های چندزبانه

#### معایب
- ❌ **سرعت پایین**: نیاز به API call
- ❌ **هزینه**: نیاز به API key و ممکن است هزینه داشته باشد
- ❌ **وابستگی به اینترنت**: نیاز به اتصال به API
- ❌ **ممکن است hallucination داشته باشد**: ممکن است موجودیت‌ها یا روابطی که در متن نیستند را استخراج کند

#### مدل‌های پشتیبانی شده

- `mistralai/Mistral-7B-Instruct-v0.2` (پیشنهادی)
- `HuggingFaceH4/zephyr-7b-beta`
- `microsoft/Phi-3-mini-4k-instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `google/gemma-2b-it`
- `meta-llama/Llama-3.2-3B-Instruct`

#### مثال

**ورودی:**
```
TP53 is a tumor suppressor gene that regulates cell cycle and apoptosis. 
Mutations in TP53 are associated with many types of cancer.
```

**خروجی:**
- موجودیت‌ها: `TP53`, `tumor suppressor gene`, `cell cycle`, `apoptosis`, `mutations`, `cancer`
- روابط:
  - `TP53 --[is a]--> tumor suppressor gene`
  - `TP53 --[regulates]--> cell cycle`
  - `TP53 --[regulates]--> apoptosis`
  - `mutations --[in]--> TP53`
  - `mutations --[associated with]--> cancer`

#### پیچیدگی زمانی
- **زمان:** O(1) برای API call (اما زمان انتظار زیاد)
- **فضا:** O(m) برای نگهداری نتایج

---

### 5. LLM Multi-pass

#### توضیحات
استخراج با چندین پاس برای افزایش recall و پیدا کردن موجودیت‌ها و روابط از دست رفته.

#### نحوه کار

**مرحله 1: استخراج اولیه**

```python
# پاس اول: استخراج کامل
response = self.hf_client.chat.completions.create(
    model=model,
    messages=history,
    temperature=0.2,
    max_tokens=4096
)
result = self._parse_llm_response(response_text)
```

**مرحله 2: Iterative Gleaning**

```python
for i in range(max_gleanings):
    followup_prompt = (
        "Review the SAME text again.\n"
        "In your previous JSON response(s) above, you already listed some entities and relationships.\n"
        "Now return ONLY additional entities and relationships that are EXPLICITLY present "
        "in the text but were NOT included before.\n"
        "If there are no additional items, respond with:\n"
        '{"entities": [], "relationships": []}\n'
    )
    
    response = self.hf_client.chat.completions.create(...)
    # اضافه کردن موجودیت‌ها و روابط جدید
```

**مرحله 3: فیلتر کردن Hallucinations**

```python
# برای متن‌های طولانی (>20 token)
if token_count > 20:
    all_entities, all_relationships = self._filter_extractions_by_text(
        text, all_entities, all_relationships
    )
```

**الگوریتم فیلتر کردن:**

```python
def _filter_extractions_by_text(text, entities, relationships):
    norm_text = self._normalize_for_match(text)
    
    # فیلتر موجودیت‌ها: فقط اگر نام موجودیت در متن باشد
    for ent in entities:
        norm_name = self._normalize_for_match(ent.get("name", ""))
        if norm_name and norm_name in norm_text:
            filtered_entities.append(ent)
    
    # فیلتر روابط: فقط اگر هر دو endpoint و relation در متن باشند
    for rel in relationships:
        if (src in valid_entity_ids and tgt in valid_entity_ids and
            relation_in_text(rel, norm_text)):
            filtered_relationships.append(rel)
```

#### مزایا
- ✅ **Recall بسیار بالا**: پیدا کردن بیشتر موجودیت‌ها و روابط
- ✅ **کاهش از دست رفتن اطلاعات**: چندین پاس برای اطمینان
- ✅ **فیلتر کردن Hallucinations**: حذف موجودیت‌ها/روابطی که در متن نیستند

#### معایب
- ❌ **زمان بسیار زیاد**: چندین API call
- ❌ **هزینه بیشتر**: چندین درخواست به API
- ❌ **ممکن است هنوز hallucination داشته باشد**: در متن‌های کوتاه فیلتر نمی‌شود

#### پارامترها
- `max_gleanings`: تعداد پاس‌های تکرار (پیش‌فرض: 2)

#### مثال

**پاس 1:**
- موجودیت‌ها: `TP53`, `cancer`
- روابط: `TP53 --[associated with]--> cancer`

**پاس 2:**
- موجودیت‌های اضافی: `BRCA1`, `apoptosis`
- روابط اضافی: `TP53 --[regulates]--> apoptosis`, `BRCA1 --[interacts with]--> TP53`

**نتیجه نهایی:** ادغام همه موجودیت‌ها و روابط

---

## الگوریتم‌های ترکیبی

### 6. Hybrid

#### توضیحات
ترکیب نتایج از چندین روش استخراج برای بهترین نتیجه.

#### نحوه کار

**مرحله 1: اجرای هر روش**

```python
for method in methods:  # ['spacy', 'llm', 'spacy_svo_enhanced', ...]
    if method == "simple":
        result = self.extract_simple(text_for_method, ...)
    elif method == "spacy":
        result = self.extract_spacy(text_for_method, ...)
    elif method == "llm":
        result = self.extract_llm(llm_text, ...)  # استفاده از متن اصلی
    ...
```

**مرحله 2: ادغام موجودیت‌ها**

```python
for ent in result.get("entities", []):
    ent_name = ent.get("name", "")
    
    if ent_name in entity_map:
        # Merge: نگه داشتن موجودیت با confidence بالاتر
        existing = entity_map[ent_name]
        existing_conf = existing.get("attributes", {}).get("confidence", 0.5)
        new_conf = ent.get("attributes", {}).get("confidence", 0.5)
        
        if new_conf > existing_conf:
            entity_map[ent_name] = ent
    else:
        entity_map[ent_name] = ent
        all_entities.append(ent)
```

**مرحله 3: ادغام روابط**

```python
for rel in result.get("relationships", []):
    source = rel.get("source")
    target = rel.get("target")
    metaedge = rel.get("metaedge", "GiG")
    key = (source, target, metaedge)
    
    if key in relationship_map:
        # Merge: ترکیب attributes و افزایش confidence
        existing = relationship_map[key]
        existing.get("attributes", {})["weight"] += rel.get("attributes", {}).get("weight", 1.0)
        existing.get("attributes", {})["confidence"] = min(1.0, 
            (existing_conf + new_conf) / 2)
    else:
        relationship_map[key] = rel
        all_relationships.append(rel)
```

**مرحله 4: فیلتر بر اساس Confidence**

```python
if confidence_threshold > 0:
    filtered_relationships = [
        rel for rel in all_relationships
        if rel.get("attributes", {}).get("confidence", 1.0) >= confidence_threshold
    ]
```

#### مزایا
- ✅ **بهترین دقت**: ترکیب نقاط قوت روش‌های مختلف
- ✅ **افزایش Recall**: پیدا کردن بیشتر موجودیت‌ها و روابط
- ✅ **افزایش Precision**: فیلتر کردن بر اساس confidence

#### معایب
- ❌ **زمان زیاد**: اجرای چندین روش
- ❌ **هزینه بیشتر**: اگر از LLM استفاده شود
- ❌ **پیچیدگی بیشتر**: نیاز به مدیریت نتایج چندگانه

#### روش‌های قابل ترکیب
- `spacy`
- `llm`
- `spacy_svo_enhanced`
- `llm_multipass`
- `simple`

#### مثال

**روش spaCy:**
- موجودیت‌ها: `TP53`, `BRCA1`
- روابط: `TP53 --[interacts with]--> BRCA1` (confidence: 0.7)

**روش LLM:**
- موجودیت‌ها: `TP53`, `BRCA1`, `apoptosis`
- روابط: `TP53 --[regulates]--> apoptosis` (confidence: 0.9)

**نتیجه Hybrid:**
- موجودیت‌ها: `TP53`, `BRCA1`, `apoptosis` (ادغام شده)
- روابط:
  - `TP53 --[interacts with]--> BRCA1` (confidence: 0.7)
  - `TP53 --[regulates]--> apoptosis` (confidence: 0.9)

---

## الگوریتم‌های تخصصی

### 7. Persian

#### توضیحات
استخراج از متن فارسی با استفاده از مدل‌های تخصصی فارسی (ParsBERT, mT5).

#### نحوه کار

**مرحله 1: نرمال‌سازی متن فارسی**

```python
if self.persian_normalizer:
    normalized_text = self.persian_normalizer.normalize(text)
    # تبدیل "ي" به "ی"، "ك" به "ک"، ...
```

**مرحله 2: استفاده از Modular Pipeline**

```python
pipeline = ModularExtractionPipeline(
    language="fa",
    enable_normalization=True,
    enable_ner=True,
    enable_relation_extraction=True,
    enable_coreference=kwargs.get("enable_coreference", False)
)
result = pipeline.process(normalized_text)
```

**مرحله 3: تبدیل به فرمت استاندارد**

```python
entities = []
for i, ent in enumerate(result.get("entities", [])):
    entities.append({
        "id": f"ENTITY_{i}",
        "name": ent.get("text", ""),
        "type": ent.get("label", "Gene"),
        "attributes": {
            "score": ent.get("score", 0.5),
            "confidence": ent.get("confidence", 0.5),
            "language": "fa"
        }
    })
```

#### مزایا
- ✅ **پشتیبانی از فارسی**: استفاده از مدل‌های تخصصی فارسی
- ✅ **نرمال‌سازی خودکار**: تبدیل کاراکترهای مختلف به فرم استاندارد
- ✅ **Coreference Resolution**: حل ارجاعات (اختیاری)

#### معایب
- ❌ **نیاز به ماژول‌های اضافی**: نیاز به نصب ماژول‌های فارسی
- ❌ **Fallback به spaCy**: در صورت خطا به spaCy برمی‌گردد

#### مثال

**ورودی:**
```
ژن TP53 در تنظیم چرخه سلولی نقش دارد. این ژن با سرطان ارتباط دارد.
```

**خروجی:**
- موجودیت‌ها: `TP53`, `چرخه سلولی`, `سرطان`
- روابط:
  - `TP53 --[نقش دارد در]--> چرخه سلولی`
  - `TP53 --[ارتباط دارد با]--> سرطان`

---

### 8. Span-based

#### توضیحات
استخراج مبتنی بر Span با استفاده از BioBERT یا SciBERT برای متون علمی.

#### نحوه کار

**مرحله 1: انتخاب Extractor**

```python
if model_type == "biobert":
    extractor = BioBERTExtractor()  # برای موجودیت‌های زیست‌پزشکی
elif model_type == "scibert":
    extractor = SciBERTExtractor()  # برای متون علمی
else:
    extractor = SpanBasedExtractor(language=language)  # انتخاب خودکار
```

**مرحله 2: استخراج موجودیت‌ها**

```python
entities_raw = extractor.extract_entities(text)
# هر موجودیت شامل: text, label, score, span (start, end)
```

**مرحله 3: استخراج روابط با BERT**

```python
rel_extractor = BERTRelationExtractor(language=language)
relationships_raw = rel_extractor.extract_relations(
    text, entities_raw, max_pairs=100
)
```

**مرحله 4: تبدیل به فرمت استاندارد**

```python
for i, ent in enumerate(entities_raw):
    entities.append({
        "id": f"ENTITY_{i}",
        "name": ent.get("text", ""),
        "type": ent.get("label", "Gene"),
        "attributes": {
            "score": ent.get("score", 0.5),
            "span": ent.get("span", (0, 0)),
            "extraction_method": "span_based"
        }
    })
```

#### مزایا
- ✅ **دقت بالا برای متون علمی**: استفاده از مدل‌های تخصصی
- ✅ **استخراج دقیق Span**: موقعیت دقیق موجودیت‌ها در متن
- ✅ **استخراج روابط با BERT**: استفاده از BERT برای روابط

#### معایب
- ❌ **نیاز به مدل‌های بزرگ**: BioBERT/SciBERT مدل‌های بزرگی هستند
- ❌ **زمان پردازش بیشتر**: نیاز به پردازش بیشتر

#### مثال

**ورودی:**
```
TP53 (tumor protein p53) is a transcription factor that regulates 
the expression of genes involved in cell cycle arrest and apoptosis.
```

**خروجی:**
- موجودیت‌ها:
  - `TP53` (span: 0-4, label: Gene, score: 0.95)
  - `tumor protein p53` (span: 6-22, label: Gene, score: 0.92)
  - `transcription factor` (span: 26-45, label: Concept, score: 0.88)
  - `cell cycle arrest` (span: 95-111, label: Process, score: 0.85)
  - `apoptosis` (span: 116-124, label: Process, score: 0.90)
- روابط:
  - `TP53 --[is a]--> transcription factor`
  - `TP53 --[regulates]--> cell cycle arrest`
  - `TP53 --[regulates]--> apoptosis`

---

### 9. With Coreference

#### توضیحات
استخراج با حل ارجاعات (Coreference Resolution) برای ادغام ارجاعات مانند "این ژن"، "آن پروتئین" با موجودیت‌های اصلی.

#### نحوه کار

**مرحله 1: استخراج با روش پایه**

```python
extraction_result = self.extract(
    text, 
    method=base_method,  # مثلاً "spacy"
    max_entities=max_entities,
    max_relationships=max_relationships
)
```

**مرحله 2: حل ارجاعات**

```python
resolver = CoreferenceResolver(language=language, spacy_model=nlp_model)
reference_map = resolver.resolve(text, extraction_result.get("entities", []))
# reference_map: {"این ژن": "TP53", "آن پروتئین": "BRCA1", ...}
```

**مرحله 3: ادغام موجودیت‌ها**

```python
merged_entities = resolver.merge_entities(
    extraction_result.get("entities", []), 
    reference_map
)
```

#### مزایا
- ✅ **حل ارجاعات**: ادغام "این ژن" با "TP53"
- ✅ **افزایش دقت**: پیدا کردن روابط بیشتر
- ✅ **کاهش تکراری‌ها**: ادغام موجودیت‌های مشابه

#### معایب
- ❌ **نیاز به ماژول Coreference**: نیاز به نصب ماژول اضافی
- ❌ **زمان پردازش بیشتر**: نیاز به پردازش اضافی

#### مثال

**ورودی:**
```
TP53 is a tumor suppressor. This gene regulates cell cycle. 
The protein interacts with BRCA1.
```

**بدون Coreference:**
- موجودیت‌ها: `TP53`, `this gene`, `the protein`, `cell cycle`, `BRCA1`
- روابط: `this gene --[regulates]--> cell cycle`, `the protein --[interacts with]--> BRCA1`

**با Coreference:**
- موجودیت‌ها: `TP53`, `cell cycle`, `BRCA1` (ادغام شده)
- روابط: `TP53 --[regulates]--> cell cycle`, `TP53 --[interacts with]--> BRCA1`

---

### 10. Long Text

#### توضیحات
پردازش متن‌های طولانی با chunking برای جلوگیری از محدودیت‌های طول متن.

#### نحوه کار

**مرحله 1: Chunking متن**

```python
strategy_map = {
    "smart": ChunkingStrategy.SMART,  # ترکیب پاراگراف و جمله
    "sliding_window": ChunkingStrategy.SLIDING_WINDOW,  # با overlap
    "sentence": ChunkingStrategy.SENTENCE,  # بر اساس جملات
    "paragraph": ChunkingStrategy.PARAGRAPH  # بر اساس پاراگراف
}

chunker = SmartChunker(
    strategy=strategy_map.get(chunking_strategy, ChunkingStrategy.SMART),
    max_tokens=max_tokens,
    overlap_ratio=chunk_overlap,
    language=language
)
chunks = chunker.chunk(text)
```

**مرحله 2: پردازش هر Chunk**

```python
for chunk in chunks:
    chunk_result = self.extract(
        chunk, 
        method=base_method,  # مثلاً "spacy"
        max_entities=max_entities,
        max_relationships=max_relationships
    )
```

**مرحله 3: ادغام نتایج**

```python
# ادغام موجودیت‌ها و روابط از همه chunkها
all_entities = merge_entities(chunk_results)
all_relationships = merge_relationships(chunk_results)
```

#### مزایا
- ✅ **پردازش متن‌های طولانی**: بدون محدودیت طول
- ✅ **استراتژی‌های مختلف Chunking**: برای نیازهای مختلف
- ✅ **Overlap برای حفظ Context**: حفظ معنی در مرزهای chunk

#### معایب
- ❌ **زمان پردازش بیشتر**: پردازش چندین chunk
- ❌ **ممکن است روابط بین chunkها از دست برود**: اگر overlap کافی نباشد

#### استراتژی‌های Chunking

1. **Smart**: ترکیب پاراگراف و جمله (پیشنهادی)
2. **Sliding Window**: با overlap برای حفظ context
3. **Sentence**: بر اساس جملات
4. **Paragraph**: بر اساس پاراگراف

#### مثال

**متن طولانی (1000 کلمه):**
```
[Chunk 1: 0-500 words]
TP53 regulates cell cycle...

[Chunk 2: 400-900 words]  # overlap: 100 words
...cell cycle progression. BRCA1 interacts...

[Chunk 3: 800-1000 words]  # overlap: 100 words
...with TP53. The complex...
```

**نتیجه:** ادغام موجودیت‌ها و روابط از همه chunkها

---

## الگوریتم‌های پیشرفته

### 11. Joint ER (Entity-Relation Extraction)

#### توضیحات
استخراج همزمان موجودیت و رابطه با Graph Structure Learning و بهینه‌سازی ساختار گراف در حین استخراج.

#### نحوه کار

**مرحله 1: استخراج اولیه**

```python
# استخراج موجودیت‌ها با spaCy
doc = self.nlp(text)
for ent in doc.ents:
    initial_entities.append({
        "id": f"ENT_{len(initial_entities)}",
        "name": ent.text,
        "type": ent.label_,
        ...
    })

# استخراج روابط با dependency parsing
for token in sent:
    if token.dep_ in ["nsubj", "dobj", "pobj", "attr", "acomp"]:
        # پیدا کردن موجودیت‌ها در head و dependent
        # ایجاد رابطه
```

**مرحله 2: ساخت گراف موقت**

```python
temp_graph = nx.MultiDiGraph()
for entity in entities:
    temp_graph.add_node(entity["id"], **entity)
for rel in relationships:
    temp_graph.add_edge(rel["source"], rel["target"], **rel)
```

**مرحله 3: بهینه‌سازی ساختار با تکرار**

```python
for iteration in range(structure_iterations):
    # محاسبه Centrality
    degree_centrality = nx.degree_centrality(temp_graph)
    betweenness_centrality = nx.betweenness_centrality(temp_graph)
    
    # به‌روزرسانی موجودیت‌ها بر اساس اهمیت
    for entity in entities:
        entity["attributes"]["degree_centrality"] = degree_centrality.get(entity["id"], 0.0)
        entity["attributes"]["betweenness_centrality"] = betweenness_centrality.get(entity["id"], 0.0)
    
    # حذف روابط ضعیف
    relationships_to_keep = []
    for rel in relationships:
        source_importance = degree_centrality.get(rel["source"], 0.0)
        target_importance = degree_centrality.get(rel["target"], 0.0)
        confidence = rel.get("attributes", {}).get("confidence", 0.5)
        
        if (source_importance + target_importance) / 2 > 0.1 or confidence > 0.6:
            relationships_to_keep.append(rel)
    
    relationships = relationships_to_keep
    
    # بازسازی گراف
    temp_graph = rebuild_graph(entities, relationships)
```

#### مزایا
- ✅ **بهینه‌سازی ساختار**: بهبود ساختار گراف در حین استخراج
- ✅ **استفاده از Centrality**: شناسایی موجودیت‌ها و روابط مهم
- ✅ **حذف روابط ضعیف**: بهبود کیفیت گراف

#### معایب
- ❌ **زمان پردازش زیاد**: چندین تکرار برای بهینه‌سازی
- ❌ **پیچیدگی بالا**: نیاز به محاسبه Centrality

#### پارامترها
- `structure_iterations`: تعداد تکرار برای بهینه‌سازی (پیش‌فرض: 3)

---

### 12. Autoregressive

#### توضیحات
تولید گراف به صورت sequential با استفاده از Transformer Encoder-Decoder.

#### نحوه کار

**مرحله 1: آماده‌سازی Prompt**

```python
prompt = f"""You are a knowledge graph extraction system. 
Extract entities and relationships from the text in an autoregressive manner.

Generate the graph step by step:
1. First, identify all entities
2. Then, for each entity, identify its relationships with other entities
3. Format as JSON

Text:
{text}

Generate entities and relationships in JSON format:
{{
  "entities": [...],
  "relationships": [...]
}}
"""
```

**مرحله 2: تولید Autoregressive**

```python
response = self.hf_client.text_generation(
    prompt,
    model=model,
    max_new_tokens=max_generation_length,
    temperature=0.3,
    return_full_text=False
)
```

**مرحله 3: Parsing و پردازش**

```python
result = self._parse_llm_response(response)
if result:
    return {
        "entities": result["entities"][:max_entities],
        "relationships": result["relationships"][:max_relationships],
        ...
    }
```

#### مزایا
- ✅ **تولید Sequential**: تولید تدریجی گراف
- ✅ **استفاده از Transformer**: استفاده از مدل‌های پیشرفته

#### معایب
- ❌ **Fallback به LLM عادی**: در صورت خطا
- ❌ **زمان پردازش زیاد**: نیاز به تولید طولانی

#### پارامترها
- `max_generation_length`: حداکثر طول تولید (پیش‌فرض: 2048)

---

### 13. EDC Framework (Extract-Define-Canonicalize)

#### توضیحات
روش سه مرحله‌ای برای ساخت گراف دانش: Extract → Define → Canonicalize.

#### نحوه کار

**مرحله 1: Extract (استخراج اولیه)**

```python
initial_result = self.extract_hybrid(
    text,
    max_entities=max_entities * 2,  # بیشتر استخراج کن
    max_relationships=max_relationships * 2,
    methods=["spacy", "llm"] if self.hf_client else ["spacy"]
)
```

**مرحله 2: Define (تعریف Schema)**

```python
schema_prompt = f"""Analyze the following extracted entities and relationships 
and define a canonical schema.

Entities: {[e.get('name', '') for e in initial_entities[:20]]}
Relationships: {[r.get('relation', '') for r in initial_relationships[:20]]}

Define canonical entity types and relation types. Return JSON:
{{
  "entity_types": ["Type1", "Type2"],
  "relation_types": ["Relation1", "Relation2"],
  "canonical_mapping": {{
    "old_relation": "canonical_relation"
  }}
}}
"""

schema_response = self.hf_client.text_generation(...)
schema_result = self._parse_llm_response(schema_response)
canonical_mapping = schema_result.get("canonical_mapping", {})
```

**مرحله 3: Canonicalize (استانداردسازی)**

```python
# استانداردسازی موجودیت‌ها
canonicalized_entities = []
entity_name_map = {}

for entity in initial_entities:
    entity_name = entity.get("name", "").strip().lower()
    normalized_name = entity_name.title()
    
    if normalized_name in entity_name_map:
        # Merge با موجودیت موجود
        existing_id = entity_name_map[normalized_name]
        # به‌روزرسانی attributes
    else:
        # موجودیت جدید
        entity_id = f"ENT_{len(canonicalized_entities)}"
        entity_name_map[normalized_name] = entity_id
        canonicalized_entities.append({...})

# استانداردسازی روابط
canonicalized_relationships = []
for rel in initial_relationships:
    relation = rel.get("relation", "").strip().lower()
    canonical_relation = canonical_mapping.get(relation, relation)
    normalized_relation = canonical_relation.title()
    
    # Map source و target به canonical entity IDs
    # اضافه کردن رابطه
```

#### مزایا
- ✅ **استانداردسازی خودکار**: یکسان‌سازی موجودیت‌ها و روابط
- ✅ **تعریف Schema**: استفاده از LLM برای تعریف schema
- ✅ **کاهش تکراری‌ها**: ادغام موجودیت‌ها و روابط مشابه

#### معایب
- ❌ **زمان پردازش زیاد**: سه مرحله پردازش
- ❌ **وابستگی به LLM**: برای مرحله Define نیاز به LLM

#### پارامترها
- `use_rag`: استفاده از RAG برای بهبود استخراج (پیش‌فرض: True)

---

### 14. Incremental

#### توضیحات
ساخت تدریجی گراف برای متن‌های طولانی با پردازش chunk به chunk و ادغام نتایج.

#### نحوه کار

**مرحله 1: تقسیم متن به Chunks**

```python
chunks = []
start = 0
text_length = len(text)

while start < text_length:
    end = min(start + chunk_size, text_length)
    chunk_text = text[start:end]
    
    # اضافه کردن context از chunk قبلی
    if start > 0 and overlap > 0:
        overlap_start = max(0, start - overlap)
        context = text[overlap_start:start]
        chunk_text = context + " " + chunk_text
    
    chunks.append({
        "text": chunk_text,
        "start": start,
        "end": end
    })
    
    start = end - overlap if overlap > 0 else end
```

**مرحله 2: پردازش Incremental**

```python
all_entities = []
all_relationships = []
entity_map = {}  # name -> entity_id
relationship_set = set()

for i, chunk in enumerate(chunks):
    # استخراج از chunk
    chunk_result = self.extract(
        chunk["text"],
        method=base_method,
        max_entities=max_entities,
        max_relationships=max_relationships
    )
    
    # ادغام موجودیت‌ها
    for entity in chunk_result.get("entities", []):
        entity_name = entity.get("name", "").strip().lower()
        
        if entity_name in entity_map:
            # موجودیت تکراری - merge attributes
            entity_id = entity_map[entity_name]
            # به‌روزرسانی attributes و track chunks
        else:
            # موجودیت جدید
            entity_id = f"ENT_{len(all_entities)}"
            entity_map[entity_name] = entity_id
            entity["id"] = entity_id
            entity["attributes"]["chunks"] = [i]
            all_entities.append(entity)
    
    # ادغام روابط
    for rel in chunk_result.get("relationships", []):
        source_name = rel.get("source", "").strip().lower()
        target_name = rel.get("target", "").strip().lower()
        
        source_id = entity_map.get(source_name)
        target_id = entity_map.get(target_name)
        
        if source_id and target_id:
            rel_key = (source_id, target_id, rel.get("relation", ""))
            if rel_key not in relationship_set:
                relationship_set.add(rel_key)
                rel["source"] = source_id
                rel["target"] = target_id
                rel["attributes"]["chunk"] = i
                all_relationships.append(rel)
            else:
                # به‌روزرسانی frequency
                for r in all_relationships:
                    if (r["source"], r["target"], r["relation"]) == rel_key:
                        r["attributes"]["frequency"] = r["attributes"].get("frequency", 1) + 1
```

#### مزایا
- ✅ **پردازش متن‌های بسیار طولانی**: بدون محدودیت طول
- ✅ **ادغام تدریجی**: ساخت گراف به صورت incremental
- ✅ **حفظ Context**: استفاده از overlap برای حفظ معنی

#### معایب
- ❌ **زمان پردازش زیاد**: پردازش چندین chunk
- ❌ **ممکن است روابط بین chunkهای دور از دست برود**: اگر overlap کافی نباشد

#### پارامترها
- `chunk_size`: اندازه هر chunk (پیش‌فرض: 500)
- `overlap`: overlap بین chunkها (پیش‌فرض: 100)
- `base_method`: روش پایه برای استخراج (spacy/simple/llm)

---

## مقایسه الگوریتم‌ها

### جدول مقایسه

| الگوریتم | سرعت | دقت | Recall | پیچیدگی | نیاز به مدل | هزینه |
|----------|------|-----|--------|---------|-------------|-------|
| **Simple** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐ | ❌ | رایگان |
| **spaCy** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ✅ | رایگان |
| **spaCy SVO** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ✅ | رایگان |
| **LLM** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ممکن است هزینه داشته باشد |
| **LLM Multi-pass** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ممکن است هزینه داشته باشد |
| **Hybrid** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ممکن است هزینه داشته باشد |
| **Persian** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | رایگان |
| **Span-based** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | رایگان |
| **With Coreference** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | رایگان |
| **Long Text** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | رایگان |
| **Joint ER** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | رایگان |
| **Autoregressive** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ممکن است هزینه داشته باشد |
| **EDC** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ممکن است هزینه داشته باشد |
| **Incremental** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | رایگان |

### راهنمای انتخاب الگوریتم

#### برای متن‌های کوتاه و ساده:
- **Simple**: اگر سرعت مهم است و متن ساختاریافته است
- **spaCy**: اگر نیاز به دقت متوسط با سرعت خوب است

#### برای متن‌های انگلیسی:
- **spaCy**: برای دقت خوب و سرعت مناسب
- **spaCy SVO Enhanced**: برای استخراج دقیق‌تر روابط
- **LLM**: برای دقت بسیار بالا

#### برای متن‌های فارسی:
- **Persian**: استفاده از مدل‌های تخصصی فارسی

#### برای متن‌های علمی:
- **Span-based**: استفاده از BioBERT/SciBERT

#### برای متن‌های طولانی:
- **Long Text**: با chunking هوشمند
- **Incremental**: برای متن‌های بسیار طولانی

#### برای حداکثر دقت:
- **LLM Multi-pass**: برای recall بسیار بالا
- **Hybrid**: ترکیب چندین روش
- **EDC**: با استانداردسازی خودکار

#### برای متن‌های با ارجاعات زیاد:
- **With Coreference**: حل ارجاعات

---

## نتیجه‌گیری

### خلاصه

سیستم تبدیل متن به گراف با **14 الگوریتم مختلف** قابلیت‌های زیر را ارائه می‌دهد:

1. **الگوریتم‌های پایه**: برای پردازش سریع و ساده
2. **الگوریتم‌های NLP**: برای دقت خوب با استفاده از مدل‌های آموزش دیده
3. **الگوریتم‌های LLM**: برای دقت بسیار بالا با استفاده از مدل‌های زبانی بزرگ
4. **الگوریتم‌های ترکیبی**: برای بهترین نتیجه با ترکیب روش‌ها
5. **الگوریتم‌های تخصصی**: برای نیازهای خاص (فارسی، علمی، ...)
6. **الگوریتم‌های پیشرفته**: برای بهینه‌سازی و استانداردسازی

### توصیه‌ها

1. **برای شروع**: از **spaCy** یا **Simple** استفاده کنید
2. **برای دقت بالا**: از **LLM** یا **Hybrid** استفاده کنید
3. **برای متن فارسی**: از **Persian** استفاده کنید
4. **برای متن‌های طولانی**: از **Long Text** یا **Incremental** استفاده کنید
5. **برای حداکثر recall**: از **LLM Multi-pass** استفاده کنید

### آینده

**پیشنهادات برای بهبود:**
- افزودن الگوریتم‌های بیشتر
- بهبود دقت الگوریتم‌های موجود
- بهینه‌سازی سرعت پردازش
- پشتیبانی از زبان‌های بیشتر
- افزودن قابلیت‌های پیشرفته‌تر

---

**تاریخ تهیه گزارش:** 17 فوریه 2026  
**نسخه:** 1.0

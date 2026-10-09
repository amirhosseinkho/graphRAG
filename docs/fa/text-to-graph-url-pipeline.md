# گزارش جامع و کد-محور بخش «تولید گراف از متن» و «استخراج متن از URL»

## 1) هدف و محدوده
این بخش از پروژه دو مسئولیت اصلی دارد:

1. دریافت ورودی `text` یا `url` و استخراج `entities/relationships`.
2. تبدیل خروجی استخراج به گراف `NetworkX.MultiDiGraph`، ذخیره در فایل، و ارسال داده‌ی مناسب برای پیش‌نمایش فرانت.

فایل‌های اصلی که این مسیر را پیاده‌سازی می‌کنند:

- `web_app.py` (endpoint و orchestration)
- `text_to_graph_service.py` (هسته‌ی extraction + graph building + post-processing)
- `url_extractor.py` (استخراج عمومی متن از صفحات وب)
- `wikipedia_extractor.py` (استخراج تخصصی ویکی‌پدیا)
- ماژول‌های پیشرفته:
  - `smart_chunker.py`
  - `hierarchical_merger.py`
  - `modular_pipeline.py`
  - `span_based_extractor.py`
  - `bert_relation_extractor.py`
  - `coreference_resolver.py`
  - `persian_models.py`

---

## 2) جریان انتها‌به‌انتها در API (`POST /api/text_to_graph`)
نقطه ورود در `web_app.py` است و به شکل زیر اجرا می‌شود:

1. خواندن JSON ورودی.
2. بررسی `text` و `url`.
3. اگر `url` وجود داشته باشد:
   - اعتبارسنجی URL با `is_valid_url`.
   - اگر URL ویکی‌پدیا باشد و `use_wikipedia_extraction=True`:
     - اجرای `WikipediaExtractor`.
     - در خطا یا خروجی ضعیف، fallback به `extract_text_from_url(..., clean_content=True)`.
   - در غیر این صورت استخراج عمومی URL.
4. خواندن پارامترهای استخراج/پردازش (method، سقف‌ها، LLM params، chunking params، ...).
5. ساخت `TextToGraphService` (با `hf_token` اگر ارسال شده باشد).
6. اجرای یکی از مسیرها:
   - `process_url_to_graph(...)`
   - `process_text_to_graph(...)`
7. اگر داده‌ی `wikipedia_entities/relationships` در درخواست وجود داشته باشد:
   - merge با خروجی extraction
   - dedup نسبی بر اساس نام/کلید رابطه
   - rebuild گراف
8. ساخت `graph_data` برای پیش‌نمایش (`nodes`/`edges`).
9. بازگشت پاسخ نهایی با `filepath`, `stats`, `graph_data`, `extraction_data`.

### ساختار پاسخ endpoint
به‌صورت عملی، پاسخ شامل این فیلدهاست:

- `success`
- `filename`, `filepath`
- `stats`
- `extraction_method`
- `resolution_summary`
- `load_url`
- `graph_data`:
  - `nodes`: `id`, `label`, `type`, `kind`, `title`
  - `edges`: `from`, `to`, `label`, `metaedge`, `relation`, `relation_meaning`, `title`
- `extraction_data`:
  - `entities`
  - `relationships`

---

## 3) استخراج متن از URL (عمومی) در `url_extractor.py`

### 3.1 اعتبارسنجی URL
تابع `is_valid_url`:

- `scheme` باید `http` یا `https` باشد.
- `netloc` باید معتبر باشد.

### 3.2 واکشی صفحه
تابع `extract_text_from_url`:

- `requests.get` با `timeout` و `User-Agent` مرورگر.
- در خطاهای `Timeout` یا `RequestException` خروجی `None`.

### 3.3 پاک‌سازی HTML (`clean_content=True`)
در `_extract_clean_content`:

1. حذف تگ‌های غیرمفید: `script/style/meta/link/noscript/iframe/...`
2. حذف `nav/header/footer/aside`.
3. حذف کلاس‌های تبلیغاتی و popup/cookie/banner.
4. یافتن ناحیه اصلی محتوا با اولویت:
   - `main/article/section`
   - `div` با کلاس‌های محتوایی
   - fallback: `body`
5. استخراج ساختاریافته:
   - headingها
   - پاراگراف‌های معنادار (طول > 30 و نسبت لینک پایین)
   - لیست‌ها

### 3.4 پاک‌سازی نهایی و truncate
- `_clean_text`: حذف فاصله/خطوط اضافه و کاراکترهای کنترلی.
- `_truncate_text_intelligently`:
  - اول پایان پاراگراف
  - سپس پایان جمله
  - سپس مرز کلمه
  - در نهایت برش مستقیم
- در صورت truncate، پسوند `[...]` به متن اضافه می‌شود.

### 3.5 fallback بدون BeautifulSoup
اگر BeautifulSoup نصب نباشد:

- حذف HTML با regex
- خروجی متنی ساده‌تر (کیفیت پایین‌تر نسبت به مسیر پاک‌سازی ساختاریافته)

---

## 4) استخراج تخصصی ویکی‌پدیا در `wikipedia_extractor.py`

### 4.1 ورودی
- `extract_from_url(url)` عنوان را از مسیر `/wiki/...` استخراج می‌کند.
- سپس `extract_from_title(title)` اجرا می‌شود.

### 4.2 استراتژی چندمرحله‌ای استخراج
در `extract_from_title` سه لایه fallback دارد:

1. **HTML parsing** (`_extract_via_html`):
   - `sections`, `infobox`, `categories`, `links`
   - تولید متن از sectionها
2. **API extraction** (`_extract_via_api`) اگر متن کافی نبود:
   - MediaWiki `action=query&prop=extracts`
   - REST `page/html`
   - REST `page/summary` (fallback)
3. **Direct HTML fallback**:
   - واکشی مستقیم `/wiki/<title>` و parse مجدد

### 4.3 پاک‌سازی متن ویکی‌پدیا
`_clean_wikipedia_text`:

- حذف markers ارجاع مثل `[1]`
- حذف citation templates
- نرمال‌سازی whitespace
- truncate هوشمند با `_truncate_text_intelligently`

### 4.4 استخراج داده ساختاریافته برای گراف
`_extract_entities_and_relations`:

- entity اصلی = عنوان صفحه (`ENTITY_0`)
- Infobox -> entity جدید + relation با کلید infobox
- Categories (تا 10 مورد) -> entity نوع `Category` + relation `BELONGS_TO`
- Internal links (تا 30 مورد) -> entity `Concept` + relation `RELATED_TO`

### 4.5 ادغام در endpoint
در `web_app.py`، اگر داده‌ی wiki موجود باشد:

- merge موجودیت‌ها بر اساس name
- افزودن `wikipedia_source` / `source='wikipedia'`
- dedup رابطه با کلید `(source, target, metaedge)`
- rebuild گراف با `build_graph`

نکته مهم: `TextToGraphService.process_url_to_graph` خودش متنی که از ویکی استخراج شده را پردازش می‌کند، اما merge صریح `wikipedia_entities/relationships` در لایه endpoint انجام می‌شود.

---

## 5) هسته‌ی تولید گراف در `TextToGraphService`

### 5.1 مقداردهی اولیه سرویس
در `__init__`:

- تلاش برای load مدل spaCy انگلیسی (`en_core_web_sm`) و فارسی (`fa_core_news_sm`)
- OpenAI client عملاً غیرفعال (کامنت شده)
- HuggingFace `InferenceClient` با `HF_TOKEN/HUGGINGFACE_API_KEY`
- مقداردهی `EntityResolution` (اگر ماژول موجود باشد)
- مقداردهی Persian normalizer (اختیاری)

### 5.2 متدهای مسیر اصلی
- `process_url_to_graph`:
  - استخراج متن URL (ویکی/عمومی)
  - سپس `process_text_to_graph`
- `process_text_to_graph`:
  - `extract(...)`
  - `build_graph(...)`
  - post-processing
  - save

---

## 6) الگوریتم روش‌های استخراج (`method`)

## 6.1 `simple` (Rule-based)
- موجودیت‌ها:
  - gene regex: حروف بزرگ/عدد
  - disease keyword matching
  - compound suffix regex
- روابط:
  - regex الگوهای `regulates/interacts with/treats/...`
- اگر source/target موجود نباشند، entity جدید می‌سازد.

## 6.2 `spacy`
- موجودیت‌ها:
  - `doc.ents`
  - noun chunkهای کوتاهِ غیر stop-word
- روابط در سه مسیر:
  1. وابستگی فعل + subject/object
  2. الگوهای preposition-based
  3. co-occurrence در جمله با فعل میانی
- نگاشت فعل به metaedge با `_map_verb_to_relation`.

## 6.3 `spacy_svo_enhanced`
- استخراج SVO با تمرکز روی token span و overlap با entity spans.
- پوشش prepositional object.
- relation متنی از lemma/verb phrase تولید می‌شود.

## 6.4 `llm`
- مسیر فعال: HuggingFace Chat Completions.
- prompt سخت‌گیرانه برای JSON و تاکید بر relation متنی واقعی (نه کد metaedge).
- parsing مقاوم خروجی با `_parse_llm_response` (ترمیم JSON نیمه‌خراب).
- `_process_llm_result`:
  - ترمیم relation خالی/کدگونه
  - فیلتر confidence
  - limit خروجی

## 6.5 `llm_multipass`
- پاس اول استخراج کامل.
- پاس‌های follow-up (`max_gleanings`) فقط برای موارد جاافتاده.
- dedup:
  - entity بر اساس `ent_id`
  - relationship بر اساس `(source,target,metaedge)`
- پس‌فیلتر متن‌مبنا (`_filter_extractions_by_text`) برای کاهش hallucination:
  - فقط entityهایی که در متن قابل تطبیق‌اند
  - فقط رابطه‌هایی که شواهد متنی relation/verb/description دارند
  - برای متن‌های خیلی کوتاه (<=20 token) این فیلتر skip می‌شود.

## 6.6 `hybrid`
- اجرای چند روش (مثل `spacy + llm`).
- merge entity بر اساس `name` (ترجیح confidence/attributes بهتر).
- merge رابطه بر اساس `(source,target,metaedge)` با ترکیب weight/description/confidence.

## 6.7 `persian`
- نرمال‌سازی فارسی.
- اجرای `ModularExtractionPipeline(language='fa')`:
  - normalization
  - NER
  - relation extraction
  - (اختیاری) coreference
- تبدیل خروجی pipeline به قالب استاندارد service.
- fallback به `extract_spacy` در خطا.

## 6.8 `span_based`
- NER با `BioBERT/SciBERT/SpanBasedExtractor`.
- relation extraction با `BERTRelationExtractor` روی جفت موجودیت‌ها.
- نگاشت خروجی به ساختار استاندارد (`id/name/type/attributes`).

## 6.9 `with_coreference`
- اول extraction با `base_method`.
- سپس `CoreferenceResolver`:
  - resolve
  - merge entityهای ارجاعی

## 6.10 `long_text`
- chunking با `SmartChunker` (`smart/sentence/paragraph/sliding_window`).
- استخراج مستقل روی هر chunk.
- merge با `HierarchicalMerger` بر اساس frequency/confidence.

## 6.11 `joint_er`
- استخراج اولیه entity/relation با spaCy + dependency.
- ساخت گراف موقت.
- چند iteration بهینه‌سازی ساختار:
  - degree/betweenness centrality
  - حذف روابط ضعیف‌تر

## 6.12 `autoregressive`
- تولید ترتیبی JSON گراف با prompt خاص.
- در خطا fallback به `llm`.

## 6.13 `edc` (Extract-Define-Canonicalize)
1. Extract: استخراج اولیه (عملاً hybrid)
2. Define: تعریف schema/canonical mapping با LLM (اختیاری)
3. Canonicalize:
   - merge موجودیت‌های هم‌نام
   - canonical relation mapping
   - dedup روابط

## 6.14 `incremental`
- chunking کاراکتری با overlap.
- extraction chunk-by-chunk با `base_method`.
- merge تدریجی موجودیت‌ها/روابط و افزایش frequency روی تکرار رابطه.

---

## 7) ماژول‌های کمکی پیشرفته

## 7.1 `SmartChunker`
- استراتژی‌ها:
  - `sentence`
  - `paragraph`
  - `smart`
  - `sliding_window`
- تشخیص زبان ساده (`fa/en`).
- sentence split با `hazm` (فارسی) یا spaCy یا regex fallback.
- تخمین token تقریبی: `words * 1.33`.

## 7.2 `HierarchicalMerger`
- تجمیع entity/relation از چند chunk.
- نرمال‌سازی متن برای مقایسه.
- محاسبه frequency و average confidence.
- حذف موارد زیر `min_confidence`.
- رتبه‌بندی بر اساس `weight * confidence`.

## 7.3 `ModularExtractionPipeline`
مراحل قابل فعال/غیرفعال:

1. `NORMALIZATION`
2. `NER`
3. `RELATION_EXTRACTION`
4. `COREFERENCE_RESOLUTION`

برای فارسی از مدل‌های `persian_models` استفاده می‌کند.

## 7.4 `SpanBasedExtractor`
- تلاش اول: `transformers pipeline("token-classification")`.
- fallback: مدل توکنی مستقیم و بازسازی spanها با B-/I- labels.
- کلاس‌های آماده:
  - `BioBERTExtractor`
  - `SciBERTExtractor`

## 7.5 `BERTRelationExtractor`
- برای جفت entityها جمله‌ی context پیدا می‌کند.
- embedding مبتنی بر BERT (`[CLS]`).
- طبقه‌بندی رابطه در این کد عمدتاً heuristic/pattern-based است (نه classifier fully fine-tuned).
- خروجی relation + metaedge mapping.

## 7.6 `CoreferenceResolver`
- انگلیسی: در صورت وجود `neuralcoref`.
- fallback الگوی ساده (فارسی/انگلیسی) + nearest-entity mapping.
- سپس merge references در entityها.

## 7.7 `persian_models`
- `PersianNERModel`:
  - تلاش برای pipeline، در خطا fallback rule-based ساده.
- `PersianRelationExtractor`:
  - مدل seq2seq (mT5) برای relation از جمله‌ی حاوی دو entity.

---

## 8) ساخت گراف و پس‌پردازش

## 8.1 `build_graph`
- نوع گراف: `nx.MultiDiGraph`.
- Node attrs:
  - `name`, `type`, `kind`, و `attributes`
- Edge attrs:
  - `metaedge`
  - `relation`
  - `relation_meaning`
  - بقیه attributes

### نکته مهم در edges
رابطه فقط زمانی اضافه می‌شود که هر دو نود (`source`, `target`) در گراف موجود باشند.

## 8.2 وزن‌دهی روابط
`_apply_relationship_weighting`:

- `frequency_weight = count(u,v,metaedge)`
- `confidence = attributes.confidence` (پیش‌فرض 0.5)
- `weight = frequency_weight * (0.5 + 0.5 * confidence)`

## 8.3 پالایش اختیاری
در `process_text_to_graph`:

- `Entity Resolution` (اگر فعال و ماژول موجود باشد)
- حذف روابط با وزن کمتر از `min_relationship_weight`
- حذف نودهای ایزوله (`remove_isolated_nodes=True`)

## 8.4 آمار
`_calculate_graph_statistics`:

- `num_nodes`, `num_edges`
- توزیع نوع نود و یال
- `avg_degree`
- `density`

## 8.5 ذخیره
`save_graph`:

- مسیر پیش‌فرض: `uploaded_graphs`
- نام پیش‌فرض: `<timestamp>_text_graph.pkl`

---

## 9) پارامترهای API مهم

پارامترهای عمومی:

- `text` یا `url`
- `method`
- `max_entities`, `max_relationships`

پارامترهای LLM:

- `llm_model`
- `hf_token`
- `confidence_threshold`
- `max_gleanings`

پارامترهای graph post-processing:

- `enable_entity_resolution`
- `enable_relationship_weighting`
- `min_relationship_weight`
- `remove_isolated_nodes`

پارامترهای روش‌های پیشرفته:

- `hybrid_methods`
- `enable_coreference`
- `chunking_strategy`, `chunk_overlap`, `max_tokens`
- `span_model_type`
- `structure_iterations`
- `max_generation_length`
- `use_rag`
- `chunk_size`, `overlap`, `base_method`

---

## 10) جمع‌بندی فنی
این بخش عملاً یک pipeline چندمسیره و قابل تنظیم ساخته است که:

- هم متن مستقیم و هم URL را پوشش می‌دهد.
- برای URL عمومی، پاک‌سازی ساختاریافته HTML انجام می‌دهد.
- برای ویکی‌پدیا، مسیر اختصاصی چندمرحله‌ای (HTML + API + fallback) دارد.
- از روش‌های extraction متنوع (rule-based، spaCy، LLM، hybrid، فارسی، span-based، long-text، incremental) پشتیبانی می‌کند.
- خروجی را به گراف چندیالی NetworkX تبدیل می‌کند و داده‌ی مناسب برای:
  - ذخیره
  - پیش‌نمایش فرانت
  - بارگذاری در چرخه GraphRAG
  تولید می‌کند.

به‌صورت عملی، این طراحی باعث می‌شود سیستم فقط مصرف‌کننده‌ی گراف از پیش‌ساخته نباشد، بلکه بتواند از متن/وب، گراف جدید بسازد و همان‌جا وارد چرخه پرسش‌وپاسخ کند.

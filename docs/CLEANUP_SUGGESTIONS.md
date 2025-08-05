# پیشنهادات پاک‌سازی فایل graphrag_service.py

## توابع قابل حذف (تکراری و غیرضروری)

### 1. توابع تولید متن زمینه تکراری

#### حذف این توابع:
```python
# این توابع تکراری هستند و می‌توانند حذف شوند
def _create_simple_context_text(self, retrieval_result: RetrievalResult) -> str:
def _create_scientific_analytical_context(self, retrieval_result: RetrievalResult) -> str:
def _create_narrative_context(self, retrieval_result: RetrievalResult) -> str:
def _create_data_driven_context(self, retrieval_result: RetrievalResult) -> str:
def _create_step_by_step_context(self, retrieval_result: RetrievalResult) -> str:
def _create_compact_direct_context(self, retrieval_result: RetrievalResult) -> str:
def _create_clinical_relevance_context(self, retrieval_result: RetrievalResult) -> str:
def _create_mechanistic_detailed_context(self, retrieval_result: RetrievalResult) -> str:
def _create_enhanced_intelligent_context_text(self, retrieval_result: RetrievalResult) -> str:
def _create_advanced_context_text(self, retrieval_result: RetrievalResult) -> str:
```

#### دلیل حذف:
- این توابع با توابع موجود در `enhanced_context_generator.py` تکراری هستند
- عملکرد مشابه دارند اما کیفیت پایین‌تری
- باعث سردرگمی در کد می‌شوند

### 2. توابع تولید پاسخ تکراری

#### حذف این توابع:
```python
# این توابع نسخه‌های قدیمی هستند
def _generate_relationship_answer(self, retrieval_result: RetrievalResult) -> str:
def _generate_drug_treatment_answer(self, retrieval_result: RetrievalResult) -> str:
def _generate_gene_function_answer(self, retrieval_result: RetrievalResult) -> str:
def _generate_disease_info_answer(self, retrieval_result: RetrievalResult) -> str:
def _generate_anatomy_expression_answer(self, retrieval_result: RetrievalResult) -> str:
def _generate_general_answer(self, retrieval_result: RetrievalResult) -> str:
def _generate_gene_cancer_answer(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes) -> str:
```

#### دلیل حذف:
- نسخه‌های قدیمی و ساده‌تر هستند
- با توابع `_generate_intelligent_*_answer()` تکراری هستند
- کیفیت پایین‌تری دارند

### 3. توابع تحلیل تکراری

#### حذف این توابع:
```python
# این توابع تحلیل تکراری هستند
def _custom_relationship_analysis(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes, drug_nodes) -> List[str]:
def _custom_drug_analysis(self, retrieval_result: RetrievalResult, drug_nodes, disease_nodes) -> List[str]:
def _custom_gene_analysis(self, retrieval_result: RetrievalResult, gene_nodes, process_nodes) -> List[str]:
def _custom_disease_analysis(self, retrieval_result: RetrievalResult, disease_nodes, gene_nodes, symptom_nodes) -> List[str]:
def _custom_anatomy_analysis(self, retrieval_result: RetrievalResult, anatomy_nodes, gene_nodes) -> List[str]:
def _custom_general_analysis(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes, drug_nodes, anatomy_nodes, process_nodes) -> List[str]:
```

#### دلیل حذف:
- با توابع `_generate_intelligent_*_answer()` تکراری هستند
- عملکرد مشابه دارند
- باعث پیچیدگی غیرضروری می‌شوند

## توابع نگه‌داری (ضروری)

### 1. توابع اصلی تولید متن زمینه:
```python
def create_context_text(self, nodes: List[GraphNode], edges: List[GraphEdge], paths: List[List[str]]) -> str:
def _create_enhanced_context_text(self, retrieval_result: RetrievalResult) -> str:
def _create_intelligent_context_text(self, retrieval_result: RetrievalResult) -> str:
def _enrich_retrieved_data(self, nodes: List[GraphNode], edges: List[GraphEdge], query: str) -> Dict[str, Any]:
def _create_biological_context(self, enriched_data: Dict, query: str) -> str:
```

### 2. توابع هوشمند تولید پاسخ:
```python
def _generate_intelligent_relationship_answer(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes, drug_nodes) -> str:
def _generate_intelligent_drug_answer(self, retrieval_result: RetrievalResult, drug_nodes, disease_nodes) -> str:
def _generate_intelligent_gene_answer(self, retrieval_result: RetrievalResult, gene_nodes, process_nodes) -> str:
def _generate_intelligent_disease_answer(self, retrieval_result: RetrievalResult, disease_nodes, gene_nodes) -> str:
def _generate_intelligent_anatomy_answer(self, retrieval_result: RetrievalResult, anatomy_nodes, gene_nodes) -> str:
def _generate_intelligent_general_answer(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes, drug_nodes, anatomy_nodes, process_nodes) -> str:
```

### 3. توابع کمکی:
```python
def _get_anatomy_significance(self, anatomy_name: str) -> str:
def _identify_central_gene(self, nodes: List[GraphNode], query: str) -> Optional[str]:
def _create_path_description(self, path: List[str], edges: List[GraphEdge]) -> str:
def _create_biological_inference(self, nodes: List[GraphNode], edges: List[GraphEdge], paths: List[List[str]], query: str) -> str:
```

## مزایای پاک‌سازی:

### 1. کاهش حجم کد:
- حذف حدود 20-25 تابع تکراری
- کاهش پیچیدگی کد
- بهبود خوانایی

### 2. بهبود نگهداری:
- کد ساده‌تر و قابل فهم‌تر
- کاهش احتمال خطا
- آسان‌تر شدن دیباگ

### 3. بهبود عملکرد:
- کاهش زمان کامپایل
- کاهش مصرف حافظه
- بهبود سرعت اجرا

### 4. یکپارچگی بهتر:
- استفاده از `enhanced_context_generator.py` به عنوان منبع اصلی
- جلوگیری از تکرار کد
- استانداردسازی روش‌ها

## مراحل پاک‌سازی:

### مرحله 1: پشتیبان‌گیری
```bash
cp graphrag_service.py graphrag_service_backup.py
```

### مرحله 2: حذف توابع تکراری
- حذف توابع تولید متن زمینه تکراری
- حذف توابع تولید پاسخ قدیمی
- حذف توابع تحلیل تکراری

### مرحله 3: به‌روزرسانی فراخوانی‌ها
- تغییر فراخوانی توابع حذف شده به توابع جدید
- به‌روزرسانی مستندات
- تست عملکرد

### مرحله 4: تست و اعتبارسنجی
- اجرای تست‌های موجود
- بررسی عملکرد سیستم
- اطمینان از عدم شکست

## نتیجه‌گیری:

با حذف این توابع تکراری:
- حجم کد حدود 30-40% کاهش می‌یابد
- کیفیت کد بهبود می‌یابد
- نگهداری آسان‌تر می‌شود
- عملکرد سیستم بهتر می‌شود 
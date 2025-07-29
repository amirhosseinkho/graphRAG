# 🧬 GraphRAG System Architecture Documentation

## 📋 فهرست مطالب
1. [معماری کلی سیستم](#معماری-کلی-سیستم)
2. [الگوریتم‌های بازیابی](#الگوریتم‌های-بازیابی)
3. [تولید متن](#تولید-متن)
4. [منطق هوش مصنوعی](#منطق-هوش-مصنوعی)
5. [ساختار فایل‌ها](#ساختار-فایل‌ها)

---

## 🏗️ معماری کلی سیستم

### 🔄 جریان پردازش اصلی
```
سوال ورودی → استخراج کلمات کلیدی → تطبیق با نودها → بازیابی → تولید پاسخ
```

### 📊 اجزای اصلی
- **GraphRAGService**: کلاس اصلی سیستم
- **RetrievalMethod**: روش‌های مختلف بازیابی
- **GenerationModel**: مدل‌های مختلف تولید متن
- **Hetionet Graph**: گراف دانش زیستی

---

## 🔍 الگوریتم‌های بازیابی

### 1. **Intelligent Semantic Search** (پیشرفته‌ترین)
**موقعیت**: `intelligent_semantic_search()` در خط 639

#### منطق کار:
```python
def intelligent_semantic_search(self, query: str, max_depth: int = 3):
    # 1. تحلیل قصد سوال
    intent = self.analyze_question_intent(query)
    
    # 2. تشخیص نوع سوال
    if self._is_gene_cancer_question(query, matched_nodes):
        return self._search_gene_cancer_relationships(query, matched_nodes, max_depth)
    
    # 3. انتخاب metaedge های مناسب
    target_metaedges = self._get_target_metaedges_for_question(intent['question_type'], query)
    
    # 4. جستجوی هدفمند
    return self._search_by_metaedges(matched_nodes, intent, target_metaedges, max_depth)
```

#### ویژگی‌های کلیدی:
- **تحلیل مفهومی**: تشخیص نوع سوال (ژن-سرطان، دارو-بیماری، و...)
- **جستجوی هدفمند**: انتخاب metaedge های مناسب
- **امتیازدهی هوشمند**: بر اساس نوع رابطه و عمق
- **جستجوی عمیق**: DFS برای روابط غیرمستقیم

### 2. **BFS Search** (جستجوی سطحی)
**موقعیت**: `bfs_search()` در خط 1298

```python
def bfs_search(self, start_node: str, max_depth: int = 2):
    queue = [(start_node, 0)]
    visited = set()
    results = []
    
    while queue:
        node, depth = queue.pop(0)
        if depth > max_depth:
            continue
            
        for neighbor in self.G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                results.append((neighbor, depth + 1))
                queue.append((neighbor, depth + 1))
    
    return results
```

### 3. **DFS Search** (جستجوی عمیق)
**موقعیت**: `dfs_search()` در خط 1316

```python
def dfs_search(self, start_node: str, max_depth: int = 2, relation_filter: str = None):
    def dfs(node, depth):
        if depth > max_depth:
            return
            
        for neighbor in self.G.neighbors(node):
            edge_data = self.G.get_edge_data(node, neighbor)
            if relation_filter and edge_data.get('metaedge') != relation_filter:
                continue
                
            results.append((neighbor, depth))
            dfs(neighbor, depth + 1)
    
    results = []
    dfs(start_node, 0)
    return results
```

### 4. **Hybrid Search** (ترکیبی)
**موقعیت**: `hybrid_search()` در خط 1348

```python
def hybrid_search(self, nodes: List[str], max_depth: int = 2):
    # ترکیب BFS و DFS
    bfs_results = []
    dfs_results = []
    
    for node in nodes:
        bfs_results.extend(self.bfs_search(node, max_depth))
        dfs_results.extend(self.dfs_search(node, max_depth))
    
    # ادغام و رتبه‌بندی نتایج
    return self._merge_and_rank_results(bfs_results, dfs_results)
```

### 5. **Multi-Method Search** (چندروشی)
**موقعیت**: `multi_method_search()` در خط 1363

```python
def multi_method_search(self, nodes: List[str], max_depth: int = 2):
    methods = [
        self.bfs_search,
        self.dfs_search,
        self.get_shortest_paths
    ]
    
    all_results = []
    for method in methods:
        for node in nodes:
            results = method(node, max_depth)
            all_results.extend(results)
    
    return self._deduplicate_and_rank(all_results)
```

### 6. **Ensemble Search** (مجموعه‌ای)
**موقعیت**: `ensemble_search()` در خط 1401

```python
def ensemble_search(self, nodes: List[str], max_depth: int = 2):
    # اجرای چندین روش و رای‌گیری
    methods_results = {
        'bfs': self.bfs_search,
        'dfs': self.dfs_search,
        'shortest_path': self.get_shortest_paths
    }
    
    votes = {}
    for method_name, method_func in methods_results.items():
        for node in nodes:
            results = method_func(node, max_depth)
            for result, depth in results:
                votes[result] = votes.get(result, 0) + 1
    
    # رتبه‌بندی بر اساس تعداد رای
    return sorted(votes.items(), key=lambda x: x[1], reverse=True)
```

---

## 🤖 تولید متن

### 1. **GPT Simulation** (پیشرفته‌ترین)
**موقعیت**: `gpt_simulation_generation()` در خط 2761

#### منطق کار:
```python
def gpt_simulation_generation(self, retrieval_result: RetrievalResult) -> str:
    # 1. تشخیص نوع سوال از محتوای بازیابی شده
    if self._is_gene_cancer_question_from_context(retrieval_result):
        return self._generate_gene_cancer_answer(retrieval_result, gene_nodes, disease_nodes)
    
    # 2. دسته‌بندی نودها
    gene_nodes = [n for n in retrieval_result.nodes if n.kind == 'Gene']
    disease_nodes = [n for n in retrieval_result.nodes if n.kind == 'Disease']
    drug_nodes = [n for n in retrieval_result.nodes if n.kind in ['Drug', 'Compound']]
    
    # 3. انتخاب تابع تولید مناسب
    if gene_nodes and disease_nodes:
        return self._generate_intelligent_relationship_answer(retrieval_result, gene_nodes, disease_nodes, drug_nodes)
    elif drug_nodes:
        return self._generate_intelligent_drug_answer(retrieval_result, drug_nodes, disease_nodes)
    elif gene_nodes:
        return self._generate_intelligent_gene_answer(retrieval_result, gene_nodes, process_nodes)
    else:
        return self._generate_intelligent_general_answer(retrieval_result, gene_nodes, disease_nodes, drug_nodes, anatomy_nodes, process_nodes)
```

### 2. **تولید پاسخ ژن-سرطان**
**موقعیت**: `_generate_gene_cancer_answer()` در خط 3112

```python
def _generate_gene_cancer_answer(self, retrieval_result: RetrievalResult, gene_nodes, disease_nodes) -> str:
    answer_parts = ["🧬 **تحلیل تخصصی ژن-سرطان:**\n"]
    
    # 1. شناسایی ژن‌های اصلی
    primary_genes = []
    for gene in gene_nodes:
        if any(famous in gene.name.lower() for famous in ['tp53', 'brca1', 'brca2']):
            primary_genes.append(gene)
    
    # 2. تحلیل تخصصی TP53
    if any('tp53' in gene.name.lower() for gene in primary_genes):
        answer_parts.append("🔬 **تحلیل تخصصی TP53:**")
        answer_parts.append("TP53 یکی از مهم‌ترین ژن‌های سرکوبگر تومور است که:")
        answer_parts.append("• در بیش از 50% سرطان‌های انسانی جهش یافته است")
        answer_parts.append("• نقش کلیدی در تنظیم چرخه سلولی و آپوپتوز دارد")
    
    # 3. تحلیل روابط
    if retrieval_result.edges:
        gene_cancer_edges = []
        for edge in retrieval_result.edges:
            # استخراج روابط ژن-سرطان
            if self._is_gene_cancer_edge(edge, retrieval_result.nodes):
                gene_cancer_edges.append(edge)
        
        answer_parts.append("**تحلیل روابط یافت شده:**")
        for edge in gene_cancer_edges[:5]:
            answer_parts.append(f"• {edge.source} → {edge.target} ({edge.relation})")
    
    return "\n".join(answer_parts)
```

### 3. **تولید پاسخ هوشمند**
**موقعیت**: `_generate_intelligent_*_answer()` در خطوط 2794-3025

#### منطق کلی:
```python
def _generate_intelligent_relationship_answer(self, retrieval_result, gene_nodes, disease_nodes, drug_nodes):
    answer_parts = []
    
    # 1. تحلیل ژن‌ها
    if gene_nodes:
        answer_parts.append("**ژن‌های یافت شده:**")
        for gene in sorted(gene_nodes, key=lambda x: x.score, reverse=True)[:5]:
            answer_parts.append(f"• **{gene.name}** (امتیاز: {gene.score:.2f})")
    
    # 2. تحلیل بیماری‌ها
    if disease_nodes:
        answer_parts.append("**بیماری‌های مرتبط:**")
        for disease in disease_nodes:
            answer_parts.append(f"• {disease.name}")
    
    # 3. تحلیل روابط
    if retrieval_result.edges:
        relations_count = {}
        for edge in retrieval_result.edges:
            relations_count[edge.relation] = relations_count.get(edge.relation, 0) + 1
        
        answer_parts.append("**روابط مهم:**")
        for relation, count in sorted(relations_count.items(), key=lambda x: x[1], reverse=True):
            answer_parts.append(f"• {relation}: {count} رابطه")
    
    return "\n".join(answer_parts)
```

---

## 🧠 منطق هوش مصنوعی

### 1. **تحلیل قصد سوال**
**موقعیت**: `analyze_question_intent()` در خط 481

```python
def analyze_question_intent(self, query: str) -> Dict[str, Any]:
    # 1. استخراج کلمات کلیدی
    keywords = self.extract_keywords(query)
    
    # 2. تشخیص نوع موجودیت‌ها
    entity_types = []
    for keyword in keywords:
        matched_nodes = self.match_tokens_to_nodes([keyword])
        for node_id in matched_nodes.values():
            node_attrs = self.G.nodes[node_id]
            entity_types.append(node_attrs.get('kind'))
    
    # 3. تشخیص نوع سوال
    question_type = self._detect_question_type(query.lower(), entity_types)
    
    # 4. انتخاب metaedge های مناسب
    target_metaedges = self._get_target_metaedges_for_question(question_type, query)
    
    return {
        'question_type': question_type,
        'entity_types': list(set(entity_types)),
        'target_metaedges': target_metaedges,
        'keywords': keywords
    }
```

### 2. **تشخیص سوال ژن-سرطان**
**موقعیت**: `_is_gene_cancer_question()` در خط 746

```python
def _is_gene_cancer_question(self, query: str, matched_nodes: Dict[str, str]) -> bool:
    query_lower = query.lower()
    
    # 1. بررسی کلمات کلیدی سرطان
    cancer_keywords = ['cancer', 'tumor', 'malignancy', 'oncology', 'carcinoma']
    has_cancer = any(keyword in query_lower for keyword in cancer_keywords)
    
    # 2. بررسی وجود ژن
    has_gene = any(self.G.nodes[node_id].get('kind') == 'Gene' 
                   for node_id in matched_nodes.values())
    
    return has_cancer and has_gene
```

### 3. **امتیازدهی هوشمند**
**موقعیت**: `_calculate_metaedge_score()` در خط 3759

```python
def _calculate_metaedge_score(self, metaedge: str, depth: int) -> float:
    # امتیاز پایه بر اساس نوع رابطه
    base_scores = {
        'DaG': 5.4,  # Disease-associates-Gene
        'DuG': 4.0,  # Disease-upregulates-Gene
        'DdG': 4.0,  # Disease-downregulates-Gene
        'GaD': 3.2,  # Gene-associates-Disease
        'GuD': 2.8,  # Gene-upregulates-Disease
        'GdD': 2.8,  # Gene-downregulates-Disease
        'GiG': 3.0,  # Gene-interacts-Gene
        'Gr>G': 2.5, # Gene-regulates-Gene
        'GcG': 2.0,  # Gene-covaries-Gene
    }
    
    base_score = base_scores.get(metaedge, 1.0)
    
    # کاهش امتیاز بر اساس عمق
    depth_penalty = 1.0 / (depth + 1)
    
    return base_score * depth_penalty
```

### 4. **جستجوی هدفمند**
**موقعیت**: `_search_by_metaedges()` در خط 3663

```python
def _search_by_metaedges(self, matched_nodes: Dict[str, str], intent: Dict, 
                         target_metaedges: List[str], max_depth: int = 2):
    results = []
    
    for node_token, node_id in matched_nodes.items():
        node_attrs = self.G.nodes[node_id]
        
        # 1. جستجوی مستقیم
        for neighbor in self.G.neighbors(node_id):
            neighbor_attrs = self.G.nodes[neighbor]
            edge_data = self.G.get_edge_data(node_id, neighbor)
            
            if edge_data and edge_data.get('metaedge') in target_metaedges:
                score = self._calculate_metaedge_score(edge_data.get('metaedge'), 1)
                results.append((neighbor, 1, score, f"Direct {edge_data.get('metaedge')}"))
        
        # 2. جستجوی معکوس
        for other_node, other_attrs in self.G.nodes(data=True):
            if other_node != node_id:
                for neighbor in self.G.neighbors(other_node):
                    if neighbor == node_id:
                        edge_data = self.G.get_edge_data(other_node, neighbor)
                        if edge_data and edge_data.get('metaedge') in target_metaedges:
                            score = self._calculate_metaedge_score(edge_data.get('metaedge'), 1) * 0.8
                            results.append((other_node, 1, score, f"Reverse {edge_data.get('metaedge')}"))
        
        # 3. جستجوی عمیق
        if max_depth > 1:
            dfs_results = self.dfs_search(node_id, max_depth)
            for found_node, depth in dfs_results:
                found_attrs = self.G.nodes[found_node]
                score = self._calculate_metaedge_score('Unknown', depth)
                results.append((found_node, depth, score, f"DFS depth {depth}"))
    
    return results
```

---

## 📁 ساختار فایل‌ها

### فایل‌های اصلی:
```
tir/
├── graphrag_service.py          # کلاس اصلی سیستم
├── main_grpahrag.py            # نقطه ورودی اصلی
├── web_app.py                  # رابط وب
├── test_*.py                   # فایل‌های تست
└── static/                     # فایل‌های استاتیک
    ├── css/
    └── js/
```

### کلاس‌های اصلی:

#### 1. **GraphRAGService** (خط 77)
- **وظیفه**: مدیریت کل سیستم
- **متدهای کلیدی**:
  - `process_query()`: پردازش کامل سوال
  - `retrieve_information()`: بازیابی اطلاعات
  - `generate_answer()`: تولید پاسخ

#### 2. **RetrievalMethod** (خط 19)
```python
class RetrievalMethod(Enum):
    BFS = "BFS"
    DFS = "DFS"
    SHORTEST_PATH = "Shortest Path"
    NEIGHBORS = "Neighbors"
    HYBRID = "Hybrid"
    MULTI_METHOD = "Multi-Method"
    ENSEMBLE = "Ensemble"
    ADAPTIVE = "Adaptive"
    INTELLIGENT = "Intelligent Semantic Search"
    NO_RETRIEVAL = "بدون بازیابی (فقط مدل)"
```

#### 3. **GenerationModel** (خط 32)
```python
class GenerationModel(Enum):
    SIMPLE = "Simple Template"
    GPT_SIMULATION = "GPT Simulation"
    CUSTOM = "Custom Model"
    HUGGINGFACE = "HuggingFace Models"
    OPENAI_GPT = "OpenAI GPT"
    ANTHROPIC_CLAUDE = "Anthropic Claude"
    GOOGLE_GEMINI = "Google Gemini"
```

### کلاس‌های داده:

#### 1. **GraphNode** (خط 42)
```python
@dataclass
class GraphNode:
    id: str
    name: str
    kind: str
    depth: int = 0
    score: float = 1.0
```

#### 2. **GraphEdge** (خط 51)
```python
@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
```

#### 3. **RetrievalResult** (خط 59)
```python
@dataclass
class RetrievalResult:
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    paths: List[List[str]]
    context_text: str
    method: str
    query: str
```

#### 4. **GenerationResult** (خط 69)
```python
@dataclass
class GenerationResult:
    answer: str
    model: str
    context_used: str
    confidence: float
```

---

## 🔄 جریان کامل پردازش

### مرحله 1: ورودی
```python
query = "How does TP53 relate to cancer?"
result = service.process_query(query, RetrievalMethod.INTELLIGENT, GenerationModel.GPT_SIMULATION)
```

### مرحله 2: استخراج کلمات کلیدی
```python
keywords = ['TP53', 'cancer', 'tp53']  # با بهبود famous_genes
```

### مرحله 3: تطبیق با نودها
```python
matched_nodes = {
    'TP53': 'Gene::TP53',
    'cancer': 'Disease::Breast Cancer',
    'tp53': 'Gene::TP53'
}
```

### مرحله 4: تشخیص نوع سوال
```python
intent = {
    'question_type': 'gene_cancer_relationship',
    'entity_types': ['Gene', 'Disease'],
    'target_metaedges': ['DaG', 'DuG', 'DdG']
}
```

### مرحله 5: بازیابی هوشمند
```python
results = [
    ('Gene::TP53', 0, 10.0, 'Primary gene'),
    ('Disease::Breast Cancer', 1, 8.64, 'Direct GaD'),
    ('Disease::Lung Cancer', 1, 7.0, 'Direct GuD')
]
```

### مرحله 6: تولید پاسخ
```python
answer = """
🧬 **تحلیل تخصصی ژن-سرطان:**

**ژن‌های اصلی یافت شده:**
• **TP53** (امتیاز: 10.00)

**سرطان‌های مرتبط:**
• Breast Cancer
• Lung Cancer

🔬 **تحلیل تخصصی TP53:**
TP53 یکی از مهم‌ترین ژن‌های سرکوبگر تومور است که:
• در بیش از 50% سرطان‌های انسانی جهش یافته است
• نقش کلیدی در تنظیم چرخه سلولی و آپوپتوز دارد
• به عنوان 'نگهبان ژنوم' شناخته می‌شود
"""
```

---

## 🎯 خلاصه منطق هوش مصنوعی

### 1. **هوش در بازیابی**:
- تشخیص نوع سوال از کلمات کلیدی
- انتخاب metaedge های مناسب
- امتیازدهی هوشمند بر اساس نوع رابطه
- جستجوی عمیق برای روابط غیرمستقیم

### 2. **هوش در تولید**:
- تشخیص نوع سوال از محتوای بازیابی شده
- انتخاب الگوی تولید مناسب
- تحلیل تخصصی برای ژن‌های مشهور
- ساختاردهی هوشمند پاسخ

### 3. **هوش در تطبیق**:
- نگاشت ژن‌های مشهور
- تطبیق فازی برای کلمات مشابه
- اولویت‌بندی روش‌های تطبیق

این سیستم ترکیبی از **Retrieval-Augmented Generation (RAG)** و **Knowledge Graph** است که برای حوزه زیستی بهینه‌سازی شده است. 
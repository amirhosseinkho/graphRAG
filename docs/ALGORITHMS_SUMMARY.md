# 🔍 خلاصه الگوریتم‌های کلیدی GraphRAG

## 📊 الگوریتم‌های بازیابی

### 1. **Intelligent Semantic Search** (خط 639)
**پیشرفته‌ترین الگوریتم بازیابی**

```python
def intelligent_semantic_search(self, query: str, max_depth: int = 3):
    # 1. تحلیل قصد سوال
    intent = self.analyze_question_intent(query)
    
    # 2. تشخیص نوع سوال خاص
    if self._is_gene_cancer_question(query, matched_nodes):
        return self._search_gene_cancer_relationships(query, matched_nodes, max_depth)
    
    # 3. انتخاب metaedge های مناسب
    target_metaedges = self._get_target_metaedges_for_question(intent['question_type'], query)
    
    # 4. جستجوی هدفمند
    return self._search_by_metaedges(matched_nodes, intent, target_metaedges, max_depth)
```

**ویژگی‌های کلیدی:**
- تشخیص نوع سوال (ژن-سرطان، دارو-بیماری، و...)
- انتخاب metaedge های مناسب بر اساس نوع سوال
- امتیازدهی هوشمند بر اساس نوع رابطه
- جستجوی عمیق برای روابط غیرمستقیم

### 2. **Gene-Cancer Relationship Search** (خط 759)
**الگوریتم تخصصی برای سوالات ژن-سرطان**

```python
def _search_gene_cancer_relationships(self, query: str, matched_nodes: Dict[str, str], max_depth: int):
    results = []
    
    # 1. شناسایی ژن‌ها و سرطان‌ها
    gene_nodes = []
    cancer_nodes = []
    for token, node_id in matched_nodes.items():
        node_attrs = self.G.nodes[node_id]
        if node_attrs.get('kind') == 'Gene':
            gene_nodes.append((token, node_id))
        elif node_attrs.get('kind') == 'Disease':
            if any(keyword in node_attrs['name'].lower() for keyword in ['cancer', 'tumor']):
                cancer_nodes.append((token, node_id))
    
    # 2. اضافه کردن ژن‌های اصلی با امتیاز بالا
    for gene_token, gene_node_id in gene_nodes:
        gene_name = self.G.nodes[gene_node_id]['name']
        results.append((gene_node_id, 0, 10.0, f"Primary gene: {gene_name}"))
    
    # 3. جستجوی روابط مستقیم ژن-سرطان
    for gene_token, gene_node_id in gene_nodes:
        for neighbor in self.G.neighbors(gene_node_id):
            neighbor_attrs = self.G.nodes[neighbor]
            if neighbor_attrs.get('kind') == 'Disease':
                edge_data = self.G.get_edge_data(gene_node_id, neighbor)
                if edge_data:
                    metaedge = edge_data.get('metaedge', 'Unknown')
                    cancer_score = 2.0 if any(keyword in neighbor_attrs['name'].lower() 
                                             for keyword in ['cancer', 'tumor']) else 1.0
                    score = self._calculate_metaedge_score(metaedge, 1) * cancer_score
                    results.append((neighbor, 1, score, f"{gene_name} related to {neighbor_attrs['name']}"))
    
    return results
```

### 3. **Metaedge-Based Search** (خط 3663)
**جستجوی هدفمند بر اساس نوع رابطه**

```python
def _search_by_metaedges(self, matched_nodes: Dict[str, str], intent: Dict, 
                         target_metaedges: List[str], max_depth: int = 2):
    results = []
    
    for node_token, node_id in matched_nodes.items():
        # 1. جستجوی مستقیم
        for neighbor in self.G.neighbors(node_id):
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
                score = self._calculate_metaedge_score('Unknown', depth)
                results.append((found_node, depth, score, f"DFS depth {depth}"))
    
    return results
```

## 🤖 الگوریتم‌های تولید متن

### 1. **GPT Simulation** (خط 2761)
**پیشرفته‌ترین الگوریتم تولید**

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

### 2. **Gene-Cancer Answer Generation** (خط 3112)
**تولید پاسخ تخصصی برای ژن-سرطان**

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
        answer_parts.append("• به عنوان 'نگهبان ژنوم' شناخته می‌شود")
    
    # 3. تحلیل روابط
    if retrieval_result.edges:
        gene_cancer_edges = []
        for edge in retrieval_result.edges:
            source_node = next((n for n in retrieval_result.nodes if n.id == edge.source), None)
            target_node = next((n for n in retrieval_result.nodes if n.id == edge.target), None)
            if source_node and target_node:
                if (source_node.kind == 'Gene' and target_node.kind == 'Disease') or \
                   (source_node.kind == 'Disease' and target_node.kind == 'Gene'):
                    gene_cancer_edges.append((source_node, target_node, edge.relation))
        
        if gene_cancer_edges:
            answer_parts.append("**تحلیل روابط یافت شده:**")
            for source, target, relation in gene_cancer_edges[:5]:
                answer_parts.append(f"• {source.name} → {target.name} ({relation})")
    
    return "\n".join(answer_parts)
```

## 🧠 الگوریتم‌های هوش مصنوعی

### 1. **Intent Analysis** (خط 481)
**تحلیل قصد سوال**

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

### 2. **Smart Scoring** (خط 3759)
**امتیازدهی هوشمند**

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

### 3. **Entity Matching** (خط 1119)
**تطبیق هوشمند موجودیت‌ها**

```python
def match_tokens_to_nodes(self, tokens: List[str]) -> Dict[str, str]:
    matched = {}
    
    # نگاشت ژن‌های مشهور
    famous_genes = {
        'tp53': ['TP53', 'P53', 'p53', 'Tumor Protein P53'],
        'brca1': ['BRCA1', 'Breast Cancer 1'],
        'brca2': ['BRCA2', 'Breast Cancer 2'],
    }
    
    for token in tokens:
        token_lower = token.lower()
        found = False
        
        # روش 1: تطبیق ژن‌های مشهور
        if token_lower in famous_genes:
            gene_variants = famous_genes[token_lower]
            for variant in gene_variants:
                for node_id, attrs in self.G.nodes(data=True):
                    if (attrs.get('kind') == 'Gene' and
                        variant.upper() in attrs['name'].upper()):
                        matched[token] = node_id
                        found = True
                        break
                if found:
                    break
        
        # روش 2: جستجوی مستقیم
        if not found:
            for node_id, attrs in self.G.nodes(data=True):
                if token_lower in attrs['name'].lower():
                    matched[token] = node_id
                    found = True
                    break
        
        # روش 3: جستجوی فازی
        if not found and len(token) >= 3:
            for node_id, attrs in self.G.nodes(data=True):
                if attrs.get('kind') == 'Gene':
                    name_lower = attrs['name'].lower()
                    if (token_lower in name_lower or
                        name_lower in token_lower):
                        matched[token] = node_id
                        found = True
                        break
    
    return matched
```

## 🔄 جریان کامل الگوریتم

### مرحله 1: ورودی و پیش‌پردازش
```python
query = "How does TP53 relate to cancer?"
keywords = extract_keywords(query)  # ['TP53', 'cancer', 'tp53']
matched_nodes = match_tokens_to_nodes(keywords)  # {'TP53': 'Gene::TP53', 'cancer': 'Disease::Breast Cancer'}
```

### مرحله 2: تحلیل قصد
```python
intent = analyze_question_intent(query)
# {
#   'question_type': 'gene_cancer_relationship',
#   'entity_types': ['Gene', 'Disease'],
#   'target_metaedges': ['DaG', 'DuG', 'DdG'],
#   'keywords': ['TP53', 'cancer', 'tp53']
# }
```

### مرحله 3: بازیابی هوشمند
```python
if _is_gene_cancer_question(query, matched_nodes):
    results = _search_gene_cancer_relationships(query, matched_nodes, max_depth)
else:
    results = _search_by_metaedges(matched_nodes, intent, target_metaedges, max_depth)
```

### مرحله 4: تولید پاسخ
```python
if _is_gene_cancer_question_from_context(retrieval_result):
    answer = _generate_gene_cancer_answer(retrieval_result, gene_nodes, disease_nodes)
else:
    answer = _generate_intelligent_relationship_answer(retrieval_result, gene_nodes, disease_nodes, drug_nodes)
```

## 🎯 خلاصه منطق هوش مصنوعی

### 1. **هوش در تشخیص نوع سوال**:
- تحلیل کلمات کلیدی
- تشخیص موجودیت‌های زیستی
- انتخاب metaedge های مناسب

### 2. **هوش در بازیابی**:
- امتیازدهی بر اساس نوع رابطه
- جستجوی هدفمند
- اولویت‌بندی نتایج

### 3. **هوش در تولید**:
- تشخیص نوع سوال از محتوا
- انتخاب الگوی تولید مناسب
- تحلیل تخصصی برای ژن‌های مشهور

### 4. **هوش در تطبیق**:
- نگاشت ژن‌های مشهور
- تطبیق فازی
- اولویت‌بندی روش‌ها

این سیستم ترکیبی از **RAG** و **Knowledge Graph** است که برای حوزه زیستی بهینه‌سازی شده و از الگوریتم‌های هوشمند برای تشخیص، بازیابی و تولید استفاده می‌کند. 
# 📁 ساختار پروژه GraphRAG

## 🗂️ فهرست فایل‌ها

```
tir/
├── 📄 graphrag_service.py          # کلاس اصلی سیستم (4171 خط)
├── 📄 main_grpahrag.py            # نقطه ورودی اصلی
├── 📄 web_app.py                  # رابط وب
├── 📄 run_graphrag.py             # اسکریپت اجرا
├── 📄 simple_demo.py              # دموی ساده
├── 📄 quick_test.py               # تست سریع
├── 📄 rebuild_graph.py            # بازسازی گراف
├── 📄 debug_graph.py              # دیباگ گراف
├── 📄 graphrag_env/               # محیط مجازی
├── 📄 requirements.txt             # وابستگی‌ها
├── 📄 setup.py                    # نصب
├── 📄 pyproject.toml              # تنظیمات پروژه
├── 📄 MANIFEST.in                 # فایل‌های پروژه
├── 📄 LICENSE                     # مجوز
├── 📄 README.md                   # راهنما
├── 📄 CONTRIBUTING.md             # مشارکت
├── 📄 API_SETUP.md               # راه‌اندازی API
├── 📄 RUN_WEB_APP.md             # اجرای وب
├── 📄 SYSTEM_STATUS.md           # وضعیت سیستم
├── 📄 IMPROVEMENTS.md            # بهبودها
├── 📄 COMPLEX_QUERIES_DOCUMENTATION.md  # مستندات سوالات پیچیده
├── 📄 PROJECT_ARCHITECTURE.md    # معماری پروژه
├── 📄 ALGORITHMS_SUMMARY.md      # خلاصه الگوریتم‌ها
├── 📄 PROJECT_STRUCTURE.md       # این فایل
├── 📄 hetionet-v1.0-edges.sif.gz # داده‌های Hetionet
├── 📄 main_grpahRAG.ipynb        # نوت‌بوک اصلی
├── 📁 static/                     # فایل‌های استاتیک
│   ├── 📁 css/
│   │   └── 📄 style.css
│   └── 📁 js/
│       └── 📄 app.js
├── 📁 templates/                  # قالب‌های HTML
│   └── 📄 index.html
└── 📄 test_*.py                   # فایل‌های تست
    ├── 📄 test_basic_functionality.py
    ├── 📄 test_complex_queries.py
    ├── 📄 test_compound_gene_relations.py
    ├── 📄 test_compound_start.py
    ├── 📄 test_comprehensive_system.py
    ├── 📄 test_final_comprehensive.py
    ├── 📄 test_final_status.py
    ├── 📄 test_graph_edges.py
    ├── 📄 test_improved_system.py
    ├── 📄 test_improvements.py
    ├── 📄 test_intelligent_search.py
    ├── 📄 test_model_only.py
    ├── 📄 test_multi_hop_debug.py
    ├── 📄 test_openai.py
    ├── 📄 test_path_finding.py
    ├── 📄 test_simple_path.py
    ├── 📄 test_simple.py
    ├── 📄 test_targeted_retrieval.py
    ├── 📄 test_tissue_disease_query.py
    ├── 📄 test_tp53_fix.py
    ├── 📄 debug_tp53_retrieval.py
    └── 📄 test_cdg_aeg_pattern.py
```

## 🏗️ معماری فایل‌ها

### 📄 فایل‌های اصلی

#### 1. **graphrag_service.py** (4171 خط)
**قلب سیستم** - کلاس اصلی GraphRAGService

**ساختار کلاس:**
```python
class GraphRAGService:
    def __init__(self, graph_data_path: str = None):
        # راه‌اندازی سیستم
    
    def initialize(self):
        # بارگذاری مدل‌ها و گراف
    
    def process_query(self, query: str, retrieval_method: RetrievalMethod, 
                     generation_model: GenerationModel, max_depth: int = 2):
        # پردازش کامل سوال
    
    def retrieve_information(self, query: str, method: RetrievalMethod, 
                           max_depth: int = 2, max_nodes: int = 10):
        # بازیابی اطلاعات
    
    def generate_answer(self, retrieval_result: RetrievalResult, 
                       model: GenerationModel):
        # تولید پاسخ
```

**متدهای کلیدی:**
- `intelligent_semantic_search()` (خط 639) - جستجوی هوشمند
- `analyze_question_intent()` (خط 481) - تحلیل قصد
- `match_tokens_to_nodes()` (خط 1119) - تطبیق موجودیت‌ها
- `gpt_simulation_generation()` (خط 2761) - تولید پاسخ
- `_search_gene_cancer_relationships()` (خط 759) - جستجوی ژن-سرطان

#### 2. **main_grpahrag.py**
**نقطه ورودی اصلی** - رابط خط فرمان

```python
def main():
    # راه‌اندازی سرویس
    service = GraphRAGService()
    
    # حلقه اصلی
    while True:
        query = input("سوال خود را بپرسید: ")
        if query.lower() == 'exit':
            break
        
        # پردازش سوال
        result = service.process_query(
            query=query,
            retrieval_method=RetrievalMethod.INTELLIGENT,
            generation_model=GenerationModel.GPT_SIMULATION
        )
        
        # نمایش نتیجه
        print(result['answer'])
```

#### 3. **web_app.py**
**رابط وب** - Flask application

```python
from flask import Flask, render_template, request, jsonify
from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

app = Flask(__name__)
service = GraphRAGService()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data.get('query', '')
    
    result = service.process_query(
        query=query,
        retrieval_method=RetrievalMethod.INTELLIGENT,
        generation_model=GenerationModel.GPT_SIMULATION
    )
    
    return jsonify(result)
```

### 📁 فایل‌های استاتیک

#### 1. **static/css/style.css**
**استایل‌های CSS** برای رابط وب

```css
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    margin: 0;
    padding: 20px;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    background: white;
    border-radius: 10px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    padding: 30px;
}

.query-form {
    margin-bottom: 30px;
}

.query-input {
    width: 100%;
    padding: 15px;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    font-size: 16px;
    transition: border-color 0.3s;
}

.result-container {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    margin-top: 20px;
}
```

#### 2. **static/js/app.js**
**جاوااسکریپت** برای تعامل با API

```javascript
document.addEventListener('DOMContentLoaded', function() {
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const resultContainer = document.getElementById('result-container');
    
    queryForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const query = queryInput.value.trim();
        if (!query) return;
        
        // نمایش loading
        resultContainer.innerHTML = '<div class="loading">در حال پردازش...</div>';
        
        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query })
            });
            
            const result = await response.json();
            
            // نمایش نتیجه
            resultContainer.innerHTML = `
                <div class="result">
                    <h3>پاسخ:</h3>
                    <div class="answer">${result.answer}</div>
                    <div class="metadata">
                        <p><strong>روش بازیابی:</strong> ${result.retrieval_method}</p>
                        <p><strong>مدل تولید:</strong> ${result.generation_model}</p>
                        <p><strong>سطح اطمینان:</strong> ${result.confidence}</p>
                    </div>
                </div>
            `;
        } catch (error) {
            resultContainer.innerHTML = '<div class="error">خطا در پردازش سوال</div>';
        }
    });
});
```

### 📁 قالب‌های HTML

#### 1. **templates/index.html**
**صفحه اصلی** رابط وب

```html
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GraphRAG System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <header>
            <h1>🧬 GraphRAG System</h1>
            <p>سیستم هوشمند پاسخ‌دهی به سوالات زیستی</p>
        </header>
        
        <main>
            <form id="query-form" class="query-form">
                <div class="input-group">
                    <input type="text" id="query-input" class="query-input" 
                           placeholder="سوال خود را بپرسید..." required>
                    <button type="submit" class="submit-btn">پرسش</button>
                </div>
            </form>
            
            <div id="result-container" class="result-container">
                <!-- نتایج اینجا نمایش داده می‌شوند -->
            </div>
        </main>
    </div>
    
    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
</body>
</html>
```

### 📄 فایل‌های تست

#### 1. **test_tp53_fix.py**
**تست اصلاحات TP53**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست اصلاحات TP53
"""

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_tp53_query():
    """تست سوال TP53"""
    print("🧪 شروع تست TP53...")
    
    # ایجاد سرویس
    service = GraphRAGService()
    
    # سوال تست
    query = "How does TP53 relate to cancer?"
    
    # پردازش سوال
    result = service.process_query(
        query=query,
        retrieval_method=RetrievalMethod.INTELLIGENT,
        generation_model=GenerationModel.GPT_SIMULATION,
        max_depth=3
    )
    
    # نمایش نتایج
    print(f"• روش بازیابی: {result.get('retrieval_method', 'N/A')}")
    print(f"• مدل تولید: {result.get('generation_model', 'N/A')}")
    
    # بررسی نودهای بازیابی شده
    retrieved_nodes = result.get('retrieved_nodes', [])
    print(f"• تعداد نودها: {len(retrieved_nodes)}")
    
    print("\n🎯 نودهای یافت شده:")
    for node in retrieved_nodes:
        print(f"  • {node['name']} ({node['kind']}) - امتیاز: {node.get('score', 'N/A')}")
    
    # بررسی پاسخ
    answer = result.get('answer', '')
    if answer:
        print(f"\n🤖 پاسخ تولید شده:")
        print(answer)
    
    return result

if __name__ == "__main__":
    test_tp53_query()
```

#### 2. **debug_tp53_retrieval.py**
**دیباگ بازیابی TP53**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
دیباگ بازیابی TP53
"""

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def debug_tp53_retrieval():
    """دیباگ بازیابی TP53"""
    print("🔍 دیباگ بازیابی TP53...")
    
    # ایجاد سرویس
    service = GraphRAGService()
    
    # سوال تست
    query = "How does TP53 relate to cancer?"
    print(f"🔍 سوال: {query}")
    
    # بررسی استخراج کلمات کلیدی
    keywords = service.extract_keywords(query)
    print(f"🔑 کلمات کلیدی استخراج شده: {keywords}")
    
    # بررسی تطبیق توکن‌ها
    matched_nodes = service.match_tokens_to_nodes(keywords)
    print(f"🎯 نودهای تطبیق یافته: {matched_nodes}")
    
    # بررسی همه نودهای ژن در گراف
    print("\n🔍 بررسی همه ژن‌های موجود در گراف:")
    gene_nodes = []
    for node_id, attrs in service.G.nodes(data=True):
        if attrs.get('kind') == 'Gene':
            gene_nodes.append((node_id, attrs['name']))
    
    print(f"📊 تعداد کل ژن‌ها: {len(gene_nodes)}")
    
    # جستجوی TP53 در گراف
    tp53_found = False
    for node_id, name in gene_nodes:
        if 'TP53' in name.upper() or 'P53' in name.upper():
            print(f"✅ TP53 یافت شد: {name} (ID: {node_id})")
            tp53_found = True
    
    return {
        'keywords': keywords,
        'matched_nodes': matched_nodes,
        'gene_nodes': gene_nodes,
        'tp53_found': tp53_found
    }

if __name__ == "__main__":
    debug_tp53_retrieval()
```

### 📄 فایل‌های پیکربندی

#### 1. **requirements.txt**
**وابستگی‌های پروژه**

```
networkx>=2.8.4
spacy>=3.4.4
flask>=2.2.3
numpy>=1.21.6
pandas>=1.4.4
matplotlib>=3.5.3
seaborn>=0.11.2
scikit-learn>=1.1.3
transformers>=4.21.3
torch>=1.12.1
openai>=0.27.8
anthropic>=0.3.11
google-generativeai>=0.3.0
```

#### 2. **setup.py**
**نصب پروژه**

```python
from setuptools import setup, find_packages

setup(
    name="graphrag",
    version="1.0.0",
    description="GraphRAG System for Biomedical Question Answering",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "networkx>=2.8.4",
        "spacy>=3.4.4",
        "flask>=2.2.3",
        "numpy>=1.21.6",
        "pandas>=1.4.4",
        "matplotlib>=3.5.3",
        "seaborn>=0.11.2",
        "scikit-learn>=1.1.3",
        "transformers>=4.21.3",
        "torch>=1.12.1",
        "openai>=0.27.8",
        "anthropic>=0.3.11",
        "google-generativeai>=0.3.0",
    ],
    python_requires=">=3.8",
)
```

#### 3. **pyproject.toml**
**تنظیمات پروژه**

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "graphrag"
version = "1.0.0"
description = "GraphRAG System for Biomedical Question Answering"
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]

[project.urls]
Homepage = "https://github.com/yourusername/graphrag"
Repository = "https://github.com/yourusername/graphrag"
Documentation = "https://github.com/yourusername/graphrag#readme"
```

### 📄 فایل‌های مستندات

#### 1. **README.md**
**راهنمای اصلی پروژه**

```markdown
# 🧬 GraphRAG System

سیستم هوشمند پاسخ‌دهی به سوالات زیستی بر اساس گراف دانش Hetionet

## 🚀 نصب و راه‌اندازی

```bash
# کلون کردن پروژه
git clone https://github.com/yourusername/graphrag.git
cd graphrag

# ایجاد محیط مجازی
python -m venv graphrag_env
source graphrag_env/bin/activate  # Linux/Mac
# یا
graphrag_env\Scripts\activate  # Windows

# نصب وابستگی‌ها
pip install -r requirements.txt

# بارگذاری مدل spaCy
python -m spacy download en_core_web_sm
```

## 📖 استفاده

### رابط خط فرمان
```bash
python main_grpahrag.py
```

### رابط وب
```bash
python web_app.py
```

## 🔧 ویژگی‌ها

- **بازیابی هوشمند**: تشخیص نوع سوال و انتخاب الگوریتم مناسب
- **تولید پاسخ تخصصی**: تحلیل تخصصی برای ژن‌های مشهور
- **پشتیبانی از چندین مدل**: GPT، Claude، Gemini
- **رابط کاربری وب**: رابط گرافیکی آسان‌استفاده

## 📚 مستندات

- [معماری پروژه](PROJECT_ARCHITECTURE.md)
- [خلاصه الگوریتم‌ها](ALGORITHMS_SUMMARY.md)
- [ساختار پروژه](PROJECT_STRUCTURE.md)
```

#### 2. **API_SETUP.md**
**راه‌اندازی API ها**

```markdown
# 🔑 راه‌اندازی API Keys

## OpenAI GPT
```python
service.set_openai_api_key("your-openai-api-key")
```

## Anthropic Claude
```python
service.set_anthropic_api_key("your-anthropic-api-key")
```

## Google Gemini
```python
service.set_gemini_api_key("your-gemini-api-key")
```

## دریافت API Keys

### OpenAI
1. به [OpenAI Platform](https://platform.openai.com/) بروید
2. حساب کاربری ایجاد کنید
3. از بخش API Keys، کلید جدید ایجاد کنید

### Anthropic
1. به [Anthropic Console](https://console.anthropic.com/) بروید
2. حساب کاربری ایجاد کنید
3. از بخش API Keys، کلید جدید ایجاد کنید

### Google Gemini
1. به [Google AI Studio](https://makersuite.google.com/) بروید
2. حساب کاربری Google ایجاد کنید
3. از بخش API Keys، کلید جدید ایجاد کنید
```

## 🔄 جریان توسعه

### 1. **توسعه ویژگی‌های جدید**
```bash
# ایجاد شاخه جدید
git checkout -b feature/new-algorithm

# توسعه کد
# ...

# تست
python test_new_feature.py

# commit و push
git add .
git commit -m "Add new algorithm"
git push origin feature/new-algorithm
```

### 2. **تست سیستم**
```bash
# تست کامل
python -m pytest test_*.py

# تست خاص
python test_tp53_fix.py

# دیباگ
python debug_tp53_retrieval.py
```

### 3. **اجرای سیستم**
```bash
# رابط خط فرمان
python main_grpahrag.py

# رابط وب
python web_app.py

# دموی ساده
python simple_demo.py
```

## 📊 آمار پروژه

- **کل خطوط کد**: ~15,000 خط
- **فایل‌های اصلی**: 25+ فایل
- **الگوریتم‌های بازیابی**: 9 روش مختلف
- **مدل‌های تولید**: 7 مدل مختلف
- **تست‌ها**: 20+ فایل تست
- **مستندات**: 10+ فایل مستندات

## 🎯 ساختار منطقی

### 1. **لایه ورودی**
- `main_grpahrag.py` - رابط خط فرمان
- `web_app.py` - رابط وب
- `simple_demo.py` - دموی ساده

### 2. **لایه منطق اصلی**
- `graphrag_service.py` - کلاس اصلی سیستم
- الگوریتم‌های بازیابی
- مدل‌های تولید

### 3. **لایه داده**
- `hetionet-v1.0-edges.sif.gz` - داده‌های Hetionet
- `rebuild_graph.py` - بازسازی گراف

### 4. **لایه تست و دیباگ**
- `test_*.py` - فایل‌های تست
- `debug_*.py` - فایل‌های دیباگ

### 5. **لایه مستندات**
- `README.md` - راهنمای اصلی
- `PROJECT_ARCHITECTURE.md` - معماری
- `ALGORITHMS_SUMMARY.md` - الگوریتم‌ها

این ساختار به گونه‌ای طراحی شده که توسعه‌دهندگان بتوانند به راحتی ویژگی‌های جدید اضافه کنند و سیستم را گسترش دهند. 
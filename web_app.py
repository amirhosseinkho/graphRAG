# -*- coding: utf-8 -*-
"""
GraphRAG Web Application - رابط وب تعاملی
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel
import json
import os
from datetime import datetime

app = Flask(__name__)

# راه‌اندازی سرویس GraphRAG با گراف Hetionet
# ابتدا بررسی می‌کنیم که آیا فایل گراف Hetionet وجود دارد
graph_files = [f for f in os.listdir('.') if f.startswith('hetionet_graph_') and f.endswith('.pkl')]
if graph_files:
    # استفاده از جدیدترین فایل گراف
    latest_graph_file = max(graph_files)
    print(f"🔧 استفاده از گراف Hetionet: {latest_graph_file}")
    graphrag_service = GraphRAGService(graph_data_path=latest_graph_file)
else:
    print("⚠️ فایل گراف Hetionet یافت نشد، استفاده از گراف نمونه")
    graphrag_service = GraphRAGService()

# تنظیم API Key های OpenAI
OPENAI_API_KEY = "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"
graphrag_service.set_openai_api_key(OPENAI_API_KEY)
print("✅ OpenAI API Key تنظیم شد")

@app.route('/')
def index():
    """صفحه اصلی"""
    return render_template('index.html')

@app.route('/api/process_query', methods=['POST'])
def process_query():
    """پردازش سوال و برگرداندن نتیجه"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        retrieval_method = data.get('retrieval_method', 'BFS')
        generation_model = data.get('generation_model', 'GPT_SIMULATION')
        max_depth = data.get('max_depth', 2)
        
        # تبدیل رشته به enum
        retrieval_enum = RetrievalMethod[retrieval_method]
        generation_enum = GenerationModel[generation_model.replace(' ', '_')]
        
        # پردازش سوال
        result = graphrag_service.process_query(
            query=query,
            retrieval_method=retrieval_enum,
            generation_model=generation_enum,
            max_depth=max_depth
        )
        
        # اضافه کردن timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/graph_info')
def graph_info():
    """اطلاعات گراف"""
    try:
        G = graphrag_service.G
        if G:
            node_types = {}
            for node, attrs in G.nodes(data=True):
                kind = attrs.get('kind', 'Unknown')
                node_types[kind] = node_types.get(kind, 0) + 1
            
            return jsonify({
                'success': True,
                'total_nodes': G.number_of_nodes(),
                'total_edges': G.number_of_edges(),
                'node_types': node_types
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Graph not loaded'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/sample_queries')
def sample_queries():
    """سوالات نمونه بر اساس ساختار Hetionet و روابط موجود"""
    samples = [
        # سوالات مربوط به بیان ژن در بافت‌ها (AeG, AuG, AdG)
        "What genes are expressed in the heart?",
        "Which genes are upregulated in the brain?",
        "What genes are downregulated in muscle tissue?",
        "How do genes express differently in liver vs kidney?",
        "What genes are expressed in the lung?",
        
        # سوالات مربوط به ژن‌ها و بیماری‌ها (DaG, DuG, DdG)
        "What genes are associated with diabetes?",
        "How does TP53 relate to cancer?",
        "Which genes are upregulated in cancer?",
        "What genes are downregulated in heart disease?",
        "What genes are associated with Alzheimer's disease?",
        
        # سوالات مربوط به داروها و درمان (CtD, CuG, CdG)
        "What drugs treat diabetes?",
        "Which compounds upregulate TP53?",
        "What drugs downregulate cancer genes?",
        "How do drugs interact with genes?",
        "What compounds bind to insulin receptor?",
        
        # سوالات مربوط به فرآیندهای زیستی (GpBP, GpMF, GpCC)
        "What genes participate in cell cycle regulation?",
        "Which genes are involved in apoptosis?",
        "What molecular functions does TP53 have?",
        "How do genes function in cellular components?",
        "What genes participate in DNA repair?",
        
        # سوالات مربوط به مسیرهای زیستی (GpPW)
        "What pathways are involved in cancer progression?",
        "Which signaling pathways regulate metabolism?",
        "How do genes participate in immune pathways?",
        "What pathways control cell growth?",
        "Which pathways involve insulin signaling?",
        
        # سوالات مربوط به تعامل ژن‌ها (GiG, Gr>G)
        "How do genes interact with each other?",
        "Which genes regulate TP53?",
        "What genes are regulated by TP53?",
        "How do genes covary in expression?",
        "What genes interact with BRCA1?",
        
        # سوالات مربوط به بیماری‌ها و علائم (DpS, DlA)
        "What symptoms are associated with diabetes?",
        "How does cancer affect different tissues?",
        "What diseases affect the heart?",
        "Which diseases localize to specific tissues?",
        "What diseases present similar symptoms?",
        
        # سوالات مربوط به عوارض جانبی داروها (CcSE)
        "What side effects does aspirin cause?",
        "How do drugs affect patient symptoms?",
        "What adverse reactions occur with diabetes drugs?",
        "Which compounds cause heart-related side effects?",
        "What side effects do cancer drugs cause?",
        
        # سوالات پیچیده و چندمرحله‌ای
        "How do drugs affect gene expression in heart tissue?",
        "What genes and pathways are involved in diabetes progression?",
        "How do genetic mutations lead to cancer development?",
        "What therapeutic targets exist for heart disease?",
        "How do genes participate in drug metabolism pathways?",
        "What biological processes are disrupted in cancer?",
        "What drugs treat diseases that affect the heart?",
        "How do genes that interact with TP53 relate to cancer?",
        "What compounds bind to genes expressed in the brain?",
        "Which diseases have symptoms related to diabetes?"
    ]
    return jsonify({'queries': samples})

if __name__ == '__main__':
    # ایجاد پوشه templates اگر وجود ندارد
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 
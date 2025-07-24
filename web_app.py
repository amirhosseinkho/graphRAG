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
    """سوالات نمونه"""
    samples = [
        "What is the relationship between HMGB3 and diabetes?",
        "What drugs treat diabetes?",
        "What genes are expressed in the heart?",
        "How does TP53 relate to cancer?",
        "What biological processes does BRCA1 regulate?",
        "Which drugs prevent heart disease?",
        "What diseases affect the brain?",
        "How do genes interact with each other?"
    ]
    return jsonify({'queries': samples})

if __name__ == '__main__':
    # ایجاد پوشه templates اگر وجود ندارد
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 
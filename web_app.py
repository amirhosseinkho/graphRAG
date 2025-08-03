# -*- coding: utf-8 -*-
"""
GraphRAG Web Application - رابط وب تعاملی
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel
import json
import os
import shutil
from datetime import datetime
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from difflib import SequenceMatcher

# Simple text processing functions without external dependencies
def simple_tokenize(text):
    """Tokenize text without external dependencies"""
    return text.lower().split()

def simple_remove_punctuation(text):
    """Remove punctuation without external dependencies"""
    import string
    return text.translate(str.maketrans('', '', string.punctuation))

# Initialize sentence transformer for semantic similarity (optional)
sentence_transformer = None
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except:
    pass

app = Flask(__name__)

# تنظیمات آپلود فایل
UPLOAD_FOLDER = 'uploaded_graphs'
ALLOWED_EXTENSIONS = {'pkl', 'sif', 'tsv', 'csv', 'txt', 'gz'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ایجاد پوشه آپلود اگر وجود ندارد
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """بررسی مجاز بودن نوع فایل"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

@app.route('/upload_graph')
def upload_graph_page():
    """صفحه آپلود گراف"""
    return render_template('upload_graph.html')

@app.route('/manage_graphs')
def manage_graphs_page():
    """صفحه مدیریت گراف‌ها"""
    return render_template('manage_graphs.html')

@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')

@app.route('/api/upload_graph', methods=['POST'])
def upload_graph():
    """آپلود فایل گراف"""
    try:
        if 'graph_file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'فایل انتخاب نشده است'
            }), 400
        
        file = request.files['graph_file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'فایل انتخاب نشده است'
            }), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_with_timestamp = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename_with_timestamp)
            
            file.save(filepath)
            
            # اگر فایل فشرده است، آن را باز کن
            if filename.endswith('.gz'):
                import gzip
                with gzip.open(filepath, 'rb') as f_in:
                    uncompressed_path = filepath[:-3]
                    with open(uncompressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(filepath)
                filepath = uncompressed_path
                filename_with_timestamp = filename_with_timestamp[:-3]
            
            return jsonify({
                'success': True,
                'message': f'فایل {filename} با موفقیت آپلود شد',
                'filename': filename_with_timestamp,
                'filepath': filepath
            })
        else:
            return jsonify({
                'success': False,
                'error': 'نوع فایل مجاز نیست. فایل‌های مجاز: pkl, sif, tsv, csv, txt, gz'
            }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/list_graphs')
def list_graphs():
    """لیست گراف‌های موجود"""
    try:
        graphs = []
        
        # گراف‌های آپلود شده
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(filepath):
                    file_size = os.path.getsize(filepath)
                    file_date = datetime.fromtimestamp(os.path.getctime(filepath))
                    graphs.append({
                        'name': filename,
                        'path': filepath,
                        'size': file_size,
                        'date': file_date.isoformat(),
                        'type': 'uploaded'
                    })
        
        # گراف‌های موجود در پوشه اصلی
        for filename in os.listdir('.'):
            if filename.endswith('.pkl') and filename.startswith('hetionet_graph_'):
                filepath = os.path.join('.', filename)
                file_size = os.path.getsize(filepath)
                file_date = datetime.fromtimestamp(os.path.getctime(filepath))
                graphs.append({
                    'name': filename,
                    'path': filepath,
                    'size': file_size,
                    'date': file_date.isoformat(),
                    'type': 'builtin'
                })
        
        # مرتب کردن بر اساس تاریخ (جدیدترین اول)
        graphs.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify({
            'success': True,
            'graphs': graphs
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/load_graph', methods=['POST'])
def load_graph():
    """بارگذاری گراف انتخاب شده"""
    try:
        data = request.get_json()
        graph_path = data.get('graph_path')
        
        if not graph_path or not os.path.exists(graph_path):
            return jsonify({
                'success': False,
                'error': 'مسیر گراف نامعتبر است'
            }), 400
        
        # بارگذاری گراف جدید
        global graphrag_service
        graphrag_service = GraphRAGService(graph_data_path=graph_path)
        graphrag_service.set_openai_api_key(OPENAI_API_KEY)
        
        return jsonify({
            'success': True,
            'message': f'گراف {os.path.basename(graph_path)} با موفقیت بارگذاری شد'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/delete_graph', methods=['POST'])
def delete_graph():
    """حذف گراف"""
    try:
        data = request.get_json()
        graph_path = data.get('graph_path')
        
        if not graph_path or not os.path.exists(graph_path):
            return jsonify({
                'success': False,
                'error': 'مسیر گراف نامعتبر است'
            }), 400
        
        # حذف فایل
        os.remove(graph_path)
        
        return jsonify({
            'success': True,
            'message': f'گراف {os.path.basename(graph_path)} با موفقیت حذف شد'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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

@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
    """مدیریت تنظیمات سیستم"""
    if request.method == 'GET':
        # دریافت تنظیمات فعلی
        try:
            config = graphrag_service.get_config()
            return jsonify({
                'success': True,
                'config': config
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    elif request.method == 'POST':
        # تغییر تنظیمات
        try:
            data = request.get_json()
            new_config = data.get('config', {})
            
            # اعمال تنظیمات جدید
            graphrag_service.set_config(**new_config)
            
            # دریافت تنظیمات به‌روز شده
            updated_config = graphrag_service.get_config()
            
            return jsonify({
                'success': True,
                'message': 'تنظیمات با موفقیت به‌روز شد',
                'config': updated_config
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

@app.route('/api/config/presets', methods=['GET'])
def config_presets():
    """پیش‌تنظیمات آماده"""
    presets = {
        'fast': {
            'name': 'سریع',
            'description': 'پاسخ سریع با محدودیت‌های کم',
            'config': {
                'max_nodes': 5,
                'max_edges': 10,
                'max_depth': 2,
                'max_paths': 3,
                'max_context_length': 1000,
                'max_answer_tokens': 500,
                'max_prompt_tokens': 2000,
                'enable_verbose_logging': False,
                'enable_biological_enrichment': False,
                'enable_smart_filtering': True
            }
        },
        'balanced': {
            'name': 'متوازن',
            'description': 'تعادل بین سرعت و کیفیت',
            'config': {
                'max_nodes': 10,
                'max_edges': 20,
                'max_depth': 3,
                'max_paths': 5,
                'max_context_length': 2000,
                'max_answer_tokens': 1000,
                'max_prompt_tokens': 4000,
                'enable_verbose_logging': True,
                'enable_biological_enrichment': True,
                'enable_smart_filtering': True
            }
        },
        'comprehensive': {
            'name': 'جامع',
            'description': 'پاسخ کامل با جزئیات بیشتر',
            'config': {
                'max_nodes': 20,
                'max_edges': 40,
                'max_depth': 4,
                'max_paths': 10,
                'max_context_length': 3000,
                'max_answer_tokens': 1500,
                'max_prompt_tokens': 6000,
                'enable_verbose_logging': True,
                'enable_biological_enrichment': True,
                'enable_smart_filtering': True
            }
        },
        'research': {
            'name': 'تحقیقاتی',
            'description': 'برای تحقیقات و تحلیل عمیق',
            'config': {
                'max_nodes': 30,
                'max_edges': 60,
                'max_depth': 5,
                'max_paths': 15,
                'max_context_length': 4000,
                'max_answer_tokens': 2000,
                'max_prompt_tokens': 8000,
                'enable_verbose_logging': True,
                'enable_biological_enrichment': True,
                'enable_smart_filtering': True
            }
        }
    }
    return jsonify({'presets': presets})

@app.route('/api/compare_texts', methods=['POST'])
def compare_texts():
    try:
        data = request.get_json()
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')
        method = data.get('method', 'cosine_tfidf')
        
        if not text1 or not text2:
            return jsonify({'error': 'هر دو متن باید وارد شوند'}), 400
        
        # Preprocess texts
        text1_processed = preprocess_text(text1)
        text2_processed = preprocess_text(text2)
        
        # Calculate similarity based on selected method
        similarity_score = 0
        method_name = ""
        
        if method == 'cosine_tfidf':
            similarity_score = cosine_similarity_tfidf(text1_processed, text2_processed)
            method_name = "شباهت کسینوسی (TF-IDF)"
        elif method == 'cosine_sbert':
            similarity_score = cosine_similarity_sbert(text1, text2)
            method_name = "شباهت کسینوسی (SBERT)"
        elif method == 'jaccard':
            similarity_score = jaccard_similarity(text1_processed, text2_processed)
            method_name = "شباهت جاکارد"
        elif method == 'levenshtein':
            similarity_score = levenshtein_similarity(text1, text2)
            method_name = "شباهت لونشتاین"
        elif method == 'sequence_matcher':
            similarity_score = sequence_matcher_similarity(text1, text2)
            method_name = "شباهت Sequence Matcher"
        elif method == 'word_overlap':
            similarity_score = word_overlap_similarity(text1_processed, text2_processed)
            method_name = "شباهت همپوشانی کلمات"
        else:
            return jsonify({'error': 'روش مقایسه نامعتبر است'}), 400
        
        # Determine quality level
        quality_level = get_quality_level(similarity_score)
        
        return jsonify({
            'similarity_score': round(similarity_score, 4),
            'method_name': method_name,
            'quality_level': quality_level,
            'text1_processed': text1_processed,
            'text2_processed': text2_processed
        })
        
    except Exception as e:
        return jsonify({'error': f'خطا در مقایسه: {str(e)}'}), 500

def preprocess_text(text):
    """پیش‌پردازش متن برای مقایسه"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def cosine_similarity_tfidf(text1, text2):
    """محاسبه شباهت کسینوسی با TF-IDF"""
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0

def cosine_similarity_sbert(text1, text2):
    """محاسبه شباهت کسینوسی با SBERT"""
    try:
        if sentence_transformer is None:
            return 0.0
        
        embeddings = sentence_transformer.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except:
        return 0.0

def jaccard_similarity(text1, text2):
    """محاسبه شباهت جاکارد"""
    try:
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    except:
        return 0.0

def levenshtein_similarity(text1, text2):
    """محاسبه شباهت لونشتاین"""
    try:
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        
        if max_len == 0:
            return 1.0
        
        similarity = 1 - (distance / max_len)
        return similarity
    except:
        return 0.0

def sequence_matcher_similarity(text1, text2):
    """محاسبه شباهت با Sequence Matcher"""
    try:
        similarity = SequenceMatcher(None, text1, text2).ratio()
        return similarity
    except:
        return 0.0

def word_overlap_similarity(text1, text2):
    """محاسبه شباهت بر اساس همپوشانی کلمات"""
    try:
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        min_length = min(len(words1), len(words2))
        
        if min_length == 0:
            return 0.0
        
        return intersection / min_length
    except:
        return 0.0

def get_quality_level(similarity_score):
    """تعیین سطح کیفیت بر اساس نمره شباهت"""
    if similarity_score >= 0.9:
        return "عالی"
    elif similarity_score >= 0.8:
        return "خیلی خوب"
    elif similarity_score >= 0.7:
        return "خوب"
    elif similarity_score >= 0.6:
        return "متوسط"
    elif similarity_score >= 0.5:
        return "ضعیف"
    else:
        return "خیلی ضعیف"

if __name__ == '__main__':
    # ایجاد پوشه templates اگر وجود ندارد
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 
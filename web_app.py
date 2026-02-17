# -*- coding: utf-8 -*-
"""
GraphRAG Web Application - رابط وب تعاملی
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel
from enhanced_graphrag_service import EnhancedGraphRAGService, TokenExtractionMethod, RetrievalAlgorithm, CommunityDetectionMethod
from text_to_graph_service import TextToGraphService
import json
import os
import shutil
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from difflib import SequenceMatcher

# OpenAI import for GPT-4o comparison
try:
    import openai
except ImportError:
    openai = None

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

# تنظیمات CORS برای جلوگیری از خطای Failed to fetch
@app.after_request
def after_request(response):
    """اضافه کردن header های CORS"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Error handler برای خطاهای عمومی
@app.errorhandler(Exception)
def handle_exception(e):
    """مدیریت خطاهای عمومی"""
    logging.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({
        'success': False,
        'error': f'خطای سرور: {str(e)}',
        'error_type': type(e).__name__
    }), 500

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
    enhanced_graphrag_service = EnhancedGraphRAGService(graph_data_path=latest_graph_file)
else:
    print("⚠️ فایل گراف Hetionet یافت نشد، استفاده از گراف نمونه")
    graphrag_service = GraphRAGService()
    enhanced_graphrag_service = EnhancedGraphRAGService()

# تنظیم API Key های OpenAI (از متغیر محیطی یا secrets.json)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# تلاش برای خواندن از secrets.json اگر متغیر محیطی خالی بود
if not OPENAI_API_KEY:
    try:
        import json as _json
        if os.path.exists('secrets.json'):
            with open('secrets.json', 'r', encoding='utf-8') as _sf:
                _secrets = _json.load(_sf) or {}
                OPENAI_API_KEY = _secrets.get('OPENAI_API_KEY', '')
    except Exception as _e:
        pass

# اگر هنوز API key تنظیم نشده، از API key پیش‌فرض استفاده نکن - فقط هشدار بده
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY تنظیم نشده است؛ تولید پاسخ با OpenAI غیرفعال خواهد بود")

if OPENAI_API_KEY:
    graphrag_service.set_openai_api_key(OPENAI_API_KEY)
    print("✅ OpenAI API Key تنظیم شد")
else:
    print("⚠️ OPENAI_API_KEY تنظیم نشده است؛ تولید پاسخ با OpenAI غیرفعال خواهد بود")

@app.route('/')
def index():
    """صفحه اصلی"""
    return render_template('index.html')

@app.route('/upload_graph')
def upload_graph_page():
    """صفحه آپلود گراف"""
    return render_template('upload_graph.html')

@app.route('/view_graph')
def view_graph_page():
    """صفحه نمایش گراف انتخاب‌شده (ویژوال + آمار)"""
    graph_name = request.args.get('graph_name', '')
    graph_path = request.args.get('graph_path', '')
    
    # Decode URL-encoded path
    if graph_path:
        from urllib.parse import unquote
        graph_path = unquote(graph_path)
    
    # Validate path exists
    if graph_path and not os.path.exists(graph_path):
        # Try to resolve relative paths
        if not os.path.isabs(graph_path):
            uploaded_path = os.path.join(UPLOAD_FOLDER, graph_path)
            if os.path.exists(uploaded_path):
                graph_path = uploaded_path
            elif os.path.exists(os.path.join('.', graph_path)):
                graph_path = os.path.join('.', graph_path)
    
    return render_template('view_graph.html', graph_name=graph_name, graph_path=graph_path)

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

@app.route('/api/text_to_graph', methods=['POST'])
def text_to_graph():
    """تبدیل متن به گراف دانش"""
    try:
        data = request.get_json()
        
        # Import URL extractor
        try:
            from url_extractor import extract_text_from_url, is_valid_url
            URL_EXTRACTOR_AVAILABLE = True
        except ImportError:
            URL_EXTRACTOR_AVAILABLE = False
            logging.warning("URL extractor not available")
        
        # Import Wikipedia extractor
        try:
            from wikipedia_extractor import WikipediaExtractor
            WIKIPEDIA_EXTRACTOR_AVAILABLE = True
        except ImportError:
            WIKIPEDIA_EXTRACTOR_AVAILABLE = False
            logging.warning("Wikipedia extractor not available")
        
        # Validate input - check for text or URL
        if not data:
            return jsonify({
                'success': False,
                'error': 'متن یا URL ورودی الزامی است'
            }), 400
        
        text = data.get('text', '').strip()
        url = data.get('url', '').strip()
        use_wikipedia_extraction = data.get('use_wikipedia_extraction', True)  # پیش‌فرض: فعال
        
        # اگر URL داده شده، متن را از URL استخراج کن
        if url:
            if not URL_EXTRACTOR_AVAILABLE:
                return jsonify({
                    'success': False,
                    'error': 'استخراج از URL در دسترس نیست. لطفاً متن را مستقیماً وارد کنید.'
                }), 400
            
            if not is_valid_url(url):
                return jsonify({
                    'success': False,
                    'error': 'URL نامعتبر است'
                }), 400
            
            # بررسی اینکه آیا URL ویکی‌پدیا است
            is_wikipedia = 'wikipedia.org' in url.lower()
            
            if is_wikipedia and WIKIPEDIA_EXTRACTOR_AVAILABLE and use_wikipedia_extraction:
                # استفاده از استخراج تخصصی ویکی‌پدیا
                try:
                    wiki_extractor = WikipediaExtractor(language='fa' if 'fa.wikipedia' in url else 'en')
                    wiki_data = wiki_extractor.extract_from_url(url)
                    
                    if "error" in wiki_data:
                        # Fallback به استخراج عادی با clean_content
                        extracted_text = extract_text_from_url(url, clean_content=True, max_length=10000)
                        if not extracted_text:
                            return jsonify({
                                'success': False,
                                'error': wiki_data.get("error", "خطا در استخراج از ویکی‌پدیا")
                            }), 400
                        text = extracted_text
                    else:
                        # استفاده از داده‌های استخراج شده از ویکی‌پدیا
                        text = wiki_data.get("text", "")
                        if not text:
                            text = wiki_extractor.get_full_text(wiki_data.get("title", ""))
                        
                        # اگر متن خالی است، از API استخراج کن
                        if not text or len(text.strip()) < 100:
                            # تلاش مجدد با API
                            api_result = wiki_extractor._extract_via_api(wiki_data.get("title", ""))
                            if api_result and api_result.get("text"):
                                text = api_result.get("text")
                        
                        # اضافه کردن موجودیت‌ها و روابط از ویکی‌پدیا به extraction_params
                        if "entities" in wiki_data and "relationships" in wiki_data:
                            # این داده‌ها بعداً در process_text_to_graph استفاده می‌شوند
                            data['wikipedia_entities'] = wiki_data.get("entities", [])
                            data['wikipedia_relationships'] = wiki_data.get("relationships", [])
                        
                        logging.info(f"Wikipedia data extracted from URL: {url} ({len(text)} characters, {len(wiki_data.get('entities', []))} entities)")
                except Exception as e:
                    logging.warning(f"Wikipedia extraction failed, falling back to regular extraction: {e}")
                    extracted_text = extract_text_from_url(url, clean_content=True, max_length=10000)
                    if not extracted_text:
                        return jsonify({
                            'success': False,
                            'error': f'خطا در استخراج از URL: {str(e)}'
                        }), 400
                    text = extracted_text
            else:
                # استخراج عادی از URL با clean_content=True برای حذف محتوای غیرضروری
                extracted_text = extract_text_from_url(url, clean_content=True, max_length=10000)
                if not extracted_text:
                    return jsonify({
                        'success': False,
                        'error': 'خطا در استخراج متن از URL. لطفاً URL را بررسی کنید.'
                    }), 400
                
                text = extracted_text
                logging.info(f"Text extracted from URL: {url} ({len(text)} characters)")
        
        # بررسی اینکه متن وجود دارد
        if not text:
            return jsonify({
                'success': False,
                'error': 'متن نمی‌تواند خالی باشد'
            }), 400
        
        # Get extraction parameters
        method = data.get('method', 'simple')
        max_entities = data.get('max_entities', 100)
        max_relationships = data.get('max_relationships', 200)
        llm_model = data.get('llm_model', 'mistralai/Mistral-7B-Instruct-v0.2')
        confidence_threshold = data.get('confidence_threshold', 0.5)
        hf_token = data.get('hf_token')  # Get token from request
        max_gleanings = data.get('max_gleanings', 2)
        enable_entity_resolution = data.get('enable_entity_resolution', True)
        enable_relationship_weighting = data.get('enable_relationship_weighting', True)
        min_relationship_weight = data.get('min_relationship_weight', 0.0)
        remove_isolated_nodes = data.get('remove_isolated_nodes', False)
        hybrid_methods = data.get('hybrid_methods', ['spacy', 'llm'])
        
        # New parameters for Persian and advanced features
        language = data.get('language', 'auto')  # auto/fa/en
        enable_coreference = data.get('enable_coreference', False)
        chunking_strategy = data.get('chunking_strategy', 'smart')  # smart/sliding_window/sentence/paragraph
        chunk_overlap = data.get('chunk_overlap', 0.2)  # 0.0 to 1.0
        max_tokens = data.get('max_tokens', 512)
        span_model_type = data.get('span_model_type', 'biobert')  # biobert/scibert/auto
        enable_preprocessing = data.get('enable_preprocessing', False)  # پیش‌پردازش متن (حذف stop words)
        
        # Validate method
        valid_methods = ['simple', 'spacy', 'spacy_svo_enhanced', 'llm', 'llm_multipass', 'hybrid',
                        'persian', 'span_based', 'with_coreference', 'long_text',
                        'joint_er', 'autoregressive', 'edc', 'incremental']
        if method not in valid_methods:
            return jsonify({
                'success': False,
                'error': f'روش استخراج نامعتبر است. روش‌های مجاز: {", ".join(valid_methods)}'
            }), 400
        
        # Initialize text to graph service
        try:
            # Ensure environment variables are loaded
            try:
                from dotenv import load_dotenv
                load_dotenv(override=True)
            except Exception:
                pass
            
            # Set HF_TOKEN from request if provided
            if hf_token:
                os.environ['HF_TOKEN'] = hf_token
                logging.info(f"HF_TOKEN set from request (length: {len(hf_token)})")
            
            text_to_graph_service = TextToGraphService(
                openai_api_key=OPENAI_API_KEY,
                spacy_model='en_core_web_sm',
                hf_token=hf_token  # Pass token directly to service
            )
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'خطا در راه‌اندازی سرویس: {str(e)}'
            }), 500
        
        # Prepare extraction parameters
        extraction_params = {
            'max_entities': max_entities,
            'max_relationships': max_relationships
        }
        
        if method in ['llm', 'llm_multipass', 'autoregressive', 'edc']:
            extraction_params['model'] = llm_model
            extraction_params['confidence_threshold'] = confidence_threshold
        
        if method == 'llm_multipass':
            extraction_params['max_gleanings'] = max_gleanings
        
        if method == 'hybrid':
            extraction_params['methods'] = hybrid_methods
            extraction_params['confidence_threshold'] = confidence_threshold
        
        # New method-specific parameters
        if method == 'joint_er':
            extraction_params['structure_iterations'] = data.get('structure_iterations', 3)
        
        if method == 'autoregressive':
            extraction_params['max_generation_length'] = data.get('max_generation_length', 2048)
        
        if method == 'edc':
            extraction_params['use_rag'] = data.get('use_rag', True)
        
        if method == 'incremental':
            extraction_params['chunk_size'] = data.get('chunk_size', 500)
            extraction_params['overlap'] = data.get('overlap', 100)
            extraction_params['base_method'] = data.get('base_method', 'spacy')
        
        # New method-specific parameters
        if method == 'persian':
            extraction_params['enable_coreference'] = enable_coreference
        
        if method == 'span_based':
            extraction_params['model_type'] = span_model_type
        
        if method == 'with_coreference':
            extraction_params['base_method'] = data.get('base_method', 'spacy')
        
        if method == 'long_text':
            extraction_params['chunking_strategy'] = chunking_strategy
            extraction_params['chunk_overlap'] = chunk_overlap
            extraction_params['max_tokens'] = max_tokens
            extraction_params['base_method'] = data.get('base_method', 'spacy')
        
        # Prepare processing parameters
        processing_params = {
            'enable_entity_resolution': enable_entity_resolution,
            'enable_relationship_weighting': enable_relationship_weighting,
            'min_relationship_weight': min_relationship_weight,
            'remove_isolated_nodes': remove_isolated_nodes,
            'enable_preprocessing': enable_preprocessing,
            'language': language
        }
        
        # Extract and build graph
        try:
            # اگر URL داده شده، از process_url_to_graph استفاده کن
            if url:
                result = text_to_graph_service.process_url_to_graph(
                    url=url,
                    method=method,
                    use_wikipedia_extraction=use_wikipedia_extraction,
                    save=True,
                    output_dir=UPLOAD_FOLDER,
                    **extraction_params,
                    **processing_params
                )
            else:
                # استفاده از process_text_to_graph برای متن مستقیم
                result = text_to_graph_service.process_text_to_graph(
                    text=text,
                    method=method,
                    save=True,
                    output_dir=UPLOAD_FOLDER,
                    **extraction_params,
                    **processing_params
                )
            
            # اگر داده‌های ویکی‌پدیا وجود دارد، آن‌ها را با نتایج استخراج ادغام کن
            if 'wikipedia_entities' in data and 'wikipedia_relationships' in data:
                wiki_entities = data.get('wikipedia_entities', [])
                wiki_relationships = data.get('wikipedia_relationships', [])
                
                if wiki_entities or wiki_relationships:
                    extraction_result = result.get('extraction_result', {})
                    existing_entities = extraction_result.get('entities', [])
                    existing_relationships = extraction_result.get('relationships', [])
                    
                    # ایجاد map برای موجودیت‌های موجود (بر اساس name)
                    entity_map = {}
                    entity_id_map = {}  # name -> id
                    for ent in existing_entities:
                        ent_name = ent.get('name', '').lower().strip()
                        if ent_name:
                            entity_map[ent_name] = ent
                            entity_id_map[ent_name] = ent.get('id')
                    
                    # ادغام موجودیت‌های ویکی‌پدیا
                    next_id = len(existing_entities)
                    for wiki_ent in wiki_entities:
                        ent_name = wiki_ent.get('name', '').strip()
                        if not ent_name:
                            continue
                        
                        ent_name_lower = ent_name.lower()
                        
                        # بررسی تکراری بودن
                        if ent_name_lower in entity_map:
                            # موجودیت موجود است - به‌روزرسانی attributes
                            existing_ent = entity_map[ent_name_lower]
                            if 'wikipedia' not in existing_ent.get('attributes', {}).get('source', ''):
                                # اضافه کردن اطلاعات ویکی‌پدیا
                                if 'attributes' not in existing_ent:
                                    existing_ent['attributes'] = {}
                                existing_ent['attributes']['wikipedia_source'] = True
                        else:
                            # موجودیت جدید - اضافه کردن
                            wiki_ent['id'] = f"ENTITY_{next_id}"
                            wiki_ent['attributes'] = wiki_ent.get('attributes', {})
                            wiki_ent['attributes']['source'] = 'wikipedia'
                            existing_entities.append(wiki_ent)
                            entity_map[ent_name_lower] = wiki_ent
                            entity_id_map[ent_name_lower] = wiki_ent['id']
                            next_id += 1
                    
                    # ادغام روابط
                    rel_set = set()
                    for rel in existing_relationships:
                        source = rel.get('source', '')
                        target = rel.get('target', '')
                        metaedge = rel.get('metaedge', '')
                        rel_set.add((source, target, metaedge))
                    
                    # تبدیل source/target در روابط ویکی‌پدیا به ID
                    for wiki_rel in wiki_relationships:
                        source_name = wiki_rel.get('source', '').strip()
                        target_name = wiki_rel.get('target', '').strip()
                        
                        # پیدا کردن ID موجودیت‌ها
                        source_id = None
                        target_id = None
                        
                        # جستجو در entity_id_map
                        for name, eid in entity_id_map.items():
                            if source_name.lower() == name or source_name.lower() in name or name in source_name.lower():
                                source_id = eid
                            if target_name.lower() == name or target_name.lower() in name or name in target_name.lower():
                                target_id = eid
                        
                        # اگر پیدا نشد، از ID مستقیم استفاده کن
                        if not source_id:
                            source_id = wiki_rel.get('source', '')
                        if not target_id:
                            target_id = wiki_rel.get('target', '')
                        
                        if source_id and target_id:
                            key = (source_id, target_id, wiki_rel.get('metaedge', ''))
                            if key not in rel_set:
                                wiki_rel['source'] = source_id
                                wiki_rel['target'] = target_id
                                wiki_rel['attributes'] = wiki_rel.get('attributes', {})
                                wiki_rel['attributes']['source'] = 'wikipedia'
                                existing_relationships.append(wiki_rel)
                                rel_set.add(key)
                    
                    # به‌روزرسانی extraction_result
                    extraction_result['entities'] = existing_entities
                    extraction_result['relationships'] = existing_relationships
                    extraction_result['wikipedia_extracted'] = True
                    extraction_result['wikipedia_stats'] = {
                        'wiki_entities': len(wiki_entities),
                        'wiki_relationships': len(wiki_relationships),
                        'merged_entities': len(existing_entities),
                        'merged_relationships': len(existing_relationships)
                    }
                    
                    # ساخت مجدد گراف با داده‌های ادغام شده
                    try:
                        graph = text_to_graph_service.build_graph(extraction_result)
                        result['graph'] = graph
                        result['extraction_result'] = extraction_result
                        logging.info(f"Merged Wikipedia data: {len(wiki_entities)} entities, {len(wiki_relationships)} relationships")
                    except Exception as e:
                        logging.warning(f"Failed to rebuild graph with Wikipedia data: {e}")
                        # استفاده از گراف قبلی
            
            # Get filename from filepath
            filename = os.path.basename(result['filepath']) if result.get('filepath') else None
            
            # Graph is saved and ready to be loaded via /api/load_graph endpoint
            # The graph will appear in the list of available graphs
            
            # Extract graph data for preview from the newly created graph
            graph = result.get('graph')
            extraction_result = result.get('extraction_result', {})
            graph_data = {
                'nodes': [],
                'edges': []
            }
            
            if graph:
                # Extract nodes from the graph with kind information
                for node_id, node_data in graph.nodes(data=True):
                    # Get kind from node data (prefer 'kind' over 'type')
                    node_kind = node_data.get('kind') or node_data.get('type', 'Unknown')
                    node_name = node_data.get('name', node_id)
                    
                    graph_data['nodes'].append({
                        'id': node_id,
                        'label': node_name,
                        'type': node_data.get('type', 'Unknown'),
                        'kind': node_kind,
                        'title': f"نام: {node_name}\nنوع: {node_data.get('type', 'Unknown')}\nKind: {node_kind}"
                    })
                
                # Extract edges from the graph with relationship information
                for source, target, edge_data in graph.edges(data=True):
                    # Get metaedge/relation from edge data
                    metaedge = edge_data.get('metaedge', 'related_to')
                    relation = edge_data.get('relation') or metaedge
                    
                    # Get relationship meaning/description
                    relation_meaning = edge_data.get('relation_meaning') or edge_data.get('description') or relation
                    
                    graph_data['edges'].append({
                        'from': source,
                        'to': target,
                        'label': relation,
                        'metaedge': metaedge,
                        'relation': relation,
                        'relation_meaning': relation_meaning,
                        'title': f"رابطه: {relation}\nمفهوم: {relation_meaning}"
                    })
            
            # Also include extraction result data for reference
            extraction_entities = extraction_result.get('entities', [])
            extraction_relationships = extraction_result.get('relationships', [])
            
            return jsonify({
                'success': True,
                'message': 'گراف با موفقیت ساخته شد',
                'filename': filename,
                'filepath': result.get('filepath'),
                'stats': result.get('stats', {}),
                'extraction_method': method,
                'resolution_summary': result.get('resolution_summary'),
                'load_url': '/api/load_graph',  # Hint for frontend to optionally load the graph
                'graph_data': graph_data,  # Graph data for preview (from newly created graph)
                'extraction_data': {  # Original extraction data for reference
                    'entities': extraction_entities,
                    'relationships': extraction_relationships
                }
            })
            
        except ValueError as e:
            # Handle validation errors - include more details
            error_msg = str(e)
            logging.error(f"Validation error in text to graph conversion: {error_msg}")
            logging.error(f"Error type: {type(e).__name__}")
            
            # Try to extract more context from the error
            import traceback
            error_trace = traceback.format_exc()
            logging.error(f"Error traceback: {error_trace}")
            
            return jsonify({
                'success': False,
                'error': error_msg,
                'error_type': 'validation_error',
                'method': method
            }), 400
        except Exception as e:
            # Handle other errors - include full traceback
            import traceback
            error_msg = str(e)
            error_trace = traceback.format_exc()
            logging.error(f"Error in text to graph conversion: {error_msg}")
            logging.error(f"Error type: {type(e).__name__}")
            logging.error(f"Full traceback: {error_trace}")
            
            return jsonify({
                'success': False,
                'error': f'خطا در تبدیل متن به گراف: {error_msg}',
                'error_type': 'server_error',
                'method': method,
                'details': error_trace[-500:] if len(error_trace) > 500 else error_trace  # Last 500 chars
            }), 500
    
    except Exception as e:
        logging.error(f"Unexpected error in text_to_graph endpoint: {e}")
        return jsonify({
            'success': False,
            'error': f'خطای غیرمنتظره: {str(e)}'
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
        if OPENAI_API_KEY:
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

@app.route('/api/graph_view_data', methods=['POST'])
def graph_view_data():
    """دریافت داده‌های گراف برای نمایش ویژوال و آماری بدون تغییر گراف فعال سیستم"""
    try:
        data = request.get_json() or {}
        graph_path = data.get('graph_path')

        if not graph_path:
            return jsonify({
                'success': False,
                'error': 'مسیر گراف مشخص نشده است'
            }), 400

        # Normalize path - handle both relative and absolute paths
        graph_path = str(graph_path).strip()

        # Decode URL-encoded path if needed
        from urllib.parse import unquote
        if '%' in graph_path:
            graph_path = unquote(graph_path)

        # تلاش برای رفع مشکلات مسیر (مثل تکرار نام پوشه یا کاراکترهای عجیب)
        original_graph_path = graph_path

        # اگر فایل مستقیماً وجود دارد، همان را استفاده کن
        if not os.path.exists(graph_path):
            # همیشه یک بار فقط بر اساس نام فایل هم امتحان می‌کنیم
            filename_only = os.path.basename(graph_path)

            candidate_paths = []

            # 1) uploaded_graphs/filename
            candidate_paths.append(os.path.join(UPLOAD_FOLDER, filename_only))

            # 2) ./filename در دایرکتوری فعلی
            candidate_paths.append(os.path.join('.', filename_only))

            # 3) اگر در مسیر رشته‌ی uploaded_graphs آمده، بعد از آن را به عنوان نام فایل بگیر
            lower_path = graph_path.lower()
            marker = 'uploaded_graphs'
            if marker in lower_path:
                idx = lower_path.rfind(marker)
                tail = graph_path[idx + len(marker):]
                tail = tail.lstrip('\\/._\u0082')
                if tail:
                    candidate_paths.append(os.path.join(UPLOAD_FOLDER, tail))

            # اولین مسیری که وجود دارد را انتخاب کن
            resolved = None
            for c in candidate_paths:
                if os.path.exists(c):
                    resolved = c
                    break

            if resolved is None:
                # تلاش ویژه برای فایل‌های text_graph:
                # اگر بخشی عددی مثل 60107_115532 در نام باشد، در بین فایل‌های uploaded_graphs جستجوی تطابق انجام می‌دهیم
                import re as _re
                numeric_match = _re.search(r'(\d{5}_\d{6})', filename_only)
                if numeric_match:
                    numeric_part = numeric_match.group(1)
                    try:
                        for f in os.listdir(UPLOAD_FOLDER):
                            # به‌دنبال همان بخش عددی وسط نام فایل می‌گردیم
                            if numeric_part in f and f.endswith('_text_graph.pkl'):
                                candidate = os.path.join(UPLOAD_FOLDER, f)
                                if os.path.exists(candidate):
                                    resolved = candidate
                                    candidate_paths.append(candidate)
                                    break
                    except Exception as e:
                        logging.error(f"خطا در جستجوی فایل در uploaded_graphs: {str(e)}")

                if resolved is None:
                    logging.error(f"مسیر گراف یافت نشد. ورودی: {original_graph_path}, امتحان‌شده: {candidate_paths}")
                    return jsonify({
                        'success': False,
                        'error': f'مسیر گراف نامعتبر است: {original_graph_path}'
                    }), 400

            graph_path = resolved

        # بارگذاری موقت گراف فقط برای نمایش (بدون تغییر graphrag_service سراسری)
        try:
            temp_service = GraphRAGService(graph_data_path=graph_path)
            G = getattr(temp_service, 'G', None)

            if G is None:
                return jsonify({
                    'success': False,
                    'error': 'گراف در فایل یافت نشد یا فایل خالی است'
                }), 500
        except Exception as e:
            logging.error(f"خطا در بارگذاری گراف از مسیر {graph_path}: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'خطا در بارگذاری گراف: {str(e)}'
            }), 500

        # محاسبه آمار پایه گراف
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0
        try:
            import networkx as nx  # برای محاسبه چگالی اگر در دسترس باشد
            density = nx.density(G)
        except Exception:
            density = None

        # شمارش انواع نود بر اساس فیلد kind یا type
        node_types = {}
        for node_id, node_data in G.nodes(data=True):
            kind = node_data.get('kind') or node_data.get('type', 'Unknown')
            node_types[kind] = node_types.get(kind, 0) + 1

        stats = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'density': density
        }

        # آماده‌سازی داده‌های گراف برای ویژوال‌سازی
        graph_data = {
            'nodes': [],
            'edges': []
        }

        for node_id, node_data in G.nodes(data=True):
            node_kind = node_data.get('kind') or node_data.get('type', 'Unknown')
            node_name = node_data.get('name', node_id)

            graph_data['nodes'].append({
                'id': node_id,
                'label': node_name,
                'type': node_data.get('type', 'Unknown'),
                'kind': node_kind,
                'title': f"نام: {node_name}\nنوع: {node_data.get('type', 'Unknown')}\nKind: {node_kind}"
            })

        for source, target, edge_data in G.edges(data=True):
            metaedge = edge_data.get('metaedge', 'related_to')
            relation = edge_data.get('relation') or metaedge
            relation_meaning = edge_data.get('relation_meaning') or edge_data.get('description') or relation

            graph_data['edges'].append({
                'from': source,
                'to': target,
                'label': relation,
                'metaedge': metaedge,
                'relation': relation,
                'relation_meaning': relation_meaning,
                'title': f"رابطه: {relation}\nمفهوم: {relation_meaning}"
            })

        return jsonify({
            'success': True,
            'stats': stats,
            'node_types': node_types,
            'graph_data': graph_data
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
        text_generation_type = data.get('text_generation_type', 'INTELLIGENT')
        max_depth = data.get('max_depth', 2)
        
        # تنظیمات پیشرفته (اختیاری)
        max_nodes = data.get('max_nodes', 20)
        max_edges = data.get('max_edges', 40)
        similarity_threshold = data.get('similarity_threshold', 0.3)
        community_detection_method = data.get('community_detection_method', 'louvain')
        advanced_retrieval_algorithm = data.get('advanced_retrieval_algorithm', 'hybrid')
        advanced_token_extraction_method = data.get('advanced_token_extraction_method', 'llm_based')
        
        # تبدیل رشته به enum
        retrieval_enum = RetrievalMethod[retrieval_method]
        generation_enum = GenerationModel[generation_model.replace(' ', '_')]
        
        # تنظیم پیکربندی پیشرفته اگر ارائه شده باشد
        if any([max_nodes != 20, max_edges != 40, similarity_threshold != 0.3]):
            graphrag_service.set_config(
                max_nodes=max_nodes,
                max_edges=max_edges,
                similarity_threshold=similarity_threshold,
                community_detection_method=community_detection_method,
                advanced_retrieval_algorithm=advanced_retrieval_algorithm,
                advanced_token_extraction_method=advanced_token_extraction_method
            )
        
        # پردازش سوال
        result = graphrag_service.process_query(
            query=query,
            retrieval_method=retrieval_enum,
            generation_model=generation_enum,
            text_generation_type=text_generation_type,
            max_depth=max_depth
        )
        
        # اضافه کردن timestamp و تنظیمات استفاده شده
        result['timestamp'] = datetime.now().isoformat()
        result['advanced_settings'] = {
            'max_nodes': max_nodes,
            'max_edges': max_edges,
            'similarity_threshold': similarity_threshold,
            'community_detection_method': community_detection_method,
            'advanced_retrieval_algorithm': advanced_retrieval_algorithm,
            'advanced_token_extraction_method': advanced_token_extraction_method
        }
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/enhanced_process_query', methods=['POST'])
def enhanced_process_query():
    """پردازش سوال با سرویس پیشرفته GraphRAG"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        token_extraction_method = data.get('token_extraction_method', 'llm_based')
        retrieval_algorithm = data.get('retrieval_algorithm', 'hybrid')
        community_detection_method = data.get('community_detection_method', 'louvain')
        max_depth = data.get('max_depth', 3)
        max_nodes = data.get('max_nodes', 20)
        max_edges = data.get('max_edges', 40)
        similarity_threshold = data.get('similarity_threshold', 0.3)
        
        # تنظیم پیکربندی سرویس پیشرفته
        enhanced_graphrag_service.set_config(
            token_extraction_method=token_extraction_method,
            retrieval_algorithm=retrieval_algorithm,
            community_detection_method=community_detection_method,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_edges=max_edges,
            similarity_threshold=similarity_threshold
        )
        
        # پردازش سوال
        result = enhanced_graphrag_service.process_query(query)
        
        # اضافه کردن timestamp
        result['timestamp'] = datetime.now().isoformat()
        result['config'] = enhanced_graphrag_service.get_config()
        
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

@app.route('/api/enhanced_graph_info')
def enhanced_graph_info():
    """اطلاعات گراف پیشرفته"""
    try:
        stats = enhanced_graphrag_service.get_graph_statistics()
        if stats:
            return jsonify({
                'success': True,
                'statistics': stats
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Enhanced graph not loaded'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/token_extraction_methods')
def token_extraction_methods():
    """دریافت روش‌های استخراج توکن"""
    methods = [
        {
            'value': 'llm_based',
            'label': 'بر اساس LLM',
            'description': 'استخراج توکن با استفاده از مدل‌های زبانی'
        },
        {
            'value': 'rule_based',
            'label': 'بر اساس قوانین',
            'description': 'استخراج توکن با استفاده از قوانین از پیش تعریف شده'
        },
        {
            'value': 'hybrid',
            'label': 'ترکیبی',
            'description': 'ترکیب روش‌های LLM و قوانین'
        },
        {
            'value': 'semantic',
            'label': 'معنایی',
            'description': 'استخراج توکن بر اساس شباهت معنایی'
        }
    ]
    return jsonify({'methods': methods})

@app.route('/api/retrieval_algorithms')
def retrieval_algorithms():
    """دریافت الگوریتم‌های بازیابی"""
    algorithms = [
        # الگوریتم‌های اصلی
        {
            'value': 'semantic_similarity',
            'label': 'شباهت معنایی (بهبود یافته)',
            'description': 'بازیابی بر اساس شباهت معنایی با افزایش دقت و پوشش ژن‌ها'
        },
        {
            'value': 'hybrid',
            'label': 'ترکیبی (بهبود یافته)',
            'description': 'ترکیب چندین الگوریتم با بهینه‌سازی عملکرد و افزایش دقت'
        },
        {
            'value': 'pagerank',
            'label': 'PageRank (بهبود یافته)',
            'description': 'رتبه‌بندی بر اساس الگوریتم PageRank با تمرکز روی ژن‌های مهم'
        },
        {
            'value': 'community_detection',
            'label': 'تشخیص جامعه (بهبود یافته)',
            'description': 'بازیابی بر اساس تشخیص جامعه‌ها با افزایش دقت خوشه‌بندی'
        },
        {
            'value': 'bfs',
            'label': 'BFS (جستجوی سطح اول - بهبود یافته)',
            'description': 'جستجوی سطح اول در گراف با افزایش عمق و بهینه‌سازی'
        },
        {
            'value': 'dfs',
            'label': 'DFS (جستجوی عمیق اول - بهبود یافته)',
            'description': 'جستجوی عمیق اول در گراف با افزایش دقت و کاهش زمان'
        },
        {
            'value': 'n_hop',
            'label': 'N-Hop (بهبود یافته)',
            'description': 'بازیابی مسیرهای N-Hop با افزایش عمق و رتبه‌بندی هوشمند'
        }
    ]
    return jsonify({'algorithms': algorithms})

@app.route('/api/community_detection_methods')
def community_detection_methods():
    """دریافت روش‌های تشخیص جامعه"""
    methods = [
        {
            'value': 'louvain',
            'label': 'Louvain',
            'description': 'الگوریتم Louvain برای تشخیص جامعه'
        },
        {
            'value': 'label_propagation',
            'label': 'Label Propagation',
            'description': 'انتشار برچسب برای تشخیص جامعه'
        },
        {
            'value': 'girvan_newman',
            'label': 'Girvan-Newman',
            'description': 'الگوریتم Girvan-Newman'
        },
        {
            'value': 'spectral',
            'label': 'Spectral',
            'description': 'روش طیفی برای تشخیص جامعه'
        }
    ]
    return jsonify({'methods': methods})

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

@app.route('/api/enhanced_config', methods=['GET', 'POST'])
def enhanced_config_endpoint():
    """مدیریت تنظیمات سرویس پیشرفته"""
    if request.method == 'GET':
        # دریافت تنظیمات فعلی
        try:
            config = enhanced_graphrag_service.get_config()
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
            enhanced_graphrag_service.set_config(**new_config)
            
            # دریافت تنظیمات به‌روز شده
            updated_config = enhanced_graphrag_service.get_config()
            
            return jsonify({
                'success': True,
                'message': 'تنظیمات پیشرفته با موفقیت به‌روز شد',
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

@app.route('/api/compare_with_gpt', methods=['POST'])
def compare_with_gpt():
    try:
        data = request.get_json()
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')
        label1 = data.get('label1', 'روش اول')
        label2 = data.get('label2', 'روش دوم')
        comparison_type = data.get('comparison_type', 'comprehensive')
        gpt_model = data.get('gpt_model', 'gpt-4o')
        
        if not text1 or not text2:
            return jsonify({'error': 'هر دو متن باید وارد شوند'}), 400
        
        # Create a comprehensive prompt for GPT
        prompt = create_gpt_comparison_prompt(text1, text2, label1, label2, comparison_type)
        
        # Check if OpenAI is available
        if openai is None:
            return jsonify({'error': 'OpenAI کتابخانه نصب نشده است. لطفاً با دستور pip install openai آن را نصب کنید.'}), 500
        
        # Call OpenAI API
        try:
            # Set the API key for this request
            openai.api_key = OPENAI_API_KEY
            
            response = openai.chat.completions.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": "شما یک متخصص ارزیابی کیفیت متن در حوزه زیست‌پزشکی، ژنتیک و پزشکی شخصی هستید. وظیفه شما مقایسه دو متن از نظر سطح علمی، ساختار تحلیلی، عمق مفهومی و کاربردپذیری بالینی است.\n\nپاسخی که دارای ویژگی‌های زیر باشد، باید امتیاز بالاتری بگیرد:\n- تحلیل دقیق مسیرهای زیستی و سیگنالینگ \n- اشاره به ژن‌های خاص با نقش بالینی \n- پیوند واضح بین عملکرد ژن و بیماری\n- توضیح در مورد کاربردهای درمانی یا تشخیصی (مانند داروهای هدفمند، بیومارکرها، مهارکننده‌ها)\n- ساختار تحلیلی منظم شامل بخش‌بندی (مثلاً: اهمیت زیستی، اهمیت بالینی، کاربردها، نتیجه‌گیری)\n\nدر مقابل، پاسخ‌هایی که فقط اطلاعات کلی، عمومی یا غیرتحلیلی می‌دهند یا صرفاً فهرستی از اسامی هستند، باید امتیاز کمتری بگیرند.\n\nارزیابی باید دقیق، تحلیلی و با تمرکز بر کیفیت علمی، عمق محتوا و ارزش کاربردی انجام شود. هدف انتخاب پاسخی است که بیشترین ارزش را برای پژوهشگر یا متخصص حوزه زیست‌پزشکی داشته باشد."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            gpt_response = response.choices[0].message.content
            
            # Parse the GPT response
            parsed_result = parse_gpt_comparison_response(gpt_response, label1, label2, comparison_type)
            parsed_result['gpt_model'] = gpt_model
            
            return jsonify(parsed_result)
            
        except Exception as e:
            return jsonify({'error': f'خطا در ارتباط با {gpt_model}: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': f'خطا در مقایسه: {str(e)}'}), 500

def create_gpt_comparison_prompt(text1, text2, label1, label2, comparison_type):
    """ایجاد پرامپ مناسب برای مقایسه با GPT-4o"""
    
    comparison_focus = {
        'comprehensive': 'کیفیت کلی، دقت، جامعیت، وضوح، و مرتبط بودن',
        'practical_specialized': 'عملی بودن، تخصصی بودن، و کاربردی بودن محتوا',
        'accuracy': 'دقت و صحت اطلاعات ارائه شده',
        'completeness': 'جامعیت و کامل بودن پاسخ',
        'clarity': 'وضوح و قابل فهم بودن متن',
        'relevance': 'مرتبط بودن با سوال اصلی'
    }
    
    focus = comparison_focus.get(comparison_type, comparison_focus['comprehensive'])
    
    prompt = f"""
    لطفاً دو متن زیر را مقایسه کنید و تحلیل دقیقی ارائه دهید. توجه ویژه به عملی بودن و تخصصی بودن محتوا داشته باشید. از ارائه هرگونه امتیاز عددی خودداری کنید و فقط تحلیل متنی ارائه دهید.
    
    **متن اول ({label1}):**
    {text1}
    
    **متن دوم ({label2}):**
    {text2}
    
    **نوع مقایسه:** {focus}
    
    **معیارهای ارزیابی مهم:**
    1. **عملی بودن**: متن باید اطلاعات کاربردی و قابل استفاده ارائه دهد
    2. **تخصصی بودن**: استفاده از اصطلاحات تخصصی و مفاهیم دقیق علمی
    3. **دقت علمی**: صحت اطلاعات و استناد به مفاهیم علمی
    4. **جامعیت**: پوشش کامل جنبه‌های مختلف موضوع
    5. **وضوح**: قابل فهم بودن برای مخاطب تخصصی
    
    لطفاً تحلیل خود را در قالب زیر ارائه دهید و از امتیاز عددی استفاده نکنید:
    
    **خلاصه مقایسه:**
    [یک خلاصه کوتاه از تفاوت‌های اصلی با تمرکز بر عملی بودن و تخصصی بودن]
    
    **تحلیل کلی:**
    [تحلیل متنی از نقاط قوت و ضعف کلی هر دو متن و دلیل برتری نسبی]
    
    **نقاط قوت {label1}:**
    [لیست نقاط قوت با تمرکز بر جنبه‌های عملی و تخصصی]
    
    **نقاط قوت {label2}:**
    [لیست نقاط قوت با تمرکز بر جنبه‌های عملی و تخصصی]
    
    **نقاط ضعف {label1}:**
    [لیست نقاط ضعف از نظر عملی بودن و تخصصی بودن]
    
    **نقاط ضعف {label2}:**
    [لیست نقاط ضعف از نظر عملی بودن و تخصصی بودن]
    
    **توصیه نهایی:**
    [توصیه کدام روش بهتر است با تأکید بر عملی بودن و تخصصی بودن]
    """
    
    return prompt

def parse_gpt_comparison_response(response, label1, label2, comparison_type):
    """تجزیه و تحلیل پاسخ GPT-4o - فقط تحلیل متنی بدون امتیاز عددی"""
    import re
    
    # Split response into sections
    sections = response.split('\n\n')
    
    summary = ""
    analysis = ""
    strengths1 = ""
    strengths2 = ""
    weaknesses1 = ""
    weaknesses2 = ""
    recommendation = ""
    
    for section in sections:
        if "خلاصه مقایسه" in section:
            summary = section.replace("**خلاصه مقایسه:**", "").strip()
        elif "تحلیل کلی" in section:
            analysis = section.replace("**تحلیل کلی:**", "").strip()
        elif "تحلیل" in section and not analysis:
            # در صورت نبود عنوان دقیق «تحلیل کلی»، هر بخش حاوی واژه تحلیل را به عنوان تحلیل کلی در نظر می‌گیریم
            analysis = re.sub(r"^\*\*[^:]+:\*\*", "", section).strip()
        elif f"نقاط قوت {label1}" in section:
            strengths1 = section.replace(f"**نقاط قوت {label1}:**", "").strip()
        elif f"نقاط قوت {label2}" in section:
            strengths2 = section.replace(f"**نقاط قوت {label2}:**", "").strip()
        elif f"نقاط ضعف {label1}" in section:
            weaknesses1 = section.replace(f"**نقاط ضعف {label1}:**", "").strip()
        elif f"نقاط ضعف {label2}" in section:
            weaknesses2 = section.replace(f"**نقاط ضعف {label2}:**", "").strip()
        elif "توصیه نهایی" in section:
            recommendation = section.replace("**توصیه نهایی:**", "").strip()
    
    # If sections are empty, use the full response
    if not summary:
        summary = response[:200] + "..." if len(response) > 200 else response
    if not analysis:
        # تلاش برای استخراج یک تحلیل کلی از روی کل پاسخ در صورت نبود بخش مشخص
        analysis = ""
    
    return {
        'summary': summary,
        'analysis': analysis,
        'strengths1': strengths1,
        'strengths2': strengths2,
        'weaknesses1': weaknesses1,
        'weaknesses2': weaknesses2,
        'recommendation': recommendation,
        'label1': label1,
        'label2': label2,
        'comparison_type': comparison_type
    }

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
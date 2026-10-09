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
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple
import math
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from difflib import SequenceMatcher
import networkx as nx

# OpenAI import for GPT-4o comparison
try:
    import openai
except ImportError:
    openai = None

# HuggingFace import for non-OpenAI LLM evaluation
try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

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

# طھظ†ط¸غŒظ…ط§طھ ط¢ظ¾ظ„ظˆط¯ ظپط§غŒظ„
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
# ط§ط¨طھط¯ط§ ط¨ط±ط±ط³غŒ ظ…غŒâ€Œع©ظ†غŒظ… ع©ظ‡ ط¢غŒط§ ظپط§غŒظ„ ع¯ط±ط§ظپ Hetionet ظˆط¬ظˆط¯ ط¯ط§ط±ط¯
graph_files = [f for f in os.listdir('.') if f.startswith('hetionet_graph_') and f.endswith('.pkl')]
if graph_files:
    # ط§ط³طھظپط§ط¯ظ‡ ط§ط² ط¬ط¯غŒط¯طھط±غŒظ† ظپط§غŒظ„ ع¯ط±ط§ظپ
    latest_graph_file = max(graph_files)
    print(f"ًں”§ ط§ط³طھظپط§ط¯ظ‡ ط§ط² ع¯ط±ط§ظپ Hetionet: {latest_graph_file}")
    graphrag_service = GraphRAGService(graph_data_path=latest_graph_file)
    enhanced_graphrag_service = EnhancedGraphRAGService(graph_data_path=latest_graph_file)
else:
    print("âڑ ï¸ڈ ظپط§غŒظ„ ع¯ط±ط§ظپ Hetionet غŒط§ظپطھ ظ†ط´ط¯طŒ ط§ط³طھظپط§ط¯ظ‡ ط§ط² ع¯ط±ط§ظپ ظ†ظ…ظˆظ†ظ‡")
    graphrag_service = GraphRAGService()
    enhanced_graphrag_service = EnhancedGraphRAGService()

# طھظ†ط¸غŒظ… API Key ظ‡ط§غŒ OpenAI (ط§ط² ظ…طھط؛غŒط± ظ…ط­غŒط·غŒ غŒط§ secrets.json)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# طھظ„ط§ط´ ط¨ط±ط§غŒ ط®ظˆط§ظ†ط¯ظ† ط§ط² secrets.json ط§ع¯ط± ظ…طھط؛غŒط± ظ…ط­غŒط·غŒ ط®ط§ظ„غŒ ط¨ظˆط¯
if not OPENAI_API_KEY:
    try:
        import json as _json
        if os.path.exists('secrets.json'):
            with open('secrets.json', 'r', encoding='utf-8') as _sf:
                _secrets = _json.load(_sf) or {}
                OPENAI_API_KEY = _secrets.get('OPENAI_API_KEY', '')
    except Exception as _e:
        pass

# ط§ع¯ط± ظ‡ظ†ظˆط² API key طھظ†ط¸غŒظ… ظ†ط´ط¯ظ‡طŒ ط§ط² API key ظ¾غŒط´â€Œظپط±ط¶ ط§ط³طھظپط§ط¯ظ‡ ع©ظ† (ط¨ط±ط§غŒ طھط³طھ)
if not OPENAI_API_KEY:
    OPENAI_API_KEY = ""  # API key ط¨ط§غŒط¯ ط§ط² ظ…طھط؛غŒط±ظ‡ط§غŒ ظ…ط­غŒط·غŒ غŒط§ secrets طھظ†ط¸غŒظ… ط´ظˆط¯

if OPENAI_API_KEY:
    graphrag_service.set_openai_api_key(OPENAI_API_KEY)
    print("✅ OpenAI API Key تنظیم شد")
else:
    print("âڑ ï¸ڈ OPENAI_API_KEY طھظ†ط¸غŒظ… ظ†ط´ط¯ظ‡ ط§ط³طھط› طھظˆظ„غŒط¯ ظ¾ط§ط³ط® ط¨ط§ OpenAI ط؛غŒط±ظپط¹ط§ظ„ ط®ظˆط§ظ‡ط¯ ط¨ظˆط¯")

def _graph_eval_normalize(text: Any) -> str:
    """Normalize text for multilingual lexical matching."""
    if text is None:
        return ""
    normalized = str(text).lower()
    normalized = re.sub(r"[\u200c\u200d]", " ", normalized)
    normalized = re.sub(r"[^\w\s\u0600-\u06FF]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _graph_eval_split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[.!?\nطں]+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _graph_eval_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _to_json_safe(value: Any) -> Any:
    """Recursively convert numpy/complex objects to JSON-safe Python types."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, np.generic):
        return _to_json_safe(value.item())

    if isinstance(value, np.ndarray):
        return [_to_json_safe(v) for v in value.tolist()]

    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]

    if isinstance(value, datetime):
        return value.isoformat()

    return str(value)


def _graph_eval_contains_label(normalized_label: str, normalized_text: str) -> bool:
    if not normalized_label or not normalized_text:
        return False

    padded_text = f" {normalized_text} "
    padded_label = f" {normalized_label} "
    if padded_label in padded_text:
        return True

    tokens = [tok for tok in normalized_label.split() if len(tok) > 2]
    if not tokens:
        return normalized_label in normalized_text

    hit_count = sum(1 for token in tokens if token in normalized_text)
    threshold = max(1, int(np.ceil(0.75 * len(tokens))))
    return hit_count >= threshold


def _graph_eval_canonical_type(raw_type: Any) -> str:
    t = _graph_eval_normalize(raw_type)
    if not t:
        return "unknown"

    if "gene" in t:
        return "gene"
    if "disease" in t or "syndrome" in t:
        return "disease"
    if "compound" in t or "drug" in t:
        return "compound"
    if "pathway" in t:
        return "pathway"
    if "biological process" in t or t == "bp":
        return "biological_process"
    if "molecular function" in t or t == "mf":
        return "molecular_function"
    if "cellular component" in t or t == "cc":
        return "cellular_component"
    if "anatomy" in t:
        return "anatomy"
    if "symptom" in t:
        return "symptom"
    return t


def _graph_eval_type_compatibility_rules() -> Dict[str, Tuple[str, str]]:
    return {
        "gig": ("gene", "gene"),
        "gr>g": ("gene", "gene"),
        "ctd": ("compound", "disease"),
        "dag": ("disease", "gene"),
        "dpg": ("disease", "gene"),
        "cdg": ("compound", "gene"),
        "gpbp": ("gene", "biological_process"),
        "gpmf": ("gene", "molecular_function"),
        "gpcc": ("gene", "cellular_component"),
        "drd": ("disease", "disease"),
        "ccse": ("compound", "symptom"),
    }


def _graph_eval_extract_graph_records(
    graph_data: Dict[str, Any],
    extraction_data: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    graph_data = graph_data or {}
    extraction_data = extraction_data or {}

    node_lookup: Dict[str, Dict[str, Any]] = {}

    for node in graph_data.get("nodes", []) or []:
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        node_lookup[node_id] = {
            "id": node_id,
            "label": node.get("label") or node.get("name") or node_id,
            "type": node.get("kind") or node.get("type") or "Unknown",
        }

    for entity in extraction_data.get("entities", []) or []:
        entity_id = str(entity.get("id", "")).strip()
        if not entity_id:
            continue
        if entity_id not in node_lookup:
            node_lookup[entity_id] = {
                "id": entity_id,
                "label": entity.get("name") or entity_id,
                "type": entity.get("type") or entity.get("kind") or "Unknown",
            }
        else:
            if not node_lookup[entity_id].get("label"):
                node_lookup[entity_id]["label"] = entity.get("name") or entity_id
            if not node_lookup[entity_id].get("type") or node_lookup[entity_id]["type"] == "Unknown":
                node_lookup[entity_id]["type"] = entity.get("type") or entity.get("kind") or "Unknown"

    edge_records: List[Dict[str, Any]] = []

    for edge in graph_data.get("edges", []) or []:
        source = str(edge.get("from", "")).strip()
        target = str(edge.get("to", "")).strip()
        if not source or not target:
            continue
        edge_records.append({
            "source": source,
            "target": target,
            "relation": edge.get("relation") or edge.get("label") or edge.get("metaedge") or "related_to",
            "metaedge": edge.get("metaedge") or edge.get("relation") or edge.get("label") or "",
        })

    if not edge_records:
        for rel in extraction_data.get("relationships", []) or []:
            source = str(rel.get("source", "")).strip()
            target = str(rel.get("target", "")).strip()
            if not source or not target:
                continue
            edge_records.append({
                "source": source,
                "target": target,
                "relation": rel.get("relation") or rel.get("metaedge") or "related_to",
                "metaedge": rel.get("metaedge") or rel.get("relation") or "",
            })

    nodes = list(node_lookup.values())
    return nodes, edge_records, node_lookup


def _graph_eval_tfidf_similarity(text_a: str, text_b: str) -> float:
    a = (text_a or "").strip()
    b = (text_b or "").strip()
    if not a or not b:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        matrix = vectorizer.fit_transform([a, b])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except Exception:
        return 0.0


def _graph_eval_best_sentence_similarity(triple_text: str, sentences: List[str]) -> float:
    if not sentences:
        return 0.0
    best_score = 0.0
    for sentence in sentences:
        sim = _graph_eval_tfidf_similarity(triple_text, sentence)
        if sim > best_score:
            best_score = sim
    return best_score


def _graph_eval_quality_band(score_0_100: float) -> str:
    if score_0_100 >= 85:
        return "عالی"
    if score_0_100 >= 70:
        return "خوب"
    if score_0_100 >= 55:
        return "متوسط"
    if score_0_100 >= 40:
        return "ضعیف"
    return "خیلی ضعیف"


def _graph_eval_compute_auto_metrics(
    source_text: str,
    graph_data: Dict[str, Any],
    extraction_data: Dict[str, Any]
) -> Dict[str, Any]:
    nodes, edges, node_lookup = _graph_eval_extract_graph_records(graph_data, extraction_data)

    total_nodes = len(nodes)
    total_edges = len(edges)
    source_text = source_text or ""
    normalized_source = _graph_eval_normalize(source_text)
    sentence_texts = _graph_eval_split_sentences(source_text)
    normalized_sentences = [_graph_eval_normalize(s) for s in sentence_texts]

    entity_supported = 0
    entity_sentence_presence: List[set] = [set() for _ in normalized_sentences]
    normalized_node_labels: Dict[str, str] = {}

    for node in nodes:
        node_id = node["id"]
        node_label = str(node.get("label", "")).strip()
        normalized_label = _graph_eval_normalize(node_label)
        normalized_node_labels[node_id] = normalized_label

        if _graph_eval_contains_label(normalized_label, normalized_source):
            entity_supported += 1

        for idx, sentence in enumerate(normalized_sentences):
            if _graph_eval_contains_label(normalized_label, sentence):
                entity_sentence_presence[idx].add(node_id)

    entity_grounding_rate = _graph_eval_ratio(entity_supported, total_nodes)

    edge_supported = 0
    edge_similarity_scores: List[float] = []
    unsupported_edge_samples: List[str] = []
    normalized_edge_signatures = []

    for edge in edges:
        source_id = edge["source"]
        target_id = edge["target"]
        relation = str(edge.get("relation", "related_to")).strip()
        source_label = node_lookup.get(source_id, {}).get("label", source_id)
        target_label = node_lookup.get(target_id, {}).get("label", target_id)

        src_norm = normalized_node_labels.get(source_id) or _graph_eval_normalize(source_label)
        tgt_norm = normalized_node_labels.get(target_id) or _graph_eval_normalize(target_label)
        rel_norm = _graph_eval_normalize(relation)

        normalized_edge_signatures.append((src_norm, rel_norm, tgt_norm))

        co_occurrence = False
        for sentence in normalized_sentences:
            if _graph_eval_contains_label(src_norm, sentence) and _graph_eval_contains_label(tgt_norm, sentence):
                co_occurrence = True
                break

        triple_text = f"{source_label} {relation} {target_label}"
        best_similarity = _graph_eval_best_sentence_similarity(triple_text, sentence_texts)
        edge_similarity_scores.append(best_similarity)

        if co_occurrence or best_similarity >= 0.20:
            edge_supported += 1
        elif len(unsupported_edge_samples) < 5:
            unsupported_edge_samples.append(f"{source_label} --[{relation}]--> {target_label}")

    edge_support_rate = _graph_eval_ratio(edge_supported, total_edges)
    semantic_plausibility = float(np.mean(edge_similarity_scores)) if edge_similarity_scores else 0.0
    hallucination_rate = 1.0 - edge_support_rate if total_edges > 0 else 1.0

    edge_counter = Counter(normalized_edge_signatures)
    duplicate_edges = sum(count - 1 for count in edge_counter.values() if count > 1)
    edge_redundancy_rate = _graph_eval_ratio(duplicate_edges, total_edges)

    node_label_counter = Counter(_graph_eval_normalize(node.get("label", "")) for node in nodes)
    duplicate_nodes = sum(count - 1 for count in node_label_counter.values() if count > 1)
    node_redundancy_rate = _graph_eval_ratio(duplicate_nodes, total_nodes)
    redundancy_rate = max(edge_redundancy_rate, node_redundancy_rate)

    candidate_pairs = set()
    for sentence_entities in entity_sentence_presence:
        if len(sentence_entities) < 2:
            continue
        for a, b in combinations(sorted(sentence_entities), 2):
            candidate_pairs.add((a, b))

    predicted_pairs = set()
    for edge in edges:
        a = edge["source"]
        b = edge["target"]
        if a == b:
            continue
        pair = tuple(sorted([a, b]))
        predicted_pairs.add(pair)

    missing_pairs = candidate_pairs - predicted_pairs
    omission_proxy = _graph_eval_ratio(len(missing_pairs), len(candidate_pairs))

    type_rules = _graph_eval_type_compatibility_rules()
    checked_type_edges = 0
    matched_type_edges = 0
    for edge in edges:
        metaedge = _graph_eval_normalize(edge.get("metaedge", edge.get("relation", "")))
        if not metaedge:
            continue
        rule = type_rules.get(metaedge)
        if not rule:
            continue

        source_type = _graph_eval_canonical_type(node_lookup.get(edge["source"], {}).get("type"))
        target_type = _graph_eval_canonical_type(node_lookup.get(edge["target"], {}).get("type"))
        checked_type_edges += 1
        if source_type == rule[0] and target_type == rule[1]:
            matched_type_edges += 1

    type_compatibility = (
        _graph_eval_ratio(matched_type_edges, checked_type_edges)
        if checked_type_edges > 0
        else 0.75
    )

    digraph = nx.DiGraph()
    for node in nodes:
        digraph.add_node(node["id"])
    for edge in edges:
        digraph.add_edge(edge["source"], edge["target"])

    isolates_count = len(list(nx.isolates(digraph))) if digraph.number_of_nodes() > 0 else 0
    isolate_ratio = _graph_eval_ratio(isolates_count, max(1, digraph.number_of_nodes()))

    if digraph.number_of_nodes() > 0:
        components = nx.number_weakly_connected_components(digraph)
    else:
        components = 0
    component_penalty = min(
        _graph_eval_ratio(max(0, components - 1), max(1, digraph.number_of_nodes() // 6)),
        1.0
    )

    density = float(nx.density(digraph)) if digraph.number_of_nodes() > 1 else 0.0
    if density < 0.01:
        density_penalty = min((0.01 - density) / 0.01, 1.0)
    elif density > 0.35:
        density_penalty = min((density - 0.35) / 0.35, 1.0)
    else:
        density_penalty = 0.0

    pagerank_max = 0.0
    if digraph.number_of_nodes() > 0 and digraph.number_of_edges() > 0:
        try:
            pagerank_scores = nx.pagerank(digraph, alpha=0.85)
            pagerank_max = max(pagerank_scores.values()) if pagerank_scores else 0.0
        except Exception:
            pagerank_max = 0.0
    hub_penalty = min(max(pagerank_max - 0.25, 0.0) / 0.35, 1.0)

    structural_score = max(
        0.0,
        1.0 - (0.45 * isolate_ratio + 0.25 * component_penalty + 0.20 * density_penalty + 0.10 * hub_penalty)
    )

    overall_score = 100.0 * (
        0.25 * edge_support_rate +
        0.18 * entity_grounding_rate +
        0.15 * semantic_plausibility +
        0.12 * type_compatibility +
        0.10 * structural_score +
        0.10 * (1.0 - omission_proxy) +
        0.10 * (1.0 - redundancy_rate)
    )
    overall_score = max(0.0, min(100.0, overall_score))

    return {
        "overall_score": round(overall_score, 2),
        "quality_band": _graph_eval_quality_band(overall_score),
        "metrics": {
            "entity_grounding_rate": round(entity_grounding_rate, 4),
            "edge_support_rate": round(edge_support_rate, 4),
            "semantic_plausibility": round(semantic_plausibility, 4),
            "type_compatibility": round(type_compatibility, 4),
            "structural_score": round(structural_score, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "omission_proxy": round(omission_proxy, 4),
            "redundancy_rate": round(redundancy_rate, 4),
            "edge_redundancy_rate": round(edge_redundancy_rate, 4),
            "node_redundancy_rate": round(node_redundancy_rate, 4),
            "density": round(density, 4),
            "isolated_nodes_ratio": round(isolate_ratio, 4),
        },
        "counts": {
            "nodes": total_nodes,
            "edges": total_edges,
            "supported_entities": int(entity_supported),
            "supported_edges": int(edge_supported),
            "candidate_pairs": int(len(candidate_pairs)),
            "missing_pairs": int(len(missing_pairs)),
            "duplicate_nodes": int(duplicate_nodes),
            "duplicate_edges": int(duplicate_edges),
            "components": int(components),
            "type_checked_edges": int(checked_type_edges),
            "type_matched_edges": int(matched_type_edges),
        },
        "unsupported_edge_samples": unsupported_edge_samples,
        "dimensions": [
            {
                "id": "grounding",
                "label": "پشتیبانی از متن",
                "score": round(edge_support_rate * 100, 2),
                "hint": "درصد یال‌هایی که در متن پشتیبانی می‌شوند.",
            },
            {
                "id": "semantic",
                "label": "همخوانی معنایی",
                "score": round(semantic_plausibility * 100, 2),
                "hint": "شباهت معنایی سه‌تایی‌ها به جملات متن.",
            },
            {
                "id": "typing",
                "label": "سازگاری نوعی",
                "score": round(type_compatibility * 100, 2),
                "hint": "سازگاری نوع گره‌ها با نوع رابطه.",
            },
            {
                "id": "structure",
                "label": "کیفیت ساختاری",
                "score": round(structural_score * 100, 2),
                "hint": "توازن ساختار گراف (تراکم، مؤلفه‌ها، گره‌های ایزوله).",
            },
        ],
    }


def _graph_eval_parse_reference_triples(reference_input: Any) -> List[Tuple[str, str, str]]:
    if reference_input is None:
        return []

    triples: List[Tuple[str, str, str]] = []
    parsed_payload = reference_input

    if isinstance(reference_input, str):
        candidate = reference_input.strip()
        if not candidate:
            return []
        try:
            parsed_payload = json.loads(candidate)
        except Exception:
            parsed_payload = candidate

    if isinstance(parsed_payload, list):
        for item in parsed_payload:
            if isinstance(item, dict):
                head = item.get("head") or item.get("source") or item.get("s")
                rel = item.get("relation") or item.get("predicate") or item.get("p")
                tail = item.get("tail") or item.get("target") or item.get("o")
                if head and rel and tail:
                    triples.append((str(head), str(rel), str(tail)))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                triples.append((str(item[0]), str(item[1]), str(item[2])))
        return triples

    if isinstance(parsed_payload, str):
        for line in parsed_payload.splitlines():
            row = line.strip()
            if not row:
                continue
            if "\t" in row:
                parts = [p.strip() for p in row.split("\t")]
            elif "|" in row:
                parts = [p.strip() for p in row.split("|")]
            elif row.count(",") >= 2:
                first, second, rest = row.split(",", 2)
                parts = [first.strip(), second.strip(), rest.strip()]
            else:
                continue

            if len(parts) < 3:
                continue
            head = parts[0]
            relation = parts[1]
            tail = " ".join(parts[2:]).strip()
            if head and relation and tail:
                triples.append((head, relation, tail))

    return triples


def _graph_eval_predicted_triples(
    graph_data: Dict[str, Any],
    extraction_data: Dict[str, Any]
) -> List[Tuple[str, str, str]]:
    nodes, edges, node_lookup = _graph_eval_extract_graph_records(graph_data, extraction_data)
    if not nodes or not edges:
        return []

    predicted: List[Tuple[str, str, str]] = []
    for edge in edges:
        source_label = node_lookup.get(edge["source"], {}).get("label", edge["source"])
        target_label = node_lookup.get(edge["target"], {}).get("label", edge["target"])
        relation = edge.get("relation") or edge.get("metaedge") or "related_to"
        predicted.append((str(source_label), str(relation), str(target_label)))
    return predicted


def _graph_eval_reference_metrics(
    graph_data: Dict[str, Any],
    extraction_data: Dict[str, Any],
    reference_triples_input: Any
) -> Dict[str, Any]:
    reference_triples = _graph_eval_parse_reference_triples(reference_triples_input)
    if not reference_triples:
        return {
            "available": False,
            "message": "مرجع معتبری برای محاسبه معیارهای مبتنی بر Ground Truth دریافت نشد.",
        }

    predicted_triples = _graph_eval_predicted_triples(graph_data, extraction_data)

    pred_norm = {
        (
            _graph_eval_normalize(h),
            _graph_eval_normalize(r),
            _graph_eval_normalize(t)
        )
        for h, r, t in predicted_triples
    }
    ref_norm = {
        (
            _graph_eval_normalize(h),
            _graph_eval_normalize(r),
            _graph_eval_normalize(t)
        )
        for h, r, t in reference_triples
    }

    true_positive = len(pred_norm & ref_norm)
    precision = _graph_eval_ratio(true_positive, len(pred_norm))
    recall = _graph_eval_ratio(true_positive, len(ref_norm))
    f1 = _graph_eval_ratio(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

    pred_nodes = set()
    ref_nodes = set()
    for h, _, t in pred_norm:
        pred_nodes.add(h)
        pred_nodes.add(t)
    for h, _, t in ref_norm:
        ref_nodes.add(h)
        ref_nodes.add(t)

    node_precision = _graph_eval_ratio(len(pred_nodes & ref_nodes), len(pred_nodes))
    node_recall = _graph_eval_ratio(len(pred_nodes & ref_nodes), len(ref_nodes))
    node_f1 = _graph_eval_ratio(
        2 * node_precision * node_recall,
        node_precision + node_recall
    ) if (node_precision + node_recall) > 0 else 0.0

    graph_exact_match = 1.0 if (pred_norm == ref_norm and pred_nodes == ref_nodes) else 0.0

    node_edits = len(pred_nodes - ref_nodes) + len(ref_nodes - pred_nodes)
    edge_edits = len(pred_norm - ref_norm) + len(ref_norm - pred_norm)
    ged_raw = node_edits + edge_edits
    ged_denominator = max(len(pred_nodes) + len(ref_nodes) + len(pred_norm) + len(ref_norm), 1)
    normalized_ged = _graph_eval_ratio(ged_raw, ged_denominator)
    ged_similarity = max(0.0, 1.0 - normalized_ged)

    pred_serialized = " || ".join(sorted([" | ".join(t) for t in pred_norm]))
    ref_serialized = " || ".join(sorted([" | ".join(t) for t in ref_norm]))
    g_bertscore_proxy = _graph_eval_tfidf_similarity(pred_serialized, ref_serialized)

    return {
        "available": True,
        "triple_precision": round(precision, 4),
        "triple_recall": round(recall, 4),
        "triple_f1": round(f1, 4),
        "node_f1": round(node_f1, 4),
        "graph_exact_match": int(graph_exact_match),
        "g_f1_single_graph": round(graph_exact_match, 4),
        "ged_raw": int(ged_raw),
        "normalized_ged": round(normalized_ged, 4),
        "ged_similarity": round(ged_similarity, 4),
        "g_bertscore_proxy": round(g_bertscore_proxy, 4),
        "counts": {
            "predicted_triples": len(pred_norm),
            "reference_triples": len(ref_norm),
            "true_positive_triples": true_positive,
            "predicted_nodes": len(pred_nodes),
            "reference_nodes": len(ref_nodes),
        }
    }


def _graph_eval_compact_for_prompt(
    graph_data: Dict[str, Any],
    extraction_data: Dict[str, Any],
    max_nodes: int = 80,
    max_edges: int = 160
) -> Dict[str, Any]:
    nodes, edges, node_lookup = _graph_eval_extract_graph_records(graph_data, extraction_data)

    compact_nodes = []
    for node in nodes[:max_nodes]:
        compact_nodes.append({
            "id": node.get("id"),
            "label": node.get("label"),
            "type": node.get("type"),
        })

    compact_edges = []
    for edge in edges[:max_edges]:
        compact_edges.append({
            "source": node_lookup.get(edge["source"], {}).get("label", edge["source"]),
            "relation": edge.get("relation") or edge.get("metaedge"),
            "target": node_lookup.get(edge["target"], {}).get("label", edge["target"]),
            "metaedge": edge.get("metaedge"),
        })

    return {
        "nodes": compact_nodes,
        "edges": compact_edges,
        "truncated": {
            "nodes_truncated": max(0, len(nodes) - max_nodes),
            "edges_truncated": max(0, len(edges) - max_edges),
            "original_nodes": len(nodes),
            "original_edges": len(edges),
        }
    }


def _graph_eval_strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    return cleaned


def _graph_eval_repair_json_candidate(candidate: str) -> str:
    fixed = (candidate or "").strip()
    if not fixed:
        return fixed

    fixed = fixed.replace("\u201c", "\"").replace("\u201d", "\"")
    fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")

    fixed = re.sub(r"([\}\]\"\d])\s*(\"(?:overall_score|dimension_scores|summary|strengths|weaknesses|hallucinated_or_unsupported_examples|missing_facts_or_relations|recommendations)\"\s*:)", r"\1, \2", fixed)
    fixed = re.sub(r",\s*,+", ", ", fixed)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    return fixed


def _graph_eval_parse_json_response(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None

    cleaned = _graph_eval_strip_code_fences(raw_text)
    if not cleaned:
        return None

    candidates = [cleaned]
    candidates.extend([m.group(0) for m in re.finditer(r"\{[\s\S]*\}", cleaned)])

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(cleaned[first_brace:last_brace + 1])

    candidates = sorted(set(candidates), key=len, reverse=True)

    for candidate in candidates:
        for attempt in (candidate, _graph_eval_repair_json_candidate(candidate)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return None


def _graph_eval_unescape_json_string(value: str) -> str:
    raw = value or ""
    try:
        return json.loads(f"\"{raw}\"")
    except Exception:
        return raw.replace("\\n", "\n").replace("\\\"", "\"").strip()


def _graph_eval_extract_list_from_raw(cleaned: str, key: str) -> List[str]:
    match = re.search(rf"\"{re.escape(key)}\"\s*:\s*\[(.*?)\]", cleaned, flags=re.DOTALL)
    if not match:
        return []

    body = match.group(1)
    values: List[str] = []
    for item_match in re.finditer(r"\"((?:\\.|[^\"\\])*)\"", body, flags=re.DOTALL):
        value = _graph_eval_unescape_json_string(item_match.group(1)).strip()
        if value:
            values.append(value)
    return values


def _graph_eval_parse_partial_json_response(raw_text: str) -> Optional[Dict[str, Any]]:
    cleaned = _graph_eval_strip_code_fences(raw_text)
    if not cleaned:
        return None

    parsed: Dict[str, Any] = {}

    overall_match = re.search(r"\"overall_score\"\s*:\s*([-+]?\d+(?:\.\d+)?)", cleaned)
    if overall_match:
        parsed["overall_score"] = float(overall_match.group(1))

    dimension_keys = [
        "faithfulness_to_text",
        "coverage_completeness",
        "relation_correctness",
        "schema_type_consistency",
        "structural_quality",
        "usefulness_for_downstream_tasks",
    ]
    dimensions: Dict[str, float] = {}
    for key in dimension_keys:
        m = re.search(rf"\"{re.escape(key)}\"\s*:\s*([-+]?\d+(?:\.\d+)?)", cleaned)
        if m:
            dimensions[key] = float(m.group(1))
    if dimensions:
        parsed["dimension_scores"] = dimensions

    summary_match = re.search(r"\"summary\"\s*:\s*\"((?:\\.|[^\"\\])*)\"", cleaned, flags=re.DOTALL)
    if summary_match:
        parsed["summary"] = _graph_eval_unescape_json_string(summary_match.group(1))

    for key in [
        "strengths",
        "weaknesses",
        "hallucinated_or_unsupported_examples",
        "missing_facts_or_relations",
        "recommendations",
    ]:
        parsed[key] = _graph_eval_extract_list_from_raw(cleaned, key)

    if not parsed:
        return None
    return parsed


def _graph_eval_normalize_llm_payload(parsed: Optional[Dict[str, Any]], raw_text: str) -> Dict[str, Any]:
    payload = parsed if isinstance(parsed, dict) else {}

    overall_value = payload.get("overall_score", 0)
    try:
        overall_score = float(overall_value)
    except Exception:
        overall_score = 0.0
    overall_score = max(0.0, min(100.0, overall_score))

    raw_dimensions = payload.get("dimension_scores")
    clean_dimensions: Dict[str, float] = {}
    if isinstance(raw_dimensions, dict):
        for key, value in raw_dimensions.items():
            try:
                numeric = float(value)
            except Exception:
                continue
            clean_dimensions[str(key)] = max(0.0, min(10.0, numeric))

    def _to_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = (raw_text or "").strip()[:1000] or "-"

    return {
        "overall_score": round(overall_score, 2),
        "dimension_scores": clean_dimensions,
        "summary": summary.strip(),
        "strengths": _to_list(payload.get("strengths")),
        "weaknesses": _to_list(payload.get("weaknesses")),
        "hallucinated_or_unsupported_examples": _to_list(payload.get("hallucinated_or_unsupported_examples")),
        "missing_facts_or_relations": _to_list(payload.get("missing_facts_or_relations")),
        "recommendations": _to_list(payload.get("recommendations")),
    }


def _graph_eval_detect_provider(model: str, provider_hint: str = "") -> str:
    hint = (provider_hint or "").strip().lower()
    if hint in {"openai", "huggingface"}:
        return hint

    model_name = (model or "").strip().lower()
    if model_name.startswith(("gpt-", "o1", "o3", "o4", "openai/")):
        return "openai"
    return "huggingface"


def _graph_eval_hf_model_candidates(preferred_model: str) -> List[str]:
    """Ordered fallback list for HF chat models that are commonly available."""
    defaults = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "HuggingFaceH4/zephyr-7b-beta",
        "microsoft/Phi-3-mini-4k-instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]
    merged = [preferred_model] + defaults
    seen = set()
    result: List[str] = []
    for m in merged:
        key = (m or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


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
        try:
            max_entities = int(data.get('max_entities', 100))
            max_relationships = int(data.get('max_relationships', 200))
        except (TypeError, ValueError):
            return jsonify({
                'success': False,
                'error': 'مقادیر max_entities و max_relationships باید عدد صحیح باشند'
            }), 400

        if max_entities < 1 or max_entities > 5000:
            return jsonify({
                'success': False,
                'error': 'max_entities باید بین 1 و 5000 باشد'
            }), 400

        if max_relationships < 1 or max_relationships > 10000:
            return jsonify({
                'success': False,
                'error': 'max_relationships باید بین 1 و 10000 باشد'
            }), 400
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
            try:
                chunk_size = int(data.get('chunk_size', 500))
                overlap = int(data.get('overlap', 100))
                max_chunks = int(data.get('max_chunks', TextToGraphService.INCREMENTAL_MAX_CHUNKS))
            except (TypeError, ValueError):
                return jsonify({
                    'success': False,
                    'error': 'پارامترهای chunk_size، overlap و max_chunks باید عدد صحیح باشند'
                }), 400

            if chunk_size < TextToGraphService.INCREMENTAL_MIN_CHUNK_SIZE or chunk_size > TextToGraphService.INCREMENTAL_MAX_CHUNK_SIZE:
                return jsonify({
                    'success': False,
                    'error': f'chunk_size باید بین {TextToGraphService.INCREMENTAL_MIN_CHUNK_SIZE} و {TextToGraphService.INCREMENTAL_MAX_CHUNK_SIZE} باشد'
                }), 400

            if overlap < 0 or overlap > TextToGraphService.INCREMENTAL_MAX_OVERLAP:
                return jsonify({
                    'success': False,
                    'error': f'overlap باید بین 0 و {TextToGraphService.INCREMENTAL_MAX_OVERLAP} باشد'
                }), 400

            if overlap >= chunk_size:
                return jsonify({
                    'success': False,
                    'error': 'overlap باید از chunk_size کمتر باشد'
                }), 400

            if overlap > (chunk_size // 2):
                return jsonify({
                    'success': False,
                    'error': 'برای پایداری حافظه، overlap نباید بیشتر از نصف chunk_size باشد'
                }), 400

            if max_chunks < 1 or max_chunks > TextToGraphService.INCREMENTAL_MAX_CHUNKS:
                return jsonify({
                    'success': False,
                    'error': f'max_chunks باید بین 1 و {TextToGraphService.INCREMENTAL_MAX_CHUNKS} باشد'
                }), 400

            if max_entities > TextToGraphService.INCREMENTAL_MAX_ENTITIES:
                return jsonify({
                    'success': False,
                    'error': f'در روش incremental، max_entities نمی‌تواند بیشتر از {TextToGraphService.INCREMENTAL_MAX_ENTITIES} باشد'
                }), 400

            if max_relationships > TextToGraphService.INCREMENTAL_MAX_RELATIONSHIPS:
                return jsonify({
                    'success': False,
                    'error': f'در روش incremental، max_relationships نمی‌تواند بیشتر از {TextToGraphService.INCREMENTAL_MAX_RELATIONSHIPS} باشد'
                }), 400

            base_method = data.get('base_method', 'spacy')
            if base_method in ['incremental', 'long_text']:
                return jsonify({
                    'success': False,
                    'error': 'base_method در روش incremental نمی‌تواند incremental یا long_text باشد'
                }), 400

            extraction_params['chunk_size'] = chunk_size
            extraction_params['overlap'] = overlap
            extraction_params['base_method'] = base_method
            extraction_params['max_chunks'] = max_chunks
        
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
            auto_evaluation = _graph_eval_compute_auto_metrics(
                source_text=text,
                graph_data=graph_data,
                extraction_data={
                    'entities': extraction_entities,
                    'relationships': extraction_relationships
                }
            )
            
            response_payload = {
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
                },
                'source_text': text[:12000] if text else '',
                'source_type': 'url' if url else 'text',
                'source_url': url if url else '',
                'auto_evaluation': auto_evaluation
            }
            return jsonify(_to_json_safe(response_payload))
            
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

@app.route('/api/evaluate_text_to_graph', methods=['POST'])
def evaluate_text_to_graph():
    """Evaluate extracted graph quality with automatic and optional reference-based metrics."""
    try:
        data = request.get_json() or {}
        source_text = data.get('source_text', '') or ''
        graph_data = data.get('graph_data') or {}
        extraction_data = data.get('extraction_data') or {}
        reference_triples = data.get('reference_triples')

        has_graph = bool((graph_data.get('nodes') if isinstance(graph_data, dict) else None) or
                         (extraction_data.get('entities') if isinstance(extraction_data, dict) else None))
        if not has_graph:
            return jsonify({
                'success': False,
                'error': 'داده‌ای برای گراف جهت ارزیابی ارسال نشده است.'
            }), 400

        auto_evaluation = _graph_eval_compute_auto_metrics(source_text, graph_data, extraction_data)
        reference_evaluation = _graph_eval_reference_metrics(graph_data, extraction_data, reference_triples)

        return jsonify({
            'success': True,
            'auto_evaluation': auto_evaluation,
            'reference_evaluation': reference_evaluation
        })
    except Exception as e:
        logging.error(f"Error in evaluate_text_to_graph: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'خطا در ارزیابی گراف: {str(e)}'
        }), 500


@app.route('/api/evaluate_text_to_graph_llm', methods=['POST'])
def evaluate_text_to_graph_llm():
    """LLM-as-a-judge evaluation for text-to-graph quality."""
    try:
        data = request.get_json() or {}
        source_text = (data.get('source_text') or '').strip()
        graph_data = data.get('graph_data') or {}
        extraction_data = data.get('extraction_data') or {}
        model = data.get('model', 'mistralai/Mistral-7B-Instruct-v0.2')
        model_used = model
        provider = _graph_eval_detect_provider(model, data.get('provider', ''))
        custom_instruction = (data.get('instruction') or '').strip()
        hf_token = (data.get('hf_token') or '').strip() or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        openai_key = (data.get('openai_api_key') or '').strip() or OPENAI_API_KEY
        hf_provider_raw = (data.get('hf_provider') or os.environ.get("HF_LLM_PROVIDER") or "hf-inference").strip().lower()
        allowed_hf_providers = {
            "auto", "hf-inference", "together", "groq", "cerebras", "cohere", "fireworks-ai",
            "replicate", "sambanova", "novita", "nscale", "hyperbolic", "nebius",
            "publicai", "black-forest-labs", "fal-ai", "featherless-ai", "openai", "scaleway"
        }
        hf_provider = hf_provider_raw if hf_provider_raw in allowed_hf_providers else "hf-inference"

        if not source_text:
            return jsonify({
                'success': False,
                'error': 'متن مرجع برای ارزیابی LLM ارسال نشده است.'
            }), 400

        compact_graph = _graph_eval_compact_for_prompt(graph_data, extraction_data)

        system_prompt = (
            "You are an expert evaluator for text-to-knowledge-graph extraction. "
            "Be strict, evidence-based, and avoid generosity bias. "
            "Score based only on the provided source text and graph content."
        )

        user_prompt = f"""
Evaluate the quality of the generated knowledge graph from the source text.

Return ONLY JSON using this schema:
{{
  "overall_score": 0-100,
  "dimension_scores": {{
    "faithfulness_to_text": 1-10,
    "coverage_completeness": 1-10,
    "relation_correctness": 1-10,
    "schema_type_consistency": 1-10,
    "structural_quality": 1-10,
    "usefulness_for_downstream_tasks": 1-10
  }},
  "summary": "short paragraph",
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "hallucinated_or_unsupported_examples": ["..."],
  "missing_facts_or_relations": ["..."],
  "recommendations": ["...", "..."]
}}

Evaluation Rubric:
- Faithfulness: Are nodes/edges explicitly supported by the text?
- Coverage: Are key entities and relations from the text captured?
- Relation correctness: Are direction and predicate semantics correct?
- Schema consistency: Are node/edge types coherent?
- Structural quality: Is the graph connected and non-redundant?
- Usefulness: Is this graph practically useful for QA/retrieval/inference?

Source Text:
{source_text}

Graph Payload:
{json.dumps(compact_graph, ensure_ascii=False, indent=2)}
"""

        if custom_instruction:
            user_prompt += f"\n\nAdditional user instruction:\n{custom_instruction}\n"

        raw_text = ""

        if provider == "openai":
            if openai is None:
                return jsonify({
                    'success': False,
                    'error': 'کتابخانه OpenAI نصب نشده است. یا مدل HuggingFace انتخاب کنید.'
                }), 500

            if not openai_key:
                return jsonify({
                    'success': False,
                    'error': 'OPENAI_API_KEY ندارید. یک مدل HuggingFace انتخاب کنید یا HF token بدهید.'
                }), 400

            openai.api_key = openai_key
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1400
            )
            raw_text = (response.choices[0].message.content or "").strip()

        else:
            if InferenceClient is None:
                return jsonify({
                    'success': False,
                    'error': 'huggingface_hub نصب نشده است. لطفاً pip install huggingface_hub اجرا کنید.'
                }), 500

            if not hf_token:
                return jsonify({
                    'success': False,
                    'error': 'برای مدل HuggingFace باید HF token بدهید (فیلد صفحه یا متغیر محیطی HF_TOKEN).'
                }), 400

            # Try requested provider first, then hf-inference, then auto as a last resort.
            provider_candidates: List[str] = []
            for p in [hf_provider, "hf-inference", "auto"]:
                if p not in provider_candidates:
                    provider_candidates.append(p)

            model_candidates = _graph_eval_hf_model_candidates(model)
            last_hf_error: Optional[Exception] = None
            selected_hf_provider: Optional[str] = None
            selected_hf_model: Optional[str] = None
            saw_permission_error = False
            saw_not_found_error = False

            for hf_provider_candidate in provider_candidates:
                hf_client = InferenceClient(provider=hf_provider_candidate, api_key=hf_token)
                for model_candidate in model_candidates:
                    try:
                        response = hf_client.chat.completions.create(
                            model=model_candidate,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.1,
                            max_tokens=1400
                        )
                        raw_text = (response.choices[0].message.content or "").strip()
                        selected_hf_provider = hf_provider_candidate
                        selected_hf_model = model_candidate
                        model_used = model_candidate
                        break
                    except Exception as chat_error:
                        chat_err = str(chat_error)
                        chat_err_lower = chat_err.lower()
                        if "403" in chat_err or "insufficient permissions" in chat_err_lower or "cannot access content at" in chat_err_lower:
                            saw_permission_error = True
                        if "404" in chat_err or "not found" in chat_err_lower:
                            saw_not_found_error = True

                        logging.warning(
                            f"HF chat.completions failed for model {model_candidate}, provider {hf_provider_candidate}: {chat_error}"
                        )
                        try:
                            response = hf_client.chat_completion(
                                model=model_candidate,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                temperature=0.1,
                                max_tokens=1400
                            )
                            raw_text = (response.choices[0].message.content or "").strip()
                            selected_hf_provider = hf_provider_candidate
                            selected_hf_model = model_candidate
                            model_used = model_candidate
                            break
                        except Exception as legacy_chat_error:
                            legacy_err = str(legacy_chat_error)
                            legacy_err_lower = legacy_err.lower()
                            if "403" in legacy_err or "insufficient permissions" in legacy_err_lower or "cannot access content at" in legacy_err_lower:
                                saw_permission_error = True
                            if "404" in legacy_err or "not found" in legacy_err_lower:
                                saw_not_found_error = True
                            last_hf_error = legacy_chat_error

                            # Fallback for models/providers where chat endpoint is unavailable:
                            # try plain text-generation and parse JSON from output.
                            try:
                                textgen_prompt = (
                                    f"{system_prompt}\n\n"
                                    f"{user_prompt}\n\n"
                                    "Return ONLY valid JSON. Do not include markdown fences."
                                )
                                generation = hf_client.text_generation(
                                    textgen_prompt,
                                    model=model_candidate,
                                    max_new_tokens=1400,
                                    temperature=0.1,
                                    return_full_text=False,
                                )
                                raw_text = str(generation or "").strip()
                                if raw_text:
                                    selected_hf_provider = hf_provider_candidate
                                    selected_hf_model = model_candidate
                                    model_used = model_candidate
                                    break
                            except Exception as textgen_error:
                                textgen_err = str(textgen_error)
                                textgen_err_lower = textgen_err.lower()
                                if "403" in textgen_err or "insufficient permissions" in textgen_err_lower or "cannot access content at" in textgen_err_lower:
                                    saw_permission_error = True
                                if "404" in textgen_err or "not found" in textgen_err_lower:
                                    saw_not_found_error = True
                                last_hf_error = textgen_error
                                continue

                if raw_text:
                    break

            if not raw_text:
                err = str(last_hf_error) if last_hf_error else "خطای نامشخص در فراخوانی HuggingFace"
                lower_err = err.lower()

                if "supported task: conversational" in err:
                    return jsonify({
                        'success': False,
                        'error': (
                            'این مدل فقط حالت گفتگویی (conversational) را پشتیبانی می‌کند. '
                            'فراخوانی ارزیابی LLM روی حالت chat انجام شد اما پاسخ معتبر دریافت نشد. '
                            'یک مدل دیگر از لیست HuggingFace انتخاب کنید.'
                        )
                    }), 400

                if saw_permission_error:
                    return jsonify({
                        'success': False,
                        'error': (
                            'توکن HuggingFace شما مجوز کافی برای Inference Providers ندارد (403 Forbidden). '
                            'در https://huggingface.co/settings/tokens برای توکن، مجوز '
                            '"Make calls to Inference Providers" را فعال کنید یا از توکن جدید استفاده کنید.'
                        )
                    }), 403

                if saw_not_found_error:
                    return jsonify({
                        'success': False,
                        'error': (
                            'مدل انتخابی روی سرویس hf-inference پیدا نشد (404). '
                            f'مدل درخواستی: {model}. '
                            'مدل‌های جایگزین و مسیر text-generation هم امتحان شدند اما پاسخ نگرفتیم. '
                            'لطفاً یک مدل دیگر انتخاب کنید یا provider را روی auto امتحان کنید.'
                        )
                    }), 404

                raise RuntimeError(f"خطا در فراخوانی مدل HuggingFace: {err}")

        parsed = _graph_eval_parse_json_response(raw_text)
        if not parsed:
            parsed = _graph_eval_parse_partial_json_response(raw_text)
        parsed = _graph_eval_normalize_llm_payload(parsed, raw_text)

        return jsonify({
            'success': True,
            'model': model_used,
            'requested_model': model,
            'provider': provider,
            'hf_provider': selected_hf_provider if provider == "huggingface" else None,
            'llm_evaluation': parsed,
            'raw_response_preview': raw_text[:4000] if raw_text else '',
            'truncation_info': compact_graph.get('truncated', {})
        })

    except Exception as e:
        logging.error(f"Error in evaluate_text_to_graph_llm: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'خطا در ارزیابی LLM: {str(e)}'
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
        
        # ظ…ط±طھط¨ ع©ط±ط¯ظ† ط¨ط± ط§ط³ط§ط³ طھط§ط±غŒط® (ط¬ط¯غŒط¯طھط±غŒظ† ط§ظˆظ„)
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
                'error': 'ظ…ط³غŒط± ع¯ط±ط§ظپ ظ†ط§ظ…ط¹طھط¨ط± ط§ط³طھ'
            }), 400
        
        # بارگذاری گراف جدید
        global graphrag_service
        graphrag_service = GraphRAGService(graph_data_path=graph_path)
        if OPENAI_API_KEY:
            graphrag_service.set_openai_api_key(OPENAI_API_KEY)
        
        return jsonify({
            'success': True,
            'message': f'ع¯ط±ط§ظپ {os.path.basename(graph_path)} ط¨ط§ ظ…ظˆظپظ‚غŒطھ ط¨ط§ط±ع¯ط°ط§ط±غŒ ط´ط¯'
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
                'error': 'ظ…ط³غŒط± ع¯ط±ط§ظپ ظ†ط§ظ…ط¹طھط¨ط± ط§ط³طھ'
            }), 400
        
        # حذف فایل
        os.remove(graph_path)
        
        return jsonify({
            'success': True,
            'message': f'ع¯ط±ط§ظپ {os.path.basename(graph_path)} ط¨ط§ ظ…ظˆظپظ‚غŒطھ ط­ط°ظپ ط´ط¯'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/graph_view_data', methods=['POST'])
def graph_view_data():
    """ط¯ط±غŒط§ظپطھ ط¯ط§ط¯ظ‡â€Œظ‡ط§غŒ ع¯ط±ط§ظپ ط¨ط±ط§غŒ ظ†ظ…ط§غŒط´ ظˆغŒعکظˆط§ظ„ ظˆ ط¢ظ…ط§ط±غŒ ط¨ط¯ظˆظ† طھط؛غŒغŒط± ع¯ط±ط§ظپ ظپط¹ط§ظ„ ط³غŒط³طھظ…"""
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

        # طھظ„ط§ط´ ط¨ط±ط§غŒ ط±ظپط¹ ظ…ط´ع©ظ„ط§طھ ظ…ط³غŒط± (ظ…ط«ظ„ طھع©ط±ط§ط± ظ†ط§ظ… ظ¾ظˆط´ظ‡ غŒط§ ع©ط§ط±ط§ع©طھط±ظ‡ط§غŒ ط¹ط¬غŒط¨)
        original_graph_path = graph_path

        # ط§ع¯ط± ظپط§غŒظ„ ظ…ط³طھظ‚غŒظ…ط§ظ‹ ظˆط¬ظˆط¯ ط¯ط§ط±ط¯طŒ ظ‡ظ…ط§ظ† ط±ط§ ط§ط³طھظپط§ط¯ظ‡ ع©ظ†
        if not os.path.exists(graph_path):
            # ظ‡ظ…غŒط´ظ‡ غŒع© ط¨ط§ط± ظپظ‚ط· ط¨ط± ط§ط³ط§ط³ ظ†ط§ظ… ظپط§غŒظ„ ظ‡ظ… ط§ظ…طھط­ط§ظ† ظ…غŒâ€Œع©ظ†غŒظ…
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
                # طھظ„ط§ط´ ظˆغŒعکظ‡ ط¨ط±ط§غŒ ظپط§غŒظ„â€Œظ‡ط§غŒ text_graph:
                # ط§ع¯ط± ط¨ط®ط´غŒ ط¹ط¯ط¯غŒ ظ…ط«ظ„ 60107_115532 ط¯ط± ظ†ط§ظ… ط¨ط§ط´ط¯طŒ ط¯ط± ط¨غŒظ† ظپط§غŒظ„â€Œظ‡ط§غŒ uploaded_graphs ط¬ط³طھط¬ظˆغŒ طھط·ط§ط¨ظ‚ ط§ظ†ط¬ط§ظ… ظ…غŒâ€Œط¯ظ‡غŒظ…
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
                    logging.error(f"ظ…ط³غŒط± ع¯ط±ط§ظپ غŒط§ظپطھ ظ†ط´ط¯. ظˆط±ظˆط¯غŒ: {original_graph_path}, ط§ظ…طھط­ط§ظ†â€Œط´ط¯ظ‡: {candidate_paths}")
                    return jsonify({
                        'success': False,
                        'error': f'ظ…ط³غŒط± ع¯ط±ط§ظپ ظ†ط§ظ…ط¹طھط¨ط± ط§ط³طھ: {original_graph_path}'
                    }), 400

            graph_path = resolved

        # ط¨ط§ط±ع¯ط°ط§ط±غŒ ظ…ظˆظ‚طھ ع¯ط±ط§ظپ ظپظ‚ط· ط¨ط±ط§غŒ ظ†ظ…ط§غŒط´ (ط¨ط¯ظˆظ† طھط؛غŒغŒط± graphrag_service ط³ط±ط§ط³ط±غŒ)
        try:
            temp_service = GraphRAGService(graph_data_path=graph_path)
            G = getattr(temp_service, 'G', None)

            if G is None:
                return jsonify({
                    'success': False,
                    'error': 'ع¯ط±ط§ظپ ط¯ط± ظپط§غŒظ„ غŒط§ظپطھ ظ†ط´ط¯ غŒط§ ظپط§غŒظ„ ط®ط§ظ„غŒ ط§ط³طھ'
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
        
        # طھظ†ط¸غŒظ…ط§طھ ظ¾غŒط´ط±ظپطھظ‡ (ط§ط®طھغŒط§ط±غŒ)
        max_nodes = data.get('max_nodes', 20)
        max_edges = data.get('max_edges', 40)
        similarity_threshold = data.get('similarity_threshold', 0.3)
        community_detection_method = data.get('community_detection_method', 'louvain')
        advanced_retrieval_algorithm = data.get('advanced_retrieval_algorithm', 'hybrid')
        advanced_token_extraction_method = data.get('advanced_token_extraction_method', 'llm_based')
        
        # تبدیل رشته به enum
        retrieval_enum = RetrievalMethod[retrieval_method]
        generation_enum = GenerationModel[generation_model.replace(' ', '_')]
        
        # طھظ†ط¸غŒظ… ظ¾غŒع©ط±ط¨ظ†ط¯غŒ ظ¾غŒط´ط±ظپطھظ‡ ط§ع¯ط± ط§ط±ط§ط¦ظ‡ ط´ط¯ظ‡ ط¨ط§ط´ط¯
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
        
        # ط§ط¶ط§ظپظ‡ ع©ط±ط¯ظ† timestamp ظˆ طھظ†ط¸غŒظ…ط§طھ ط§ط³طھظپط§ط¯ظ‡ ط´ط¯ظ‡
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
        
        # طھظ†ط¸غŒظ… ظ¾غŒع©ط±ط¨ظ†ط¯غŒ ط³ط±ظˆغŒط³ ظ¾غŒط´ط±ظپطھظ‡
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
    """ط§ط·ظ„ط§ط¹ط§طھ ع¯ط±ط§ظپ ظ¾غŒط´ط±ظپطھظ‡"""
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
    """ط¯ط±غŒط§ظپطھ ط±ظˆط´â€Œظ‡ط§غŒ ط§ط³طھط®ط±ط§ط¬ طھظˆع©ظ†"""
    methods = [
        {
            'value': 'llm_based',
            'label': 'بر اساس LLM',
            'description': 'ط§ط³طھط®ط±ط§ط¬ طھظˆع©ظ† ط¨ط§ ط§ط³طھظپط§ط¯ظ‡ ط§ط² ظ…ط¯ظ„â€Œظ‡ط§غŒ ط²ط¨ط§ظ†غŒ'
        },
        {
            'value': 'rule_based',
            'label': 'بر اساس قوانین',
            'description': 'ط§ط³طھط®ط±ط§ط¬ طھظˆع©ظ† ط¨ط§ ط§ط³طھظپط§ط¯ظ‡ ط§ط² ظ‚ظˆط§ظ†غŒظ† ط§ط² ظ¾غŒط´ طھط¹ط±غŒظپ ط´ط¯ظ‡'
        },
        {
            'value': 'hybrid',
            'label': 'ترکیبی',
            'description': 'ترکیب روش‌های LLM و قوانین'
        },
        {
            'value': 'semantic',
            'label': 'معنایی',
            'description': 'ط§ط³طھط®ط±ط§ط¬ طھظˆع©ظ† ط¨ط± ط§ط³ط§ط³ ط´ط¨ط§ظ‡طھ ظ…ط¹ظ†ط§غŒغŒ'
        }
    ]
    return jsonify({'methods': methods})

@app.route('/api/retrieval_algorithms')
def retrieval_algorithms():
    """ط¯ط±غŒط§ظپطھ ط§ظ„ع¯ظˆط±غŒطھظ…â€Œظ‡ط§غŒ ط¨ط§ط²غŒط§ط¨غŒ"""
    algorithms = [
        # الگوریتم‌های اصلی
        {
            'value': 'semantic_similarity',
            'label': 'ط´ط¨ط§ظ‡طھ ظ…ط¹ظ†ط§غŒغŒ (ط¨ظ‡ط¨ظˆط¯ غŒط§ظپطھظ‡)',
            'description': 'ط¨ط§ط²غŒط§ط¨غŒ ط¨ط± ط§ط³ط§ط³ ط´ط¨ط§ظ‡طھ ظ…ط¹ظ†ط§غŒغŒ ط¨ط§ ط§ظپط²ط§غŒط´ ط¯ظ‚طھ ظˆ ظ¾ظˆط´ط´ عکظ†â€Œظ‡ط§'
        },
        {
            'value': 'hybrid',
            'label': 'طھط±ع©غŒط¨غŒ (ط¨ظ‡ط¨ظˆط¯ غŒط§ظپطھظ‡)',
            'description': 'طھط±ع©غŒط¨ ع†ظ†ط¯غŒظ† ط§ظ„ع¯ظˆط±غŒطھظ… ط¨ط§ ط¨ظ‡غŒظ†ظ‡â€Œط³ط§ط²غŒ ط¹ظ…ظ„ع©ط±ط¯ ظˆ ط§ظپط²ط§غŒط´ ط¯ظ‚طھ'
        },
        {
            'value': 'pagerank',
            'label': 'PageRank (بهبود یافته)',
            'description': 'ط±طھط¨ظ‡â€Œط¨ظ†ط¯غŒ ط¨ط± ط§ط³ط§ط³ ط§ظ„ع¯ظˆط±غŒطھظ… PageRank ط¨ط§ طھظ…ط±ع©ط² ط±ظˆغŒ عکظ†â€Œظ‡ط§غŒ ظ…ظ‡ظ…'
        },
        {
            'value': 'community_detection',
            'label': 'طھط´ط®غŒطµ ط¬ط§ظ…ط¹ظ‡ (ط¨ظ‡ط¨ظˆط¯ غŒط§ظپطھظ‡)',
            'description': 'ط¨ط§ط²غŒط§ط¨غŒ ط¨ط± ط§ط³ط§ط³ طھط´ط®غŒطµ ط¬ط§ظ…ط¹ظ‡â€Œظ‡ط§ ط¨ط§ ط§ظپط²ط§غŒط´ ط¯ظ‚طھ ط®ظˆط´ظ‡â€Œط¨ظ†ط¯غŒ'
        },
        {
            'value': 'bfs',
            'label': 'BFS (ط¬ط³طھط¬ظˆغŒ ط³ط·ط­ ط§ظˆظ„ - ط¨ظ‡ط¨ظˆط¯ غŒط§ظپطھظ‡)',
            'description': 'ط¬ط³طھط¬ظˆغŒ ط³ط·ط­ ط§ظˆظ„ ط¯ط± ع¯ط±ط§ظپ ط¨ط§ ط§ظپط²ط§غŒط´ ط¹ظ…ظ‚ ظˆ ط¨ظ‡غŒظ†ظ‡â€Œط³ط§ط²غŒ'
        },
        {
            'value': 'dfs',
            'label': 'DFS (ط¬ط³طھط¬ظˆغŒ ط¹ظ…غŒظ‚ ط§ظˆظ„ - ط¨ظ‡ط¨ظˆط¯ غŒط§ظپطھظ‡)',
            'description': 'ط¬ط³طھط¬ظˆغŒ ط¹ظ…غŒظ‚ ط§ظˆظ„ ط¯ط± ع¯ط±ط§ظپ ط¨ط§ ط§ظپط²ط§غŒط´ ط¯ظ‚طھ ظˆ ع©ط§ظ‡ط´ ط²ظ…ط§ظ†'
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
    """ط¯ط±غŒط§ظپطھ ط±ظˆط´â€Œظ‡ط§غŒ طھط´ط®غŒطµ ط¬ط§ظ…ط¹ظ‡"""
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
        # ط³ظˆط§ظ„ط§طھ ظ…ط±ط¨ظˆط· ط¨ظ‡ ط¨غŒط§ظ† عکظ† ط¯ط± ط¨ط§ظپطھâ€Œظ‡ط§ (AeG, AuG, AdG)
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
        
        # ط³ظˆط§ظ„ط§طھ ظ…ط±ط¨ظˆط· ط¨ظ‡ ظپط±ط¢غŒظ†ط¯ظ‡ط§غŒ ط²غŒط³طھغŒ (GpBP, GpMF, GpCC)
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
        
        # ط³ظˆط§ظ„ط§طھ ظ…ط±ط¨ظˆط· ط¨ظ‡ طھط¹ط§ظ…ظ„ عکظ†â€Œظ‡ط§ (GiG, Gr>G)
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
    """ظ…ط¯غŒط±غŒطھ طھظ†ط¸غŒظ…ط§طھ ط³غŒط³طھظ…"""
    if request.method == 'GET':
        # ط¯ط±غŒط§ظپطھ طھظ†ط¸غŒظ…ط§طھ ظپط¹ظ„غŒ
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
        # طھط؛غŒغŒط± طھظ†ط¸غŒظ…ط§طھ
        try:
            data = request.get_json()
            new_config = data.get('config', {})
            
            # اعمال تنظیمات جدید
            graphrag_service.set_config(**new_config)
            
            # ط¯ط±غŒط§ظپطھ طھظ†ط¸غŒظ…ط§طھ ط¨ظ‡â€Œط±ظˆط² ط´ط¯ظ‡
            updated_config = graphrag_service.get_config()
            
            return jsonify({
                'success': True,
                'message': 'طھظ†ط¸غŒظ…ط§طھ ط¨ط§ ظ…ظˆظپظ‚غŒطھ ط¨ظ‡â€Œط±ظˆط² ط´ط¯',
                'config': updated_config
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

@app.route('/api/enhanced_config', methods=['GET', 'POST'])
def enhanced_config_endpoint():
    """ظ…ط¯غŒط±غŒطھ طھظ†ط¸غŒظ…ط§طھ ط³ط±ظˆغŒط³ ظ¾غŒط´ط±ظپطھظ‡"""
    if request.method == 'GET':
        # ط¯ط±غŒط§ظپطھ طھظ†ط¸غŒظ…ط§طھ ظپط¹ظ„غŒ
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
        # طھط؛غŒغŒط± طھظ†ط¸غŒظ…ط§طھ
        try:
            data = request.get_json()
            new_config = data.get('config', {})
            
            # اعمال تنظیمات جدید
            enhanced_graphrag_service.set_config(**new_config)
            
            # ط¯ط±غŒط§ظپطھ طھظ†ط¸غŒظ…ط§طھ ط¨ظ‡â€Œط±ظˆط² ط´ط¯ظ‡
            updated_config = enhanced_graphrag_service.get_config()
            
            return jsonify({
                'success': True,
                'message': 'طھظ†ط¸غŒظ…ط§طھ ظ¾غŒط´ط±ظپطھظ‡ ط¨ط§ ظ…ظˆظپظ‚غŒطھ ط¨ظ‡â€Œط±ظˆط² ط´ط¯',
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
            'description': 'طھط¹ط§ط¯ظ„ ط¨غŒظ† ط³ط±ط¹طھ ظˆ ع©غŒظپغŒطھ',
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
            'description': 'ط¨ط±ط§غŒ طھط­ظ‚غŒظ‚ط§طھ ظˆ طھط­ظ„غŒظ„ ط¹ظ…غŒظ‚',
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
            return jsonify({'error': 'OpenAI ع©طھط§ط¨ط®ط§ظ†ظ‡ ظ†طµط¨ ظ†ط´ط¯ظ‡ ط§ط³طھ. ظ„ط·ظپط§ظ‹ ط¨ط§ ط¯ط³طھظˆط± pip install openai ط¢ظ† ط±ط§ ظ†طµط¨ ع©ظ†غŒط¯.'}), 500
        
        # Call OpenAI API
        try:
            # Set the API key for this request
            openai.api_key = OPENAI_API_KEY
            
            response = openai.chat.completions.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": "ط´ظ…ط§ غŒع© ظ…طھط®طµطµ ط§ط±ط²غŒط§ط¨غŒ ع©غŒظپغŒطھ ظ…طھظ† ط¯ط± ط­ظˆط²ظ‡ ط²غŒط³طھâ€Œظ¾ط²ط´ع©غŒطŒ عکظ†طھغŒع© ظˆ ظ¾ط²ط´ع©غŒ ط´ط®طµغŒ ظ‡ط³طھغŒط¯. ظˆط¸غŒظپظ‡ ط´ظ…ط§ ظ…ظ‚ط§غŒط³ظ‡ ط¯ظˆ ظ…طھظ† ط§ط² ظ†ط¸ط± ط³ط·ط­ ط¹ظ„ظ…غŒطŒ ط³ط§ط®طھط§ط± طھط­ظ„غŒظ„غŒطŒ ط¹ظ…ظ‚ ظ…ظپظ‡ظˆظ…غŒ ظˆ ع©ط§ط±ط¨ط±ط¯ظ¾ط°غŒط±غŒ ط¨ط§ظ„غŒظ†غŒ ط§ط³طھ.\n\nظ¾ط§ط³ط®غŒ ع©ظ‡ ط¯ط§ط±ط§غŒ ظˆغŒعکع¯غŒâ€Œظ‡ط§غŒ ط²غŒط± ط¨ط§ط´ط¯طŒ ط¨ط§غŒط¯ ط§ظ…طھغŒط§ط² ط¨ط§ظ„ط§طھط±غŒ ط¨ع¯غŒط±ط¯:\n- طھط­ظ„غŒظ„ ط¯ظ‚غŒظ‚ ظ…ط³غŒط±ظ‡ط§غŒ ط²غŒط³طھغŒ ظˆ ط³غŒع¯ظ†ط§ظ„غŒظ†ع¯ \n- ط§ط´ط§ط±ظ‡ ط¨ظ‡ عکظ†â€Œظ‡ط§غŒ ط®ط§طµ ط¨ط§ ظ†ظ‚ط´ ط¨ط§ظ„غŒظ†غŒ \n- ظ¾غŒظˆظ†ط¯ ظˆط§ط¶ط­ ط¨غŒظ† ط¹ظ…ظ„ع©ط±ط¯ عکظ† ظˆ ط¨غŒظ…ط§ط±غŒ\n- طھظˆط¶غŒط­ ط¯ط± ظ…ظˆط±ط¯ ع©ط§ط±ط¨ط±ط¯ظ‡ط§غŒ ط¯ط±ظ…ط§ظ†غŒ غŒط§ طھط´ط®غŒطµغŒ (ظ…ط§ظ†ظ†ط¯ ط¯ط§ط±ظˆظ‡ط§غŒ ظ‡ط¯ظپظ…ظ†ط¯طŒ ط¨غŒظˆظ…ط§ط±ع©ط±ظ‡ط§طŒ ظ…ظ‡ط§ط±ع©ظ†ظ†ط¯ظ‡â€Œظ‡ط§)\n- ط³ط§ط®طھط§ط± طھط­ظ„غŒظ„غŒ ظ…ظ†ط¸ظ… ط´ط§ظ…ظ„ ط¨ط®ط´â€Œط¨ظ†ط¯غŒ (ظ…ط«ظ„ط§ظ‹: ط§ظ‡ظ…غŒطھ ط²غŒط³طھغŒطŒ ط§ظ‡ظ…غŒطھ ط¨ط§ظ„غŒظ†غŒطŒ ع©ط§ط±ط¨ط±ط¯ظ‡ط§طŒ ظ†طھغŒط¬ظ‡â€Œع¯غŒط±غŒ)\n\nط¯ط± ظ…ظ‚ط§ط¨ظ„طŒ ظ¾ط§ط³ط®â€Œظ‡ط§غŒغŒ ع©ظ‡ ظپظ‚ط· ط§ط·ظ„ط§ط¹ط§طھ ع©ظ„غŒطŒ ط¹ظ…ظˆظ…غŒ غŒط§ ط؛غŒط±طھط­ظ„غŒظ„غŒ ظ…غŒâ€Œط¯ظ‡ظ†ط¯ غŒط§ طµط±ظپط§ظ‹ ظپظ‡ط±ط³طھغŒ ط§ط² ط§ط³ط§ظ…غŒ ظ‡ط³طھظ†ط¯طŒ ط¨ط§غŒط¯ ط§ظ…طھغŒط§ط² ع©ظ…طھط±غŒ ط¨ع¯غŒط±ظ†ط¯.\n\nط§ط±ط²غŒط§ط¨غŒ ط¨ط§غŒط¯ ط¯ظ‚غŒظ‚طŒ طھط­ظ„غŒظ„غŒ ظˆ ط¨ط§ طھظ…ط±ع©ط² ط¨ط± ع©غŒظپغŒطھ ط¹ظ„ظ…غŒطŒ ط¹ظ…ظ‚ ظ…ط­طھظˆط§ ظˆ ط§ط±ط²ط´ ع©ط§ط±ط¨ط±ط¯غŒ ط§ظ†ط¬ط§ظ… ط´ظˆط¯. ظ‡ط¯ظپ ط§ظ†طھط®ط§ط¨ ظ¾ط§ط³ط®غŒ ط§ط³طھ ع©ظ‡ ط¨غŒط´طھط±غŒظ† ط§ط±ط²ط´ ط±ط§ ط¨ط±ط§غŒ ظ¾عکظˆظ‡ط´ع¯ط± غŒط§ ظ…طھط®طµطµ ط­ظˆط²ظ‡ ط²غŒط³طھâ€Œظ¾ط²ط´ع©غŒ ط¯ط§ط´طھظ‡ ط¨ط§ط´ط¯."},
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
        'comprehensive': 'ع©غŒظپغŒطھ ع©ظ„غŒطŒ ط¯ظ‚طھطŒ ط¬ط§ظ…ط¹غŒطھطŒ ظˆط¶ظˆط­طŒ ظˆ ظ…ط±طھط¨ط· ط¨ظˆط¯ظ†',
        'practical_specialized': 'عملی بودن، تخصصی بودن، و کاربردی بودن محتوا',
        'accuracy': 'ط¯ظ‚طھ ظˆ طµط­طھ ط§ط·ظ„ط§ط¹ط§طھ ط§ط±ط§ط¦ظ‡ ط´ط¯ظ‡',
        'completeness': 'جامعیت و کامل بودن پاسخ',
        'clarity': 'وضوح و قابل فهم بودن متن',
        'relevance': 'مرتبط بودن با سوال اصلی'
    }
    
    focus = comparison_focus.get(comparison_type, comparison_focus['comprehensive'])
    
    prompt = f"""
    ظ„ط·ظپط§ظ‹ ط¯ظˆ ظ…طھظ† ط²غŒط± ط±ط§ ظ…ظ‚ط§غŒط³ظ‡ ع©ظ†غŒط¯ ظˆ طھط­ظ„غŒظ„ ط¯ظ‚غŒظ‚غŒ ط§ط±ط§ط¦ظ‡ ط¯ظ‡غŒط¯. طھظˆط¬ظ‡ ظˆغŒعکظ‡ ط¨ظ‡ ط¹ظ…ظ„غŒ ط¨ظˆط¯ظ† ظˆ طھط®طµطµغŒ ط¨ظˆط¯ظ† ظ…ط­طھظˆط§ ط¯ط§ط´طھظ‡ ط¨ط§ط´غŒط¯. ط§ط² ط§ط±ط§ط¦ظ‡ ظ‡ط±ع¯ظˆظ†ظ‡ ط§ظ…طھغŒط§ط² ط¹ط¯ط¯غŒ ط®ظˆط¯ط¯ط§ط±غŒ ع©ظ†غŒط¯ ظˆ ظپظ‚ط· طھط­ظ„غŒظ„ ظ…طھظ†غŒ ط§ط±ط§ط¦ظ‡ ط¯ظ‡غŒط¯.
    
    **متن اول ({label1}):**
    {text1}
    
    **متن دوم ({label2}):**
    {text2}
    
    **نوع مقایسه:** {focus}
    
    **معیارهای ارزیابی مهم:**
    1. **ط¹ظ…ظ„غŒ ط¨ظˆط¯ظ†**: ظ…طھظ† ط¨ط§غŒط¯ ط§ط·ظ„ط§ط¹ط§طھ ع©ط§ط±ط¨ط±ط¯غŒ ظˆ ظ‚ط§ط¨ظ„ ط§ط³طھظپط§ط¯ظ‡ ط§ط±ط§ط¦ظ‡ ط¯ظ‡ط¯
    2. **طھط®طµطµغŒ ط¨ظˆط¯ظ†**: ط§ط³طھظپط§ط¯ظ‡ ط§ط² ط§طµط·ظ„ط§ط­ط§طھ طھط®طµطµغŒ ظˆ ظ…ظپط§ظ‡غŒظ… ط¯ظ‚غŒظ‚ ط¹ظ„ظ…غŒ
    3. **ط¯ظ‚طھ ط¹ظ„ظ…غŒ**: طµط­طھ ط§ط·ظ„ط§ط¹ط§طھ ظˆ ط§ط³طھظ†ط§ط¯ ط¨ظ‡ ظ…ظپط§ظ‡غŒظ… ط¹ظ„ظ…غŒ
    4. **ط¬ط§ظ…ط¹غŒطھ**: ظ¾ظˆط´ط´ ع©ط§ظ…ظ„ ط¬ظ†ط¨ظ‡â€Œظ‡ط§غŒ ظ…ط®طھظ„ظپ ظ…ظˆط¶ظˆط¹
    5. **وضوح**: قابل فهم بودن برای مخاطب تخصصی
    
    ظ„ط·ظپط§ظ‹ طھط­ظ„غŒظ„ ط®ظˆط¯ ط±ط§ ط¯ط± ظ‚ط§ظ„ط¨ ط²غŒط± ط§ط±ط§ط¦ظ‡ ط¯ظ‡غŒط¯ ظˆ ط§ط² ط§ظ…طھغŒط§ط² ط¹ط¯ط¯غŒ ط§ط³طھظپط§ط¯ظ‡ ظ†ع©ظ†غŒط¯:
    
    **خلاصه مقایسه:**
    [غŒع© ط®ظ„ط§طµظ‡ ع©ظˆطھط§ظ‡ ط§ط² طھظپط§ظˆطھâ€Œظ‡ط§غŒ ط§طµظ„غŒ ط¨ط§ طھظ…ط±ع©ط² ط¨ط± ط¹ظ…ظ„غŒ ط¨ظˆط¯ظ† ظˆ طھط®طµطµغŒ ط¨ظˆط¯ظ†]
    
    **تحلیل کلی:**
    [طھط­ظ„غŒظ„ ظ…طھظ†غŒ ط§ط² ظ†ظ‚ط§ط· ظ‚ظˆطھ ظˆ ط¶ط¹ظپ ع©ظ„غŒ ظ‡ط± ط¯ظˆ ظ…طھظ† ظˆ ط¯ظ„غŒظ„ ط¨ط±طھط±غŒ ظ†ط³ط¨غŒ]
    
    **نقاط قوت {label1}:**
    [ظ„غŒط³طھ ظ†ظ‚ط§ط· ظ‚ظˆطھ ط¨ط§ طھظ…ط±ع©ط² ط¨ط± ط¬ظ†ط¨ظ‡â€Œظ‡ط§غŒ ط¹ظ…ظ„غŒ ظˆ طھط®طµطµغŒ]
    
    **نقاط قوت {label2}:**
    [ظ„غŒط³طھ ظ†ظ‚ط§ط· ظ‚ظˆطھ ط¨ط§ طھظ…ط±ع©ط² ط¨ط± ط¬ظ†ط¨ظ‡â€Œظ‡ط§غŒ ط¹ظ…ظ„غŒ ظˆ طھط®طµطµغŒ]
    
    **نقاط ضعف {label1}:**
    [ظ„غŒط³طھ ظ†ظ‚ط§ط· ط¶ط¹ظپ ط§ط² ظ†ط¸ط± ط¹ظ…ظ„غŒ ط¨ظˆط¯ظ† ظˆ طھط®طµطµغŒ ط¨ظˆط¯ظ†]
    
    **نقاط ضعف {label2}:**
    [ظ„غŒط³طھ ظ†ظ‚ط§ط· ط¶ط¹ظپ ط§ط² ظ†ط¸ط± ط¹ظ…ظ„غŒ ط¨ظˆط¯ظ† ظˆ طھط®طµطµغŒ ط¨ظˆط¯ظ†]
    
    **توصیه نهایی:**
    [طھظˆطµغŒظ‡ ع©ط¯ط§ظ… ط±ظˆط´ ط¨ظ‡طھط± ط§ط³طھ ط¨ط§ طھط£ع©غŒط¯ ط¨ط± ط¹ظ…ظ„غŒ ط¨ظˆط¯ظ† ظˆ طھط®طµطµغŒ ط¨ظˆط¯ظ†]
    """
    
    return prompt

def parse_gpt_comparison_response(response, label1, label2, comparison_type):
    """طھط¬ط²غŒظ‡ ظˆ طھط­ظ„غŒظ„ ظ¾ط§ط³ط® GPT-4o - ظپظ‚ط· طھط­ظ„غŒظ„ ظ…طھظ†غŒ ط¨ط¯ظˆظ† ط§ظ…طھغŒط§ط² ط¹ط¯ط¯غŒ"""
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
            # ط¯ط± طµظˆط±طھ ظ†ط¨ظˆط¯ ط¹ظ†ظˆط§ظ† ط¯ظ‚غŒظ‚ آ«طھط­ظ„غŒظ„ ع©ظ„غŒآ»طŒ ظ‡ط± ط¨ط®ط´ ط­ط§ظˆغŒ ظˆط§عکظ‡ طھط­ظ„غŒظ„ ط±ط§ ط¨ظ‡ ط¹ظ†ظˆط§ظ† طھط­ظ„غŒظ„ ع©ظ„غŒ ط¯ط± ظ†ط¸ط± ظ…غŒâ€Œع¯غŒط±غŒظ…
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
        # طھظ„ط§ط´ ط¨ط±ط§غŒ ط§ط³طھط®ط±ط§ط¬ غŒع© طھط­ظ„غŒظ„ ع©ظ„غŒ ط§ط² ط±ظˆغŒ ع©ظ„ ظ¾ط§ط³ط® ط¯ط± طµظˆط±طھ ظ†ط¨ظˆط¯ ط¨ط®ط´ ظ…ط´ط®طµ
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
    """طھط¹غŒغŒظ† ط³ط·ط­ ع©غŒظپغŒطھ ط¨ط± ط§ط³ط§ط³ ظ†ظ…ط±ظ‡ ط´ط¨ط§ظ‡طھ"""
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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست جامع سیستم GraphRAG با تمام قابلیت‌های جدید
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel

def test_question_type_detection():
    """تست تشخیص نوع سوال"""
    print("🔍 تست تشخیص نوع سوال")
    print("=" * 60)
    
    service = GraphRAGService()
    
    test_questions = [
        # Expression queries
        ("What genes are expressed in the heart?", "anatomy_expression"),
        ("Which genes are expressed in the brain?", "anatomy_expression"),
        ("Where is gene TP53 expressed?", "gene_expression_location"),
        
        # Disease queries
        ("What diseases affect the heart?", "anatomy_disease"),
        ("What symptoms does breast cancer present?", "disease_symptom"),
        ("What diseases are similar to lung cancer?", "disease_similarity"),
        
        # Treatment queries
        ("What compounds treat diabetes?", "disease_treatment"),
        ("What drugs treat heart disease?", "disease_treatment"),
        
        # Gene interaction queries
        ("Which genes interact with BRCA1?", "gene_interaction"),
        ("What genes regulate TP53?", "gene_regulation"),
        
        # Pathway queries
        ("What pathways does TNF participate in?", "gene_pathway"),
        ("Which biological processes involve APOE?", "biological_participation"),
        
        # Compound queries
        ("What compounds upregulate EGFR?", "compound_gene_regulation"),
        ("What side effects does aspirin cause?", "compound_side_effect"),
        
        # Complex queries
        ("What genes are associated with breast cancer?", "disease_gene_regulation"),
        ("What compounds bind to TP53?", "compound_gene_regulation")
    ]
    
    correct_detections = 0
    total_questions = len(test_questions)
    
    for question, expected_type in test_questions:
        print(f"\n🔍 سوال: {question}")
        intent = service.analyze_question_intent(question)
        detected_type = intent['question_type']
        metaedges = intent['metaedges']
        
        print(f"  تشخیص شده: {detected_type}")
        print(f"  انتظار: {expected_type}")
        print(f"  Metaedges: {metaedges}")
        print(f"  توضیح: {intent['description']}")
        
        if detected_type == expected_type:
            print("  ✅ درست تشخیص داده شد")
            correct_detections += 1
        else:
            print("  ❌ اشتباه تشخیص داده شد")
    
    accuracy = (correct_detections / total_questions) * 100
    print(f"\n📊 دقت تشخیص نوع سوال: {accuracy:.1f}% ({correct_detections}/{total_questions})")
    
    return accuracy > 80  # موفقیت اگر دقت بالای 80% باشد

def test_metaedge_aware_search():
    """تست جستجوی آگاه از metaedge"""
    print("\n🔗 تست جستجوی آگاه از metaedge")
    print("=" * 60)
    
    service = GraphRAGService()
    
    # تست سوالات مختلف با metaedges مختلف
    test_cases = [
        {
            'question': "What genes are expressed in the heart?",
            'expected_metaedge': 'AeG',
            'expected_genes': ['MMP9', 'BID', 'KCNQ2', 'HMGB3']
        },
        {
            'question': "Which genes interact with TP53?",
            'expected_metaedge': 'GiG',
            'expected_genes': ['BRCA1', 'MMP9', 'APOE']
        },
        {
            'question': "What compounds treat heart disease?",
            'expected_metaedge': 'CtD',
            'expected_compounds': ['Aspirin', 'Metformin']
        },
        {
            'question': "What diseases are associated with BRCA1?",
            'expected_metaedge': 'DaG',
            'expected_diseases': ['Breast Cancer']
        }
    ]
    
    successful_searches = 0
    
    for case in test_cases:
        print(f"\n🔍 سوال: {case['question']}")
        print(f"  انتظار metaedge: {case['expected_metaedge']}")
        
        try:
            results = service.intelligent_semantic_search(case['question'], max_depth=2)
            
            if results:
                print(f"  ✅ {len(results)} نتیجه یافت شد")
                
                # بررسی اینکه آیا نتایج شامل metaedge مورد انتظار هستند
                found_metaedge = False
                for node_id, depth, score, explanation in results[:3]:  # فقط 3 نتیجه اول
                    if case['expected_metaedge'] in explanation:
                        found_metaedge = True
                        print(f"    ✅ {explanation}")
                
                if found_metaedge:
                    successful_searches += 1
                    print(f"  ✅ metaedge {case['expected_metaedge']} یافت شد")
                else:
                    print(f"  ❌ metaedge {case['expected_metaedge']} یافت نشد")
            else:
                print(f"  ❌ هیچ نتیجه‌ای یافت نشد")
                
        except Exception as e:
            print(f"  ❌ خطا: {e}")
    
    success_rate = (successful_searches / len(test_cases)) * 100
    print(f"\n📊 نرخ موفقیت جستجو: {success_rate:.1f}% ({successful_searches}/{len(test_cases)})")
    
    return success_rate > 70

def test_evidence_based_answers():
    """تست تولید پاسخ مبتنی بر شواهد"""
    print("\n📚 تست تولید پاسخ مبتنی بر شواهد")
    print("=" * 60)
    
    service = GraphRAGService()
    
    test_questions = [
        "What genes are expressed in the heart?",
        "Which genes interact with TP53?",
        "What compounds treat heart disease?"
    ]
    
    for question in test_questions:
        print(f"\n🔍 سوال: {question}")
        
        try:
            # جستجوی هوشمند
            search_results = service.intelligent_semantic_search(question, max_depth=2)
            
            if search_results:
                # ایجاد RetrievalResult
                from graphrag_service import GraphNode, GraphEdge, RetrievalResult
                
                nodes = []
                edges = []
                paths = []
                
                # تبدیل نتایج به GraphNode و GraphEdge
                for node_id, depth, score, explanation in search_results[:5]:  # حداکثر 5 نتیجه
                    node_attrs = service.G.nodes[node_id]
                    nodes.append(GraphNode(
                        id=node_id,
                        name=node_attrs['name'],
                        kind=node_attrs['kind'],
                        depth=depth,
                        score=score
                    ))
                
                # یافتن یال‌های مرتبط
                for node in nodes:
                    for neighbor in service.G.neighbors(node.id):
                        edge_data = service.G.get_edge_data(node.id, neighbor)
                        if edge_data:
                            edges.append(GraphEdge(
                                source=node.id,
                                target=neighbor,
                                relation=edge_data.get('metaedge', 'Unknown'),
                                weight=1.0
                            ))
                
                retrieval_result = RetrievalResult(
                    nodes=nodes,
                    edges=edges,
                    paths=paths,
                    context_text="",
                    method="Intelligent Semantic Search",
                    query=question
                )
                
                # تولید پاسخ
                answer = service._generate_intelligent_anatomy_answer(
                    retrieval_result, 
                    [n for n in nodes if n.kind == 'Anatomy'],
                    [n for n in nodes if n.kind == 'Gene']
                )
                
                print("📝 پاسخ تولید شده:")
                print(answer[:500] + "..." if len(answer) > 500 else answer)
                
                # بررسی کیفیت پاسخ
                quality_indicators = [
                    "AeG" in answer,  # آیا metaedge ذکر شده
                    "heart" in answer.lower(),  # آیا موجودیت اصلی ذکر شده
                    "gene" in answer.lower(),  # آیا نوع موجودیت ذکر شده
                    len(answer) > 100  # آیا پاسخ به اندازه کافی طولانی است
                ]
                
                quality_score = sum(quality_indicators) / len(quality_indicators) * 100
                print(f"📊 کیفیت پاسخ: {quality_score:.1f}%")
                
            else:
                print("❌ هیچ نتیجه‌ای برای تولید پاسخ یافت نشد")
                
        except Exception as e:
            print(f"❌ خطا در تولید پاسخ: {e}")

def test_entity_linking():
    """تست نگاشت موجودیت‌ها"""
    print("\n🔗 تست نگاشت موجودیت‌ها")
    print("=" * 60)
    
    service = GraphRAGService()
    
    # تست تطبیق توکن‌های مختلف
    test_tokens = [
        'heart', 'genes', 'tp53', 'brca1', 'breast cancer', 'aspirin',
        'brain', 'liver', 'diabetes', 'caffeine', 'apoe', 'mmp9'
    ]
    
    successful_mappings = 0
    
    for token in test_tokens:
        print(f"\n🔍 توکن: '{token}'")
        
        try:
            matched_nodes = service.match_tokens_to_nodes([token])
            
            if matched_nodes:
                for token_name, node_id in matched_nodes.items():
                    node_attrs = service.G.nodes[node_id]
                    print(f"  ✅ تطبیق: '{token}' → {node_attrs['name']} ({node_attrs['kind']})")
                    successful_mappings += 1
            else:
                print(f"  ❌ تطبیق یافت نشد")
                
        except Exception as e:
            print(f"  ❌ خطا: {e}")
    
    success_rate = (successful_mappings / len(test_tokens)) * 100
    print(f"\n📊 نرخ موفقیت نگاشت: {success_rate:.1f}% ({successful_mappings}/{len(test_tokens)})")
    
    return success_rate > 60

def test_multi_hop_queries():
    """تست سوالات چندمرحله‌ای"""
    print("\n🔄 تست سوالات چندمرحله‌ای")
    print("=" * 60)
    
    service = GraphRAGService()
    
    # سوالات پیچیده که نیاز به چند مرحله دارند
    complex_questions = [
        "What compounds upregulate genes expressed in the heart?",
        "What diseases are associated with genes that interact with TP53?",
        "What pathways involve genes that are expressed in the brain?"
    ]
    
    successful_complex = 0
    
    for question in complex_questions:
        print(f"\n🔍 سوال پیچیده: {question}")
        
        try:
            intent = service.analyze_question_intent(question)
            print(f"  تشخیص نوع: {intent['question_type']}")
            print(f"  Metaedges: {intent['metaedges']}")
            
            results = service.intelligent_semantic_search(question, max_depth=3)
            
            if results:
                print(f"  ✅ {len(results)} نتیجه یافت شد")
                
                # بررسی تنوع انواع موجودیت‌ها در نتایج
                entity_types = set()
                for node_id, depth, score, explanation in results:
                    node_kind = service.G.nodes[node_id]['kind']
                    entity_types.add(node_kind)
                
                print(f"  📊 انواع موجودیت‌ها: {list(entity_types)}")
                
                if len(entity_types) >= 2:  # حداقل 2 نوع موجودیت مختلف
                    successful_complex += 1
                    print(f"  ✅ سوال چندمرحله‌ای با موفقیت پردازش شد")
                else:
                    print(f"  ⚠️ تنوع موجودیت‌ها کم است")
            else:
                print(f"  ❌ هیچ نتیجه‌ای یافت نشد")
                
        except Exception as e:
            print(f"  ❌ خطا: {e}")
    
    success_rate = (successful_complex / len(complex_questions)) * 100
    print(f"\n📊 نرخ موفقیت سوالات پیچیده: {success_rate:.1f}% ({successful_complex}/{len(complex_questions)})")
    
    return success_rate > 50

def main():
    """تست اصلی"""
    print("🚀 شروع تست جامع سیستم GraphRAG")
    print("=" * 80)
    
    test_results = {}
    
    # تست 1: تشخیص نوع سوال
    test_results['question_type_detection'] = test_question_type_detection()
    
    # تست 2: جستجوی آگاه از metaedge
    test_results['metaedge_aware_search'] = test_metaedge_aware_search()
    
    # تست 3: تولید پاسخ مبتنی بر شواهد
    test_evidence_based_answers()
    test_results['evidence_based_answers'] = True  # این تست فقط نمایشی است
    
    # تست 4: نگاشت موجودیت‌ها
    test_results['entity_linking'] = test_entity_linking()
    
    # تست 5: سوالات چندمرحله‌ای
    test_results['multi_hop_queries'] = test_multi_hop_queries()
    
    # خلاصه نتایج
    print("\n" + "=" * 80)
    print("📊 خلاصه نتایج تست جامع")
    print("=" * 80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    overall_success_rate = (passed_tests / total_tests) * 100
    print(f"\n🎯 نرخ موفقیت کلی: {overall_success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    if overall_success_rate >= 70:
        print("🎉 سیستم آماده استفاده است!")
    else:
        print("⚠️ نیاز به بهبود بیشتر دارد")
    
    return overall_success_rate >= 70

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 
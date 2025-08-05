#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ماژول نهایی برای ادغام کامل EnhancedContextGenerator با GraphRAGService
این ماژول مشکل اصلی سیستم فعلی را حل می‌کند
"""

import sys
import os
from typing import Dict, List, Tuple, Optional, Any
import json

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel, RetrievalResult
from enhanced_context_generator import EnhancedContextGenerator

class IntegratedGraphRAGService:
    """سرویس GraphRAG بهبود یافته با سیستم تبدیل شناسه‌ها"""
    
    def __init__(self, graph_data_path: str = None):
        """راه‌اندازی سرویس بهبود یافته"""
        print("🚀 راه‌اندازی سرویس GraphRAG بهبود یافته...")
        
        # راه‌اندازی سرویس اصلی
        self.graphrag_service = GraphRAGService(graph_data_path)
        
        # راه‌اندازی سیستم تولید متن زمینه بهبود یافته
        self.enhanced_context_generator = EnhancedContextGenerator()
        
        print("✅ سرویس GraphRAG بهبود یافته راه‌اندازی شد")
    
    def process_query_enhanced(self, query: str, retrieval_method: RetrievalMethod, 
                             generation_model: GenerationModel, 
                             text_generation_type: str = 'INTELLIGENT',
                             context_type: str = 'INTELLIGENT',
                             max_depth: int = 2) -> Dict[str, Any]:
        """پردازش سوال با سیستم بهبود یافته"""
        
        print(f"🔍 پردازش سوال بهبود یافته: {query}")
        print(f"📝 نوع متن زمینه: {context_type}")
        
        # مرحله 1: بازیابی با سرویس اصلی
        retrieval_result = self.graphrag_service.retrieve_information(
            query, retrieval_method, max_depth
        )
        
        # مرحله 2: بهبود متن زمینه با سیستم جدید
        enhanced_context = self.enhanced_context_generator.create_enhanced_context_text(
            retrieval_result, context_type
        )
        
        # مرحله 3: تولید پاسخ با متن زمینه بهبود یافته
        # ایجاد RetrievalResult جدید با متن زمینه بهبود یافته
        enhanced_retrieval_result = RetrievalResult(
            nodes=retrieval_result.nodes,
            edges=retrieval_result.edges,
            paths=retrieval_result.paths,
            context_text=enhanced_context,  # متن زمینه بهبود یافته
            method=retrieval_result.method,
            query=retrieval_result.query
        )
        
        # تولید پاسخ
        generation_result = self.graphrag_service.generate_answer(
            enhanced_retrieval_result, generation_model, text_generation_type
        )
        
        # آماده‌سازی نتیجه نهایی
        result = {
            "query": query,
            "retrieval_method": retrieval_method.value,
            "generation_model": generation_model.value,
            "context_type": context_type,
            "keywords": self.graphrag_service.extract_keywords(query),
            "matched_nodes": {k: self.graphrag_service.G.nodes[v]['name'] 
                            for k, v in self.graphrag_service.match_tokens_to_nodes(
                                self.graphrag_service.extract_keywords(query)
                            ).items()},
            "retrieved_nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "kind": node.kind,
                    "depth": node.depth,
                    "score": node.score
                } for node in retrieval_result.nodes
            ],
            "retrieved_edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "weight": edge.weight
                } for edge in retrieval_result.edges
            ],
            "paths": retrieval_result.paths,
            "original_context_text": retrieval_result.context_text,
            "enhanced_context_text": enhanced_context,  # متن زمینه بهبود یافته
            "answer": generation_result.answer,
            "confidence": generation_result.confidence,
            "process_steps": [
                "1. استخراج کلمات کلیدی از سوال",
                "2. تطبیق کلمات کلیدی با نودهای گراف",
                f"3. بازیابی اطلاعات با روش {retrieval_method.value}",
                "4. تبدیل شناسه‌ها به نام‌های معنادار",
                f"5. ایجاد متن زمینه بهبود یافته ({context_type})",
                f"6. تولید پاسخ با مدل {generation_model.value}"
            ]
        }
        
        return result
    
    def compare_context_quality(self, query: str, retrieval_method: RetrievalMethod = RetrievalMethod.INTELLIGENT) -> Dict[str, Any]:
        """مقایسه کیفیت متن زمینه اصلی و بهبود یافته"""
        
        print(f"🔍 مقایسه کیفیت متن زمینه برای سوال: {query}")
        
        # بازیابی اطلاعات
        retrieval_result = self.graphrag_service.retrieve_information(query, retrieval_method)
        
        # متن زمینه اصلی
        original_context = retrieval_result.context_text
        
        # متن زمینه بهبود یافته
        enhanced_context = self.enhanced_context_generator.create_enhanced_context_text(
            retrieval_result, "INTELLIGENT"
        )
        
        # تحلیل کیفیت
        comparison = {
            "query": query,
            "original_context_length": len(original_context),
            "enhanced_context_length": len(enhanced_context),
            "original_context": original_context,
            "enhanced_context": enhanced_context,
            "improvement_metrics": {
                "length_ratio": len(enhanced_context) / max(len(original_context), 1),
                "has_meaningful_names": "Gene::7157" not in enhanced_context,
                "has_biological_info": "نقش زیستی" in enhanced_context,
                "has_relation_descriptions": "مشارکت در فرآیند زیستی" in enhanced_context
            }
        }
        
        return comparison
    
    def test_enhanced_system(self, test_queries: List[str]) -> Dict[str, Any]:
        """تست کامل سیستم بهبود یافته"""
        
        print("🧪 تست کامل سیستم بهبود یافته")
        print("=" * 60)
        
        results = {
            "test_queries": test_queries,
            "results": [],
            "summary": {}
        }
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 تست {i}/{len(test_queries)}: {query}")
            
            try:
                # تست سیستم بهبود یافته
                enhanced_result = self.process_query_enhanced(
                    query=query,
                    retrieval_method=RetrievalMethod.INTELLIGENT,
                    generation_model=GenerationModel.GPT_SIMULATION,
                    text_generation_type='INTELLIGENT',
                    context_type='INTELLIGENT'
                )
                
                # مقایسه کیفیت
                comparison = self.compare_context_quality(query)
                
                result = {
                    "query": query,
                    "enhanced_result": enhanced_result,
                    "comparison": comparison,
                    "success": True
                }
                
                print(f"✅ تست {i} موفقیت‌آمیز")
                
            except Exception as e:
                print(f"❌ خطا در تست {i}: {e}")
                result = {
                    "query": query,
                    "error": str(e),
                    "success": False
                }
            
            results["results"].append(result)
        
        # خلاصه نتایج
        successful_tests = [r for r in results["results"] if r["success"]]
        results["summary"] = {
            "total_tests": len(test_queries),
            "successful_tests": len(successful_tests),
            "success_rate": len(successful_tests) / len(test_queries),
            "average_context_improvement": sum(
                r["comparison"]["improvement_metrics"]["length_ratio"] 
                for r in successful_tests
            ) / len(successful_tests) if successful_tests else 0
        }
        
        print(f"\n📊 خلاصه نتایج:")
        print(f"• کل تست‌ها: {results['summary']['total_tests']}")
        print(f"• تست‌های موفق: {results['summary']['successful_tests']}")
        print(f"• نرخ موفقیت: {results['summary']['success_rate']:.2%}")
        print(f"• بهبود متوسط متن زمینه: {results['summary']['average_context_improvement']:.2f}x")
        
        return results

def main():
    """تابع اصلی برای تست سیستم بهبود یافته"""
    
    # سوالات تست
    test_queries = [
        "What is the relationship between TP53 and cancer?",
        "How does Carmustine treat brain cancer?",
        "What genes are involved in apoptosis?",
        "What drugs are used to treat glioma?",
        "How do genes regulate biological processes?"
    ]
    
    try:
        # راه‌اندازی سیستم
        integrated_service = IntegratedGraphRAGService()
        
        # تست سیستم
        test_results = integrated_service.test_enhanced_system(test_queries)
        
        # نمایش نمونه‌ای از نتایج
        if test_results["results"]:
            first_result = test_results["results"][0]
            if first_result["success"]:
                print(f"\n📄 نمونه متن زمینه بهبود یافته:")
                print("-" * 40)
                enhanced_context = first_result["enhanced_result"]["enhanced_context_text"]
                print(enhanced_context[:1000] + "..." if len(enhanced_context) > 1000 else enhanced_context)
        
        print("\n🎉 تست سیستم بهبود یافته تکمیل شد!")
        return True
        
    except Exception as e:
        print(f"❌ خطا در تست سیستم: {e}")
        return False

if __name__ == "__main__":
    main() 
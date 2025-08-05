#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست ساده سیستم بهبود یافته
"""

import sys
import os
from typing import Dict, List, Any

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_context_generator import EnhancedContextGenerator
from graphrag_service import GraphRAGService, RetrievalMethod, GenerationModel, GraphNode, GraphEdge, RetrievalResult

def simple_test():
    """تست ساده سیستم بهبود یافته"""
    print("🧬 تست ساده سیستم بهبود یافته")
    print("=" * 60)
    
    try:
        # راه‌اندازی سیستم‌ها
        print("📋 راه‌اندازی سیستم‌ها...")
        enhanced_generator = EnhancedContextGenerator()
        graphrag_service = GraphRAGService()
        
        print("✅ سیستم‌ها راه‌اندازی شدند")
        
        # سوال تست
        test_query = "What is the relationship between TP53 and cancer?"
        
        print(f"\n🔍 سوال تست: {test_query}")
        
        # بازیابی اطلاعات
        print("\n📊 بازیابی اطلاعات...")
        retrieval_result = graphrag_service.retrieve_information(
            test_query, RetrievalMethod.INTELLIGENT, max_depth=2
        )
        
        print(f"✅ {len(retrieval_result.nodes)} نود و {len(retrieval_result.edges)} یال بازیابی شد")
        
        # نمایش نودهای بازیابی شده
        print("\n📋 نودهای بازیابی شده:")
        for i, node in enumerate(retrieval_result.nodes[:5], 1):  # فقط 5 نود اول
            print(f"{i}. {node.id} -> {node.name} ({node.kind})")
        
        # نمایش یال‌های بازیابی شده
        print("\n🔗 یال‌های بازیابی شده:")
        for i, edge in enumerate(retrieval_result.edges[:5], 1):  # فقط 5 یال اول
            print(f"{i}. {edge.source} -> {edge.relation} -> {edge.target}")
        
        # ایجاد متن زمینه بهبود یافته
        print("\n📄 ایجاد متن زمینه بهبود یافته...")
        enhanced_context = enhanced_generator.create_enhanced_context_text(
            retrieval_result, "INTELLIGENT"
        )
        
        print("\n📄 متن زمینه بهبود یافته:")
        print("-" * 40)
        print(enhanced_context[:1500] + "..." if len(enhanced_context) > 1500 else enhanced_context)
        
        # مقایسه با متن زمینه اصلی
        print("\n📊 مقایسه کیفیت:")
        original_length = len(retrieval_result.context_text)
        enhanced_length = len(enhanced_context)
        
        print(f"• طول متن زمینه اصلی: {original_length} کاراکتر")
        print(f"• طول متن زمینه بهبود یافته: {enhanced_length} کاراکتر")
        print(f"• نسبت بهبود: {enhanced_length / max(original_length, 1):.2f}x")
        
        # بررسی ویژگی‌های بهبود
        improvements = {
            "has_meaningful_names": "Gene::7157" not in enhanced_context,
            "has_biological_info": "نقش زیستی" in enhanced_context,
            "has_relation_descriptions": "مشارکت در فرآیند زیستی" in enhanced_context,
            "has_clinical_info": "اهمیت بالینی" in enhanced_context
        }
        
        print("\n✅ ویژگی‌های بهبود:")
        for feature, has_feature in improvements.items():
            status = "✅" if has_feature else "❌"
            print(f"{status} {feature}")
        
        print("\n🎉 تست ساده تکمیل شد!")
        return True
        
    except Exception as e:
        print(f"❌ خطا در تست: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_query():
    """تست سوال خاص"""
    print("\n🔍 تست سوال خاص")
    print("=" * 40)
    
    try:
        enhanced_generator = EnhancedContextGenerator()
        
        # داده‌های نمونه
        sample_nodes = [
            GraphNode(id="Gene::7157", name="TP53", kind="Gene", depth=0, score=1.0),
            GraphNode(id="Disease::DOID:162", name="malignant glioma", kind="Disease", depth=1, score=0.8),
            GraphNode(id="Compound::DB00262", name="Carmustine", kind="Compound", depth=2, score=0.6)
        ]
        
        sample_edges = [
            GraphEdge(source="Gene::7157", target="Disease::DOID:162", relation="DaG", weight=1.0),
            GraphEdge(source="Compound::DB00262", target="Disease::DOID:162", relation="CtD", weight=0.9)
        ]
        
        sample_result = RetrievalResult(
            nodes=sample_nodes,
            edges=sample_edges,
            paths=[],
            context_text="",
            method="Test",
            query="What is the relationship between TP53 and cancer treatment?"
        )
        
        # ایجاد متن زمینه بهبود یافته
        enhanced_context = enhanced_generator.create_enhanced_context_text(
            sample_result, "INTELLIGENT"
        )
        
        print("📄 متن زمینه بهبود یافته برای سوال خاص:")
        print("-" * 40)
        print(enhanced_context)
        
        return True
        
    except Exception as e:
        print(f"❌ خطا در تست سوال خاص: {e}")
        return False

if __name__ == "__main__":
    # تست ساده
    success1 = simple_test()
    
    # تست سوال خاص
    success2 = test_specific_query()
    
    if success1 and success2:
        print("\n🎉 تمام تست‌ها موفقیت‌آمیز بودند!")
    else:
        print("\n⚠️ برخی تست‌ها با مشکل مواجه شدند.") 
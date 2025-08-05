#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست متن زمینه‌ای بهبود یافته
"""

from graphrag_service import GraphRAGService, RetrievalResult, GraphNode, GraphEdge, GenerationModel

def test_enhanced_context():
    """تست متن زمینه‌ای بهبود یافته"""
    
    # ایجاد نمونه سرویس
    service = GraphRAGService()
    service.initialize()
    
    # ایجاد داده‌های نمونه برای تست
    nodes = [
        GraphNode(id="TP53", name="TP53", kind="Gene", depth=0, score=1.0),
        GraphNode(id="SMAD2", name="SMAD2", kind="Gene", depth=1, score=0.8),
        GraphNode(id="malignant_glioma", name="malignant glioma", kind="Disease", depth=1, score=0.9),
        GraphNode(id="brain", name="brain", kind="Anatomy", depth=2, score=0.7),
        GraphNode(id="apoptosis", name="apoptosis", kind="Biological Process", depth=2, score=0.6)
    ]
    
    edges = [
        GraphEdge(source="TP53", target="SMAD2", relation="GiG", weight=1.0),
        GraphEdge(source="TP53", target="malignant_glioma", relation="DaG", weight=0.9),
        GraphEdge(source="SMAD2", target="malignant_glioma", relation="DaG", weight=0.8),
        GraphEdge(source="malignant_glioma", target="brain", relation="DlA", weight=0.7),
        GraphEdge(source="TP53", target="apoptosis", relation="GpBP", weight=0.9)
    ]
    
    paths = [
        ["TP53", "SMAD2", "malignant_glioma"],
        ["TP53", "malignant_glioma", "brain"],
        ["TP53", "apoptosis"]
    ]
    
    # ایجاد نتیجه بازیابی
    retrieval_result = RetrievalResult(
        nodes=nodes,
        edges=edges,
        paths=paths,
        context_text="",
        method="Intelligent",
        query="What is the relationship between TP53 and cancer?"
    )
    
    print("🧬 تست متن زمینه‌ای بهبود یافته")
    print("=" * 50)
    
    # تست متن زمینه‌ای هوشمند بهبود یافته
    print("\n📝 متن زمینه‌ای هوشمند بهبود یافته:")
    enhanced_context = service._create_enhanced_intelligent_context_text(retrieval_result)
    print(enhanced_context)
    
    print("\n" + "=" * 50)
    
    # تست متن زمینه‌ای مسیر زیستی
    print("\n🛤️ متن زمینه‌ای مسیر زیستی:")
    pathway_context = service._create_biological_pathway_context(retrieval_result)
    print(pathway_context)
    
    print("\n" + "=" * 50)
    
    # تست متن زمینه‌ای مکانیسمی
    print("\n⚙️ متن زمینه‌ای مکانیسمی:")
    mechanistic_context = service._create_mechanistic_detailed_context(retrieval_result)
    print(mechanistic_context)
    
    print("\n" + "=" * 50)
    
    # تست توضیحات metaedge
    print("\n🔗 تست توضیحات metaedge:")
    from graphrag_service import METAEDGE_DESCRIPTIONS
    for relation in ["DaG", "GiG", "DlA", "GpBP"]:
        desc = METAEDGE_DESCRIPTIONS.get(relation, relation)
        print(f"• {relation}: {desc}")
    
    print("\n" + "=" * 50)
    
    # تست نقش‌های زیستی
    print("\n🧬 تست نقش‌های زیستی:")
    from graphrag_service import BIOLOGICAL_ROLES
    for gene in ["TP53", "SMAD2", "BRCA1"]:
        role = BIOLOGICAL_ROLES.get(gene, "نقش نامشخص")
        print(f"• {gene}: {role}")
    
    print("\n" + "=" * 50)
    
    # تست بیماری‌های مهم
    print("\n🏥 تست بیماری‌های مهم:")
    from graphrag_service import DISEASE_SIGNIFICANCE
    for disease in ["malignant glioma", "breast cancer", "lung cancer"]:
        significance = DISEASE_SIGNIFICANCE.get(disease, disease)
        print(f"• {disease}: {significance}")

if __name__ == "__main__":
    test_enhanced_context() 
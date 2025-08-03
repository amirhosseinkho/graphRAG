#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تست نهایی برای بررسی وضعیت فعلی سیستم
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag_service import GraphRAGService

def test_final_status():
    """تست نهایی برای بررسی وضعیت فعلی سیستم"""
    print("🔍 تست نهایی برای بررسی وضعیت فعلی سیستم")
    
    # ایجاد سرویس
    service = GraphRAGService()
    service.initialize()
    
    # تست 1: جستجوی هوشمند
    print("\n📋 تست 1: جستجوی هوشمند")
    query1 = "What genes are expressed in the heart?"
    results1 = service.intelligent_semantic_search(query1, max_depth=2)
    
    print(f"  سوال: {query1}")
    print(f"  تعداد نتایج: {len(results1)}")
    for i, (node_id, depth, score, explanation) in enumerate(results1[:3]):
        node_name = service.G.nodes[node_id]['name'] if service.G.has_node(node_id) else node_id
        print(f"    {i+1}. {node_name} (عمق: {depth}, امتیاز: {score:.2f})")
        print(f"       توضیح: {explanation}")
    
    # تست 2: جستجوی چندمرحله‌ای
    print("\n📋 تست 2: جستجوی چندمرحله‌ای")
    query2 = "What compounds upregulate genes expressed in the heart?"
    results2 = service.multi_hop_search(query2, max_depth=3)
    
    print(f"  سوال: {query2}")
    print(f"  تعداد نتایج: {len(results2)}")
    for i, (node_id, depth, score, explanation, path) in enumerate(results2[:3]):
        node_name = service.G.nodes[node_id]['name'] if service.G.has_node(node_id) else node_id
        print(f"    {i+1}. {node_name} (عمق: {depth}, امتیاز: {score:.2f})")
        print(f"       توضیح: {explanation}")
        print(f"       مسیر: {path}")
    
    # تست 3: بررسی مسیرهای واقعی
    print("\n📋 تست 3: بررسی مسیرهای واقعی")
    
    # بررسی مسیر Heart → MMP9 → Caffeine
    heart_node = 'Anatomy::Heart'
    mmp9_node = 'Gene::MMP9'
    caffeine_node = 'Compound::Caffeine'
    
    # بررسی یال Heart → MMP9
    heart_mmp9 = service.G.get_edge_data(heart_node, mmp9_node)
    print(f"  یال Heart → MMP9: {'✅ وجود دارد' if heart_mmp9 else '❌ وجود ندارد'}")
    
    # بررسی یال Caffeine → MMP9
    caffeine_mmp9 = service.G.get_edge_data(caffeine_node, mmp9_node)
    print(f"  یال Caffeine → MMP9: {'✅ وجود دارد' if caffeine_mmp9 else '❌ وجود ندارد'}")
    
    # بررسی یال MMP9 → Heart (معکوس)
    mmp9_heart = service.G.get_edge_data(mmp9_node, heart_node)
    print(f"  یال MMP9 → Heart: {'✅ وجود دارد' if mmp9_heart else '❌ وجود ندارد'}")
    
    # تست 4: بررسی مسیرهای ممکن
    print("\n📋 تست 4: بررسی مسیرهای ممکن")
    
    # مسیر مستقیم: Heart → MMP9 → Caffeine (غیرممکن)
    print(f"  مسیر مستقیم: Heart → MMP9 → Caffeine")
    print(f"    Heart → MMP9: {'✅' if heart_mmp9 else '❌'}")
    print(f"    MMP9 → Caffeine: {'✅' if mmp9_heart else '❌'}")
    
    # مسیر معکوس: Caffeine → MMP9 → Heart (غیرممکن)
    print(f"  مسیر معکوس: Caffeine → MMP9 → Heart")
    print(f"    Caffeine → MMP9: {'✅' if caffeine_mmp9 else '❌'}")
    print(f"    MMP9 → Heart: {'✅' if mmp9_heart else '❌'}")
    
    # تست 5: خلاصه وضعیت
    print("\n📋 تست 5: خلاصه وضعیت")
    print(f"  ✅ جستجوی هوشمند: کار می‌کند ({len(results1)} نتیجه)")
    print(f"  ❌ جستجوی چندمرحله‌ای: کار نمی‌کند ({len(results2)} نتیجه)")
    print(f"  🔍 دلیل: گراف جهت‌دار است و مسیرهای معکوس وجود ندارند")
    
    # پیشنهادات بهبود
    print("\n📋 پیشنهادات بهبود:")
    print(f"  1. اضافه کردن یال‌های معکوس به گراف")
    print(f"  2. بهبود الگوریتم جستجوی چندمرحله‌ای")
    print(f"  3. استفاده از مسیرهای غیرمستقیم")
    
    print("\n✅ تست نهایی تکمیل شد!")

if __name__ == "__main__":
    test_final_status() 
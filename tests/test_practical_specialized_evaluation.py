#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test for practical and specialized evaluation functionality
"""

import requests
import json

def test_practical_specialized_evaluation():
    """Test the practical and specialized evaluation feature"""
    
    # Test data
    test_data = {
        "text1": "ژن TP53 یک ژن سرکوب‌کننده تومور است که نقش مهمی در تنظیم چرخه سلولی و آپوپتوز دارد. جهش‌های این ژن در بسیاری از سرطان‌ها مشاهده می‌شود و منجر به از دست رفتن عملکرد طبیعی آن می‌شود.",
        "text2": "TP53 یک ژن مهم در پیشگیری از سرطان است که پروتئین p53 را کد می‌کند. این پروتئین در پاسخ به آسیب DNA فعال می‌شود و می‌تواند باعث توقف چرخه سلولی یا مرگ سلولی شود.",
        "label1": "روش تخصصی",
        "label2": "روش عمومی",
        "comparison_type": "practical_specialized",
        "gpt_model": "gpt-4o-mini"
    }
    
    try:
        # Send request to the API
        response = requests.post(
            'http://localhost:5000/api/compare_with_gpt',
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Test successful!")
            print(f"Score 1 ({result['label1']}): {result['score1']}/100")
            print(f"Score 2 ({result['label2']}): {result['score2']}/100")
            print(f"Summary: {result['summary'][:100]}...")
            print(f"Recommendation: {result['recommendation'][:100]}...")
            
            # Check if the system is prioritizing practical/specialized content
            if result['score1'] > result['score2']:
                print("✅ System correctly prioritized specialized content!")
            else:
                print("⚠️ System did not prioritize specialized content as expected")
                
        else:
            print(f"❌ Test failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the server. Make sure the web app is running.")
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")

if __name__ == "__main__":
    test_practical_specialized_evaluation() 
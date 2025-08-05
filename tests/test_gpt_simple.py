#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test for GPT comparison API
"""

import requests
import json

def test_gpt_api():
    """Test the GPT comparison API endpoint"""
    
    # Test data
    test_data = {
        "text1": "TP53 is a tumor suppressor gene that plays a crucial role in preventing cancer development.",
        "text2": "TP53 gene helps prevent cancer by controlling cell growth and death.",
        "label1": "Graph Method",
        "label2": "Direct Method", 
        "comparison_type": "comprehensive"
    }
    
    try:
        print("🧪 Testing GPT Comparison API...")
        print(f"Text 1: {test_data['text1']}")
        print(f"Text 2: {test_data['text2']}")
        print(f"Labels: {test_data['label1']} vs {test_data['label2']}")
        
        # Send POST request to the API
        response = requests.post(
            'http://localhost:5000/api/compare_with_gpt',
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ GPT Comparison Test Successful!")
            print(f"Summary: {result.get('summary', 'N/A')}")
            print(f"Score 1: {result.get('score1', 'N/A')}/10")
            print(f"Score 2: {result.get('score2', 'N/A')}/10")
            print(f"Recommendation: {result.get('recommendation', 'N/A')}")
            return True
        else:
            print(f"❌ Test failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the web application. Make sure it's running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    test_gpt_api() 
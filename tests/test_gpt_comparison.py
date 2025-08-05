#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for GPT comparison feature
"""

import requests
import json

def test_gpt_comparison():
    """Test the GPT comparison API endpoint"""
    
    # Test data
    test_data = {
        "text1": "TP53 is a tumor suppressor gene that plays a crucial role in preventing cancer development. It regulates cell cycle and apoptosis.",
        "text2": "TP53 gene helps prevent cancer by controlling cell growth and death. It's important for tumor suppression.",
        "label1": "Graph Retrieval Method",
        "label2": "Direct Generation Method", 
        "comparison_type": "comprehensive"
    }
    
    try:
        # Send POST request to the API
        response = requests.post(
            'http://localhost:5000/api/compare_with_gpt',
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ GPT Comparison Test Successful!")
            print(f"Summary: {result.get('summary', 'N/A')}")
            print(f"Score 1: {result.get('score1', 'N/A')}/10")
            print(f"Score 2: {result.get('score2', 'N/A')}/10")
            print(f"Recommendation: {result.get('recommendation', 'N/A')}")
        else:
            print(f"❌ Test failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the web application. Make sure it's running on http://localhost:5000")
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")

if __name__ == "__main__":
    print("🧪 Testing GPT Comparison Feature...")
    test_gpt_comparison() 
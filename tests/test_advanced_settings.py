#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test for advanced settings functionality in evaluation page
"""

import requests
import json

def test_advanced_settings():
    """Test the advanced settings functionality"""
    
    # Test data with advanced settings
    test_data = {
        "query": "ژن TP53 چه نقشی در سرطان دارد؟",
        "retrieval_method": "BFS",
        "generation_model": "OPENAI_GPT_4O_MINI",
        "text_generation_type": "INTELLIGENT",
        "max_depth": 2,
        # تنظیمات پیشرفته
        "max_nodes": 15,
        "max_edges": 30,
        "similarity_threshold": 0.4,
        "community_detection_method": "label_propagation",
        "advanced_retrieval_algorithm": "pagerank",
        "advanced_token_extraction_method": "semantic"
    }
    
    try:
        # Send request to the API
        response = requests.post(
            'http://localhost:5000/api/process_query',
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Test successful!")
            print(f"Query: {test_data['query']}")
            print(f"Answer: {result['result']['answer'][:100]}...")
            
            # Check if advanced settings were applied
            if 'advanced_settings' in result['result']:
                advanced_settings = result['result']['advanced_settings']
                print("✅ Advanced settings applied:")
                print(f"   Max Nodes: {advanced_settings['max_nodes']}")
                print(f"   Max Edges: {advanced_settings['max_edges']}")
                print(f"   Similarity Threshold: {advanced_settings['similarity_threshold']}")
                print(f"   Community Detection: {advanced_settings['community_detection_method']}")
                print(f"   Advanced Algorithm: {advanced_settings['advanced_retrieval_algorithm']}")
                print(f"   Token Extraction: {advanced_settings['advanced_token_extraction_method']}")
            else:
                print("⚠️ Advanced settings not found in response")
                
        else:
            print(f"❌ Test failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the server. Make sure the web app is running.")
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")

def test_default_settings():
    """Test with default settings for comparison"""
    
    # Test data with default settings
    test_data = {
        "query": "ژن TP53 چه نقشی در سرطان دارد؟",
        "retrieval_method": "BFS",
        "generation_model": "OPENAI_GPT_4O_MINI",
        "text_generation_type": "INTELLIGENT",
        "max_depth": 2
        # No advanced settings - should use defaults
    }
    
    try:
        # Send request to the API
        response = requests.post(
            'http://localhost:5000/api/process_query',
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Default settings test successful!")
            print(f"Query: {test_data['query']}")
            print(f"Answer: {result['result']['answer'][:100]}...")
            
            # Check if default settings were used
            if 'advanced_settings' in result['result']:
                advanced_settings = result['result']['advanced_settings']
                print("✅ Default settings used:")
                print(f"   Max Nodes: {advanced_settings['max_nodes']} (default: 20)")
                print(f"   Max Edges: {advanced_settings['max_edges']} (default: 40)")
                print(f"   Similarity Threshold: {advanced_settings['similarity_threshold']} (default: 0.3)")
                
        else:
            print(f"❌ Default settings test failed with status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Default settings test failed with error: {str(e)}")

if __name__ == "__main__":
    print("🧪 Testing Advanced Settings Functionality")
    print("=" * 50)
    
    test_advanced_settings()
    test_default_settings()
    
    print("\n" + "=" * 50)
    print("✅ Advanced settings test completed!") 
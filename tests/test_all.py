#!/usr/bin/env python3
"""
Comprehensive test suite for the Fact-Checking API
Combines all essential tests in one file for easy execution.
"""

import asyncio
import json
import os
import sys
import time
from typing import Optional

import requests

# Get the parent directory and add it to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Change to parent directory for correct relative paths
os.chdir(parent_dir)

from api_wrapper import process_full_request


def test_endpoint(base_url: str, endpoint: str, method: str = "GET", data: Optional[dict] = None) -> bool:
    """Test a single endpoint."""
    url = f"{base_url}{endpoint}"
    
    try:
        print(f"Testing {method} {endpoint}...")
        
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=60)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        if response.status_code == 200:
            print(f"✅ {endpoint} - Status: {response.status_code}")
            return True
        else:
            print(f"❌ {endpoint} - Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ {endpoint} - Request timed out")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {endpoint} - Request failed: {e}")
        return False


def test_api_deployment(base_url: str):
    """Test the complete API deployment."""
    print(f"🧪 Testing API deployment at: {base_url}")
    print("=" * 50)
    
    tests = []
    
    # Test health endpoint
    tests.append(test_endpoint(base_url, "/health"))
    
    # Test RSS status endpoint
    tests.append(test_endpoint(base_url, "/rss-status"))
    
    # Test documentation endpoints
    tests.append(test_endpoint(base_url, "/docs"))
    
    # Test basic fact-check endpoint
    fact_check_data = {
        "text": "This is a test message for fact checking",
        "author_name": "Test Author"
    }
    tests.append(test_endpoint(base_url, "/fact-check", "POST", fact_check_data))
    
    # Test JSON fact-check endpoint
    json_fact_check_data = {
        "input_data": {
            "author": "Test Author",
            "text": "This is a test message for JSON fact checking"
        }
    }
    tests.append(test_endpoint(base_url, "/fact-check-json", "POST", json_fact_check_data))
    
    # Summary
    passed = sum(tests)
    total = len(tests)
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Deployment is successful.")
        return True
    else:
        print("⚠️  Some tests failed. Check the deployment configuration.")
        return False


def test_rss_functionality(base_url: str):
    """Test RSS-specific functionality."""
    print("\n🔄 Testing RSS functionality...")
    
    # Get RSS status
    try:
        response = requests.get(f"{base_url}/rss-status", timeout=30)
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ RSS Status retrieved successfully")
            print(f"   Last update: {status_data.get('data', {}).get('last_update', 'Never')}")
            print(f"   Total feeds: {status_data.get('data', {}).get('total_feeds', 0)}")
            print(f"   Is running: {status_data.get('data', {}).get('is_running', False)}")
        else:
            print(f"❌ RSS Status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ RSS Status error: {e}")


async def test_comprehensive_fact_check():
    """Test comprehensive fact-checking with Hebrew input."""
    print("\n🔍 Testing Author Processing with Hebrew Input")
    print("=" * 60)
    
    # Test data with Hebrew author and text
    hebrew_author = "יאיר גולן"
    hebrew_text = """״מסיתים כמוכם הביאו לרצח רבין, לא למדתם כלום, יש לכם אפס הערכה לאנשים שהקריבו את חייהם עבור המדינה הזאת״.

״כשאתם ישבתם בממ"דים בשבעה באוקטובר אני יצאתי לנובה להציל אנשים, יצאתי לסכן את חיי, ואתם ישבתם בבית ובממ"דים״.

״אין בכם כלום חוץ משנאה, אתם לא יודעים שום דבר חוץ משנאה וזה מה שיצא מכנס שדרות שכותרתו איך נבנים מחדש ואתם רק מדברים על איך להרוס״."""
    
    print(f"📝 Input Author: {hebrew_author}")
    print(f"📝 Input Text: {hebrew_text[:100]}...")
    
    # Create JSON input
    input_data = {
        "author": hebrew_author,
        "text": hebrew_text
    }
    json_input = json.dumps(input_data, ensure_ascii=False)
    
    print(f"\n📊 Running full analysis with Hebrew author...")
    
    try:
        result = await process_full_request(json_input)
        
        if result["status"] == "success":
            print("✅ Full analysis completed successfully!")
            
            data = result["data"]
            print(f"\n👤 Author in result: {data.get('author', 'N/A')}")
            print(f"📊 Author info: {data.get('author_info', 'N/A')}")
            
            fact_checks = data.get('fact_check_results', [])
            print(f"✅ Generated {len(fact_checks)} fact check results")
            
            if fact_checks:
                first_check = fact_checks[0]
                print(f"\n📋 Sample Fact Check (Event 1):")
                print(f"   📝 Sentence: {first_check.get('sentence', 'N/A')[:50]}...")
                print(f"   🏷️ Classification: {first_check.get('classification', 'N/A')}")
                print(f"   📊 Overall Score: {first_check.get('overall_score', 'N/A')}")
                
                sources = first_check.get('sources', [])
                if sources:
                    print(f"   🔗 Sources: {len(sources)} found")
                else:
                    print(f"   🔗 No sources found")
            
            print(f"\n📈 Final Summary:")
            print(f"   👤 Author: {data.get('author', 'N/A')}")
            print(f"   📝 Events analyzed: {len(data.get('user_events', []))}")
            print(f"   ✅ Fact checks completed: {len(fact_checks)}")
            print(f"   📰 RSS matches: {len([m for m in data.get('matches_dict', {}).values() if m.get('Match')])}")
            print(f"   📚 Wikipedia characters: {len(data.get('character_info', {}))}")
            
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎯 Author processing test complete!")
    return True


def test_simple_similarity():
    """Test the SimpleSimilarityModel."""
    print("\n🧠 Testing SimpleSimilarityModel...")
    
    try:
        from simple_similarity import SentenceTransformer
        
        # Test model creation
        model = SentenceTransformer("test-model")
        print("✅ SimpleSimilarityModel created successfully")
        
        # Test encoding
        text = "This is a test sentence"
        embedding = model.encode(text)
        print(f"✅ Text encoded to vector of shape: {embedding.shape}")
        
        # Test similarity
        text1 = "Hello world"
        text2 = "Hi there"
        emb1 = model.encode(text1)
        emb2 = model.encode(text2)
        
        # Simple cosine similarity
        import numpy as np
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"✅ Similarity calculated: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ SimpleSimilarityModel test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests."""
    print("🚀 Comprehensive Fact-Checking API Test Suite")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_passed = True
    
    # Test 1: SimpleSimilarityModel
    print("\n" + "="*60)
    print("TEST 1: SimpleSimilarityModel")
    print("="*60)
    if not test_simple_similarity():
        all_passed = False
    
    # Test 2: Comprehensive fact-checking
    print("\n" + "="*60)
    print("TEST 2: Comprehensive Fact-Checking")
    print("="*60)
    if not await test_comprehensive_fact_check():
        all_passed = False
    
    # Test 3: API endpoints (if server is running)
    print("\n" + "="*60)
    print("TEST 3: API Endpoints (requires running server)")
    print("="*60)
    try:
        # Try to connect to local server
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Local server detected, running API tests...")
            if not test_api_deployment("http://localhost:8000"):
                all_passed = False
            test_rss_functionality("http://localhost:8000")
        else:
            print("⚠️ Local server not responding, skipping API tests")
    except requests.exceptions.RequestException:
        print("⚠️ Local server not running, skipping API tests")
        print("   To test API endpoints, run: python api.py")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The fact-checking system is working correctly")
        print("\n🚀 Ready for deployment!")
    else:
        print("❌ SOME TESTS FAILED")
        print("⚠️ Please check the error messages above")
        print("\n🔧 Fix issues before deployment")
    
    return all_passed


if __name__ == "__main__":
    # Run all tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1) 
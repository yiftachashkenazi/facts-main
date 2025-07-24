#!/usr/bin/env python3
"""
Test script to verify the fact-checking API deployment
"""

import json
import sys
import time
from typing import Optional

import requests


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

def main():
    """Main test function."""
    if len(sys.argv) != 2:
        print("Usage: python test_deployment.py <base_url>")
        print("Example: python test_deployment.py https://your-app.onrender.com")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    
    print("🚀 Fact-Checking API Deployment Test")
    print(f"Target URL: {base_url}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run main API tests
    success = test_api_deployment(base_url)
    
    # Test RSS functionality
    test_rss_functionality(base_url)
    
    if success:
        print("\n🎯 Deployment test completed successfully!")
        print("\nNext steps:")
        print("1. Monitor the RSS collector status")
        print("2. Test with real fact-checking data")
        print("3. Set up monitoring and alerts")
        sys.exit(0)
    else:
        print("\n💥 Deployment test failed!")
        print("Please check the logs and configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
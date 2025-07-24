#!/usr/bin/env python3

import asyncio
import json
import time

import requests


def test_api_with_author():
    """Test the API with the user's exact JSON input."""
    
    # Test data - exact input from user
    test_data = {
        "author": "יאיר גולן",
        "text": "״מסיתים כמוכם הביאו לרצח רבין, לא למדתם כלום, יש לכם אפס הערכה לאנשים שהקריבו את חייהם עבור המדינה הזאת״.\n\n״כשאתם ישבתם בממ\"דים בשבעה באוקטובר אני יצאתי לנובה להציל אנשים, יצאתי לסכן את חיי, ואתם ישבתם בבית לבטח ועכשיו צועקים לי בוגד, תתביישו לכם״.\n\n״אין בכם כלום חוץ משנאה, אתם לא יודעים שום דבר חוץ משנאה וזה מה שיצא מכנס שדרות שכותרתו איך נבנים מחדש כשעדיין לא הפסקנו להתפרק. זה מה שאתם עושים - פירוק מדינת ישראל. תמשיכו לפרק״."
    }
    
    print("🔍 Testing API with Author Information")
    print("=" * 60)
    print(f"📝 Input Author: {test_data['author']}")
    print(f"📝 Input Text: {test_data['text'][:100]}...")
    
    # Test 1: Using the new /fact-check endpoint with separate fields
    print("\n🧪 Test 1: Using /fact-check endpoint with separate fields")
    try:
        response = requests.post(
            "http://localhost:8000/fact-check",
            json={
                "text": test_data["text"],
                "author_name": test_data["author"]
            },
            timeout=600  # 10 minute timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"📊 Status: {result['status']}")
            
            if result['status'] == 'success' and result['fact_check_output']:
                output = result['fact_check_output']
                print(f"👤 Author in result: {output.get('author')}")
                print(f"📊 Author info name: {output.get('author_info', {}).get('name', 'N/A')}")
                print(f"📈 Events analyzed: {output.get('summary', {}).get('total_events_analyzed', 0)}")
                print(f"✅ Fact checks completed: {output.get('summary', {}).get('fact_checks_completed', 0)}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    
    # Test 2: Using the new /fact-check-json endpoint with JSON input
    print("\n🧪 Test 2: Using /fact-check-json endpoint with JSON input")
    try:
        response = requests.post(
            "http://localhost:8000/fact-check-json",
            json={
                "input_data": test_data
            },
            timeout=600  # 10 minute timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ JSON API call successful!")
            print(f"📊 Status: {result['status']}")
            
            if result['status'] == 'success' and result['fact_check_output']:
                output = result['fact_check_output']
                print(f"👤 Author in result: {output.get('author')}")
                print(f"📊 Author info name: {output.get('author_info', {}).get('name', 'N/A')}")
                print(f"📈 Events analyzed: {output.get('summary', {}).get('total_events_analyzed', 0)}")
                print(f"✅ Fact checks completed: {output.get('summary', {}).get('fact_checks_completed', 0)}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 API author testing complete!")

if __name__ == "__main__":
    print("🚀 Starting API server test...")
    print("⚠️  Make sure the API server is running on localhost:8000")
    print("   Run: python api.py")
    print()
    
    # Wait a moment for user to start server if needed
    input("Press Enter when the API server is ready...")
    
    test_api_with_author() 
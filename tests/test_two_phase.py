#!/usr/bin/env python3

import asyncio
import json

from api_wrapper import process_calibration_request, process_full_request


async def test_two_phase_system():
    """Test the two-phase processing system."""
    
    # Test with JSON input containing author
    test_input = {
        "author": "Benjamin Netanyahu", 
        "text": "היועצת המשפטית לממשלה פוגעת בביטחון המדינה ומונעת מינויים חשובים."
    }
    
    print("🔄 Testing Two-Phase Fact-Checking System")
    print("=" * 50)
    
    # Phase 1: Calibration
    print("\n📊 Phase 1: Running Calibration Analysis...")
    calibration_result = await process_calibration_request(json.dumps(test_input, ensure_ascii=False))
    
    if calibration_result["status"] == "success":
        print("✅ Calibration phase completed successfully!")
        print(f"📈 Summary: {calibration_result['data']['summary']}")
        print(f"👤 Author: {calibration_result['data']['author']}")
        print(f"📝 Events analyzed: {calibration_result['data']['summary']['total_events_analyzed']}")
        print(f"📰 RSS matches: {calibration_result['data']['summary']['rss_matches']}")
        print(f"📚 Wikipedia characters: {calibration_result['data']['summary']['wikipedia_characters']}")
        print(f"⚖️ Calibration events: {calibration_result['data']['summary']['calibration_events']}")
    else:
        print(f"❌ Calibration failed: {calibration_result['error']}")
        return
    
    print("\n" + "=" * 50)
    
    # Phase 2: Full Analysis
    print("\n🔍 Phase 2: Running Full Fact-Check Analysis...")
    full_result = await process_full_request(json.dumps(test_input, ensure_ascii=False))
    
    if full_result["status"] == "success":
        print("✅ Full analysis completed successfully!")
        print(f"📊 Final Summary: {full_result['data']['summary']}")
        print(f"✅ Fact checks completed: {full_result['data']['summary']['fact_checks_completed']}")
        
        # Show first fact check result if available
        if full_result['data']['fact_check_results']:
            first_check = full_result['data']['fact_check_results'][0]
            print(f"\n📋 Sample Fact Check (Event {first_check['user_event_id']}):")
            if 'analysis' in first_check and 'analysis_results' in first_check['analysis']:
                analysis = first_check['analysis']['analysis_results']
                if analysis and len(analysis) > 0:
                    first_analysis = analysis[0]
                    print(f"   📝 Sentence: {first_analysis.get('sentence_text', 'N/A')[:100]}...")
                    print(f"   🏷️ Classification: {first_analysis.get('classification', 'N/A')}")
                    print(f"   📊 Overall Score: {first_analysis.get('overall_accuracy_score', 'N/A')}")
    else:
        print(f"❌ Full analysis failed: {full_result['error']}")
    
    print("\n" + "=" * 50)
    print("🎯 Two-phase processing complete!")
    print("📁 Check 'calibration_output.json' for Phase 1 results")
    print("📁 Check 'fact_check_output.json' for Phase 2 results")

if __name__ == "__main__":
    asyncio.run(test_two_phase_system()) 
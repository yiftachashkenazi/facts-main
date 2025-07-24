#!/usr/bin/env python3

import asyncio
import json

from api_wrapper import process_full_request


async def test_comprehensive_fact_check():
    """Test that comprehensive data is passed to the fact checker."""
    
    # Test with the exact JSON input from the user
    test_input = {
        "author": "יאיר גולן",
        "text": "״מסיתים כמוכם הביאו לרצח רבין, לא למדתם כלום, יש לכם אפס הערכה לאנשים שהקריבו את חייהם עבור המדינה הזאת״.\n\n״כשאתם ישבתם בממ\"דים בשבעה באוקטובר אני יצאתי לנובה להציל אנשים, יצאתי לסכן את חיי, ואתם ישבתם בבית לבטח ועכשיו צועקים לי בוגד, תתביישו לכם״.\n\n״אין בכם כלום חוץ משנאה, אתם לא יודעים שום דבר חוץ משנאה וזה מה שיצא מכנס שדרות שכותרתו איך נבנים מחדש כשעדיין לא הפסקנו להתפרק. זה מה שאתם עושים - פירוק מדינת ישראל. תמשיכו לפרק״."
    }
    
    print("🔍 Testing Author Processing with Hebrew Input")
    print("=" * 60)
    print(f"📝 Input Author: {test_input['author']}")
    print(f"📝 Input Text: {test_input['text'][:100]}...")
    
    print("\n📊 Running full analysis with Hebrew author...")
    full_result = await process_full_request(json.dumps(test_input, ensure_ascii=False))
    
    if full_result["status"] == "success":
        print("✅ Full analysis completed successfully!")
        
        # Check author in the result
        result_data = full_result['data']
        print(f"\n👤 Author in result: {result_data.get('author')}")
        print(f"📊 Author info: {result_data.get('author_info', {}).get('name', 'N/A')}")
        
        # Check if we have fact check results
        if result_data.get('fact_check_results'):
            print(f"✅ Generated {len(result_data['fact_check_results'])} fact check results")
            
            # Show the first fact check to verify comprehensive data was used
            first_check = result_data['fact_check_results'][0]
            print(f"\n📋 Sample Fact Check (Event {first_check['user_event_id']}):")
            
            if 'analysis' in first_check:
                if 'error' in first_check['analysis']:
                    print(f"   ❌ Error: {first_check['analysis']['error']}")
                elif 'analysis_results' in first_check['analysis']:
                    analysis = first_check['analysis']['analysis_results']
                    if analysis and len(analysis) > 0:
                        first_analysis = analysis[0]
                        print(f"   📝 Sentence: {first_analysis.get('sentence_text', 'N/A')[:100]}...")
                        print(f"   🏷️ Classification: {first_analysis.get('classification', 'N/A')}")
                        print(f"   📊 Overall Score: {first_analysis.get('overall_accuracy_score', 'N/A')}")
                        
                        # Check if reasoning mentions author context
                        reasoning = first_analysis.get('reasoning', '')
                        if reasoning:
                            print(f"   💭 Reasoning length: {len(reasoning)} characters")
                            print(f"   📚 Has author context: {'יאיר גולן' in reasoning or 'author' in reasoning.lower()}")
                        
                        # Check sources
                        sources = first_analysis.get('sources', [])
                        if sources:
                            print(f"   🔗 Sources found: {len(sources)}")
                        else:
                            print("   🔗 No sources found")
                    else:
                        print("   ⚠️ No analysis results found")
                else:
                    print("   ⚠️ Unexpected analysis structure")
            else:
                print("   ⚠️ No analysis field found")
        else:
            print("⚠️ No fact check results generated")
            
        # Show summary
        summary = result_data.get('summary', {})
        print(f"\n📈 Final Summary:")
        print(f"   👤 Author: {result_data.get('author', 'None')}")
        print(f"   📝 Events analyzed: {summary.get('total_events_analyzed', 0)}")
        print(f"   ✅ Fact checks completed: {summary.get('fact_checks_completed', 0)}")
        print(f"   📰 RSS matches: {summary.get('rss_matches', 0)}")
        print(f"   📚 Wikipedia characters: {summary.get('wikipedia_characters', 0)}")
        
    else:
        print(f"❌ Full analysis failed: {full_result['error']}")
    
    print("\n" + "=" * 60)
    print("🎯 Author processing test complete!")

if __name__ == "__main__":
    asyncio.run(test_comprehensive_fact_check()) 
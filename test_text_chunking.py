#!/usr/bin/env python3
"""
Test script for text chunking functionality with Hebrew texts.
Tests with 600-word and 1000-word Hebrew texts.
"""

import asyncio
import json
import time

from api_wrapper import process_full_request, set_global_model
from simple_similarity import SentenceTransformer
from utils.text_chunker import chunk_text_by_words, count_words

# 600-word Hebrew text
HEBREW_TEXT_600 = """
היועצת המשפטית לממשלה פוגעת בביטחון המדינה ומונעת מינויים חשובים. זה נושא רציני מאוד שצריך לטפל בו מיידית. הממשלה צריכה לפעול בתוקף כדי להבטיח שהמערכת המשפטית תשרת את האינטרסים הלאומיים ולא תפגע בביטחון המדינה.

מסיתים כמוכם הביאו לרצח רבין, לא למדתם כלום מההיסטוריה. יש לכם אפס הערכה לאנשים שהקריבו את חייהם עבור המדינה הזאת. השמאל הקיצוני ממשיך להסית נגד הממשלה הנבחרת ולפגוע בדמוקרטיה. הם מנסים לערער את יסודות המדינה ולהביא לכאוס פוליטי.

כשאתם ישבתם בממדים בשבעה באוקטובר אני יצאתי לנובה להציל אנשים. יצאתי לסכן את חיי כדי להגן על אזרחי ישראל, ואתם ישבתם בבית לבטח ועכשיו צועקים לי בוגד. אין לכם מושג מה זה אמת ומה זה להקריב למען המדינה. הצבא הישראלי נלחם כל יום כדי להגן עלינו.

אין בכם כלום חוץ משנאה וקיטוב. אתם לא יודעים שום דבר חוץ משנאה פוליטית וזה מה שיצא מכנס שדרות שכותרתו איך נבנים מחדש כשעדיין לא הפסקנו להתפרק. זה מה שאתם עושים כל הזמן - פירוק מדינת ישראל מבפנים. תמשיכו להרוס את מה שבנינו כאן עשרות שנים.

הממשלה הנוכחית נבחרה באופן דמוקרטי והיא מייצגת את רצון העם. האופוזיציה חייבת לכבד את התוצאות ולפעול במסגרת הדמוקרטית. המחאות ההמוניות נגד הרפורמה המשפטית הן ביטוי לרצון לשינוי במערכת שכבר לא משרתת את הציבור הרחב.

הרפורמה המשפטית נחוצה כדי להחזיר את האיזון בין הרשויות. בית המשפט העליון התערב יותר מדי בהחלטות הממשלה הנבחרת. העם בחר וזכותו לקבל את מה שהובטח לו בבחירות. השינוי הזה יביא לחיזוק הדמוקרטיה הישראלית.

המצב הביטחוני מצריך מנהיגות חזקה ויציבה. איראן מהווה איום קיומי על ישראל וצריך להתמודד איתה בכל האמצעים הנדרשים. חמאס וחיזבאללה הם שלוחים של איראן ויש לפעול נגדהם בתוקף. ביטחון ישראל חשוב מכל שיקול אחר.

הכלכלה הישראלית חזקה ויציבה למרות האתגרים. ההיטק הישראלי מוביל בעולם וממשיך לצמוח. יש לעודד יזמות וחדשנות ולהפחית רגולציה מיותרת. השקעות זרות ממשיכות להגיע למדינה.

חינוך הערכים חשוב לא פחות מהחינוך האקדמי. יש ללמד את ילדינו אהבת המדינה וההיסטוריה שלנו. הזהות היהודית והציונית היא הבסיס לקיומנו כאן. אנחנו חייבים לשמר את המורשת שלנו.
"""

# 1000-word Hebrew text
HEBREW_TEXT_1000 = """
המצב הביטחוני במזרח התיכון מחמיר מיום ליום והאיומים על ישראל הולכים וגוברים. איראן ממשיכה לפתח את התוכנית הגרעינית שלה ומאיימת להשמיד את ישראל. המשטר האיראני מממן ארגוני טרור ברחבי האזור ומנסה ליצור חזית רצופה נגד ישראל. זה מצב שישראל לא יכולה לעמוד בו ויש לפעול בתוקף.

החזית הצפונית עם לבנון וסוריה הפכה לאיום ישיר. חיזבאללה הצטייד בעשרות אלפי רקטות וטילים המכוונים לערים ישראליות. הארגון הטרוריסטי מנסה לבנות מנהרות התקפה ולחדור לשטח ישראלי. צה"ל פועל יום יום כדי לסכל את המזימות האלה ולהגן על אזרחי הצפון.

הרצועה הדרומית אינה פחות מאתגרת. חמאס המשיך לבנות את יכולותיו הצבאיות למרות המבצעים הקודמים. הארגון מפתח נשק חדש וחופר מנהרות התקפה חדשות. הגדר התת-קרקעית שישראל בונה נועדה למנוע את החדירות הללו, אבל המאמץ הטכנולוגי והכספי הוא עצום.

מול כל האתגרים הביטחוניים האלה, ישראל צריכה לחזק את הכוחות הביטחוניים שלה. ההשקעה בצה"ל ובמערכות הביטחון חייבת לגדול. טכנולוגיות חדשות כמו מערכות יירוט מתקדמות, מזל"טים אוטונומיים ומערכות סייבר מתקדמות הן קריטיות להגנה על המדינה.

הממד הכלכלי של הביטחון הוא משמעותי מאוד. תקציב הביטחון צובר ממדים גדולים והציבור נדרש להבין את החשיבות שבכך. ללא השקעה נאותה בביטחון, לא תהיה מדינה שתוכל לפרוח כלכלית וחברתיו. זו השקעה הכרחית לקיומנו כאן.

הקשרים הבינלאומיים הם חלק מהותי מהאסטרטגיה הביטחונית. ברית חזקה עם ארצות הברית היא יסוד מרכזי בביטחון ישראל. הסכמי אברהם פתחו צוות חדש של שיתופי פעולה עם מדינות ערב מתונות. הרחבת המעגל הזה חיונית להכלת האיום האיראני.

השקעה בחינוך ובמחקר היא גם חלק מהאסטרטגיה הביטחונית. מדענים ומהנדסים ישראליים מובילים בעולם בתחומי ההגנה והטכנולוגיה הצבאית. האוניברסיטאות הישראליות מפתחות טכנולוגיות מתקדמות המשרתות את הביטחון הלאומי. יש לחזק את התחומים האלה.

המורל והכוחשות הנפש של החברה הישראלית הם נכסים אסטרטגיים חשובים. עם ישראל הוכיח לאורך השנים עמידות ונחישות מול איומים. הרוח הלוחמת והנכונות להקרבה למען המדינה הן תכונות שיש לשמר ולטפח בדורות הבאים.

הקהילה הבינלאומית חייבת להבין את המציאות הביטחונית שבה חיה ישראל. המדינה מוקפת באויבים השואפים להשמידה ואין לה ברירה אלא להיות חזקה ונחושה. כל ויתור או חולשה נתפסים כהזמנה לתקיפה. זו מציאות קשה אבל זו המציאות.

למרות כל האתגרים, ישראל ממשיכה לפרוח ולהתפתח. הכלכלה הישראלית חזקה והחדשנות הטכנולוגית ממשיכה להוביל בעולם. תושבי ישראל יודעים לחיות במציאות מורכבת זו ולבנות עתיד טוב למדינה. זו עוצמתה של ישראל.

הפוליטיקה הפנימית לא צריכה להעמיד בסכנה את הביטחון הלאומי. המחלוקות הפוליטיות הן דבר טבעי בדמוקרטיה, אבל כשמדובר בביטחון המדינה יש לפעול באחדות. האויבים שלנו מנסים לנצל את הפילוגים הפנימיים ואסור לתת להם את היד.

חיזוק הרתיעה הישראלית הוא המפתח לשמירה על השקט. האויבים צריכים לדעת שכל תוקפנות תזכה למענה קשה ומיידי. ישראל לא תהסס להגן על עצמה ועל אזרחיה בכל האמצעים הנדרשים. זו המסר שצריך להעביר בבירור לכל האזור.

העתיד של ישראל תלוי ביכולתה לשמור על העדפה איכותית מול איומים כמותיים. צה"ל חייב להישאר הצבא המתקדם והמקצועי ביותר באזור. ההשקעה בכוח האדם, בטכנולוגיה ובחדשנות תקבע את העתיד הביטחוני של המדינה. זו המשימה החשובה ביותר עבורנו.
"""


async def initialize_model():
    """Initialize and set the global model."""
    print("🔧 Initializing SentenceTransformer model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    set_global_model(model)
    print("✅ Model initialized and set globally")
    return model


async def test_chunking_600_words():
    """Test text chunking with 600-word Hebrew text."""
    print("🔍 Testing Text Chunking with 600-word Hebrew Text")
    print("=" * 60)
    
    word_count = count_words(HEBREW_TEXT_600)
    print(f"📊 Text length: {word_count} words")
    
    # Test chunking logic
    chunks = chunk_text_by_words(HEBREW_TEXT_600, target_words=250, max_chunk_words=300)
    print(f"📋 Split into {len(chunks)} chunks:")
    
    for i, (chunk_text, chunk_words) in enumerate(chunks, 1):
        print(f"   Chunk {i}: {chunk_words} words")
        print(f"   Preview: {chunk_text[:100]}...")
        print()
    
    # Test with author name
    test_input = {
        "author": "בנימין נתניהו",
        "text": HEBREW_TEXT_600
    }
    
    print("🚀 Processing with fact-checking system...")
    start_time = time.time()
    
    try:
        result = await process_full_request(json.dumps(test_input, ensure_ascii=False))
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.1f} seconds")
        
        if result.get("is_chunked"):
            print(f"📊 Chunking Summary:")
            print(f"   Total chunks: {result['total_chunks']}")
            print(f"   Chunks processed: {result['chunking_summary']['chunks_processed']}")
            print(f"   Chunks failed: {result['chunking_summary']['chunks_failed']}")
            print(f"   Original word count: {result['original_word_count']}")
            
            print(f"📈 Analysis Summary:")
            print(f"   Events analyzed: {result['summary']['total_events_analyzed']}")
            print(f"   Fact checks completed: {result['summary']['fact_checks_completed']}")
            print(f"   RSS matches: {result['summary']['rss_matches']}")
            print(f"   Wikipedia characters: {result['summary']['wikipedia_characters']}")
            
            print(f"🔍 Fact Check Results: {len(result.get('fact_check_results', []))}")
            
            # Show sample fact check results
            fact_checks = result.get('fact_check_results', [])
            if fact_checks:
                print(f"\n📋 Sample Fact Check Results:")
                for i, fact_check in enumerate(fact_checks[:3], 1):  # Show first 3
                    chunk_idx = fact_check.get('chunk_index', 'N/A')
                    print(f"   {i}. From chunk {chunk_idx}: {fact_check.get('user_event_quote', '')[:50]}...")
        else:
            print("❓ Text was not chunked (unexpected for 600 words)")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    return result


async def test_chunking_1000_words():
    """Test text chunking with 1000-word Hebrew text."""
    print("\n🔍 Testing Text Chunking with 1000-word Hebrew Text")
    print("=" * 60)
    
    word_count = count_words(HEBREW_TEXT_1000)
    print(f"📊 Text length: {word_count} words")
    
    # Test chunking logic
    chunks = chunk_text_by_words(HEBREW_TEXT_1000, target_words=250, max_chunk_words=300)
    print(f"📋 Split into {len(chunks)} chunks:")
    
    for i, (chunk_text, chunk_words) in enumerate(chunks, 1):
        print(f"   Chunk {i}: {chunk_words} words")
        print(f"   Preview: {chunk_text[:100]}...")
        print()
    
    # Test with author name
    test_input = {
        "author": "יאיר לפיד",
        "text": HEBREW_TEXT_1000
    }
    
    print("🚀 Processing with fact-checking system...")
    start_time = time.time()
    
    try:
        result = await process_full_request(json.dumps(test_input, ensure_ascii=False))
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.1f} seconds")
        
        if result.get("is_chunked"):
            print(f"📊 Chunking Summary:")
            print(f"   Total chunks: {result['total_chunks']}")
            print(f"   Chunks processed: {result['chunking_summary']['chunks_processed']}")
            print(f"   Chunks failed: {result['chunking_summary']['chunks_failed']}")
            print(f"   Original word count: {result['original_word_count']}")
            
            print(f"📈 Analysis Summary:")
            print(f"   Events analyzed: {result['summary']['total_events_analyzed']}")
            print(f"   Fact checks completed: {result['summary']['fact_checks_completed']}")
            print(f"   RSS matches: {result['summary']['rss_matches']}")
            print(f"   Wikipedia characters: {result['summary']['wikipedia_characters']}")
            
            print(f"🔍 Fact Check Results: {len(result.get('fact_check_results', []))}")
            
            # Show sample fact check results
            fact_checks = result.get('fact_check_results', [])
            if fact_checks:
                print(f"\n📋 Sample Fact Check Results:")
                for i, fact_check in enumerate(fact_checks[:5], 1):  # Show first 5
                    chunk_idx = fact_check.get('chunk_index', 'N/A')
                    print(f"   {i}. From chunk {chunk_idx}: {fact_check.get('user_event_quote', '')[:50]}...")
        else:
            print("❓ Text was not chunked (unexpected for 1000 words)")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    return result


async def main():
    """Run both tests."""
    print("🧪 Text Chunking Test Suite")
    print("Testing Hebrew text processing with chunking functionality")
    print()
    
    # Initialize model first
    await initialize_model()
    print()
    
    # Test 1: 600 words
    result_600 = await test_chunking_600_words()
    
    # Test 2: 1000 words  
    result_1000 = await test_chunking_1000_words()
    
    # Summary
    print("\n🏁 Test Summary")
    print("=" * 40)
    
    if result_600 and result_600.get("is_chunked"):
        print(f"✅ 600-word test: Successfully chunked into {result_600['total_chunks']} parts")
    else:
        print("❌ 600-word test: Failed or not chunked")
    
    if result_1000 and result_1000.get("is_chunked"):
        print(f"✅ 1000-word test: Successfully chunked into {result_1000['total_chunks']} parts")
    else:
        print("❌ 1000-word test: Failed or not chunked")
    
    print("\n📝 Both tests completed!")


if __name__ == "__main__":
    asyncio.run(main()) 
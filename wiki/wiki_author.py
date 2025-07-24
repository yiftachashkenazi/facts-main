import wikipedia
import logging

# הגדרת שפת ברירת מחדל
wikipedia.set_lang("he")

# לוגים (אם רוצים להרחיב בהמשך)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_author_wiki_summary(name: str) -> dict:
    """
    מביא תקציר ושם דף ויקיפדיה עבור שם המחבר.
    :param name: שם המחבר
    :return: dict עם תקציר, כותרת, כתובת URL, סטטוס
    """
    try:
        results = wikipedia.search(name, results=3)
        if not results:
            return {
                "status": "not_found",
                "summary": "",
                "title": "",
                "url": ""
            }

        page = wikipedia.page(results[0])
        return {
            "status": "success",
            "summary": page.summary,
            "title": page.title,
            "url": page.url
        }

    except wikipedia.exceptions.DisambiguationError as e:
        return {
            "status": "disambiguation",
            "summary": "",
            "title": name,
            "url": "",
            "options": e.options[:5]
        }

    except wikipedia.exceptions.PageError:
        return {
            "status": "not_found",
            "summary": "",
            "title": "",
            "url": ""
        }

    except Exception as e:
        return {
            "status": "error",
            "summary": "",
            "title": "",
            "url": "",
            "error": str(e)
        }
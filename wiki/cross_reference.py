import re
from datetime import datetime

def find_paragraphs_with_cross_references(author_wiki: dict, wiki_results: dict) -> list:
    """
    הצלבה בין תוכן הערך של המחבר לבין תוכן של ערכים אחרים.
    מחזיר פסקאות שמופיע בהן גם שמו של המחבר או שהמחבר מופיע בתוכן הערכים האחרים.
    """
    if author_wiki.get("status") != "success":
        return []

    author_name = author_wiki.get("title", "").strip()
    author_content = author_wiki.get("content", "")
    related_paragraphs = []

    for name, info in wiki_results.items():
        if info.get("status") != "success":
            continue

        content = info.get("content", "")
        paragraphs = content.split("\n\n")
        for para in paragraphs:
            if re.search(re.escape(author_name), para, re.IGNORECASE) or \
               re.search(re.escape(name), author_content, re.IGNORECASE):
                related_paragraphs.append({
                    "related_to": name,
                    "source_page": info.get("title", name),
                    "paragraph": para.strip()
                })

    return related_paragraphs


def find_recent_paragraphs_from_all(wiki_results: dict) -> list:
    """
    מוצא פסקאות שמכילות את השנה הנוכחית בכל ערך ויקיפדיה.
    """
    current_year = str(datetime.now().year)
    pattern = re.compile(fr"\b{current_year}\b")

    recent_paragraphs = []
    for name, info in wiki_results.items():
        if info.get("status") != "success":
            continue

        content = info.get("content", "")
        paragraphs = content.split("\n\n")
        for para in paragraphs:
            if pattern.search(para):
                recent_paragraphs.append({
                    "related_to": name,
                    "source_page": info.get("title", name),
                    "paragraph": para.strip()
                })

    return recent_paragraphs
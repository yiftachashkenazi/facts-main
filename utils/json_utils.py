import json
import re

def extract_json_block(text: str) -> str:
    """
    Return only the substring from the first '{' to the last '}' inclusive.
    If no such braces exist, return the stripped original text.
    Useful when Gemini prepends explanations or code fences around JSON.
    """
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text.strip()

def parse_json_any(text: str):
    """
    Robustly parse JSON from potentially malformed or concatenated content.
    Handles:
      • A proper JSON array or object
      • Multiple JSON objects separated by '},{' or '},\n{'
      • Trailing commas inside arrays/objects
    Returns:
      • a dict  – single object
      • a list  – multiple objects
    """
    text = text.strip()
    # ---- If we detect several root objects separated by commas, wrap into [...] ----
    if text.startswith("{") and re.search(r"}\s*,\s*{", text):
        text_wrapped = f"[{text}]"
    else:
        text_wrapped = text

    try:
        return json.loads(text_wrapped)
    except json.JSONDecodeError:
        pass   # fall through to incremental parsing

    # Attempt to parse multiple concatenated JSON objects/arrays
    decoder = json.JSONDecoder()
    objs = []
    s = text
    while s:
        s = s.lstrip()
        if not s:
            break
        try:
            obj, idx = decoder.raw_decode(s)
            objs.append(obj)
            s = s[idx:]
        except json.JSONDecodeError:
            break
    if objs:
        return objs[0] if len(objs) == 1 else objs
    # Last‑ditch: remove dangling commas and retry
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(cleaned) 
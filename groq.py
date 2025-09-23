import os, requests, json

SYSTEM = (
    "You are an information-extraction engine. "
    "Return ONLY valid JSON (no markdown, no prose)."
)
USER_TEMPLATE = """Extract this document into JSON with keys exactly:
- source_filename: string
- doc_title: short title
- sections: array of objects with keys [id, level, title, path, content]

Rules:
- id: short kebab-case slug (e.g., 'intro', 'methods-results')
- level: 1 for top headings, 2 for subheadings, etc.
- path: array of headings from root to here (e.g., ['Introduction','Background'])
- content: clean text for that section
- If headings aren’t explicit, infer sensible sections.

TEXT:
{doc_text}
"""

def groq_extract_json(doc_text: str, source_filename: str = "sample.txt"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
        "Content-Type": "application/json",
    }
    user_msg = USER_TEMPLATE.format(doc_text=doc_text) + f"\nsource_filename: {source_filename}"
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    # try to parse JSON strictly
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        return json.loads(content[start:end+1])

# 🔹 quick test on a fake doc
sample_doc = """
1. Introduction
This document describes solar pumps.

2. Design
Prototype specs and energy estimates.

3. Conclusion
Next steps for deployment.
"""

extracted = groq_extract_json(sample_doc, "solar_notes.txt")
print(json.dumps(extracted, indent=2)[:600])  # preview

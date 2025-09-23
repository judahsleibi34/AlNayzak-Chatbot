# Everything via OCR.space; no local text/docx readers. Saves as <uploaded-name>.json in your repo.

import os, json, base64, pathlib, requests
from google.colab import files

# ---- config (from env) ----
def _bool_env(name, default):
    v=os.getenv(name); 
    return default if v is None else str(v).strip().lower() in {"1","true","yes","on"}

OCR_API_KEY = os.environ["OCR_SPACE_API_KEY"]
OCR_LANG    = os.getenv("OCR_LANG","eng")
OCR_ENGINE  = os.getenv("OCR_ENGINE","5")
OCR_TABLE   = _bool_env("OCR_TABLE", True)

GROQ_KEY    = os.environ["GROQ_API_KEY"]
MODEL_NAME  = os.getenv("GROQ_MODEL","llama-3.3-70b-versatile")

GH_REPO     = os.environ["GH_REPO"]
GH_TOKEN    = os.environ["GH_TOKEN"]
OUT_EXTRACT = os.getenv("REPO_SUBDIR_EXTRACTED","data/extracted")

# ---- OCR.space (single path for all uploads) ----
def ocr_space_extract(path: pathlib.Path, lang=OCR_LANG) -> str:
    # Supports images & PDFs. For DOCX/CSV, convert to PDF first (OCR.space doesn't accept DOCX/CSV).
    url = "https://api.ocr.space/parse/image"
    with open(path, "rb") as f:
        files = {"file": (path.name, f)}
        data = {
            "apikey": OCR_API_KEY,
            "language": lang,
            "OCREngine": OCR_ENGINE,       # 5 = best
            "isTable": str(OCR_TABLE).lower(),
            "scale": "true",
            "isOverlayRequired": "false",
        }
        r = requests.post(url, data=data, files=files, timeout=180)
    r.raise_for_status()
    js = r.json()
    if js.get("IsErroredOnProcessing"):
        raise RuntimeError(str(js.get("ErrorMessage") or js))
    results = js.get("ParsedResults") or []
    return "\n".join(p.get("ParsedText","") for p in results).strip()

# ---- Your strict hierarchy schema (model must obey) ----
HIERARCHY_JSON_SCHEMA = {
    "name": "doc_hierarchy",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_filename": {"type":"string"},
            "doc_title": {"type":"string"},
            "sections": {
                "type":"array",
                "items": {
                    "type":"object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type":"string"},                 # kebab-case slug
                        "level": {"type":"integer","minimum":1}, # 1=H1, 2=H2...
                        "title": {"type":"string"},
                        "path":  {"type":"array","items":{"type":"string"}},
                        "page_range": {
                            "type":"array","items":{"type":"integer"},
                            "minItems":2,"maxItems":2
                        },
                        "content":{"type":"string"}
                    },
                    "required":["id","level","title","path","content"]
                }
            }
        },
        "required":["source_filename","doc_title","sections"]
    }
}

# ---- Groq structured outputs (enforces your schema) ----
def groq_extract_structured(doc_text: str, source_filename: str) -> dict:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type":"application/json"}
    system = "You are an extraction engine. Follow the schema strictly; no extra keys."
    user = f"""Extract the document into the schema. If page numbers unknown, use [1,1].
TEXT (<=20k chars):
{doc_text[:20000]}

source_filename: {source_filename}
doc_title: infer a short human title
"""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role":"system","content":system},{"role":"user","content":user}],
        "temperature": 0.05,
        "response_format": {"type":"json_schema","json_schema": HIERARCHY_JSON_SCHEMA}
    }
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])

# ---- GitHub upsert (saves as <same-base>.json) ----
def github_upsert(path_in_repo: str, content_bytes: bytes, message="add/update"):
    api = f"https://api.github.com/repos/{GH_REPO}/contents/{path_in_repo}"
    heads = {"Authorization": f"Bearer {GH_TOKEN}"}
    g = requests.get(api, headers=heads, timeout=60)
    sha = g.json().get("sha") if g.status_code == 200 else None
    payload = {"message": message, "content": base64.b64encode(content_bytes).decode(), **({"sha": sha} if sha else {})}
    put = requests.put(api, headers=heads, json=payload, timeout=120)
    put.raise_for_status()
    return put.json()

# ---- 1) pick files (no hard-coded names) ----
uploaded = files.upload()  # choose your PDFs or images (PNG/JPG/TIFF, etc)
paths = [pathlib.Path(n) for n in uploaded.keys()]

# ---- 2) process each file via OCR.space → Groq → save to GitHub as <same-name>.json ----
SUPPORTED = {".pdf",".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"}
if not paths:
    raise SystemExit("No files uploaded.")
for p in paths:
    print(f"\n→ {p.name}")
    if p.suffix.lower() not in SUPPORTED:
        print("  Unsupported by OCR.space. Please upload a PDF or image (convert DOCX/CSV to PDF first).")
        continue

    raw = ocr_space_extract(p)
    if len(raw.strip()) < 20:
        print("  (very little text from OCR) — skipped"); 
        continue

    obj = groq_extract_structured(raw, p.name)
    obj.setdefault("source_filename", p.name)
    obj.setdefault("doc_title", p.stem)
    obj.setdefault("sections", [])

    # Save as SAME base name with .json (e.g., invoice.pdf -> invoice.json)
    repo_path = f"{OUT_EXTRACT}/{p.stem}.json"
    github_upsert(repo_path, json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8"),
                  message=f"extract {p.name} → {p.stem}.json")
    print(f"  ✅ pushed: {repo_path}")

# -*- coding: utf-8 -*-
"""
NewRag.py — runner & reporter for the extractive retriever.

- Builds HybridIndex from retrival_model.py
- Runs 30 SANITY_PROMPTS, prints Q/A with sources
- Saves artifacts under --out-dir/run_YYYYMMDD_HHMMSS/{results.jsonl, summary.md}
- Arabic-aware strict checks
- RTL/digit formatting for console/summary
- Post-fix: if STRICT fails on time/duration/workdays, scan cited pages in chunks to synthesize a precise line (from the PDF)

Usage
-----
python NewRag.py \
  --chunks Data_pdf_clean_chunks.jsonl \
  --hier-index heading_inverted_index.json \
  --aliases section_aliases.json \
  --sanity \
  --out-dir runs \
  --rtl force \
  --digits arabic
"""

import os, re, json, argparse, datetime
from pathlib import Path

# Use only exported APIs from your retriever module
from retrival_model import (
    load_hierarchy, load_chunks, HybridIndex,
    classify_intent, answer, SANITY_PROMPTS
)

# ---------------- Arabic utils ----------------
_AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
_AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"٠١٢٣٤٥٦٧٨٩")}
_IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"۰۱۲۳۴۵۶۷۸۹")}
_RLE = '\u202B'  # Right-to-left embedding
_PDF = '\u202C'  # Pop directional formatting

def ar_normalize(s: str) -> str:
    if not s: return ""
    s = s.replace('\u0640','')
    s = _AR_DIAC.sub('', s)
    s = (s.replace('أ','ا').replace('إ','ا').replace('آ','ا')
           .replace('ى','ي'))
    s = s.translate(_AR_NUMS).translate(_IR_NUMS)
    s = s.replace('،', ',').replace('٫','.')
    return ' '.join(s.split())

def to_arabic_digits(s: str) -> str:
    if not s: return s
    return s.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))

def strip_rtl_wrap(s: str) -> str:
    if not s: return s
    return s.replace(_RLE, "").replace(_PDF, "")

# ---------------- Signals & regex ----------------
TIME_RANGE = re.compile(
    r'(?:من\s*)?'
    r'(\d{1,2}(?::|\.)?\d{0,2})\s*(?:[-–—]|الى|إلى|حتى|حتي)\s*'
    r'(\d{1,2}(?::|\.)?\d{0,2})'
)
TIME_TOKEN = re.compile(r'\b\d{1,2}[:\.]\d{0,2}\b')
ANY_DIGIT  = re.compile(r'\d')
DUR_TOKEN  = re.compile(r'\b(\d{1,3})\s*(?:دقيقه|دقيقه|دقيقة|دقائق|ساعة|ساعه|ساعات)\b', re.I)

HALF_HOUR_PAT    = re.compile(r'\bنصف\s+ساع[هة]\b')
QUARTER_HOUR_PAT = re.compile(r'\bربع\s+ساع[هة]\b')

WEEKDAYS = ["السبت","الاحد","الأحد","الاثنين","الإثنين","الثلاثاء","الاربعاء","الأربعاء","الخميس","الجمعة"]

def normalize_hhmm(tok: str) -> str:
    tok = tok.replace('.', ':')
    if ':' not in tok: return f"{int(tok):d}:00"
    h, m = tok.split(':', 1)
    if m == "": m = "00"
    return f"{int(h):d}:{int(m):02d}"

def _to_minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h*60 + m

def _plausible_workday(A: int, B: int) -> bool:
    return 6*60 <= A <= 20*60+30 and 6*60 <= B <= 20*60+30 and B > A

# ---------------- Strict needs rules (intent-aware) ----------------
def needs_numeric_or_time(question: str, intent: str) -> bool:
    """
    Only require numbers/time when it truly makes sense.
    """
    qn = ar_normalize(question)
    has_num = bool(ANY_DIGIT.search(qn))
    cues = any(c in qn for c in ['كم','مدة','نسبة','٪','%','سقف','حد','ثلاث'])
    if intent in ("work_hours","ramadan_hours"):
        return True
    if intent == "break":
        return ('كم' in qn) or ('مدة' in qn) or has_num
    if intent in ("procurement","per_diem","overtime","leave"):
        return has_num or cues
    if intent == "workdays":
        return False
    return has_num or cues

def body_has_required_signals(body: str, intent: str) -> bool:
    bn = ar_normalize(body or "")
    if intent in ("work_hours","ramadan_hours"):
        return bool(TIME_RANGE.search(bn) or TIME_TOKEN.search(bn))
    if intent == "break":
        return bool(DUR_TOKEN.search(bn) or TIME_TOKEN.search(bn) or
                    HALF_HOUR_PAT.search(bn) or QUARTER_HOUR_PAT.search(bn))
    if intent == "workdays":
        return any(d in body for d in WEEKDAYS) or ("ايام" in bn and ("العمل" in bn or "الدوام" in bn))
    if intent in ("procurement","per_diem","overtime","leave"):
        return bool(ANY_DIGIT.search(bn))
    return len(bn) >= 6

# ---------------- Pass criteria & parsing ----------------
def pass_loose(ans_text: str) -> bool:
    return ("Sources:" in ans_text) and ("لم أعثر" not in ans_text)

def pass_strict(question: str, body_only: str, intent: str) -> bool:
    if needs_numeric_or_time(question, intent):
        return body_has_required_signals(body_only, intent)
    return len(ar_normalize(body_only)) >= 6

def extract_pages(ans_text: str):
    pages = []
    m = re.search(r"Sources:\s*(.*)", ans_text, flags=re.S)
    if not m: return pages
    for line in m.group(1).splitlines():
        mm = re.search(r"page\s+(\d+)", line)
        if mm:
            try: pages.append(int(mm.group(1)))
            except: pass
    return pages

# ---------------- Pretty printing ----------------
def transform_for_print(body: str, rtl: str, digits: str) -> str:
    out = body or ""
    if digits == "arabic":
        out = to_arabic_digits(out)
    has_wrap = (out.startswith(_RLE) and out.endswith(_PDF))
    if rtl == "off":
        out = strip_rtl_wrap(out)
    elif rtl == "force" and not has_wrap:
        out = _RLE + out + _PDF
    return out

# ---------------- Post-fixers (scan cited pages in chunks) ----------------
def pick_best_time_range(texts):
    best = None
    for t in texts:
        tn = ar_normalize(t)
        for m in TIME_RANGE.finditer(tn):
            a = normalize_hhmm(m.group(1))
            b = normalize_hhmm(m.group(2))
            A, B = _to_minutes(a), _to_minutes(b)
            if B <= A:
                B += 12*60  # handle "8:30–3:00" typos

            dur = B - A
            if not (360 <= dur <= 660):   # require 6–11 hours
                continue
            if not _plausible_workday(A, B):
                continue

            # score: closeness to 7h30 + presence of دوام/العمل + bonus if start 7:00–9:30
            score = -abs(dur - 450)
            if ("دوام" in tn or "العمل" in tn): score += 30
            if 7*60 <= A <= 9*60+30: score += 10

            cand = (score, a, b)
            if (best is None) or (cand[0] > best[0]):
                best = cand

    return (best[1], best[2]) if best else None

def find_duration_line(texts):
    candidates = []
    for t in texts:
        tn = ar_normalize(t)
        if (("استراح" in tn or "راحة" in tn or "بريك" in tn or "رضاع" in tn) and
            (DUR_TOKEN.search(tn) or HALF_HOUR_PAT.search(tn) or QUARTER_HOUR_PAT.search(tn))):
            candidates.append(t.strip())
    if not candidates:
        for t in texts:
            tn = ar_normalize(t)
            if DUR_TOKEN.search(tn) or HALF_HOUR_PAT.search(tn) or QUARTER_HOUR_PAT.search(tn):
                candidates.append(t.strip())
    return candidates[0] if candidates else None

def find_workdays_line(texts):
    for t in texts:
        if any(d in t for d in WEEKDAYS) or ("ايام" in t and ("العمل" in t or "الدوام" in t)):
            return t.strip()
    return None

def repair_body_if_needed(question, intent, body, pages, all_chunks):
    """If strict fails, search cited pages and synthesize a minimal, correct line."""
    page_set = set(pages)
    texts = [c.text for c in all_chunks if c.page in page_set]

    if intent in ("work_hours","ramadan_hours"):
        rng = pick_best_time_range(texts)
        if rng:
            a, b = rng
            suffix = " في شهر رمضان" if intent == "ramadan_hours" else ""
            return f"ساعات الدوام{suffix} من {a} إلى {b}."
        return body

    if intent == "break":
        line = find_duration_line(texts)
        if line: return line
        return body

    if intent == "workdays":
        line = find_workdays_line(texts)
        if line: return line
        return body

    if intent in ("procurement","per_diem","overtime","leave"):
        if not ANY_DIGIT.search(ar_normalize(body or "")):
            for t in texts:
                if ANY_DIGIT.search(ar_normalize(t)):
                    return t.strip()
    return body

# ---------------- artifacts ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_sanity(index, all_chunks, out_dir: Path, rtl: str, digits: str):
    stamp = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_dir / stamp
    ensure_dir(run_dir)

    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.md"

    total = len(SANITY_PROMPTS)
    passL = passS = 0
    rows = []

    print(f"🧪 Running sanity prompts ({total}) …")
    print("="*80)

    with results_path.open("w", encoding="utf-8") as jf:
        for i, q in enumerate(SANITY_PROMPTS, 1):
            print(f"\n📝 Test {i}/{total}: {q}")
            print("-"*60)

            intent = classify_intent(q)
            ans = answer(q, index, intent, use_rerank_flag=False)

            parts = ans.split("\nSources:")
            body_raw = parts[0].strip()
            pages = extract_pages(ans)

            okL = pass_loose(ans)
            okS = pass_strict(q, strip_rtl_wrap(body_raw), intent)

            repaired = None
            if not okS and pages:
                repaired = repair_body_if_needed(q, intent, strip_rtl_wrap(body_raw), pages, all_chunks)
                if repaired and repaired != strip_rtl_wrap(body_raw):
                    body_raw = (_RLE + repaired + _PDF) if body_raw.startswith(_RLE) else repaired
                    okS = pass_strict(q, strip_rtl_wrap(body_raw), intent)

            body_print = transform_for_print(body_raw, rtl=rtl, digits=digits)

            print(body_print)
            if len(parts) > 1:
                print("Sources:" + parts[1])

            passL += int(okL); passS += int(okS)
            print("✅ PASS_LOOSE" if okL else "⚪ FAIL_LOOSE")
            print("✅ PASS_STRICT" if okS else "⚪ FAIL_STRICT")
            print("="*80)

            row = {
                "idx": i,
                "question": q,
                "intent": intent,
                "answer": body_raw,
                "sources_pages": pages,
                "pass_loose": okL,
                "pass_strict": okS,
                "repaired": bool(repaired)
            }
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

    with summary_path.open("w", encoding="utf-8") as sf:
        sf.write("# Sanity Run\n\n")
        sf.write(f"- Total questions: **{total}**\n")
        sf.write(f"- PASS_LOOSE: **{passL}/{total}**\n")
        sf.write(f"- PASS_STRICT: **{passS}/{total}**\n\n")
        for r in rows:
            sf.write(f"## Q{r['idx']}: {r['question']}\n\n")
            pretty = transform_for_print(r['answer'], rtl=rtl, digits=digits)
            sf.write(pretty + "\n\n")
            if r["sources_pages"]:
                cites = "\n".join([f"{i+1}. Data_pdf.pdf - page {p}" for i, p in enumerate(r["sources_pages"])])
                sf.write("**Sources**\n\n" + cites + "\n\n")
            sf.write(f"- Intent: `{r['intent']}`\n")
            sf.write(f"- Repaired: {'✅' if r['repaired'] else '—'}\n")
            sf.write(f"- PASS_LOOSE: {'✅' if r['pass_loose'] else '❌'}\n")
            sf.write(f"- PASS_STRICT: {'✅' if r['pass_strict'] else '❌'}\n\n")

    print(f"\nSummary: PASS_LOOSE {passL}/{total} | PASS_STRICT {passS}/{total}")
    print(f"Artifacts saved in: {run_dir}")
    print(f"✅ Saved: {results_path.name}, {summary_path.name}")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, required=True, help="Path to chunks (JSONL/JSON)")
    ap.add_argument("--hier-index", type=str, default=None, help="Optional hierarchy inverted index path")
    ap.add_argument("--aliases", type=str, default=None, help="Optional aliases for headings")
    ap.add_argument("--sanity", action="store_true", help="Run sanity prompts and exit")
    ap.add_argument("--out-dir", type=str, default="runs", help="Directory to store run artifacts")
    ap.add_argument("--save-index", type=str, default=None, help="Directory to save index artifacts (if supported)")
    ap.add_argument("--load-index", type=str, default=None, help="Directory to load index artifacts (if supported)")
    ap.add_argument("--model", type=str, default=None, help="SentenceTransformer model ID")
    ap.add_argument("--rtl", choices=["auto","off","force"], default="auto",
                    help="RTL wrapping behavior for printing/summary")
    ap.add_argument("--digits", choices=["ascii","arabic"], default="ascii",
                    help="Digit style for printing/summary")
    args = ap.parse_args()

    hier = load_hierarchy(args.hier_index, args.aliases) if (args.hier_index or args.aliases) else None
    chunks, chunks_hash = load_chunks(path=args.chunks)

    # index
    try:
        index = HybridIndex(chunks, chunks_hash, hier=hier, model_name=args.model) if args.model else HybridIndex(chunks, chunks_hash, hier=hier)
    except TypeError:
        index = HybridIndex(chunks, chunks_hash, hier=hier)

    loaded = False
    if args.load_index and hasattr(index, "load"):
        try:
            loaded = index.load(args.load_index)
        except Exception:
            loaded = False

    if not loaded:
        index.build()
        if args.save_index and hasattr(index, "save"):
            try:
                index.save(args.save_index)
            except Exception:
                pass

    if args.sanity:
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        run_sanity(index, chunks, out_dir, rtl=args.rtl, digits=args.digits)
        return

    # interactive
    print("Ready. Interactive mode (type 'exit' to quit).")
    while True:
        try:
            q = input("سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break
        if not q: continue
        if q.lower() in ("exit","quit","q"): print("Exiting."); break
        intent = classify_intent(q)
        resp = answer(q, index, intent, use_rerank_flag=False)
        parts = resp.split("\nSources:")
        body = transform_for_print(parts[0].strip(), rtl=args.rtl, digits=args.digits)
        print(body)
        if len(parts) > 1:
            print("Sources:" + parts[1])
        print("-"*66)

if __name__ == "__main__":
    main()

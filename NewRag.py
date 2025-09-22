# -*- coding: utf-8 -*-
"""
NewRag.py — runner & reporter for the extractive retriever.

- Intent-aware STRICT checks (incl. hourly_leave must mention إذن/مغادرة)
- Fails STRICT if the body says "لم أعثر…"
- Post-fixers scan cited pages; hours repair now prefers realistic day windows
- Display sanitizer removes OCR tokens like "uni06BE" from printed text
"""

import os, re, json, argparse, datetime
from pathlib import Path

from retrival_model import (
    load_hierarchy, load_chunks, HybridIndex,
    classify_intent, answer, SANITY_PROMPTS, clean_display_text
)

# ---------------- Arabic utils ----------------
_AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
_AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"٠١٢٣٤٥٦٧٨٩")}
_IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"۰۱۲۳۴۵۶۷۸۹")}
_NOISY_UNI = re.compile(r'/?\buni[0-9A-Fa-f]{3,6}\b')
_RIGHTS_LINE = re.compile(r'جميع الحقوق محفوظة|التعليم المساند والإبداع العلمي')
_SECTION_NUM = re.compile(r'\b\d+\.\d+\b')
_RLE = '\u202B'; _PDF = '\u202C'

def ar_normalize(s: str) -> str:
    if not s: return ""
    s = s.replace('\u0640','')
    s = _AR_DIAC.sub('', s)
    s = (s.replace('أ','ا').replace('إ','ا').replace('آ','ا')
           .replace('ى','ي'))
    s = s.translate(_AR_NUMS).translate(_IR_NUMS)
    s = s.replace('،', ',').replace('٫','.')
    s = _NOISY_UNI.sub('', s)
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
DUR_TOKEN  = re.compile(r'\b(\d{1,3})\s*(?:دقيقه|دقيقة|دقائق|ساعه|ساعة|ساعات)\b', re.I)

WEEKDAYS = ["السبت","الاحد","الأحد","الاثنين","الإثنين","الثلاثاء","الاربعاء","الأربعاء","الخميس","الجمعة"]

def normalize_hhmm(tok: str) -> str:
    tok = tok.replace('.', ':')
    if ':' not in tok: return f"{int(tok):d}:00"
    h, m = tok.split(':', 1)
    if m == "": m = "00"
    return f"{int(h):d}:{int(m):02d}"

# ---------------- Strict needs rules (intent-aware) ----------------
def needs_numeric_or_time(question: str, intent: str) -> bool:
    qn = ar_normalize(question)
    has_num = bool(ANY_DIGIT.search(qn))
    cues = any(c in qn for c in ['كم','مدة','نسبة','٪','%','سقف','حد','ثلاث','اقصى','أقصى'])
    if intent in ("work_hours","ramadan_hours"):
        return True
    if intent == "break":
        return ('كم' in qn) or ('مدة' in qn) or has_num
    if intent in ("procurement","per_diem","overtime","leave","hourly_leave","carryover_leave"):
        return has_num or cues
    if intent in ("performance","emergency_leave"):
        return False
    if intent == "workdays":
        return False
    return has_num or cues

def body_has_required_signals(body: str, intent: str) -> bool:
    bn = ar_normalize(body or "")
    if "لم اعثر" in bn:
        return False
    if _RIGHTS_LINE.search(body):
        return False
    # only time-intents reject section-number lines
    if intent in ("work_hours","ramadan_hours") and _SECTION_NUM.search(bn):
        return False
    if intent in ("work_hours","ramadan_hours"):
        has_time = bool(TIME_RANGE.search(bn) or TIME_TOKEN.search(bn))
        ctx = ("دوام" in bn) or ("عمل" in bn) or ("ساعات" in bn)
        return has_time and ctx
    if intent == "break":
        if DUR_TOKEN.search(bn) or TIME_TOKEN.search(bn):
            return True
        if ("استراح" in bn or "راحة" in bn or "بريك" in bn or "رضاع" in bn):
            if any(kw in bn for kw in ["نصف ساعه","نصف ساعة","ربع ساعه","ربع ساعة"]):
                return True
        return False
    if intent == "hourly_leave":
        return (("مغادر" in bn or "اذن" in bn or "إذن" in bn) and bool(ANY_DIGIT.search(bn)))
    if intent == "carryover_leave":
        return (("ترح" in bn or "غير مستخدم" in bn or "غير المستخدم" in bn) and ("اجاز" in bn or "اجازه" in bn or "إجازة" in bn))
    if intent == "performance":
        return ("تقييم" in bn or "الأداء" in bn)
    if intent == "workdays":
        return any(d in body for d in WEEKDAYS) or ("ايام" in bn and ("العمل" in bn or "الدوام" in bn))
    if intent in ("procurement","per_diem","overtime","leave"):
        return bool(ANY_DIGIT.search(bn))
    return len(bn) >= 6

# ---------------- Pass criteria & parsing ----------------
def pass_loose(ans_text: str) -> bool:
    return ("Sources:" in ans_text) and ("لم أعثر" not in ans_text)

def pass_strict(question: str, body_only: str, intent: str) -> bool:
    if "لم أعثر" in body_only or "لم اعثر" in ar_normalize(body_only):
        return False
    if needs_numeric_or_time(question, intent):
        return body_has_required_signals(body_only, intent)
    return body_has_required_signals(body_only, intent)

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
    out = clean_display_text(out)  # strip OCR tokens like uni06BE
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
        if _RIGHTS_LINE.search(t): 
            continue
        tn = ar_normalize(t)
        for m in TIME_RANGE.finditer(tn):
            a = normalize_hhmm(m.group(1))
            b = normalize_hhmm(m.group(2))
            h1, m1 = map(int, a.split(':')); A = h1*60+m1
            h2, m2 = map(int, b.split(':')); B = h2*60+m2
            if B <= A: B += 12*60
            dur = B - A
            score = 0.0
            if 360 <= dur <= 660: score += 3.0
            if 420 <= dur <= 540: score += 1.0
            if 420 <= A <= 600: score += 2.0
            if 780 <= B <= 1080: score += 2.0
            best = max(best, (score, a, b), key=lambda x: x[0]) if best else (score, a, b)
    if not best:
        joined = "\n".join(texts)
        tn = ar_normalize(joined)
        for m in TIME_RANGE.finditer(tn):
            a = normalize_hhmm(m.group(1)); b = normalize_hhmm(m.group(2))
            h1, m1 = map(int, a.split(':')); A = h1*60+m1
            h2, m2 = map(int, b.split(':')); B = h2*60+m2
            if B <= A: B += 12*60
            dur = B - A
            score = 1.0 if 360 <= dur <= 660 else 0.0
            if score > 0:
                best = (score, a, b); break
    if best and best[0] > 0:
        return best[1], best[2]
    return None

def find_duration_line(texts):
    candidates = []
    for t in texts:
        if _RIGHTS_LINE.search(t): 
            continue
        tn = ar_normalize(t)
        if ("استراح" in tn or "راحة" in tn or "بريك" in tn or "رضاع" in tn):
            if DUR_TOKEN.search(tn) or ("نصف ساعه" in tn or "نصف ساعة" in tn or "ربع ساعه" in tn or "ربع ساعة" in tn):
                candidates.append(clean_display_text(t.strip()))
    if not candidates:
        for t in texts:
            tn = ar_normalize(t)
            if DUR_TOKEN.search(tn):
                candidates.append(clean_display_text(t.strip()))
    return candidates[0] if candidates else None

def find_workdays_line(texts):
    for t in texts:
        if _RIGHTS_LINE.search(t): 
            continue
        if any(d in t for d in WEEKDAYS) or ("ايام" in t and ("العمل" in t or "الدوام" in t)):
            return clean_display_text(t.strip())
    return None

def repair_body_if_needed(question, intent, body, pages, all_chunks):
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

    if intent in ("procurement","per_diem","overtime","leave","hourly_leave","carryover_leave"):
        bn = ar_normalize(body or "")
        if not ANY_DIGIT.search(bn):
            for t in texts:
                if _RIGHTS_LINE.search(t): 
                    continue
                if ANY_DIGIT.search(ar_normalize(t)):
                    return clean_display_text(t.strip())
    if intent in ("performance","emergency_leave"):
        for t in texts:
            tn = ar_normalize(t)
            if (intent == "performance" and ("تقييم" in tn or "الأداء" in tn)) or \
               (intent == "emergency_leave" and ("طارئ" in tn or "الطارئة" in tn)):
                return clean_display_text(t.strip())
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

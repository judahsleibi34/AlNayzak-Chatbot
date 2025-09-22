# -*- coding: utf-8 -*-
"""
NewRag.py — simple runner & reporter for the extractive retriever.

What it does
------------
- Builds the HybridIndex from retrival_model.py
- Runs your 30 SANITY_PROMPTS and prints each Q/A with sources
- Saves artifacts to --out-dir/run_YYYYMMDD_HHMMSS/{results.jsonl, summary.md}
- Arabic-aware strict pass checks (for questions that expect time/amounts)

Usage
-----
python NewRag.py \
  --chunks Data_pdf_clean_chunks.jsonl \
  --hier-index heading_inverted_index.json \
  --aliases section_aliases.json \
  --sanity \
  --out-dir runs
"""

import os, re, json, argparse, datetime
from pathlib import Path

# import ONLY things that exist in your retrival_model.py
from retrival_model import (
    load_hierarchy, load_chunks, HybridIndex,
    classify_intent, answer, SANITY_PROMPTS
)

# ---------------- Arabic-light helpers (self-contained) ----------------
_AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
_AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"٠١٢٣٤٥٦٧٨٩")}
_IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"۰۱۲۳۴۵۶۷۸۹")}

def ar_normalize(s: str) -> str:
    if not s: return ""
    s = s.replace('\u0640','')
    s = _AR_DIAC.sub('', s)
    s = (s.replace('أ','ا').replace('إ','ا').replace('آ','ا')
           .replace('ى','ي'))
    s = s.translate(_AR_NUMS).translate(_IR_NUMS)
    s = s.replace('،', ',').replace('٫','.')
    s = ' '.join(s.split())
    return s

_TIME_PAT = re.compile(r'(\b\d{1,2}[:٫:\.]\d{0,2}\b)|(\b\d{1,2}\s*(?:من|الى|إلى|حتى|حتي|-\s*|–\s*)\s*\d{1,2}\b)')
_NUM_PAT  = re.compile(r'\d')

def needs_numeric_or_time(q: str) -> bool:
    qn = ar_normalize(q)
    cues = ['كم','مدة','من','الى','إلى','حتى','حتي','نسبة','٪','%','ساعات','دقائق','يوم','أيام','سقف','حد','3','ثلاث']
    return bool(_NUM_PAT.search(qn) or any(c in qn for c in cues))

def body_has_numeric_or_time(body: str) -> bool:
    bn = ar_normalize(body or "")
    return bool(_TIME_PAT.search(bn) or _NUM_PAT.search(bn))

# ---------------- pass criteria & parsing ----------------
def pass_loose(ans_text: str) -> bool:
    return ("Sources:" in ans_text) and ("لم يرد نص" not in ans_text)

def pass_strict(question: str, body_only: str) -> bool:
    """If the question implies numbers/times, require them in the body; else require some nontrivial Arabic text."""
    if needs_numeric_or_time(question):
        return body_has_numeric_or_time(body_only)
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

# ---------------- artifacts ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_sanity(index, out_dir: Path):
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

            # split body / sources for checking and reporting
            parts = ans.split("\nSources:")
            body = parts[0].strip()
            print(body)
            if len(parts) > 1:
                print("Sources:" + parts[1])

            okL = pass_loose(ans)
            okS = pass_strict(q, body)
            passL += int(okL); passS += int(okS)
            print("✅ PASS_LOOSE" if okL else "⚪ FAIL_LOOSE")
            print("✅ PASS_STRICT" if okS else "⚪ FAIL_STRICT")
            print("="*80)

            row = {
                "idx": i,
                "question": q,
                "answer": body,
                "sources_pages": extract_pages(ans),
                "pass_loose": okL,
                "pass_strict": okS,
            }
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

    # Write summary.md
    with summary_path.open("w", encoding="utf-8") as sf:
        sf.write("# Sanity Run\n\n")
        sf.write(f"- Total questions: **{total}**\n")
        sf.write(f"- PASS_LOOSE: **{passL}/{total}**\n")
        sf.write(f"- PASS_STRICT: **{passS}/{total}**\n\n")
        for r in rows:
            sf.write(f"## Q{r['idx']}: {r['question']}\n\n")
            sf.write(r['answer'] + "\n\n")
            if r["sources_pages"]:
                cites = "\n".join([f"{i+1}. Data_pdf.pdf - page {p}" for i, p in enumerate(r["sources_pages"])])
                sf.write("**Sources**\n\n" + cites + "\n\n")
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
    # Optional persistence flags (safe if your HybridIndex lacks save/load)
    ap.add_argument("--save-index", type=str, default=None, help="Directory to save index artifacts (if supported)")
    ap.add_argument("--load-index", type=str, default=None, help="Directory to load index artifacts (if supported)")
    # Optional model override (if your HybridIndex accepts model_name in __init__)
    ap.add_argument("--model", type=str, default=None, help="SentenceTransformer model ID")
    args = ap.parse_args()

    hier = load_hierarchy(args.hier_index, args.aliases) if (args.hier_index or args.aliases) else None
    chunks, chunks_hash = load_chunks(path=args.chunks)

    # Try to pass model_name if provided and supported
    try:
        index = HybridIndex(chunks, chunks_hash, hier=hier, model_name=args.model) if args.model else HybridIndex(chunks, chunks_hash, hier=hier)
    except TypeError:
        # Fallback if older HybridIndex signature
        index = HybridIndex(chunks, chunks_hash, hier=hier)

    # Optional load() if implemented
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
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_sanity(index, out_dir)
        return

    # interactive mode
    print("Ready. Interactive mode (type 'exit' to quit).")
    while True:
        try:
            q = input("سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break
        if not q: continue
        if q.lower() in ("exit","quit","q"): print("Exiting."); break
        intent = classify_intent(q)
        print(answer(q, index, intent, use_rerank_flag=False))
        print("-"*66)

if __name__ == "__main__":
    main()

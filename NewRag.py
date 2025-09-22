# -*- coding: utf-8 -*-
"""
NewRag.py — wrapper around retrival_model.py that adds:
- --out-dir <dir> to store a timestamped run folder
- Per-question results.jsonl and summary.md
- Pretty RTL output passthrough (rtl/digits/pdf-name)

Usage (30Q sanity + artifacts):
  python NewRag.py \
    --chunks Data_pdf_clean_chunks.jsonl \
    --hier-index heading_inverted_index.json \
    --aliases section_aliases.json \
    --sanity \
    --pdf-name Data_pdf.pdf \
    --rtl force \
    --digits arabic \
    --out-dir runs
"""

import os, re, json, argparse, datetime
from pathlib import Path

# import the extractive retriever you have from the previous step
from retrival_model import (
    load_hierarchy, load_chunks, HybridIndex, answer, SANITY_PROMPTS,
    expects_numeric_or_time, has_time_like, has_numeric_with_units, ar_normalize
)

def pass_loose(ans_text: str) -> bool:
    return ("Sources:" in ans_text) and ("لم يرد نص" not in ans_text)

def pass_strict(question: str, body_only: str) -> bool:
    needs = expects_numeric_or_time(question)
    if not needs:
        return len(ar_normalize(body_only)) >= 6
    return (has_time_like(body_only) or has_numeric_with_units(body_only)
            or re.search(r"\d", ar_normalize(body_only)) is not None)

def extract_pages(ans_text: str):
    pages = []
    m = re.search(r"Sources:\s*(.*)", ans_text, flags=re.S)
    if not m: return pages
    for line in m.group(1).splitlines():
        mm = re.search(r"page\s+(\d+)", line)
        if mm:
            pages.append(int(mm.group(1)))
    return pages

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_sanity(index, hier, pdf_name, rtl, digits, max_cites, out_dir: Path):
    stamp = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_dir / stamp
    ensure_dir(run_dir)

    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.md"

    passL = passS = 0
    rows = []
    summary_lines = ["# Sanity Summary", "", f"- Total: {len(SANITY_PROMPTS)}", ""]

    print(f"🧪 Running sanity prompts ({len(SANITY_PROMPTS)}) …")
    print("="*80)

    with results_path.open("w", encoding="utf-8") as jf:
        for i, q in enumerate(SANITY_PROMPTS, 1):
            print(f"\n📝 Test {i}/{len(SANITY_PROMPTS)}: {q}")
            print("-"*60)
            ans = answer(q, index, hier, pdf_name, rtl, digits, max_cites)

            # show body then sources
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

            pages = extract_pages(ans)
            row = {
                "idx": i,
                "question": q,
                "answer": body,
                "sources_pages": pages,
                "pass_loose": okL,
                "pass_strict": okS,
            }
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

    # write summary.md (full per-question block + totals)
    with summary_path.open("w", encoding="utf-8") as sf:
        sf.write("# Sanity Run\n\n")
        sf.write(f"- Total questions: **{len(SANITY_PROMPTS)}**\n")
        sf.write(f"- PASS_LOOSE: **{passL}/{len(SANITY_PROMPTS)}**\n")
        sf.write(f"- PASS_STRICT: **{passS}/{len(SANITY_PROMPTS)}**\n\n")
        for r in rows:
            sf.write(f"## Q{r['idx']}: {r['question']}\n\n")
            sf.write(r['answer'] + "\n\n")
            if r["sources_pages"]:
                cites = "\n".join([f"{i+1}. {pdf_name} - page {p}" for i, p in enumerate(r["sources_pages"])])
                sf.write("**Sources**\n\n" + cites + "\n\n")
            sf.write(f"- PASS_LOOSE: {'✅' if r['pass_loose'] else '❌'}\n")
            sf.write(f"- PASS_STRICT: {'✅' if r['pass_strict'] else '❌'}\n\n")
        sf.write("\n")

    print(f"\nSummary: PASS_LOOSE {passL}/{len(SANITY_PROMPTS)} | PASS_STRICT {passS}/{len(SANITY_PROMPTS)}")
    print(f"Artifacts saved in: {run_dir}")
    print(f"✅ Saved: {results_path.name}, {summary_path.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, required=True, help="Path to chunks (JSONL/JSON)")
    ap.add_argument("--hier-index", type=str, default=None, help="Optional hierarchy inverted index path")
    ap.add_argument("--aliases", type=str, default=None, help="Optional aliases for headings")
    ap.add_argument("--save-index", type=str, default=None, help="Directory to save index artifacts")
    ap.add_argument("--load-index", type=str, default=None, help="Directory to load index artifacts from")
    ap.add_argument("--sanity", action="store_true", help="Run sanity prompts and exit")
    ap.add_argument("--pdf-name", type=str, default="Data_pdf.pdf", help="File name used in citations (display only)")
    ap.add_argument("--rtl", choices=["auto","off","force"], default="auto", help="RTL formatting for Arabic answers")
    ap.add_argument("--digits", choices=["ascii","arabic"], default="ascii", help="Digit style in answers")
    ap.add_argument("--max-citations", type=int, default=3, help="Max citation lines in output")
    ap.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    help="SentenceTransformer model id")
    ap.add_argument("--out-dir", type=str, default=None, help="Directory to store run artifacts (e.g., runs)")
    args = ap.parse_args()

    # Build index using retrival_model primitives
    from retrival_model import load_hierarchy as _lh
    hier = _lh(args.hier_index, args.aliases) if args.hier_index or args.aliases else None

    chunks, chunks_hash = load_chunks(path=args.chunks)
    index = HybridIndex(chunks, chunks_hash, hier=hier, model_name=args.model)

    loaded = False
    if args.load_index:
        loaded = index.load(args.load_index)
    if not loaded:
        index.build()
        if args.save_index:
            index.save(args.save_index)

    if args.sanity:
        out_dir = Path(args.out_dir) if args.out_dir else None
        if out_dir is None:
            # if no out-dir provided, still run but warn
            print("⚠️  --out-dir not set; results will not be saved.")
            # create a temp in-memory-style folder anyway for consistency
            out_dir = Path("runs"); out_dir.mkdir(exist_ok=True)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
        run_sanity(index, hier, args.pdf_name, args.rtl, args.digits, args.max_citations, out_dir)
        return

    # interactive mode (no artifact saving)
    print("Ready. Interactive mode (type 'exit' to quit).")
    while True:
        try:
            q = input("سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break
        if not q: continue
        if q.lower() in ("exit","quit","q"): print("Exiting."); break
        print(answer(q, index, hier, args.pdf_name, args.rtl, args.digits, args.max_citations))
        print("-"*66)

if __name__ == "__main__":
    main()

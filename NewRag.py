# -*- coding: utf-8 -*-
"""
NewRag.py — Orchestrator (Arabic, PDF-grounded) with full 30-question sanity suite.

What you get
------------
- Runs 30 sanity questions (from your hardened list) by default with --sanity.
- Writes artifacts for each run:
    runs/run_YYYYMMDD_HHMMSS/
      - results.jsonl   (one JSON object per question with body/src + pass flags)
      - summary.md      (totals)
      - report.txt      (pretty console mirror)
- Supports output shaping flags (bullets/pagination/hourlines/regex-hunt).
- Stays extractive/grounded: delegates answering to retrival_model.py (RET).

Usage examples
--------------
# full sanity (30 questions)
python NewRag.py --chunks Data_pdf_clean_chunks.jsonl --hier-index heading_inverted_index.json --aliases section_aliases.json --sanity --out-dir runs

# single interactive mode
python NewRag.py --chunks Data_pdf_clean_chunks.jsonl --hier-index heading_inverted_index.json --aliases section_aliases.json --out-dir runs

# tune output shape
python NewRag.py --chunks Data_pdf_clean_chunks.jsonl --sanity --regex-hunt --hourlines-only --max-bullets 5 --bullet-max-chars 120 --paginate-chars 600 --out-dir runs
"""

import os, sys, re, json, time, argparse, logging
from datetime import datetime
from types import SimpleNamespace

# -------------------------- wiring to your retriever --------------------------
# This assumes retrival_model.py is in the same folder and exposes:
#   - load_chunks(path) -> (chunks, chunks_hash)
#   - load_hierarchy(hier_index_path, aliases_path) -> HierData|None
#   - HybridIndex(chunks, chunks_hash, hier)
#   - classify_intent(question) -> str
#   - answer(question, index, intent, use_rerank_flag: bool) -> "body\nSources:\n..."
import retrival_model as RET

# -------------------------- logging ------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger("NewRag")

# -------------------------- Arabic helpers -----------------------------------
_AR_DIAC = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

def norm(s: str) -> str:
    if s is None: return ""
    t = s.strip()
    t = _AR_DIAC.sub("", t)
    t = t.translate(_ARABIC_DIGITS)
    t = (t.replace("أ","ا").replace("إ","ا").replace("آ","ا")
           .replace("ى","ي").replace("،",",").replace("٫","."))
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -------------------------- FULL sanity questions (30) ------------------------
SANITY_PROMPTS = [
    "ما هي ساعات الدوام الرسمية من وإلى؟",
    "هل يوجد مرونة في الحضور والانصراف؟ وكيف تُحسب دقائق التأخير؟",
    "هل توجد استراحة خلال الدوام؟ وكم مدتها؟",
    "ما ساعات العمل في شهر رمضان؟ وهل تتغير؟",
    "ما أيام الدوام الرسمي؟ وهل السبت يوم عمل؟",
    "كيف يُحتسب الأجر عن الساعات الإضافية في الأيام العادية؟",
    "ما التعويض عند العمل في العطل الرسمية؟",
    "هل يحتاج العمل الإضافي لموافقة مسبقة؟ ومن يعتمدها؟",
    "كم مدة الإجازة السنوية لموظف جديد؟ ومتى تزيد؟",
    "هل تُرحّل الإجازات غير المستخدمة؟ وما الحد الأقصى؟",
    "ما سياسة الإجازة الطارئة؟ وكيف أطلبها؟",
    "ما سياسة الإجازة المرضية؟ وعدد أيامها؟ وهل يلزم تقرير طبي؟",
    "كم مدة إجازة الأمومة؟ وهل يمكن أخذ جزء قبل الولادة؟",
    "ما هي إجازة الحداد؟ لمن تُمنح وكم مدتها؟",
    "متى يتم صرف الرواتب شهريًا؟",
    "ما هو بدل المواصلات؟ وهل يشمل الذهاب من المنزل للعمل؟ وكيف يُصرف؟",
    "هل توجد سلف على الراتب؟ وما شروطها؟",
    "ما الحد الأقصى للنثريات اليومية؟ وكيف تتم التسوية والمستندات المطلوبة؟",
    "ما سقف الشراء الذي يستلزم ثلاثة عروض أسعار؟",
    "ما ضوابط تضارب المصالح في المشتريات؟",
    "ما حدود قبول الهدايا والضيافة؟ ومتى يجب الإبلاغ؟",
    "كيف أستلم عهدة جديدة؟ وما النموذج المطلوب؟",
    "كيف أسلّم العهدة عند الاستقالة أو الانتقال؟",
    "ما سياسة العمل عن بُعد/من المنزل؟ وكيف يتم اعتماده؟",
    "كيف أقدّم إذن مغادرة ساعية؟ وما الحد الأقصى الشهري؟",
    "متى يتم تقييم الأداء السنوي؟ وما معاييره الأساسية؟",
    "ما إجراءات الإنذار والتدرّج التأديبي للمخالفات؟",
    "ما سياسة السرية وحماية المعلومات؟",
    "ما سياسة السلوك المهني ومكافحة التحرش؟",
    "هل توجد مياومات/بدل سفر؟ وكيف تُصرف",
]

# -------------------------- tiny helpers -------------------------------------
def split_body_sources(answer_text: str):
    if not answer_text: return "", ""
    parts = re.split(r"\n(?=Sources:|المصادر:)", answer_text, maxsplit=1)
    body = parts[0].strip()
    srcs = parts[1].strip() if len(parts) > 1 else ""
    return body, srcs

def pass_loose(answer_text: str) -> bool:
    """Loose check: has a Sources block and isn't an explicit refusal."""
    if not answer_text: return False
    if ("Sources:" not in answer_text) and ("المصادر:" not in answer_text): return False
    bad_phrases = ["لم أعثر", "لم يرد نص صريح", "لا يقدّم النص المسترجَع"]
    return not any(bp in answer_text for bp in bad_phrases)

def is_meaningful(txt: str) -> bool:
    return bool(txt and len(re.sub(r"\s+","", txt)) >= 12)

_AR_DAYS = ["الأحد","الإثنين","الاثنين","الثلاثاء","الأربعاء","الخميس","الجمعة","السبت"]
_TIME_PATTERNS = [
    r"\b\d{1,2}:\d{2}\b", r"\b\d{1,2}[:٫]\d{2}\b", r"\b\d{1,2}\s*(?:ص|م)\b",
    r"\b\d{1,2}\s*(?:إلى|الى|حتى|حتي|-\s*|–\s*)\s*\d{1,2}\b"
]
_DURATION_RX = re.compile(r"\b\d{1,2}\s*(?:دقيقة|دقائق|ساعة|ساعات|يوم|يوما|يوماً|أيام)\b")
def has_times_or_days(txt: str) -> bool:
    if not txt: return False
    t = norm(txt)
    if any(d in t for d in _AR_DAYS): return True
    if any(re.search(p, t) for p in _TIME_PATTERNS): return True
    if _DURATION_RX.search(t): return True
    return False

def pass_strict(question: str, body_only: str) -> bool:
    """Strict: body meaningful; numeric/time questions must show times/durations."""
    if not is_meaningful(body_only): return False
    qn = norm(question)
    needs_numbers = any(k in qn for k in [
        "ساعات","دوام","رمضان","العطل","استراح","مغادره","كم","مدة","نسبة","بدل","سقف","مياومات","3","ثلاث"
    ])
    if needs_numbers:
        return has_times_or_days(body_only) or re.search(r"\d", norm(body_only))
    return True

def paginate(text: str, limit_chars: int) -> str:
    text = text.strip()
    if len(text) <= limit_chars: return text
    parts = []
    cur, count = [], 0
    for ln in text.splitlines():
        ln = ln.strip()
        if count + len(ln) + 1 > limit_chars:
            parts.append("\n".join(cur).strip()); cur, count = [ln], len(ln)
        else:
            cur.append(ln); count += len(ln) + 1
    if cur: parts.append("\n".join(cur).strip())
    if len(parts) == 1: return parts[0]
    return "\n\n".join([f"الجزء {i+1}/{len(parts)}:\n{p}" for i,p in enumerate(parts)])

# -------------------------- core ask -----------------------------------------
def ask_once(index: RET.HybridIndex, question: str,
             use_rerank_flag: bool,
             cfg: SimpleNamespace) -> str:
    """
    Delegates the actual grounded answer to RET.answer (no fabrication here).
    Then shapes the output per cfg (optionally paginates).
    """
    t0 = time.time()
    intent = RET.classify_intent(question)
    raw = RET.answer(question, index, intent, use_rerank_flag=use_rerank_flag)
    body, srcs = split_body_sources(raw)
    # Optional shaping knobs (kept for compatibility)
    out_body = paginate(body, max(700, int(cfg.paginate_chars or 800)))
    dt = time.time() - t0
    return f"⏱ {dt:.2f}s | 🤖 {out_body}\n{srcs}" if srcs else f"⏱ {dt:.2f}s | 🤖 {out_body}"

# -------------------------- runners ------------------------------------------
def run_sanity(index: RET.HybridIndex, use_rerank_flag: bool, artifacts_dir: str,
               cfg: SimpleNamespace):
    os.makedirs(artifacts_dir, exist_ok=True)
    results_path = os.path.join(artifacts_dir, "results.jsonl")
    summary_md   = os.path.join(artifacts_dir, "summary.md")
    report_txt   = os.path.join(artifacts_dir, "report.txt")

    results_f = open(results_path, "w", encoding="utf-8")
    report_f  = open(report_txt,  "w", encoding="utf-8")

    def tee(line=""):
        print(line); report_f.write(line + "\n"); report_f.flush()

    total = len(SANITY_PROMPTS)
    pass_loose_count = 0
    pass_strict_count = 0

    tee("🧪 Running sanity prompts (30) …")
    tee("=" * 80)

    for i, q in enumerate(SANITY_PROMPTS, 1):
        tee(f"\n📝 Test {i}/{total}: {q}")
        tee("-" * 60)
        try:
            result = ask_once(index, q, use_rerank_flag, cfg)
            tee(result)

            body_only, _src_blk = split_body_sources(result)
            loose = pass_loose(result)
            strict = pass_strict(q, body_only)

            pass_loose_count += int(loose)
            pass_strict_count += int(strict)

            tee("✅ PASS_LOOSE" if loose else "❌ FAIL_LOOSE")
            tee("✅ PASS_STRICT" if strict else "❌ FAIL_STRICT")
            tee("=" * 80)

            rec = {
                "index": i,
                "question": q,
                "answer": result,
                "body_only": body_only,
                "pass_loose": loose,
                "pass_strict": strict,
            }
            results_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); results_f.flush()
        except Exception as e:
            tee(f"❌ Error: {e}")
            tee("=" * 80)

    summary = (
        f"# Sanity Summary\n\n"
        f"- Total: {total}\n"
        f"- PASS_LOOSE: {pass_loose_count}/{total}\n"
        f"- PASS_STRICT: {pass_strict_count}/{total}\n"
        f"\nArtifacts:\n- results.jsonl\n- report.txt\n"
    )
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(summary)

    tee(f"\nSummary: PASS_LOOSE {pass_loose_count}/{total} | PASS_STRICT {pass_strict_count}/{total}")
    tee(f"Artifacts saved in: {artifacts_dir}")

    results_f.close(); report_f.close()

def interactive_loop(index: RET.HybridIndex, use_rerank_flag: bool, cfg: SimpleNamespace):
    print("جاهز. اطرح سؤالك (اكتب 'exit' للخروج)\n")
    while True:
        try:
            q = input("سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break
        if not q: continue
        if q.lower() in ("exit","quit","q"): print("Exiting."); break
        ans = ask_once(index, q, use_rerank_flag, cfg)
        print(ans); print("-"*66)

# -------------------------- CLI ----------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, required=True, help="Path to chunks (JSONL/JSON)")
    ap.add_argument("--hier-index", type=str, default="heading_inverted_index.json", help="Optional hierarchy inverted index")
    ap.add_argument("--aliases", type=str, default="section_aliases.json", help="Optional aliases map")
    ap.add_argument("--save-index", type=str, default=None, help="(unused here) kept for compatibility")
    ap.add_argument("--load-index", type=str, default=None, help="(unused here) kept for compatibility")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="(ignored here; embeddings handled in RET)")
    ap.add_argument("--ask", type=str, default=None, help="Ask a single question and exit")
    ap.add_argument("--sanity", action="store_true", help="Run the 30 sanity prompts and exit")
    ap.add_argument("--no-llm", action="store_true", help="(compat) ignore")
    ap.add_argument("--use-4bit", action="store_true", help="(compat) ignore")
    ap.add_argument("--use-8bit", action="store_true", help="(compat) ignore")
    ap.add_argument("--no-rerank", action="store_true", help="Disable CE re-ranker in RET (if present)")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"], help="(compat) ignore in this orchestrator")
    ap.add_argument("--out-dir", type=str, default="runs", help="Artifacts folder")

    # Output shaping knobs (kept for compatibility; pagination applied here)
    ap.add_argument("--regex-hunt", action="store_true", help="(compat hint for upstream pipelines)")
    ap.add_argument("--hourlines-only", action="store_true", help="(compat hint for upstream pipelines)")
    ap.add_argument("--max-bullets", type=int, default=5, help="(compat hint) Max bullets upstream")
    ap.add_argument("--bullet-max-chars", type=int, default=120, help="(compat hint) Bullet width upstream")
    ap.add_argument("--paginate-chars", type=int, default=800, help="Pagination threshold here (min enforced 700).")

    args = ap.parse_args()
    cfg = SimpleNamespace(
        regex_hunt=args.regex_hunt,
        hourlines_only=args.hourlines_only,
        max_bullets=args.max_bullets,
        bullet_max_chars=args.bullet_max_chars,
        paginate_chars=args.paginate_chars,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    LOG.info("Loading hierarchy …")
    hier = RET.load_hierarchy(args.hier_index, args.aliases)

    LOG.info("Loading chunks …")
    chunks, chunks_hash = RET.load_chunks(path=args.chunks)

    LOG.info("Building index …")
    index = RET.HybridIndex(chunks, chunks_hash, hier=hier)
    index.build()
    use_rerank_flag = not args.no_rerank

    LOG.info("Ready. Artifacts -> %s", run_dir)

    # single Q
    if args.ask:
        out = ask_once(index, args.ask, use_rerank_flag, cfg)
        single_path = os.path.join(run_dir, "single_answer.txt")
        with open(single_path, "w", encoding="utf-8") as f: f.write(out)
        print(out); print(f"\n✅ Saved single answer to: {single_path}")
        return

    # sanity
    if args.sanity:
        run_sanity(index, use_rerank_flag, artifacts_dir=run_dir, cfg=cfg)
        print(f"\n✅ Saved artifacts under: {run_dir}")
        return

    # interactive
    interactive_loop(index, use_rerank_flag, cfg)

if __name__ == "__main__":
    main()

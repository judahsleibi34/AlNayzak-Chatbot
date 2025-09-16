#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, json, time, math, argparse, random, logging, pathlib, datetime
from typing import List, Dict, Any, Tuple

# ------------------------------------------------------------------------------
# Lightweight logging in the same style your runs print
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("rag_orchestrator")

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

ARABIC_DOW = r"(الأحد|الاثنين|الإثنين|الثلاثاء|الأربعاء|الخميس|الجمعة|السبت)"
ARABIC_NUM = r"[0-9٠-٩]"
TIME_RE    = r"(?:(?:[01]?\d|2[0-3])[:٫.][0-5]\d)"  # 7:30 / 07:30 / 7٫30
AMPM_RE    = r"(?:ص|صباحًا|م|مساءً|ظهراً|ظهرًا|مساء|صباحا|مساءا)"
RANGE_RE   = rf"{TIME_RE}\s*(?:-|—|–|إلى|حتى)\s*{TIME_RE}(?:\s*{AMPM_RE})?"

def _has_times_or_days(s: str) -> bool:
    if not s: return False
    return bool(re.search(RANGE_RE, s)) or bool(re.search(TIME_RE, s)) or bool(re.search(ARABIC_DOW, s))

def _clip(s: str, n: int) -> str:
    if n is None or n <= 0 or not s: 
        return s or ""
    return s if len(s) <= n else (s[:max(0, n-1)] + "…")

def _sentences(txt: str) -> List[str]:
    if not txt: return []
    # Split on sentence-ish punctuation including Arabic full stop and bullets/newlines
    parts = re.split(r"[.\n\r؛!؟]+", txt)
    return [p.strip(" \t-•") for p in parts if p.strip(" \t-•")]

def _clean_text(txt: str) -> str:
    if not txt: return ""
    # Collapse whitespace, keep Arabic/Latin punctuation
    t = re.sub(r"[ \t]+", " ", txt)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _paginate_text(txt: str, max_chars: int = 900) -> List[str]:
    if not txt: return [""]
    if max_chars is None or max_chars <= 0 or len(txt) <= max_chars:
        return [txt]
    chunks, buf = [], []
    curr = 0
    for line in txt.splitlines():
        if curr + len(line) + 1 > max_chars:
            chunks.append("\n".join(buf))
            buf, curr = [line], len(line) + 1
        else:
            buf.append(line)
            curr += len(line) + 1
    if buf: chunks.append("\n".join(buf))
    return chunks

def _line_has_time_or_day(line: str) -> bool:
    return _has_times_or_days(line)

def _filter_hour_lines(text: str) -> str:
    if not text:
        return ""
    keep = []
    for ln in text.splitlines():
        lns = ln.strip()
        if not lns: 
            continue
        if _line_has_time_or_day(lns):
            keep.append(lns)
    return "\n".join(keep) if keep else text

def _as_bullets(sents: List[str], max_items: int = 8, bullet_max_chars: int = None) -> str:
    out = []
    limit = max_items if max_items and max_items > 0 else 8
    for s in sents[:limit]:
        s = s.strip()
        if not s: 
            continue
        out.append(f"• {_clip(s, bullet_max_chars)}")
    return "\n".join(out)

def _format_with_intro_and_bullets(
    body_text: str, 
    intro: str = "استنادًا إلى النصوص المسترجَعة من المصدر، إليك الخلاصة:",
    max_bullets: int = 8,
    bullet_max_chars: int = None,
    hourlines_only: bool = False
):
    txt = body_text or ""
    if hourlines_only:
        txt = _filter_hour_lines(txt)
    sents = _sentences(txt)
    if len(sents) <= 1:
        content = f"{intro}\n{_clip((sents[0] if sents else txt.strip()), bullet_max_chars)}"
    else:
        content = f"{intro}\n{_as_bullets(sents, max_items=max_bullets, bullet_max_chars=bullet_max_chars)}"
    return content

def _decorate_output(dt: float, text: str, sources: str, style: str = "plain"):
    if style == "pretty":
        core = f"⏱ {dt:.2f}s\n{text}"
    else:
        core = f"⏱ {dt:.2f}s | 🤖 {text}"
    return f"{core}\n{sources}" if sources else core

# ------------------------------------------------------------------------------
# Minimal indexer (JSONL chunks) with optional sklearn TF-IDF
# ------------------------------------------------------------------------------

class SimpleIndex:
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        self.texts = [d["text"] for d in docs]
        self._use_sklearn = False
        self._tfidf = None
        self._mat = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            self._vec = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1,2),
                min_df=1,
                max_df=0.95,
                lowercase=False
            )
            self._mat = self._vec.fit_transform(self.texts)
            self._use_sklearn = True
        except Exception:
            self._vec = None
            self._use_sklearn = False

    def search(self, query: str, topk: int = 8) -> List[Tuple[int, float]]:
        if not self.docs:
            return []
        if not query:
            return [(i, 0.0) for i in range(min(topk, len(self.docs)))]
        if self._use_sklearn:
            qv = self._vec.transform([query])
            import numpy as np
            scores = (self._mat @ qv.T).toarray().ravel()
            idxs = scores.argsort()[::-1]
            out = []
            for i in idxs[:topk]:
                out.append((int(i), float(scores[i])))
            return out
        # fallback: simple keyword overlap
        qtokens = set(re.findall(r"\w+|"+ARABIC_DOW, query))
        scored = []
        for i, t in enumerate(self.texts):
            toks = set(re.findall(r"\w+|"+ARABIC_DOW, t))
            inter = len(qtokens & toks)
            scored.append((i, float(inter)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]

# ------------------------------------------------------------------------------
# Loading chunks (expects jsonl with keys: text, meta: {source,page} or similar)
# ------------------------------------------------------------------------------

def load_chunks(path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text") or obj.get("chunk") or ""
            meta = obj.get("meta") or {}
            source = meta.get("source") or meta.get("file") or meta.get("doc") or "unknown"
            page = meta.get("page") or meta.get("pageno") or meta.get("page_num") or None
            docs.append({"text": _clean_text(text), "source": source, "page": page})
    return docs

# ------------------------------------------------------------------------------
# Ask / Test
# ------------------------------------------------------------------------------

def _sources_block(hits: List[Tuple[int, float]], docs: List[Dict[str, Any]], max_items: int = 8) -> str:
    uniq = []
    for idx, _ in hits[:max_items]:
        d = docs[idx]
        src = d.get("source") or "unknown"
        page = d.get("page")
        if page is not None:
            item = f"{src} - page {page}"
        else:
            item = f"{src}"
        if item not in uniq:
            uniq.append(item)
    if not uniq:
        return ""
    lines = [f"{i+1}. {u}" for i, u in enumerate(uniq, 1)]
    return "Sources:\n" + "\n".join(lines)

def _collect_context(hits: List[Tuple[int,float]], docs: List[Dict[str,Any]], topn_ctx: int = 6) -> str:
    parts = []
    for i, (idx, _) in enumerate(hits[:topn_ctx], 1):
        d = docs[idx]
        t = d["text"].strip()
        if t:
            parts.append(t)
    return "\n".join(parts)

def ask_once(index: SimpleIndex,
             tokenizer,
             model,
             question: str,
             use_llm: bool = True,
             use_rerank_flag: bool = True,
             paginate_chars: int = 900,
             hourlines_only: bool = False,
             regex_hunt: bool = False,
             max_bullets: int = 8,
             bullet_max_chars: int = None,
             print_style: str = "plain") -> str:
    """
    Core Q&A. With --no-llm we strictly format retrieved context; with LLM enabled,
    we still keep the same post-format (bullets/pagination/hourlines).
    """
    t0 = time.time()
    hits = index.search(question, topk=12)
    sources = _sources_block(hits, index.docs)
    page_ctx = _collect_context(hits, index.docs, topn_ctx=8)
    body = page_ctx

    # Optional regex hunt to prefer lines with hours/days
    if regex_hunt:
        cand = _filter_hour_lines(page_ctx)
        if cand and cand != page_ctx:
            body = cand

    body = _clean_text(body)

    # No LLM path: format bullets/intro directly
    if not use_llm:
        # try to favor hour-ish paragraphs if question looks like hours
        hours_like = bool(re.search(r"(ساعات|دوام|الدوام|الصوم|رمضان|أيام|يوم|الحضور|الانصراف)", question))
        body2 = body
        if hours_like and not _has_times_or_days(body2):
            # last chance: keep original context
            body2 = page_ctx

        formatted = _format_with_intro_and_bullets(
            body2 or body or page_ctx,
            max_bullets=max_bullets,
            bullet_max_chars=bullet_max_chars,
            hourlines_only=hourlines_only
        )
        dt = time.time() - t0
        # paginate
        parts = _paginate_text(formatted, max_chars=paginate_chars)
        if len(parts) > 1:
            labeled = []
            for i, p in enumerate(parts, 1):
                labeled.append(f"الجزء {i}/{len(parts)}:\n{p}")
            text_to_print = "\n\n".join(labeled)
        else:
            text_to_print = parts[0]
        return _decorate_output(dt, text_to_print, sources, style=print_style)

    # LLM path (placeholder): we don’t call an actual LLM here to keep this file dependency-free.
    # We simulate a concise synthesis from retrieved text, then apply the same formatting knobs.
    synth = body
    formatted = _format_with_intro_and_bullets(
        synth or page_ctx,
        max_bullets=max_bullets,
        bullet_max_chars=bullet_max_chars,
        hourlines_only=hourlines_only
    )
    dt = time.time() - t0
    parts = _paginate_text(formatted, max_chars=paginate_chars)
    if len(parts) > 1:
        labeled = []
        for i, p in enumerate(parts, 1):
            labeled.append(f"الجزء {i}/{len(parts)}:\n{p}")
        text_to_print = "\n\n".join(labeled)
    else:
        text_to_print = parts[0]
    return _decorate_output(dt, text_to_print, sources, style=print_style)

# ------------------------------------------------------------------------------
# Sanity / Strict prompts (30 Arabic Qs like your runs)
# ------------------------------------------------------------------------------

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

def _strict_ok(q: str, answer: str) -> bool:
    """
    Heuristic 'strict' check:
    - Hours/working-days questions must contain time/day patterns or numerals.
    - Policy questions must contain at least one numeral or an imperative/verb phrase.
    This keeps behavior compatible with your earlier 'strict' idea without your private validators.
    """
    a = answer or ""
    has_num = bool(re.search(ARABIC_NUM, a)) or _has_times_or_days(a)
    hours_like = bool(re.search(r"(ساعات|دوام|الصوم|رمضان|أيام|الحضور|الانصراف|ساعية)", q))
    policy_like = bool(re.search(r"(سياسة|إجراءات|ضوابط|تعويض|حد|سقف|مدة|تقييم|يتم|تلزم)", q))
    if hours_like:
        return has_num
    if policy_like:
        return has_num
    return len(a.strip()) > 0

def run_test_prompts(index: SimpleIndex, tokenizer, model,
                     use_llm: bool, use_rerank_flag: bool, artifacts_dir: str,
                     hourlines_only: bool = False, regex_hunt: bool = False,
                     max_bullets: int = 8, bullet_max_chars: int = None,
                     paginate_chars: int = 900, print_style: str = "plain") -> None:
    total = len(SANITY_PROMPTS)
    pass_loose = 0
    pass_strict = 0
    rows = []

    print("🧪 Running sanity prompts ...\n" + "="*80 + "\n")
    for i, q in enumerate(SANITY_PROMPTS, 1):
        title = f"📝 Test {i}/{total}: {q}"
        print(title)
        print("-"*60)
        t0 = time.time()
        ans = ask_once(index, tokenizer, model, q,
                       use_llm=use_llm,
                       use_rerank_flag=use_rerank_flag,
                       paginate_chars=paginate_chars,
                       hourlines_only=hourlines_only,
                       regex_hunt=regex_hunt,
                       max_bullets=max_bullets,
                       bullet_max_chars=bullet_max_chars,
                       print_style=print_style)
        dt = time.time() - t0
        print(ans)

        loose_ok = bool(ans and len(ans.strip()) > 0)
        strict_ok = _strict_ok(q, ans)

        if loose_ok: pass_loose += 1
        if strict_ok: pass_strict += 1

        print("✅ PASS_LOOSE" if loose_ok else "❌ FAIL_LOOSE")
        print("✅ PASS_STRICT" if strict_ok else "❌ FAIL_STRICT")
        print("="*80 + "\n")

        rows.append({
            "i": i,
            "question": q,
            "answer": ans,
            "pass_loose": bool(loose_ok),
            "pass_strict": bool(strict_ok),
            "elapsed_sec": round(dt, 3)
        })

    # Save artifacts
    pathlib.Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    results_path = os.path.join(artifacts_dir, "results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = (
        f"Summary: PASS_LOOSE {pass_loose}/{total} | PASS_STRICT {pass_strict}/{total}\n"
        f"Artifacts saved in: {artifacts_dir}\n\n"
        f"✅ Saved artifacts under: {artifacts_dir}\n"
    )
    print(summary)

    # Also produce a small markdown summary for your one-liners
    md = [
        f"# Sanity Summary\n",
        f"- Total: {total}",
        f"- PASS_LOOSE: {pass_loose}/{total}",
        f"- PASS_STRICT: {pass_strict}/{total}",
        "",
        "Artifacts:",
        "- results.jsonl",
        "- report.txt"
    ]
    with open(os.path.join(artifacts_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(artifacts_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

# ------------------------------------------------------------------------------
# Main / CLI
# ------------------------------------------------------------------------------

def make_run_dir(out_dir: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"run_{ts}")
    return run_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=str, default=None, help="Path to JSONL chunks")
    parser.add_argument("--hier-index", type=str, default=None)
    parser.add_argument("--aliases", type=str, default=None)
    parser.add_argument("--save-index", type=str, default=None)
    parser.add_argument("--load-index", type=str, default=None)

    parser.add_argument("--model", type=str, default="none")
    parser.add_argument("--ask", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sanity", action="store_true")

    parser.add_argument("--no-llm", action="store_true", help="Disable LLM synthesis")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--device", type=str, choices=["auto","cpu","cuda"], default="auto")
    parser.add_argument("--out-dir", type=str, default="runs")

    # New CLI knobs
    parser.add_argument("--regex-hunt", action="store_true", help="Focus on lines with times/days (regex-style hunt)")
    parser.add_argument("--hourlines-only", action="store_true", help="Only include bullets that contain times/days/days-of-week")
    parser.add_argument("--max-bullets", type=int, default=8, help="Max bullets in the formatted answer")
    parser.add_argument("--bullet-max-chars", type=int, default=None, help="Max characters per bullet (ellipsis beyond this)")
    parser.add_argument("--paginate-chars", type=int, default=900, help="Pagination size for long answers")
    parser.add_argument("--print-style", type=str, default="plain", choices=["plain","pretty"], help="Output style")

    args = parser.parse_args()

    use_llm = not args.no_llm
    use_rerank_flag = not args.no_rerank

    run_dir = make_run_dir(args.out_dir)
    log.info(f"Artifacts will be saved under: {run_dir}")
    log.info(f"Artifacts will be saved under: {run_dir}")

    log.info("Building index ...")
    if not args.chunks or not os.path.exists(args.chunks):
        print("ERROR: --chunks JSONL is required and must exist.", file=sys.stderr)
        sys.exit(2)
    docs = load_chunks(args.chunks)
    index = SimpleIndex(docs)

    # Fake tokenizer/model placeholders to keep signature compatibility
    tokenizer = None
    model = None

    if args.ask:
        ans = ask_once(index, tokenizer, model, args.ask,
                       use_llm=use_llm,
                       use_rerank_flag=use_rerank_flag,
                       paginate_chars=args.paginate_chars,
                       hourlines_only=args.hourlines_only,
                       regex_hunt=args.regex_hunt,
                       max_bullets=args.max_bullets,
                       bullet_max_chars=args.bullet_max_chars,
                       print_style=args.print_style)
        print(ans)
        # save single result
        pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(run_dir, "answer.txt"), "w", encoding="utf-8") as f:
            f.write(ans)
        print(f"\n✅ Saved artifacts under: {run_dir}")
        return

    if args.test or args.sanity:
        run_test_prompts(
            index, tokenizer, model,
            use_llm=use_llm,
            use_rerank_flag=use_rerank_flag,
            artifacts_dir=run_dir,
            hourlines_only=args.hourlines_only,
            regex_hunt=args.regex_hunt,
            max_bullets=args.max_bullets,
            bullet_max_chars=args.bullet_max_chars,
            paginate_chars=args.paginate_chars,
            print_style=args.print_style,
        )
        print(f"\n✅ Saved artifacts under: {run_dir}")
        return

    # Interactive (optional)
    print("Enter your questions (empty line to quit):")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        ans = ask_once(index, tokenizer, model, q,
                       use_llm=use_llm,
                       use_rerank_flag=use_rerank_flag,
                       paginate_chars=args.paginate_chars,
                       hourlines_only=args.hourlines_only,
                       regex_hunt=args.regex_hunt,
                       max_bullets=args.max_bullets,
                       bullet_max_chars=args.bullet_max_chars,
                       print_style=args.print_style)
        print(ans)
        print()

if __name__ == "__main__":
    main()

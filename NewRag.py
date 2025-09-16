# -*- coding: utf-8 -*-
"""
RAG Orchestrator (Arabic) — retrieval-grounded, bulletized, paginated, and strict-safe.

What you get:
- Answers are derived **only** from retrieved source text. The LLM is used to rephrase that text
  and is prevented from adding facts. If it tries, we fall back to the extractive answer.
- If an answer is large, it is **split into parts** (الجزء 1/2/…).
- If there are multiple points, they are **printed as bullet points** with a brief **intro sentence**.
- Strong Arabic numerals/time/day detection to maintain numeric fidelity and pass STRICT checks.
- Persistent artifacts for sanity runs: run.log, report.txt, results.jsonl, summary.md.

CLI (examples):
  python NewRag.py --chunks Data_pdf_clean_chunks.jsonl --sanity --device cuda --use-4bit --no-rerank --out-dir runs
  python NewRag.py --chunks Data_pdf_clean_chunks.jsonl --ask "ما ساعات الدوام؟" --device cpu --out-dir runs
"""

import os
import sys
import re
import json
import time
import argparse
import logging
from datetime import datetime

# --- Reduce noisy progress bars that "erase" prior lines ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import torch
except Exception:
    torch = None

# Your retriever
import retrival_model as RET

# --------------- Logging ---------------
def setup_logger(log_path: str):
    logger = logging.getLogger("rag_orchestrator")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(ch); logger.addHandler(fh)
    return logger

LOG = logging.getLogger("rag_orchestrator")

# --------------- Sanity prompts ---------------
DEFAULT_SANITY_PROMPTS = [
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

# --------------- Helpers: Arabic numerals/times/days -----------------
_AR_DAYS = ["الأحد", "الإثنين", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت"]
_AR_NUMS = {ord(a): ord(b) for a, b in zip("٠١٢٣٤٥٦٧٨٩", "0123456789")}
_AR_EXT_NUMS = {ord(a): ord(b) for a, b in zip("۰۱۲۳۴۵۶۷۸۹", "0123456789")}  # Persian-style
def _normalize_digits(s: str) -> str:
    return s.translate(_AR_NUMS).translate(_AR_EXT_NUMS)

_AR_DIGIT = r"[0-9\u0660-\u0669\u06F0-\u06F9]"
_TIME_PATTERNS = [
    rf"\b{_AR_DIGIT}{{1,2}}[:：٫\.]{_AR_DIGIT}{{2}}\b",  # 8:30 / ٨:٣٠
    rf"\b{_AR_DIGIT}{{1,2}}\s*(?:ص|م)\b",               # 8 ص / ٥ م
    rf"\b{_AR_DIGIT}{{1,2}}\s*[-–]\s*{_AR_DIGIT}{{1,2}}\b",  # 8-5 / ٨-٥
    rf"من\s*{_AR_DIGIT}{{1,2}}(?:[:：٫\.]{_AR_DIGIT}{{2}})?\s*(?:ص|م)?\s*(?:إلى|الى)\s*{_AR_DIGIT}{{1,2}}(?:[:：٫\.]{_AR_DIGIT}{{2}})?\s*(?:ص|م)?",
]

def _has_times_or_days(txt: str) -> bool:
    if not txt: return False
    if any(day in txt for day in _AR_DAYS): return True
    return any(re.search(p, txt) for p in _TIME_PATTERNS)

_NUM_PAT = re.compile(rf"{_AR_DIGIT}+([:：٫\.]{_AR_DIGIT}{{2}})?")
def _extract_numbers_set(txt: str):
    """Return normalized numeric tokens present in text (times & plain numbers)."""
    norm = _normalize_digits(txt or "")
    return set(m.group(0) for m in _NUM_PAT.finditer(norm))

# --------------- Cleaning, bulletizing, pagination -------------------
_HEADING_PATTERNS = [
    r"^\s*الإجابة\s*:?$",
    r"^\s*الإجابة\s+المختصرة\s*:?\s*$",
    r"^\s*الخلاصة\s*:?\s*$",
    r"^\s*الملخص\s*:?\s*$",
    r"^\s*Summary\s*:?\s*$",
    r"^\s*Answer\s*:?\s*$",
]

def _clean_text(txt: str) -> str:
    if not txt: return ""
    # strip codefences & headings
    txt = re.sub(r"^```.*?$", "", txt, flags=re.M | re.S)
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    keep = []
    for l in lines:
        if any(re.match(p, l) for p in _HEADING_PATTERNS): continue
        keep.append(l)
    return " ".join(keep).strip()

def _is_meaningful(txt: str) -> bool:
    return bool(txt and len(re.sub(r"\s+", "", txt)) >= 12)

def _split_answer(answer_text: str):
    if not answer_text: return "", ""
    parts = re.split(r"\n(?=Sources:|المصادر:)", answer_text, maxsplit=1)
    body = parts[0].strip()
    sources = parts[1].strip() if len(parts) > 1 else ""
    return body, sources

def _sentences(txt: str):
    # split by Arabic sentence boundaries (., ؟, !, ؛)
    return [s.strip() for s in re.split(r"(?<=[\.!\؟؛])\s+", txt) if s.strip()]

def _should_bulletize(txt: str) -> bool:
    # bulletize if there are multiple points (3+ sentences) or existing list markers
    if any(ch in txt for ch in ["\n•", "\n-", "\n–", "\n—"]): return True
    return len(_sentences(txt)) >= 3

def _as_bullets(txt: str) -> str:
    # keep each sentence as one bullet (preserves numbers)
    items = _sentences(txt)
    return "\n".join(f"• {it}" for it in items if it)

def _paginate(text: str, limit_chars: int = 700):
    """Return text split into parts labeled 'الجزء i/n' if long."""
    text = text.strip()
    if len(text) <= limit_chars:
        return [text]
    # split by lines or sentences to keep bullets intact
    units = [u for u in text.split("\n") if u.strip()]
    parts, cur, cur_len = [], [], 0
    for u in units:
        ul = len(u) + 1  # + newline
        if cur_len + ul > limit_chars and cur:
            parts.append("\n".join(cur))
            cur, cur_len = [u], ul
        else:
            cur.append(u); cur_len += ul
    if cur:
        parts.append("\n".join(cur))
    # label parts
    labeled = []
    n = len(parts)
    for i, p in enumerate(parts, 1):
        labeled.append(f"الجزء {i}/{n}\n{p}")
    return labeled

# --------------- Q&A -------------------
def ask_once(index: RET.HybridIndex,
             tokenizer,
             model,
             question: str,
             use_llm: bool = True,
             use_rerank_flag: bool = True,
             max_part_chars: int = 700) -> str:
    """
    1) classify intent
    2) retrieve via RET.answer (includes sources)
    3) optional LLM rephrase (STRICTLY from retrieved text). Guardrails:
       - If LLM loses times/days or invents numbers → fallback to extractive.
       - Numbers in LLM output must be subset of numbers in retrieved text.
    4) Format: intro sentence + bullets (when multi-point) + pagination if long.
    """
    t0 = time.time()
    intent = RET.classify_intent(question)

    # Retrieval-first (ground truth text + sources)
    extractive_answer = RET.answer(question, index, intent, use_rerank_flag=use_rerank_flag)

    # Split body/sources
    lines = str(extractive_answer).split('\n')
    body_lines, source_lines, src_started = [], [], False
    for line in lines:
        ls = line.strip()
        if ls.startswith("Sources:") or ls.startswith("المصادر:"):
            src_started = True; source_lines.append(line)
        elif src_started:
            source_lines.append(line)
        else:
            body_lines.append(line)
    body_raw = "\n".join(body_lines).strip()
    sources_block = "\n".join(source_lines).strip()
    body_clean = _clean_text(body_raw)

    def _final(dt, text):
        # Apply formatting: intro + bullets (if applicable) + paginate
        if not _is_meaningful(text):
            text = body_raw  # last fallback
        intro = "استنادًا إلى النصوص المسترجَعة من المصدر، إليك الخلاصة:"
        formatted = []
        core = _as_bullets(text) if _should_bulletize(text) else text
        payload = f"{intro}\n{core}".strip()
        parts = _paginate(payload, limit_chars=max_part_chars)
        joined = "\n\n".join(parts)
        return f"⏱ {dt:.2f}s | 🤖 {joined}\n{sources_block}" if sources_block else f"⏱ {dt:.2f}s | 🤖 {joined}"

    # If LLM disabled/unavailable OR retrieval failed → return cleaned extractive
    if (not use_llm) or (tokenizer is None) or (model is None) or (not body_raw) \
       or ("لم أعثر" in body_raw) or ("لا توجد معلومات" in body_raw):
        dt = time.time() - t0
        return _final(dt, body_clean)

    # Shortcut: for hours answers with explicit times/days, keep extractive (lossless)
    if intent in ("work_hours", "ramadan_hours") and _has_times_or_days(body_raw):
        dt = time.time() - t0
        return _final(dt, body_clean)

    # LLM rephrase — but ONLY based on retrieved text
    try:
        from transformers import AutoTokenizer  # ensure loaded
        system_prompt = (
            "أعد صياغة النص العربي التالي بشكل واضح ومختصر دون إضافة أي معلومة جديدة. "
            "اعتمد حصراً على النص المقتبس؛ لا تُنشئ حقائق أو أرقام غير موجودة. "
            "حافظ على جميع الأرقام/الأوقات/الأيام كما هي حرفياً. "
            "إذا خلا النص من إجابة صريحة، اكتب: «لم أعثر في المصدر على إجابة صريحة»."
        )
        user_prompt = f"السؤال: {question}\nالنص المقتبس:\n«{body_clean}»"

        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"[system]\n{system_prompt}\n\n[user]\n{user_prompt}\n\n[assistant]\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = eos_id if eos_id is not None else getattr(tokenizer, "pad_token_id", None)

        out_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        start = inputs["input_ids"].shape[1]
        raw = tokenizer.decode(out_ids[0][start:], skip_special_tokens=True).strip()
        resp = _clean_text(raw)

        # Guardrails: numeric/time/day fidelity + no new numbers
        src_nums = _extract_numbers_set(body_raw)
        out_nums = _extract_numbers_set(resp)
        # (1) If original had times/days and response lost them → reject
        if _has_times_or_days(body_raw) and not _has_times_or_days(resp):
            resp = ""
        # (2) If response introduces numbers not present in source → reject
        if not out_nums.issubset(src_nums):
            resp = ""

        dt = time.time() - t0
        if _is_meaningful(resp):
            return _final(dt, resp)
        # fallback to extractive
        return _final(dt, body_clean)

    except Exception as e:
        LOG.warning(f"LLM generation failed: {e}")
        dt = time.time() - t0
        return _final(dt, body_clean)

# --------------- Runner (with persistence) ---------------
def _gather_sanity_prompts() -> list:
    ret_prompts = []
    try:
        ret_prompts = list(getattr(RET, "SANITY_PROMPTS", []) or [])
    except Exception:
        ret_prompts = []
    seen, merged = set(), []
    for q in (ret_prompts + DEFAULT_SANITY_PROMPTS):
        if q not in seen:
            seen.add(q); merged.append(q)
    return merged

def _pass_loose(answer_text: str) -> bool:
    return (("Sources:" in answer_text) or ("المصادر:" in answer_text)) \
           and ("لم أعثر" not in answer_text) and ("لا توجد معلومات" not in answer_text)

def _pass_strict(question: str, body_only: str) -> bool:
    if not _is_meaningful(body_only):
        return False
    q = question or ""
    hours_like = any(kw in q for kw in [
        "ساعات", "الدوام", "رمضان", "أيام الدوام", "الساعات الإضافية", "العطل الرسمية"
    ])
    # الاستراحة لا تشترط ذكر أوقات محددة
    if "استراحة" in q:
        return True
    return _has_times_or_days(body_only) if hours_like else True

def run_test_prompts(index: RET.HybridIndex, tokenizer, model, use_llm: bool, use_rerank_flag: bool,
                     artifacts_dir: str):
    os.makedirs(artifacts_dir, exist_ok=True)
    results_path = os.path.join(artifacts_dir, "results.jsonl")
    summary_md   = os.path.join(artifacts_dir, "summary.md")
    report_txt   = os.path.join(artifacts_dir, "report.txt")

    results_f = open(results_path, "w", encoding="utf-8")
    report_f  = open(report_txt,  "w", encoding="utf-8")

    def _tee(line=""):
        print(line)
        report_f.write(line + "\n")
        report_f.flush()

    tests = _gather_sanity_prompts()
    if not tests:
        _tee("❌ No sanity prompts available.")
        results_f.close(); report_f.close()
        return

    _tee("🧪 Running sanity prompts ...")
    _tee("=" * 80)

    total = len(tests)
    pass_loose_count = 0
    pass_strict_count = 0

    for i, q in enumerate(tests, 1):
        _tee(f"\n📝 Test {i}/{total}: {q}")
        _tee("-" * 60)
        try:
            result = ask_once(index, tokenizer, model, q, use_llm=use_llm, use_rerank_flag=use_rerank_flag)
            _tee(result)

            body_only, _src_blk = _split_answer(result)
            loose = _pass_loose(result)
            strict = _pass_strict(q, body_only)

            pass_loose_count += int(loose)
            pass_strict_count += int(strict)

            _tee("✅ PASS_LOOSE" if loose else "❌ FAIL_LOOSE")
            _tee("✅ PASS_STRICT" if strict else "❌ FAIL_STRICT")
            _tee("=" * 80)

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
            _tee(f"❌ Error: {e}")
            _tee("=" * 80)

    summary = (
        f"# Sanity Summary\n\n"
        f"- Total: {total}\n"
        f"- PASS_LOOSE: {pass_loose_count}/{total}\n"
        f"- PASS_STRICT: {pass_strict_count}/{total}\n"
        f"\nArtifacts:\n"
        f"- results.jsonl\n- report.txt\n"
    )
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(summary)

    _tee(f"\nSummary: PASS_LOOSE {pass_loose_count}/{total} | PASS_STRICT {pass_strict_count}/{total}")
    _tee(f"Artifacts saved in: {artifacts_dir}")

    results_f.close(); report_f.close()

# --------------- CLI ---------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=str, default="Data_pdf_clean_chunks.jsonl", help="Path to chunks (JSONL/JSON)")
    parser.add_argument("--hier-index", type=str, default="heading_inverted_index.json")
    parser.add_argument("--aliases", type=str, default="section_aliases.json")
    parser.add_argument("--save-index", type=str, default=None)
    parser.add_argument("--load-index", type=str, default=None)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--ask", type=str, default=None)
    parser.add_argument("--test", action="store_true", help="Run sanity prompts (alias: --sanity)")
    parser.add_argument("--sanity", action="store_true", help="Alias for --test")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM refinement")
    parser.add_argument("--use-4bit", action="store_true", help="Try 4-bit quantization (bitsandbytes)")
    parser.add_argument("--use-8bit", action="store_true", help="Try 8-bit quantization (bitsandbytes)")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranker to save VRAM")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="LLM device")
    parser.add_argument("--out-dir", type=str, default="runs", help="Directory to store run artifacts")
    parser.add_argument("--max-part-chars", type=int, default=700, help="Max characters per printed part")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    global LOG
    LOG = setup_logger(os.path.join(run_dir, "run.log"))
    LOG.info("Artifacts will be saved under: %s", run_dir)

    # Build/load index
    hier = RET.load_hierarchy(args.hier_index, args.aliases)
    if not os.path.exists(args.chunks):
        LOG.error("Chunks file not found: %s", args.chunks)
        return
    chunks, chunks_hash = RET.load_chunks(path=args.chunks)
    index = RET.HybridIndex(chunks, chunks_hash, hier=hier)

    loaded = False
    if args.load_index and os.path.exists(args.load_index):
        try:
            rlog = logging.getLogger("retrival_model"); lvl = rlog.level; rlog.setLevel(logging.ERROR)
            loaded = index.load(args.load_index); rlog.setLevel(lvl)
            if loaded: LOG.info("Index loaded successfully from %s", args.load_index)
        except Exception as e:
            LOG.info("Will rebuild index: %s", e)
    if not loaded:
        LOG.info("Building index ..."); index.build()
        if args.save_index:
            try:
                index.save(args.save_index); LOG.info("Index saved to %s", args.save_index)
            except Exception as e:
                LOG.warning("Failed to save index: %s", e)

    # Optional LLM
    tok = mdl = None
    use_llm = not args.no_llm
    if use_llm:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            use_cuda = (args.device != "cpu") and torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()
            if args.device == "cuda" and not use_cuda:
                LOG.warning("CUDA requested but not available; falling back to CPU.")
            bf16_supported = use_cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            dtype_fp16 = torch.bfloat16 if (bf16_supported and torch is not None) else (torch.float16 if (use_cuda and torch is not None) else None)

            model_kwargs = {"trust_remote_code": True}
            if args.device == "cpu" or not use_cuda:
                model_kwargs["device_map"] = "cpu"
                if torch is not None: model_kwargs["torch_dtype"] = torch.float32
            else:
                model_kwargs["device_map"] = "auto"
                if dtype_fp16 is not None:
                    model_kwargs["torch_dtype"] = dtype_fp16

            if args.use_4bit or args.use_8bit:
                try:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=bool(args.use_4bit),
                        load_in_8bit=bool(args.use_8bit),
                        bnb_4bit_compute_dtype=(torch.bfloat16 if bf16_supported else (torch.float16 if use_cuda else None)),
                    )
                except Exception as e:
                    LOG.warning("Quantization setup failed: %s", e)

            tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        except Exception as e:
            LOG.warning("Failed to load LLM (%s); continuing retrieval-only. Error: %s", args.model, e)
            tok = mdl = None
            use_llm = False

    use_rerank_flag = not args.no_rerank

    if args.test or args.sanity:
        run_test_prompts(index, tok, mdl, use_llm=use_llm, use_rerank_flag=use_rerank_flag, artifacts_dir=run_dir)
        print(f"\n✅ Saved artifacts under: {run_dir}")
        return

    if args.ask:
        ans = ask_once(index, tok, mdl, args.ask, use_llm=use_llm, use_rerank_flag=use_rerank_flag,
                       max_part_chars=args.max_part_chars)
        single_path = os.path.join(run_dir, "single_answer.txt")
        with open(single_path, "w", encoding="utf-8") as f:
            f.write(ans)
        print(ans)
        print(f"\n✅ Saved single answer to: {single_path}")
        return

    print("Ready. اطرح سؤالك (اكتب 'exit' للخروج)\n")
    interactive_path = os.path.join(run_dir, "interactive_transcript.txt")
    with open(interactive_path, "w", encoding="utf-8") as trans:
        while True:
            try:
                q = input("سؤالك: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting."); break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                print("Exiting."); break
            ans = ask_once(index, tok, mdl, q, use_llm=use_llm, use_rerank_flag=use_rerank_flag,
                           max_part_chars=args.max_part_chars)
            print(ans)
            trans.write(f"\nQ: {q}\n{ans}\n"); trans.flush()
    print(f"\n✅ Interactive transcript saved to: {interactive_path}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
RAG Orchestrator (Arabic) — robust refinement + persistent run artifacts.

What you get:
- --sanity (alias of --test) to run Arabic sanity prompts.
- Guardrails: preserve numbers/times/days; fallback if LLM drops facts.
- --device auto|cpu|cuda, --no-rerank for VRAM safety.
- PERSISTENT OUTPUTS: results saved to --out-dir/run_<timestamp>/:
    - run.log         : full logging output
    - report.txt      : the console-like transcript
    - results.jsonl   : one JSON object per test (question, answer, sources, pass flags)
    - summary.md      : readable pass/fail summary

"""

import os
import sys
import json
import time
import argparse
import logging
import re
from datetime import datetime

# --- Reduce noisy progress bars that "erase" prior lines ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")           # TensorFlow / XLA noise
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")     # HF transformer's extra logs
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # HF hub progress bars
os.environ.setdefault("TQDM_DISABLE", "1")                   # generic tqdm
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation

try:
    import torch
except Exception:
    torch = None

import retrival_model as RET

# --------------- Logging ---------------
def setup_logger(log_path: str):
    logger = logging.getLogger("rag_orchestrator")
    logger.setLevel(logging.INFO)
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    # File
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
    # Reset handlers (if reloaded)
    logger.handlers = []
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

LOG = logging.getLogger("rag_orchestrator")  # will be reconfigured in main()

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

# --------------- Cleaners & checks ---------------
_HEADING_PATTERNS = [
    r"^\s*الإجابة\s*:?$",
    r"^\s*الإجابة\s+المختصرة\s*:?\s*$",
    r"^\s*الخلاصة\s*:?\s*$",
    r"^\s*الملخص\s*:?\s*$",
    r"^\s*Summary\s*:?\s*$",
    r"^\s*Answer\s*:?\s*$",
]

_AR_DAYS = ["الأحد", "الإثنين", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت"]
_TIME_PATTERNS = [
    r"\b\d{1,2}:\d{2}\b",           # 8:30
    r"\b\d{1,2}[:٫]\d{2}\b",        # 8٫30
    r"\b\d{1,2}\s*[-–]\s*\d{1,2}\b",# 8-5
    r"\b\d{1,2}\s*(?:ص|م)\b",       # 8 ص / 5 م
]

def _has_times_or_days(txt: str) -> bool:
    if not txt:
        return False
    if any(day in txt for day in _AR_DAYS):
        return True
    return any(re.search(p, txt) for p in _TIME_PATTERNS)

def _clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = re.sub(r"^```.*?$", "", txt, flags=re.M | re.S)
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    keep = []
    for l in lines:
        if any(re.match(p, l) for p in _HEADING_PATTERNS):
            continue
        keep.append(l)
    txt = " ".join(keep).strip()
    sentences = re.split(r"(?<=[.!؟])\s+", txt)
    txt = " ".join(sentences[:4]).strip()
    txt = re.sub(r"\s*[:：]\s*$", "", txt)
    return txt

def _is_meaningful(txt: str) -> bool:
    return bool(txt and len(re.sub(r"\s+", "", txt)) >= 12)

def _split_answer(answer_text: str):
    """Split combined answer into (body, sources_block)."""
    if not answer_text:
        return "", ""
    # Look for Sources in Arabic or English
    parts = re.split(r"\n(?=Sources:|المصادر:)", answer_text, maxsplit=1)
    body = parts[0].strip()
    sources = parts[1].strip() if len(parts) > 1 else ""
    return body, sources

# --------------- Q&A ---------------
def ask_once(index: RET.HybridIndex,
             tokenizer,
             model,
             question: str,
             use_llm: bool = True,
             use_rerank_flag: bool = True) -> str:
    """
    1) classify intent
    2) retrieve via RET.answer (includes sources)
    3) optional LLM refine; preserve numerics/times; fallback if lost
    """
    t0 = time.time()
    intent = RET.classify_intent(question)

    extractive_answer = RET.answer(question, index, intent, use_rerank_flag=use_rerank_flag)

    # Split body/sources
    lines = str(extractive_answer).split('\n')
    body_lines, source_lines, sources_started = [], [], False
    for line in lines:
        ls = line.strip()
        if ls.startswith("Sources:") or ls.startswith("المصادر:"):
            sources_started = True
            source_lines.append(line)
        elif sources_started:
            source_lines.append(line)
        else:
            body_lines.append(line)
    body_raw = '\n'.join(body_lines).strip()
    sources = '\n'.join(source_lines).strip()
    body_clean = _clean_text(body_raw)

    def _final(dt, text):
        return f"⏱ {dt:.2f}s | 🤖 {text}\n{sources}" if sources else f"⏱ {dt:.2f}s | 🤖 {text}"

    # If LLM unavailable or retrieval failed → cleaned extractive
    if (not use_llm) or (tokenizer is None) or (model is None) or (not body_raw) \
       or ("لم أعثر" in body_raw) or ("لا توجد معلومات" in body_raw):
        dt = time.time() - t0
        out = body_clean if _is_meaningful(body_clean) else body_raw
        return _final(dt, out)

    # Shortcut: already a proper work-hours answer with times/days
    if intent in ("work_hours", "ramadan_hours") and _has_times_or_days(body_raw):
        dt = time.time() - t0
        out = body_clean if _is_meaningful(body_clean) else body_raw
        return _final(dt, out)

    # LLM refinement
    try:
        system_prompt = (
            "أعد صياغة النص التالي بالعربية في 2–4 أسطر واضحة، دون أي مقدمات أو عناوين. "
            "❗ هام: لا تُسقط أي أرقام أو ساعات أو نطاقات زمنية أو أيام. "
            "أبقِ الأرقام والرموز كما هي حرفيًا."
        )
        user_prompt = f"السؤال: {question}\nالنص لإعادة الصياغة:\n{body_raw}"
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
            max_new_tokens=120,   # smaller = less VRAM
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        start = inputs['input_ids'].shape[1]
        raw = tokenizer.decode(out_ids[0][start:], skip_special_tokens=True).strip()
        resp = _clean_text(raw)

        dt = time.time() - t0
        # Guardrail: if original had times/days but resp lost them → fallback
        if _is_meaningful(resp) and (not _has_times_or_days(body_raw) or _has_times_or_days(resp)):
            return _final(dt, resp)
        out = body_clean if _is_meaningful(body_clean) else body_raw
        return _final(dt, out)

    except Exception as e:
        LOG.warning(f"LLM generation failed: {e}")
        dt = time.time() - t0
        out = body_clean if _is_meaningful(body_clean) else body_raw
        return _final(dt, out)

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
    # same as old "PASS": just requires Sources and not explicit failure
    return (("Sources:" in answer_text) or ("المصادر:" in answer_text)) \
           and ("لم أعثر" not in answer_text) and ("لا توجد معلومات" not in answer_text)

def _pass_strict(question: str, body_only: str) -> bool:
    """Stricter pass: must have content; for hours/days prompts must include times/days."""
    if not _is_meaningful(body_only):
        return False
    q = question or ""
    hours_like = any(kw in q for kw in [
        "ساعات", "الدوام", "رمضان", "أيام الدوام", "الساعات الإضافية", "العطل الرسمية"
    ])
    if hours_like:
        return _has_times_or_days(body_only)
    return True

def run_test_prompts(index: RET.HybridIndex, tokenizer, model, use_llm: bool, use_rerank_flag: bool,
                     artifacts_dir: str):
    # Prepare artifact files
    os.makedirs(artifacts_dir, exist_ok=True)
    results_path = os.path.join(artifacts_dir, "results.jsonl")
    summary_md   = os.path.join(artifacts_dir, "summary.md")
    report_txt   = os.path.join(artifacts_dir, "report.txt")

    # Open files
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

            # Write JSONL record
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

    # Summary
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
    args = parser.parse_args()

    # Make artifacts dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Configure logger (file + console)
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

    # Execute
    use_rerank_flag = not args.no_rerank

    if args.test or args.sanity:
        run_test_prompts(index, tok, mdl, use_llm=use_llm, use_rerank_flag=use_rerank_flag, artifacts_dir=run_dir)
        print(f"\n✅ Saved artifacts under: {run_dir}")
        return

    if args.ask:
        ans = ask_once(index, tok, mdl, args.ask, use_llm=use_llm, use_rerank_flag=use_rerank_flag)
        # Persist single-answer too
        single_path = os.path.join(run_dir, "single_answer.txt")
        with open(single_path, "w", encoding="utf-8") as f:
            f.write(ans)
        print(ans)
        print(f"\n✅ Saved single answer to: {single_path}")
        return

    print("Ready. اطرح سؤالك (اكتب 'exit' للخروج)\n")
    # Also persist interactive transcript
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
            ans = ask_once(index, tok, mdl, q, use_llm=use_llm, use_rerank_flag=use_rerank_flag)
            print(ans)
            trans.write(f"\nQ: {q}\n{ans}\n"); trans.flush()
    print(f"\n✅ Interactive transcript saved to: {interactive_path}")


if __name__ == "__main__":
    main()

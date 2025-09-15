# -*- coding: utf-8 -*-
"""
RAG Orchestrator (Arabic) — runs sanity prompts easily.
- Adds --sanity flag (alias of --test).
- If RET.SANITY_PROMPTS is missing, falls back to DEFAULT_SANITY_PROMPTS (below).
- Retrieval-first; optional LLM refinement via Transformers (can be disabled with --no-llm).
"""

import os
import time
import argparse
import logging

# ---- Optional: reduce TF/XLA noise (safe if TF not installed) ----
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all,1=info,2=warning,3=error

# Optional torch (for dtype/device checks)
try:
    import torch
except Exception:
    torch = None

# Your retriever module (as in your original code)
import retrival_model as RET

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger("rag_orchestrator")

# ---------------- Built-in sanity prompts (fallback) ----------------
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

# ---------------- Answer Handler (Minimal interference) ----------------
def ask_once(index: RET.HybridIndex,
             tokenizer,
             model,
             question: str,
             use_llm: bool = True) -> str:
    """
    One Q&A round:
      1) classify intent via RET.classify_intent
      2) get extractive answer via RET.answer (includes sources)
      3) optional: refine wording with LLM, preserving sources
    """
    t0 = time.time()
    intent = RET.classify_intent(question)

    # Retrieval-first
    extractive_answer = RET.answer(question, index, intent, use_rerank_flag=True)

    # If LLM disabled/unavailable → return extractive
    if not use_llm or tokenizer is None or model is None:
        dt = time.time() - t0
        if isinstance(extractive_answer, str) and extractive_answer.startswith("⏱"):
            return extractive_answer
        return f"⏱ {dt:.2f}s | 🤖 {extractive_answer}"

    # For work-hours style outputs that are already clean, don't over-process
    if intent in ("work_hours", "ramadan_hours") and ("ساعات الدوام" in extractive_answer) and ("من" in extractive_answer) and ("إلى" in extractive_answer):
        dt = time.time() - t0
        if extractive_answer.startswith("⏱"):
            return extractive_answer
        return f"⏱ {dt:.2f}s | 🤖 {extractive_answer}"

    # Split body/sources to preserve citations
    lines = extractive_answer.split('\n')
    body_lines, source_lines = [], []
    sources_started = False
    for line in lines:
        ls = line.strip()
        if ls.startswith("Sources:") or ls.startswith("المصادر:"):
            sources_started = True
            source_lines.append(line)
        elif sources_started:
            source_lines.append(line)
        else:
            body_lines.append(line)

    body = '\n'.join(body_lines).strip()
    sources = '\n'.join(source_lines).strip()

    # If retrieval failed, skip LLM
    if not body or "لم أعثر" in body or "لا توجد معلومات" in body:
        dt = time.time() - t0
        if extractive_answer.startswith("⏱"):
            return extractive_answer
        return f"⏱ {dt:.2f}s | 🤖 {extractive_answer}"

    # LLM refinement (Arabic)
    try:
        from transformers import PreTrainedTokenizerBase
        system_prompt = "أعد صياغة الإجابة التالية بشكل واضح ومختصر باللغة العربية:"
        user_prompt = f"السؤال: {question}\nالإجابة: {body}"

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Not all tokenizers expose apply_chat_template
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"[system]\n{system_prompt}\n\n[user]\n{user_prompt}\n\n[assistant]\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        # Move to device
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = eos_id if eos_id is not None else getattr(tokenizer, "pad_token_id", None)

        out_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        start = inputs['input_ids'].shape[1]
        resp = tokenizer.decode(out_ids[0][start:], skip_special_tokens=True).strip()
        # Keep the first line concise
        resp_line = resp.split('\n')[0].strip() if resp else ""

        dt = time.time() - t0
        if resp_line and len(resp_line) > 5:
            return f"⏱ {dt:.2f}s | 🤖 {resp_line}\n{sources}" if sources else f"⏱ {dt:.2f}s | 🤖 {resp_line}"
    except Exception as e:
        LOG.warning(f"LLM generation failed: {e}")

    # Fallback: extractive
    dt = time.time() - t0
    if extractive_answer.startswith("⏱"):
        return extractive_answer
    return f"⏱ {dt:.2f}s | 🤖 {extractive_answer}"


def _gather_sanity_prompts() -> list:
    """Merge RET.SANITY_PROMPTS (if any) with DEFAULT_SANITY_PROMPTS, preserving order and uniqueness."""
    ret_prompts = []
    try:
        ret_prompts = list(getattr(RET, "SANITY_PROMPTS", []) or [])
    except Exception:
        ret_prompts = []
    seen = set()
    merged = []
    for q in (ret_prompts + DEFAULT_SANITY_PROMPTS):
        if q not in seen:
            seen.add(q)
            merged.append(q)
    return merged


def run_test_prompts(index: RET.HybridIndex, tokenizer, model, use_llm: bool):
    """
    Run sanity prompts (merged list). PASS heuristic: includes 'Sources:' and not a generic fail string.
    """
    test_prompts = _gather_sanity_prompts()
    if not test_prompts:
        print("❌ No sanity prompts available.")
        return

    print("🧪 Running sanity prompts ...")
    print("=" * 80)

    passed = 0
    total = len(test_prompts)
    for i, q in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}/{total}: {q}")
        print("-" * 60)
        try:
            result = ask_once(index, tokenizer, model, q, use_llm=use_llm)
            print(result)
            ok = ("Sources:" in result or "المصادر:" in result) and ("لم أعثر" not in result) and ("لا توجد معلومات" not in result)
            print("✅ PASS" if ok else "❌ FAIL")
            passed += int(ok)
        except Exception as e:
            print(f"❌ Error: {e}")
        print("=" * 80)

    print(f"\nSummary: PASS {passed}/{total}")


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default="Data_pdf_clean_chunks.jsonl",
                    help="Path to chunks (JSONL or JSON) used by retriever")
    ap.add_argument("--hier-index", type=str, default="heading_inverted_index.json",
                    help="Optional hierarchical inverted index path")
    ap.add_argument("--aliases", type=str, default="section_aliases.json",
                    help="Optional section aliases path")
    ap.add_argument("--save-index", type=str, default=None,
                    help="Directory to save index artifacts (embeddings/FAISS/TF-IDF)")
    ap.add_argument("--load-index", type=str, default=None,
                    help="Directory to load index artifacts from")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="HF model id for optional refinement")
    ap.add_argument("--ask", type=str, default=None,
                    help="Ask a single question then exit")
    ap.add_argument("--test", action="store_true",
                    help="Run sanity prompts (alias: --sanity)")
    ap.add_argument("--sanity", action="store_true",
                    help="Alias for --test (runs sanity prompts)")
    ap.add_argument("--no-llm", action="store_true",
                    help="Disable LLM refinement (retrieval-only)")
    ap.add_argument("--use-4bit", action="store_true",
                    help="Try 4-bit quantization (requires bitsandbytes)")
    ap.add_argument("--use-8bit", action="store_true",
                    help="Try 8-bit quantization (requires bitsandbytes)")
    args = ap.parse_args()

    # Build/load index via RET
    hier = RET.load_hierarchy(args.hier_index, args.aliases)

    if not os.path.exists(args.chunks):
        LOG.error("Chunks file not found: %s", args.chunks)
        return

    chunks, chunks_hash = RET.load_chunks(path=args.chunks)
    index = RET.HybridIndex(chunks, chunks_hash, hier=hier)

    loaded = False
    if args.load_index and os.path.exists(args.load_index):
        try:
            # Reduce noise while loading
            retrieval_logger = logging.getLogger("retrival_model")
            original_level = retrieval_logger.level
            retrieval_logger.setLevel(logging.ERROR)

            loaded = index.load(args.load_index)

            retrieval_logger.setLevel(original_level)
            if loaded:
                LOG.info("Index loaded successfully from %s", args.load_index)
        except Exception as e:
            LOG.info(f"Will rebuild index: {e}")

    if not loaded:
        LOG.info("Building index ...")
        index.build()
        if args.save_index:
            try:
                index.save(args.save_index)
                LOG.info("Index saved to %s", args.save_index)
            except Exception as e:
                LOG.warning(f"Failed to save index: {e}")

    # Optional LLM
    tok = mdl = None
    use_llm = not args.no_llm
    if use_llm:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

            # Default dtype/device
            bf16_supported = False
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            dtype_fp16 = None
            if torch is not None:
                dtype_fp16 = torch.bfloat16 if bf16_supported else torch.float16

            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": dtype_fp16 or None,
            }

            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = "cpu"
                if torch is not None:
                    model_kwargs["torch_dtype"] = torch.float32

            if args.use_4bit or args.use_8bit:
                try:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True if args.use_4bit else False,
                        load_in_8bit=True if args.use_8bit else False,
                        bnb_4bit_compute_dtype=(torch.bfloat16 if (torch is not None and bf16_supported) else (torch.float16 if torch is not None else None)),
                    )
                    model_kwargs["quantization_config"] = quant_config
                except Exception as e:
                    LOG.warning(f"Quantization setup failed: {e}")

            tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        except Exception as e:
            LOG.warning(f"Failed to load LLM (%s); continuing retrieval-only. Error: %s", args.model, e)
            tok = mdl = None
            use_llm = False

    # Run sanity prompts (alias: --sanity)
    if args.test or args.sanity:
        run_test_prompts(index, tok, mdl, use_llm=use_llm)
        return

    # Single-question mode
    if args.ask:
        print(ask_once(index, tok, mdl, args.ask, use_llm=use_llm))
        return

    # Interactive loop
    print("Ready. اطرح سؤالك (اكتب 'exit' للخروج)\n")
    while True:
        try:
            q = input("سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("Exiting.")
            break
        print(ask_once(index, tok, mdl, q, use_llm=use_llm))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
RAG Orchestrator (Arabic) — uses RET.SANITY_PROMPTS from retrival_model.py
- Reuses the retriever's sanity prompts (single source of truth)
- Adds --sanity flag as an alias for --test
- Optional LLM refinement through Transformers (can be disabled via --no-llm)
"""

import os
import time
import argparse
import logging
import torch

# Your retriever module
import retrival_model as RET

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger("rag_orchestrator")

# ---------------- Answer Handler (Minimal interference) ----------------

def ask_once(index: RET.HybridIndex,
             tokenizer,
             model,
             question: str,
             use_llm: bool = True) -> str:
    """
    Runs one Q&A round:
      1) classifies intent with RET.classify_intent
      2) gets extractive answer via RET.answer (includes sources)
      3) optional: refines wording with LLM, preserving sources
    """
    t0 = time.time()
    intent = RET.classify_intent(question)

    # Use your retrieval system first
    extractive_answer = RET.answer(question, index, intent, use_rerank_flag=True)

    # If not using LLM (or tokenizer/model missing), return extractive answer
    if not use_llm or tokenizer is None or model is None:
        dt = time.time() - t0
        if extractive_answer.startswith("⏱"):
            return extractive_answer
        return f"⏱ {dt:.2f}s | 🤖 {extractive_answer}"

    # If it's already a clean work-hours statement, don't over-process
    if intent in ("work_hours", "ramadan_hours") and ("ساعات الدوام" in extractive_answer) and ("من" in extractive_answer) and ("إلى" in extractive_answer):
        dt = time.time() - t0
        if extractive_answer.startswith("⏱"):
            return extractive_answer
        return f"⏱ {dt:.2f}s | 🤖 {extractive_answer}"

    # Split body and sources to keep citations intact
    lines = extractive_answer.split('\n')
    body_lines = []
    source_lines = []
    sources_started = False
    for line in lines:
        ls = line.strip()
        if ls.startswith("Sources:") or ls.startswith("المصادر:"):
            sources_started = True
            source_lines.append(line)
        elif sources_started and (ls == "" or ls[:1].isdigit() or "Data_pdf.pdf" in ls):
            source_lines.append(line)
        elif not sources_started:
            body_lines.append(line)
    body = '\n'.join(body_lines).strip()
    sources = '\n'.join(source_lines).strip()

    # If retrieval failed, don't try LLM
    if not body or "لم أعثر" in body or "لا توجد معلومات" in body:
        dt = time.time() - t0
        if extractive_answer.startswith("⏱"):
            return extractive_answer
        return f"⏱ {dt:.2f}s | 🤖 {extractive_answer}"

    # LLM refinement (Arabic)
    try:
        system_prompt = "أعد صياغة الإجابة التالية بشكل واضح ومختصر باللغة العربية:"
        user_prompt = f"السؤال: {question}\nالإجابة: {body}"

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        out_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        resp = tokenizer.decode(out_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        resp = resp.split('\n')[0].strip()

        dt = time.time() - t0
        if resp and len(resp) > 5:
            return f"⏱ {dt:.2f}s | 🤖 {resp}\n{sources}" if sources else f"⏱ {dt:.2f}s | 🤖 {resp}"
    except Exception as e:
        LOG.warning(f"LLM generation failed: {e}")

    # Fallback to extractive answer
    dt = time.time() - t0
    if extractive_answer.startswith("⏱"):
        return extractive_answer
    return f"⏱ {dt:.2f}s | 🤖 {extractive_answer}"


def run_test_prompts(index: RET.HybridIndex, tokenizer, model, use_llm: bool):
    """
    Runs all sanity prompts defined in retrival_model.py (RET.SANITY_PROMPTS).
    Uses a simple heuristic for "pass": has "Sources:" and not a generic fail.
    """
    test_prompts = getattr(RET, "SANITY_PROMPTS", [])
    if not test_prompts:
        print("❌ No SANITY_PROMPTS found in retrival_model.py")
        return

    print("🧪 Running sanity prompts from retrival_model.SANITY_PROMPTS ...")
    print("=" * 80)

    passed = 0
    total = len(test_prompts)
    for i, q in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}/{total}: {q}")
        print("-" * 60)
        try:
            result = ask_once(index, tokenizer, model, q, use_llm=use_llm)
            print(result)
            ok = ("Sources:" in result) and ("لم أعثر" not in result)
            passed += int(ok)
            print("✅ PASS" if ok else "❌ FAIL")
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
                    help="Run all sanity prompts (from retrival_model.py)")
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
    use_llm = not args.no-llm if hasattr(args, "no-llm") else not args.no_llm  # guard in case of dash attr
    use_llm = not args.no_llm  # final
    if use_llm:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            }

            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = "cpu"
                model_kwargs["torch_dtype"] = torch.float32

            if args.use_4bit or args.use_8bit:
                try:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True if args.use_4bit else False,
                        load_in_8bit=True if args.use_8bit else False,
                        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                    )
                    model_kwargs["quantization_config"] = quant_config
                except Exception as e:
                    LOG.warning(f"Quantization setup failed: {e}")

            tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        except Exception as e:
            LOG.warning(f"Failed to load LLM ({args.model}); continuing retrieval-only. Error: {e}")
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

    # Interactive mode
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

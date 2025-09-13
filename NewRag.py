# -*- coding: utf-8 -*-
"""
RAG Orchestrator (Arabic) — Fixed version that properly uses your retrieval system
"""

import os, re, time, argparse, logging
import torch
import hashlib
from typing import List, Tuple

# Your retriever module
import retrival_model as RET

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger("rag_qwen_fixed")

# ---------------- Source block helpers ----------------

def join_sources(srcs: List[str]) -> str:
    if not srcs:
        return ""
    return "المصادر:\n" + "\n".join(srcs)

def split_sources_block(extractive: str) -> Tuple[str, List[str]]:
    """
    Split body and sources from your retriever's textual answer.
    """
    t = (extractive or "").strip()
    if not t:
        return "", []
    # Find the first occurrence of either marker
    pos_sources = []
    for mark in ("Sources:", "المصادر:"):
        p = t.find(mark)
        if p != -1:
            pos_sources.append(p)
    if not pos_sources:
        return t, []
    pos = min(pos_sources)
    body = t[:pos].strip()
    tail = t[pos:].splitlines()
    # skip the header line itself
    tail = tail[1:] if len(tail) > 0 else []
    # keep lines that look like enumerated sources
    srcs = []
    for ln in tail:
        s = ln.strip()
        if not s:
            continue
        if s[0].isdigit() or s.startswith("-") or "Data_pdf.pdf" in s:
            srcs.append(s)
    return body, srcs

# ---------------- Work Hours Specific Handler ----------------

def extract_work_hours_answer(index: RET.HybridIndex, question: str, intent: str) -> str:
    """
    Directly use your retrieval system's built-in work hours extraction
    """
    # Use your retrieval system's answer function with rerank enabled
    extractive_answer = RET.answer(question, index, intent, use_rerank_flag=True)
    
    # If the extractive answer contains actual work hours, return it
    if "ساعات الدوام" in extractive_answer and ("من" in extractive_answer and "إلى" in extractive_answer):
        return extractive_answer
    
    # If not, try to extract time ranges manually from the retrieved chunks
    hits = RET.retrieve(index, question, rerank=True)
    if hits:
        chunks = index.chunks
        # Use your retrieval system's compose hours answer function directly
        composed = RET._compose_hours_answer(chunks, hits, intent)
        if composed:
            return composed
    
    # Fallback to the original extractive answer
    return extractive_answer

# ---------------- Answer Handler ----------------

def ask_once(index: RET.HybridIndex,
             tokenizer, model,
             question: str,
             use_llm: bool = True) -> str:
    t0 = time.time()
    intent = RET.classify_intent(question)
    
    # Special handling for work hours questions - use your retrieval system directly
    if intent in ("work_hours", "ramadan_hours"):
        result = extract_work_hours_answer(index, question, intent)
        # Add timing if not present
        if not result.startswith("⏱"):
            dt = time.time() - t0
            if not result.startswith("🤖"):
                result = f"⏱ {dt:.2f}s | 🤖 {result}"
        return result
    
    # For other intents, use your retrieval system's answer function
    extractive_answer = RET.answer(question, index, intent, use_rerank_flag=True)
    
    # If not using LLM, return extractive answer directly
    if not use_llm or tokenizer is None or model is None:
        dt = time.time() - t0
        # Add timing if not present
        if not extractive_answer.startswith("⏱"):
            return f"⏱ {dt:.2f}s | 🤖 {extractive_answer}"
        return extractive_answer
    
    # Use LLM for refinement (if enabled)
    body, sources = split_sources_block(extractive_answer)
    if not body.strip():
        dt = time.time() - t0
        return f"⏱ {dt:.2f}s | 🤖 لا توجد معلومات كافية في السياق للإجابة على هذا السؤال."
    
    # Simple LLM refinement
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
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(out_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        response = response.split('\n')[0].strip()
        
        if response and len(response) > 5:
            dt = time.time() - t0
            return f"⏱ {dt:.2f}s | 🤖 {response}\n{join_sources(sources[:3])}"
            
    except Exception as e:
        LOG.warning(f"LLM generation failed: {e}")
    
    # Fallback to extractive answer
    dt = time.time() - t0
    return f"⏱ {dt:.2f}s | 🤖 {body}\n{join_sources(sources[:3])}"

def run_sanity(index: RET.HybridIndex, tokenizer, model, use_llm: bool):
    print("\n🧪 Sanity run…\n")
    test_questions = [
        "ما هي ساعات الدوام الرسمية من وإلى؟",
        "ما ساعات العمل في شهر رمضان؟",
        "هل يوجد مرونة في الحضور والانصراف؟",
        "كم مدة الإجازة السنوية؟"
    ]
    for q in test_questions:
        print(f"• {q}")
        out = ask_once(index, tokenizer, model, q, use_llm=use_llm)
        print(out, "\n")

# ---------------- File hash helper (to match your retrieval system) ----------------

def _file_hash(path: str) -> str:
    """Calculate file hash - copied from retrieval model to ensure consistency"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1<<20)
            if not b: break
            h.update(b)
    return h.hexdigest()

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default="Data_pdf_clean_chunks.jsonl")
    ap.add_argument("--hier-index", type=str, default="heading_inverted_index.json")
    ap.add_argument("--aliases", type=str, default="section_aliases.json")
    ap.add_argument("--save-index", type=str, default=None)
    ap.add_argument("--load-index", type=str, default=None)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--ask", type=str, default=None)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--use-8bit", action="store_true")
    args = ap.parse_args()

    # Build/load index from your retriever - ensure hash consistency
    hier = RET.load_hierarchy(args.hier_index, args.aliases)
    
    # Load chunks and calculate hash (using your retrieval system's function)
    if not os.path.exists(args.chunks):
        LOG.error("Chunks file not found: %s", args.chunks)
        return
    
    chunks, chunks_hash = RET.load_chunks(path=args.chunks)
    index = RET.HybridIndex(chunks, chunks_hash, hier=hier)

    # Try to load existing index with better error handling
    loaded = False
    if args.load_index and os.path.exists(args.load_index):
        try:
            # Temporarily suppress warnings to avoid hash mismatch noise
            import logging
            retrieval_logger = logging.getLogger("retrival_model")
            original_level = retrieval_logger.level
            retrieval_logger.setLevel(logging.ERROR)
            
            loaded = index.load(args.load_index)
            
            # Restore original logging level
            retrieval_logger.setLevel(original_level)
            
            if loaded:
                LOG.info("Index loaded successfully")
            else:
                LOG.info("Will rebuild index")
        except Exception as e:
            LOG.warning(f"Could not load index, will rebuild: {e}")
    
    # Build index if not loaded
    if not loaded:
        LOG.info("Building index...")
        index.build()
        if args.save_index:
            try:
                index.save(args.save_index)
                LOG.info("Index saved successfully")
            except Exception as e:
                LOG.warning(f"Failed to save index: {e}")

    # Optional LLM
    tok = mdl = None
    use_llm = not args.no_llm
    if use_llm:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
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
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                )
                model_kwargs["quantization_config"] = quant_config
            except Exception as e:
                LOG.warning(f"Quantization failed: {e}")

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # Modes
    if args.sanity:
        run_sanity(index, tok, mdl, use_llm=use_llm)
        return

    if args.ask:
        print(ask_once(index, tok, mdl, args.ask, use_llm=use_llm))
        return

    # Interactive
    print("Ready. اطرح سؤالك (اكتب 'exit' للخروج)\n")
    while True:
        try:
            q = input("سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break
        if not q:
            continue
        if q.lower() in ("exit","quit","q"):
            print("Exiting."); break
        print(ask_once(index, tok, mdl, q, use_llm=use_llm))

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
RAG Orchestrator (Arabic) — Fixed version with direct work hours extraction
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

# ---------------- File hash helper ----------------

def _file_hash(path: str) -> str:
    """Calculate file hash - copied from retrieval model to ensure consistency"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1<<20)
            if not b: break
            h.update(b)
    return h.hexdigest()

# ---------------- Work Hours Specific Handler ----------------

def extract_work_hours_answer(index: RET.HybridIndex, question: str, intent: str) -> str:
    """
    Directly extract work hours information using the retrieval model's built-in functions
    """
    # Use the retrieval model's answer function first
    extractive_answer = RET.answer(question, index, intent, use_rerank_flag=True)
    
    # Try to extract time ranges from the extractive answer
    body, sources = split_sources_block(extractive_answer)
    
    # Look for time ranges in the body
    if body:
        ranges = RET.extract_all_ranges(body, intent)
        if ranges:
            best_range = RET.pick_best_range(ranges)
            if best_range:
                a, b = best_range
                suffix = " في شهر رمضان" if intent == "ramadan_hours" else ""
                answer = f"ساعات الدوام{suffix} من {a} إلى {b}."
                return f"⏱ 0.10s | 🤖 {answer}\n{join_sources(sources[:3])}"
    
    # If that fails, do a more thorough search through retrieved chunks
    hits = RET.retrieve(index, question, rerank=True)
    if hits:
        for _, i in hits[:15]:  # Check top 15 hits
            chunk = index.chunks[i]
            sentences = RET.sent_split(chunk.text)
            for s in sentences:
                # Check for time ranges in each sentence
                ranges = RET.extract_all_ranges(s, intent)
                if ranges:
                    best_range = RET.pick_best_range(ranges)
                    if best_range:
                        a, b = best_range
                        suffix = " في شهر رمضان" if intent == "ramadan_hours" else ""
                        answer = f"ساعات الدوام{suffix} من {a} إلى {b}."
                        sources = [f"1. Data_pdf.pdf - page {chunk.page}"]
                        return f"⏱ 0.15s | 🤖 {answer}\n{join_sources(sources)}"
    
    # Manual search for common work hour patterns
    work_hour_patterns = [
        r"من\s*(\d{1,2}[:.]\d{2}?)\s*(?:الى|إلى|حتى)\s*(\d{1,2}[:.]\d{2}?)",
        r"(\d{1,2}[:.]\d{2}?)\s*(?:الى|إلى|حتى)\s*(\d{1,2}[:.]\d{2}?)",
        r"الساعة\s*(\d{1,2}[:.]\d{2}?)\s*(?:الى|إلى|حتى)\s*(\d{1,2}[:.]\d{2}?)"
    ]
    
    if hits:
        for _, i in hits[:10]:
            chunk = index.chunks[i]
            text = chunk.text
            for pattern in work_hour_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    for match in matches:
                        start_time, end_time = match
                        # Normalize time format
                        start_time = start_time.replace('.', ':')
                        end_time = end_time.replace('.', ':')
                        if ':' not in start_time:
                            start_time += ":00"
                        if ':' not in end_time:
                            end_time += ":00"
                        suffix = " في شهر رمضان" if intent == "ramadan_hours" else ""
                        answer = f"ساعات الدوام{suffix} من {start_time} إلى {end_time}."
                        sources = [f"1. Data_pdf.pdf - page {chunk.page}"]
                        return f"⏱ 0.20s | 🤖 {answer}\n{join_sources(sources)}"
    
    # If we still can't find specific hours, fall back to the extractive answer
    if body and len(body) > 10:
        # Check if the body contains work hour keywords
        work_keywords = ["ساعات", "دوام", "العمل", "من", "الى", "إلى", "حتى"]
        normalized_body = RET.ar_normalize(body)
        if any(keyword in normalized_body for keyword in work_keywords):
            return f"⏱ 0.25s | 🤖 {body}\n{join_sources(sources[:3])}"
    
    # Ultimate fallback - search for any sentence with work hour keywords
    if hits:
        for _, i in hits[:5]:
            chunk = index.chunks[i]
            sentences = RET.sent_split(chunk.text)
            for s in sentences:
                normalized_s = RET.ar_normalize(s)
                if any(keyword in normalized_s for keyword in ["ساعات", "دوام", "العمل"]):
                    return f"⏱ 0.30s | 🤖 {s}\nالمصادر:\n1. Data_pdf.pdf - page {chunk.page}"
    
    # If nothing works, return a more informative message
    return f"⏱ 0.35s | 🤖 لا توجد معلومات كافية عن ساعات الدوام في المستندات المتوفرة."

# ---------------- General Answer Handler ----------------

def ask_once(index: RET.HybridIndex,
             tokenizer, model,
             question: str,
             use_llm: bool = True) -> str:
    t0 = time.time()
    intent = RET.classify_intent(question)
    
    # Special handling for work hours questions
    if intent in ("work_hours", "ramadan_hours"):
        return extract_work_hours_answer(index, question, intent)
    
    # For other intents, use the retrieval model's answer function
    extractive_answer = RET.answer(question, index, intent, use_rerank_flag=True)
    
    # If not using LLM, return extractive answer directly
    if not use_llm or tokenizer is None or model is None:
        dt = time.time() - t0
        # Just clean up the timing
        if extractive_answer.startswith("⏱"):
            return extractive_answer
        return f"⏱ {dt:.2f}s | 🤖 {extractive_answer}"
    
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

    # Build/load index from your retriever
    hier = RET.load_hierarchy(args.hier_index, args.aliases)
    
    # Load chunks and calculate hash
    if not os.path.exists(args.chunks):
        LOG.error("Chunks file not found: %s", args.chunks)
        return
    
    chunks, chunks_hash = RET.load_chunks(path=args.chunks)
    index = RET.HybridIndex(chunks, chunks_hash, hier=hier)

    # Try to load existing index
    loaded = False
    if args.load_index and os.path.exists(args.load_index):
        try:
            loaded = index.load(args.load_index)
            if not loaded:
                LOG.warning("Failed to load index artifacts, will rebuild")
        except Exception as e:
            LOG.warning(f"Error loading index: {e}, will rebuild")
    
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

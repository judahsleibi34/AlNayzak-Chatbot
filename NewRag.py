# -*- coding: utf-8 -*-
"""
RAG Orchestrator (Arabic) — Fixed version with built-in sanity testing
"""

import os, re, time, argparse, logging
import torch
import hashlib
from typing import List, Tuple

# Your retriever module
import retrival_model as RET

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger("rag_qwen_fixed")

# ---------------- Answer Handler (Minimal interference) ----------------

def ask_once(index: RET.HybridIndex,
             tokenizer, model,
             question: str,
             use_llm: bool = True) -> str:
    t0 = time.time()
    intent = RET.classify_intent(question)
    
    # Use your retrieval system directly - minimal interference
    extractive_answer = RET.answer(question, index, intent, use_rerank_flag=True)
    
    # If not using LLM, return retrieval system's answer directly
    if not use_llm or tokenizer is None or model is None:
        dt = time.time() - t0
        # If retrieval already provided timing, use it; otherwise add timing
        if extractive_answer.startswith("⏱"):
            return extractive_answer
        return f"⏱ {dt:.2f}s | 🤖 {extractive_answer}"
    
    # If retrieval system already gave a good work hours answer, don't interfere
    if intent in ("work_hours", "ramadan_hours") and "ساعات الدوام" in extractive_answer and "من" in extractive_answer and "إلى" in extractive_answer:
        dt = time.time() - t0
        if extractive_answer.startswith("⏱"):
            return extractive_answer
        return f"⏱ {dt:.2f}s | 🤖 {extractive_answer}"
    
    # For other cases, try LLM refinement but preserve sources
    # Extract body and sources from retrieval output
    lines = extractive_answer.split('\n')
    body_lines = []
    source_lines = []
    sources_started = False
    
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("Sources:") or line_stripped.startswith("المصادر:"):
            sources_started = True
            source_lines.append(line_stripped)
        elif sources_started and (line_stripped == "" or line_stripped[0].isdigit() or "Data_pdf.pdf" in line_stripped):
            source_lines.append(line_stripped)
        elif not sources_started:
            body_lines.append(line)
    
    body = '\n'.join(body_lines).strip()
    sources = '\n'.join(source_lines).strip()
    
    # If retrieval found nothing useful, don't try LLM
    if not body.strip() or "لم أعثر" in body or "لا توجد معلومات" in body:
        dt = time.time() - t0
        if extractive_answer.startswith("⏱"):
            return extractive_answer
        return f"⏱ {dt:.2f}s | 🤖 {extractive_answer}"
    
    # Use LLM for refinement
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
            if sources:
                return f"⏱ {dt:.2f}s | 🤖 {response}\n{sources}"
            else:
                return f"⏱ {dt:.2f}s | 🤖 {response}"
            
    except Exception as e:
        LOG.warning(f"LLM generation failed: {e}")
    
    # Fallback to retrieval system's answer
    dt = time.time() - t0
    if extractive_answer.startswith("⏱"):
        return extractive_answer
    return f"⏱ {dt:.2f}s | 🤖 {extractive_answer}"

def run_test_prompts(index: RET.HybridIndex, tokenizer, model, use_llm: bool):
    """Run all 30 sanity prompts as test cases"""
    print("🧪 Running all 30 test prompts from your sanity suite...")
    print("=" * 80)
    
    # All 30 sanity prompts from your retrieval system
    test_prompts = [
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
        "هل توجد مياومات/بدل سفر؟ وكيف تُصرف"
    ]
    
    for i, q in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {q}")
        print("-" * 60)
        try:
            result = ask_once(index, tokenizer, model, q, use_llm=use_llm)
            print(result)
        except Exception as e:
            print(f"❌ Error: {e}")
        print("=" * 80)

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
    ap.add_argument("--test", action="store_true", help="Run all 30 built-in test prompts")
    ap.add_argument("--sanity", action="store_true", help="Run all 30 built-in test prompts (alias for --test)")
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

    # Try to load existing index with better error handling
    loaded = False
    if args.load_index and os.path.exists(args.load_index):
        try:
            # Temporarily suppress warnings
            import logging
            retrieval_logger = logging.getLogger("retrival_model")
            original_level = retrieval_logger.level
            retrieval_logger.setLevel(logging.ERROR)
            
            loaded = index.load(args.load_index)
            
            # Restore original logging level
            retrieval_logger.setLevel(original_level)
            
            if loaded:
                LOG.info("Index loaded successfully")
        except Exception as e:
            LOG.info(f"Will rebuild index: {e}")
    
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

    # Test mode - run all 30 sanity prompts (support both --test and --sanity)
    if args.test or args.sanity:
        run_test_prompts(index, tok, mdl, use_llm=use_llm)
        return

    # Single question mode
    if args.ask:
        print(ask_once(index, tok, mdl, args.ask, use_llm=use_llm))
        return

    # Interactive mode
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

# -*- coding: utf-8 -*-
"""
RAG pipeline that makes the retriever decide, and Qwen2.5 only rewrite.

- Uses your existing `retrival_model.py` (HybridIndex + answer())
- Stage 1 (extractive): get intent-aware answer + sources from your retriever
- Stage 2 (editor): Qwen2.5-7B-Instruct rewrites into one cohesive Arabic paragraph,
  strictly constrained to the given evidence. No new facts allowed.
- Safety: validates numbers/time ranges; if mismatch, falls back to extractive.

Usage (Colab/CLI):
  # 1) Build/load index quickly
  python rag_qwen_editor.py --chunks Data_pdf_clean_chunks.jsonl --pdf Data_pdf.pdf \
      --load-index .artifact --save-index .artifact --model Qwen/Qwen2.5-7B-Instruct --sanity

  # 2) Ask one question
  python rag_qwen_editor.py --chunks Data_pdf_clean_chunks.jsonl --pdf Data_pdf.pdf \
      --load-index .artifact --model Qwen/Qwen2.5-7B-Instruct \
      --ask "ما هي ساعات الدوام الرسمية من وإلى؟"

  # 3) Chat loop
  python rag_qwen_editor.py --chunks Data_pdf_clean_chunks.jsonl --pdf Data_pdf.pdf \
      --load-index .artifact --model Qwen/Qwen2.5-7B-Instruct --chat

Dependencies:
  pip install -U transformers accelerate bitsandbytes sentence-transformers faiss-cpu scikit-learn joblib
"""

import os, re, time, json, argparse, logging
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---- import your retriever (you shared this file) ----
# it provides: load_chunks, load_hierarchy, HybridIndex, classify_intent, answer, ar_normalize
import retrival_model as RET

LOG = logging.getLogger("rag_qwen_editor")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# -------------------- small utils --------------------

RTL_L = "\u202B"
RTL_R = "\u202C"

def strip_rtl(s: str) -> str:
    return s.replace(RTL_L, "").replace(RTL_R, "").strip()

def split_sources_block(extractive: str) -> Tuple[str, List[str]]:
    """
    Extract the 'Sources:' lines from your retriever's answer.
    Returns (answer_without_sources, sources_lines)
    """
    txt = strip_rtl(extractive)
    parts = txt.split("Sources:\n", 1)
    if len(parts) == 2:
        body, src_block = parts[0].strip(), parts[1].strip()
        src_lines = [ln.strip() for ln in src_block.splitlines() if ln.strip()]
        return body, src_lines
    return txt, []

NUM_RE = re.compile(r"\d[\d\.:\/]*")

def extract_numbers(text: str) -> List[str]:
    # normalize Arabic/Indic digits via retriever normalizer
    t = RET.ar_normalize(text)
    return NUM_RE.findall(t)

def all_numbers_in_evidence(answer_text: str, evidence_text: str) -> bool:
    ans_nums = set(extract_numbers(answer_text))
    if not ans_nums:
        return True
    ev_nums = set(extract_numbers(evidence_text))
    # every number in the answer must appear in the evidence
    return ans_nums.issubset(ev_nums)

def join_sources(src_lines: List[str]) -> str:
    if not src_lines:
        return ""
    return "المصادر:\n" + "\n".join(src_lines)

# --------------- LLM: Qwen2.5 editor-only ---------------

def load_qwen(model_name: str, device_hint: Optional[str] = None):
    """
    Load Qwen2.5 Instruct safely on T4 with 4-bit quant (bitsandbytes).
    """
    if device_hint is None:
        device_hint = "cuda" if torch.cuda.is_available() else "cpu"

    LOG.info("Loading LLM: %s", model_name)
    use_4bit = (device_hint == "cuda")
    quant = None
    if use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )

    # Qwen often has eos as pad. We'll set attention_mask explicitly in generate().
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if device_hint == "cuda" else torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=quant
    )

    try:
        if torch.cuda.is_available() and torch.__version__ >= "2.0":
            model = torch.compile(model, mode="reduce-overhead")
            LOG.info("Model compiled for faster inference")
    except Exception as e:
        LOG.warning("torch.compile skipped: %s", e)

    model.eval()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    device = model.device
    LOG.info("LLM on: %s", device)
    return tokenizer, model, device


EDITOR_SYSTEM = (
    "أنت محرّر عربي محافظ. مهمتك فقط إعادة صياغة الجواب التالي في فقرة عربية واحدة واضحة، "
    "دون إضافة أي معلومة جديدة غير موجودة في النص المقتبس. "
    "يُمنع إدخال أرقام أو نسب أو مدد زمنية غير مذكورة ضمن الأدلة. "
    "إن كان النص يقول «لا توجد معلومات كافية»، فأعد صياغته بوضوح بنفس المعنى. "
    "لا تغيّر المعطيات الرقمية ولا تبتكر تفاصيل جديدة."
)

EDITOR_USER_TMPL = (
    "الأدلة المقتبسة (لا تُضِف غيرها):\n"
    "-----\n"
    "{evidence}\n"
    "-----\n\n"
    "الجواب المستخلص الذي يجب إعادة صياغته دون إضافة حقائق:\n"
    "<<<\n{extractive}\n>>>"
)

def llm_rewrite_to_paragraph(tokenizer, model, device, extractive_body: str, evidence_text: str,
                             max_new_tokens: int = 220) -> str:
    """
    Ask Qwen to rewrite extractive answer into one paragraph, constrained to evidence.
    """
    # Use chat format if available, otherwise simple prompt
    sys_msg = EDITOR_SYSTEM
    user_msg = EDITOR_USER_TMPL.format(evidence=evidence_text, extractive=extractive_body)

    # Build a simple chat-style prompt that works across Qwen variants
    prompt = (
        f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=3000
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            temperature=0.1,       # conservative
            top_p=0.9,
            do_sample=False,       # deterministic to reduce drift
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # strip the chat preamble if echoed
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    return strip_rtl(text).strip()


# --------------- Pipeline (retriever decides) ---------------

def build_or_load_index(chunks_path: str,
                        hier_index_path: Optional[str],
                        aliases_path: Optional[str],
                        load_dir: Optional[str],
                        save_dir: Optional[str]) -> RET.HybridIndex:
    hier = RET.load_hierarchy(hier_index_path, aliases_path)
    chunks, chunks_hash = RET.load_chunks(path=chunks_path)
    index = RET.HybridIndex(chunks, chunks_hash, hier=hier)

    loaded = False
    if load_dir and os.path.isdir(load_dir):
        loaded = index.load(load_dir)

    if not loaded:
        LOG.info("Building dense+TF-IDF indexes (first run)…")
        index.build()
        if save_dir:
            index.save(save_dir)
    else:
        LOG.info("Index loaded from artifacts")

    return index

def ask_once(index: RET.HybridIndex,
             tokenizer, model, device,
             question: str,
             pdf_name: str,
             use_llm: bool = True) -> str:
    """
    1) Use your retriever to produce extractive answer + sources (decisive step).
    2) Optionally rewrite it to a single paragraph (editor step).
    3) Validate: if rewrite adds numbers not in evidence, fall back to extractive.
    """
    t0 = time.time()
    intent = RET.classify_intent(question)
    extractive = RET.answer(question, index, intent, use_rerank_flag=False)  # you can set True if CE is on

    body, sources = split_sources_block(extractive)
    # Build a compact evidence text: concat top 3–5 chunks' page refs (already in your extractive)
    evidence_text = body  # conservative: use the same text the answer came from

    # If no LLM editing requested, return extractive as-is
    if not use_llm:
        dt = time.time() - t0
        return f"⏱ {dt:.2f}s | 🤖 {body}\n{join_sources(sources)}"

    # Editor step (strict)
    rewritten = llm_rewrite_to_paragraph(tokenizer, model, device, body, evidence_text)

    # Guardrail: ensure no new numbers in rewritten that aren't present in evidence
    if not all_numbers_in_evidence(rewritten, evidence_text):
        LOG.warning("Editor introduced numbers not in evidence. Falling back to extractive.")
        final_body = body
    else:
        final_body = rewritten

    dt = time.time() - t0
    return f"⏱ {dt:.2f}s | 🤖 {final_body}\n{join_sources(sources)}"

# ---------------- CLI / main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default="Data_pdf_clean_chunks.jsonl", help="Chunks JSONL/JSON")
    ap.add_argument("--pdf", type=str, default="Data_pdf.pdf", help="Original PDF name (for display only)")
    ap.add_argument("--hier-index", type=str, default="heading_inverted_index.json", help="Optional hierarchy index")
    ap.add_argument("--aliases", type=str, default="section_aliases.json", help="Optional aliases")
    ap.add_argument("--load-index", type=str, default=".artifact", help="Load index dir if exists")
    ap.add_argument("--save-index", type=str, default=".artifact", help="Save index dir after build")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="LLM model id")
    ap.add_argument("--no-llm", action="store_true", help="Return extractive answer only (no rewrite)")
    ap.add_argument("--ask", type=str, default=None, help="Ask one question and exit")
    ap.add_argument("--sanity", action="store_true", help="Run retriever sanity prompts")
    ap.add_argument("--chat", action="store_true", help="Enter chat loop")
    args = ap.parse_args()

    # 1) Build/Load retriever
    index = build_or_load_index(
        chunks_path=args.chunks,
        hier_index_path=args.hier_index,
        aliases_path=args.aliases,
        load_dir=args.load_index,
        save_dir=args.save_index
    )

    # 2) Load LLM editor unless disabled
    tok = mdl = dev = None
    if not args.no_llm:
        tok, mdl, dev = load_qwen(args.model)

    def _answer(q: str) -> str:
        return ask_once(index, tok, mdl, dev, q, args.pdf, use_llm=(not args.no_llm))

    # 3) Modes
    if args.ask:
        print(_answer(args.ask))
        return

    if args.sanity:
        print("🧪 Sanity run…\n")
        # reuse your sanity set for apples-to-apples
        prompts = RET.SANITY_PROMPTS if hasattr(RET, "SANITY_PROMPTS") else [
            "ما هي ساعات الدوام الرسمية من وإلى؟",
            "ما ساعات العمل في شهر رمضان؟ وهل تتغير؟",
            "ما سقف الشراء الذي يستلزم ثلاثة عروض أسعار؟"
        ]
        for q in prompts:
            print(f"• {q}")
            print(_answer(q))
            print()
        return

    if args.chat:
        print("💬 Chat mode. اكتب 'خروج' للإنهاء.\n")
        while True:
            try:
                q = input("سؤالك: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nتم الإنهاء."); break
            if not q:
                continue
            if q.lower() in ("خروج", "exit", "quit", "q"):
                print("وداعًا!"); break
            print(_answer(q))
            print()
        return

    # default: small demo
    demo_qs = [
        "ما هي ساعات الدوام الرسمية من وإلى؟",
        "ما ساعات العمل في شهر رمضان؟ وهل تتغير؟",
        "ما سقف الشراء الذي يستلزم ثلاثة عروض أسعار؟"
    ]
    for q in demo_qs:
        print(f"• {q}")
        print(_answer(q))
        print()

if __name__ == "__main__":
    main()

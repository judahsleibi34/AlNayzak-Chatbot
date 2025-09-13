# -*- coding: utf-8 -*-
"""
RAG pipeline where the retriever decides and Qwen2.5 only rewrites
AFTER an intent-aware validation that the extractive answer actually
addresses the question.

Usage:
  pip install -U transformers accelerate bitsandbytes sentence-transformers faiss-cpu scikit-learn joblib

  python rag_qwen_editor.py \
    --chunks Data_pdf_clean_chunks.jsonl \
    --pdf Data_pdf.pdf \
    --load-index .artifact --save-index .artifact \
    --model Qwen/Qwen2.5-7B-Instruct --sanity

  python rag_qwen_editor.py --load-index .artifact \
    --model Qwen/Qwen2.5-7B-Instruct \
    --ask "ما هي ساعات الدوام الرسمية من وإلى؟"

  # Pure extractive (no LLM rewrite):
  python rag_qwen_editor.py --load-index .artifact --no-llm --ask "..."
"""

import os, re, time, json, argparse, logging
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---- your retriever (from the code you sent) ----
import retrival_model as RET

LOG = logging.getLogger("rag_qwen_editor")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

RTL_L = "\u202B"
RTL_R = "\u202C"
NUM_RE = re.compile(r"\d[\d\.:\/]*")

def strip_rtl(s: str) -> str:
    return s.replace(RTL_L, "").replace(RTL_R, "").strip()

def split_sources_block(extractive: str) -> Tuple[str, List[str]]:
    txt = strip_rtl(extractive)
    parts = txt.split("Sources:\n", 1)
    if len(parts) == 2:
        body, src_block = parts[0].strip(), parts[1].strip()
        src_lines = [ln.strip() for ln in src_block.splitlines() if ln.strip()]
        return body, src_lines
    return txt, []

def extract_numbers(text: str) -> List[str]:
    t = RET.ar_normalize(text)
    return NUM_RE.findall(t)

def all_numbers_in_evidence(answer_text: str, evidence_text: str) -> bool:
    ans_nums = set(extract_numbers(answer_text))
    if not ans_nums:
        return True
    ev_nums = set(extract_numbers(evidence_text))
    return ans_nums.issubset(ev_nums)

def join_sources(src_lines: List[str]) -> str:
    if not src_lines:
        return ""
    return "المصادر:\n" + "\n".join(src_lines)

# ---------- Intent validation (hard gate) ----------

def _has_time_range(text: str, intent: str) -> bool:
    ranges = RET.extract_all_ranges(text, intent)
    return bool(ranges)

def _has_any(text: str, words: List[str]) -> bool:
    n = RET.ar_normalize(text)
    return any(w in n for w in words)

def extractive_answers_intent(question: str, intent: str, body_text: str) -> bool:
    """Return True only if the extractive body really answers the intent."""
    nbody = RET.ar_normalize(body_text)

    if intent in ("work_hours", "ramadan_hours"):
        # Must include a plausible time range like "من 8:00 إلى 4:00"
        if _has_time_range(body_text, intent):
            return True
        # Avoid false positives like "كشف الساعات" without ranges
        return False

    if intent == "workdays":
        return _has_any(nbody, ["ايام العمل","ايام الدوام","السبت","الاحد","الأحد","الخميس"])

    if intent == "overtime":
        return _has_any(nbody, ["احتساب","نسبة","أجر","اجر","اضافي","موافقه","موافقة","اعتماد","125"])

    if intent == "leave":
        return _has_any(nbody, ["إجاز","اجاز","سنويه","سنوية","مرضية","مرضيه","طارئة","امومه","أمومة","حداد"])

    if intent == "procurement":
        return _has_any(nbody, ["عروض","عرض","ثلاث","3","سقف","حد","شيكل","₪","دينار","دولار","شراء","مشتريات"])

    if intent == "per_diem":
        return _has_any(nbody, ["مياومات","مياومه","بدل سفر","بدل","سفر","نفقات","ايصالات","فواتير"])

    if intent in ("flex","break"):
        # Presence of relevant tokens
        if intent == "flex":
            return _has_any(nbody, ["مرون","تأخير","تاخير","الحضور","الانصراف","خصم","بصمة","بصمه"])
        if intent == "break":
            return _has_any(nbody, ["استراح","راحة","بريك","رضاع","ساعه","ساعة","دقائق"])
        return False

    # general: accept
    return True

# --------------- LLM: Qwen2.5 editor-only ---------------

EDITOR_SYSTEM = (
    "أنت محرّر عربي محافظ. مهمتك إعادة صياغة الجواب التالي في فقرة عربية واحدة واضحة، "
    "دون إضافة أي معلومة جديدة أو أرقام غير واردة في الأدلة. "
    "إذا كانت الأدلة لا تجيب على السؤال، اكتب: «لا توجد معلومات كافية في السياق للإجابة». "
    "لا تغيّر الأرقام أو المدد أو النسب."
)

EDITOR_USER_TMPL = (
    "السؤال:\n{question}\n\n"
    "الأدلة المقتبسة (التزم بها فقط):\n"
    "-----\n{evidence}\n-----\n\n"
    "الجواب المستخلص الذي يجب إعادة صياغته دون إضافة حقائق:\n"
    "<<<\n{extractive}\n>>>"
)

def load_qwen(model_name: str, device_hint: Optional[str] = None):
    if device_hint is None:
        device_hint = "cuda" if torch.cuda.is_available() else "cpu"

    LOG.info("Loading LLM: %s", model_name)
    quant = None
    if device_hint == "cuda":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
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

    return tokenizer, model, model.device

def llm_rewrite_to_paragraph(tokenizer, model, device, question: str,
                             extractive_body: str, evidence_text: str,
                             max_new_tokens: int = 200) -> str:
    sys_msg = EDITOR_SYSTEM
    user_msg = EDITOR_USER_TMPL.format(
        question=question, evidence=evidence_text, extractive=extractive_body
    )

    prompt = (
        f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    enc = tokenizer(prompt, return_tensors="pt", padding=True,
                    truncation=True, max_length=3000)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,             # deterministic to reduce drift
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    # IMPORTANT: do NOT skip special tokens so we can strip preamble cleanly
    text = tokenizer.decode(out[0], skip_special_tokens=False)

    # Cut everything before the last assistant tag
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    # Remove any end tags
    text = text.replace("<|im_end|>", "")
    # Strip leftover meta labels if any
    text = text.strip()
    return strip_rtl(text)

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
    t0 = time.time()
    intent = RET.classify_intent(question)
    extractive = RET.answer(question, index, intent, use_rerank_flag=False)

    body, sources = split_sources_block(extractive)
    evidence_text = body  # conservative: evidence == extractive body

    # HARD GATE: if extractive does not satisfy the intent, declare insufficient.
    if not extractive_answers_intent(question, intent, body):
        dt = time.time() - t0
        fail_msg = "لا توجد معلومات كافية في السياق للإجابة على هذا السؤال."
        return f"⏱ {dt:.2f}s | 🤖 {fail_msg}\n{join_sources(sources)}"

    if not use_llm:
        dt = time.time() - t0
        return f"⏱ {dt:.2f}s | 🤖 {body}\n{join_sources(sources)}"

    # Editor step
    rewritten = llm_rewrite_to_paragraph(tokenizer, model, device, question, body, evidence_text)

    # Guard: prevent new numbers
    if not all_numbers_in_evidence(rewritten, evidence_text):
        LOG.warning("Editor introduced numbers not in evidence. Falling back to extractive.")
        final_body = body
    else:
        final_body = rewritten

    dt = time.time() - t0
    return f"⏱ {dt:.2f}s | 🤖 {final_body}\n{join_sources(sources)}"

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default="Data_pdf_clean_chunks.jsonl", help="Chunks JSONL/JSON")
    ap.add_argument("--pdf", type=str, default="Data_pdf.pdf", help="Original PDF (display name only)")
    ap.add_argument("--hier-index", type=str, default="heading_inverted_index.json", help="Optional hierarchy index")
    ap.add_argument("--aliases", type=str, default="section_aliases.json", help="Optional aliases")
    ap.add_argument("--load-index", type=str, default=".artifact", help="Load index dir if exists")
    ap.add_argument("--save-index", type=str, default=".artifact", help="Save index dir after build")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="LLM editor model id")
    ap.add_argument("--no-llm", action="store_true", help="Disable editor (extractive only)")
    ap.add_argument("--ask", type=str, default=None, help="Ask one question and exit")
    ap.add_argument("--sanity", action="store_true", help="Run sanity prompts")
    ap.add_argument("--chat", action="store_true", help="Interactive chat")
    args = ap.parse_args()

    index = build_or_load_index(
        chunks_path=args.chunks,
        hier_index_path=args.hier_index,
        aliases_path=args.aliases,
        load_dir=args.load_index,
        save_dir=args.save_index
    )

    tok = mdl = dev = None
    if not args.no_llm:
        tok, mdl, dev = load_qwen(args.model)

    def _answer(q: str) -> str:
        return ask_once(index, tok, mdl, dev, q, args.pdf, use_llm=(not args.no_llm))

    if args.ask:
        print(_answer(args.ask)); return

    if args.sanity:
        print("🧪 Sanity run…\n")
        prompts = getattr(RET, "SANITY_PROMPTS", [
            "ما هي ساعات الدوام الرسمية من وإلى؟",
            "ما ساعات العمل في شهر رمضان؟ وهل تتغير؟",
            "ما سقف الشراء الذي يستلزم ثلاثة عروض أسعار؟",
        ])
        for q in prompts:
            print(f"• {q}")
            print(_answer(q)); print()
        return

    if args.chat:
        print("💬 اكتب 'خروج' للإنهاء.\n")
        while True:
            try:
                q = input("سؤالك: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nتم الإنهاء."); break
            if not q: continue
            if q.lower() in ("خروج","exit","quit","q"): print("وداعًا!"); break
            print(_answer(q)); print()
        return

    # default demo
    demo_qs = [
        "ما هي ساعات الدوام الرسمية من وإلى؟",
        "ما ساعات العمل في شهر رمضان؟ وهل تتغير؟",
        "ما سقف الشراء الذي يستلزم ثلاثة عروض أسعار؟"
    ]
    for q in demo_qs:
        print(f"• {q}")
        print(_answer(q)); print()

if __name__ == "__main__":
    main()

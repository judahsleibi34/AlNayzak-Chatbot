# -*- coding: utf-8 -*-
"""
RAG Orchestrator (Arabic) with:
- Extractive retriever (your retrival_model.py) as source of truth
- Intent miner: scans top-K hits for sentences that actually answer the question
- Conservative LLM editor (Qwen2.5-7B-Instruct) to rewrite into ONE Arabic paragraph
  without inventing facts/numbers; falls back to extractive text if numbers drift

Usage
-----
# Build/load index and run sanity:
python rag_qwen_editor.py --chunks Data_pdf_clean_chunks.jsonl --save-index .artifact --load-index .artifact --model Qwen/Qwen2.5-7B-Instruct --sanity

# Ask a single question:
python rag_qwen_editor.py --load-index .artifact --model Qwen/Qwen2.5-7B-Instruct --ask "ما هي ساعات الدوام الرسمية من وإلى؟"

# Extractive only (no LLM rewrite):
python rag_qwen_editor.py --load-index .artifact --no-llm --ask "ما ساعات العمل في شهر رمضان؟ وهل تتغير؟"
"""

import os, re, time, argparse, logging, math
import torch

# ---- your retriever module ----
import retrival_model as RET

from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger("rag_qwen")

# ----------------------- Utilities -----------------------

AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"٠١٢٣٤٥٦٧٨٩")}
IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"۰۱۲۳۴۵۶۷۸۹")}

def norm_digits(s: str) -> str:
    return s.translate(AR_NUMS).translate(IR_NUMS)

NUM_RE = re.compile(r'(?<![\w/])(?:\d+(?:[.:]\d+)*)(?![\w/])')

def numbers_in(s: str) -> List[str]:
    return NUM_RE.findall(norm_digits(s or ""))

def all_numbers_in_evidence(candidate: str, evidence: str) -> bool:
    """True if every number in candidate exists in evidence (prevents hallucinated numerics)."""
    cnc = set(numbers_in(candidate))
    if not cnc:
        return True  # nothing numeric to police
    ev = set(numbers_in(evidence))
    return cnc.issubset(ev)

def join_sources(srcs: List[str]) -> str:
    if not srcs:
        return ""
    # Arabic label to match your UI
    return "المصادر:\n" + "\n".join(srcs)

def split_sources_block(extractive: str) -> Tuple[str, List[str]]:
    """Parse the body and sources from RET.answer output."""
    text = extractive.strip()
    # Accept 'Sources:' or 'المصادر:'
    marker_idx = max(text.rfind("\nSources:"), text.rfind("\nالمصادر:"))
    if marker_idx == -1:
        return text, []
    body = text[:marker_idx].strip()
    tail = text[marker_idx:].splitlines()
    # Drop header
    lines = [ln.strip() for ln in tail[1:] if ln.strip()]
    return body, lines

# ----------------------- Intent satisfaction checks -----------------------

def sentence_satisfies_intent(s: str, intent: str) -> bool:
    sn = RET.ar_normalize(s)
    if intent in ("work_hours", "ramadan_hours"):
        return bool(RET.extract_all_ranges(s, intent))
    if intent == "workdays":
        return any(w in sn for w in ["ايام العمل","ايام الدوام","السبت","الاحد","الأحد","الخميس"])
    if intent == "overtime":
        return any(w in sn for w in ["احتساب","نسبة","أجر","اجر","اضافي","موافق","موافقة","اعتماد","125"])
    if intent == "leave":
        return any(w in sn for w in ["إجاز","اجاز","سنويه","سنوية","مرضية","مرضيه","طارئة","امومه","أمومة","حداد"])
    if intent == "procurement":
        return any(w in sn for w in ["عروض","عرض","ثلاث","3","سقف","حد","شيكل","₪","دينار","دولار","شراء","مشتريات"])
    if intent == "per_diem":
        return any(w in sn for w in ["مياومات","مياومه","بدل سفر","بدل","سفر","نفقات","ايصالات","فواتير"])
    if intent == "flex":
        return any(w in sn for w in ["مرون","تأخير","تاخير","الحضور","الانصراف","خصم","بصمة","بصمه"])
    if intent == "break":
        return any(w in sn for w in ["استراح","راحة","بريك","رضاع","ساعه","ساعة","دقائق"])
    return True

def extractive_answers_intent(question: str, intent: str, body: str) -> bool:
    if not body.strip():
        return False
    if intent in ("work_hours", "ramadan_hours"):
        return bool(RET.extract_all_ranges(body, intent))
    # Generic: body must contain at least one intent signal
    return sentence_satisfies_intent(body, intent)

# ----------------------- Intent Miner -----------------------

def mine_intent_evidence(index: RET.HybridIndex, question: str, intent: str,
                         topk: int = 12, max_snippets: int = 3) -> Tuple[str, List[str]]:
    """
    Look through top-K retrieved chunks and pull 1–3 sentences that actually satisfy the intent.
    Returns (joined_body, sources). If none found -> ("", []).
    """
    hits = RET.retrieve(index, question, rerank=True)
    if not hits:
        return "", []

    picked, srcs, seen = [], [], set()
    for _, i in hits[:topk]:
        ch = index.chunks[i]
        for s in RET.sent_split(ch.text):
            ss = s.strip()
            if not ss or ss in seen:
                continue
            if sentence_satisfies_intent(ss, intent):
                picked.append(ss)
                srcs.append(f"{len(srcs)+1}. Data_pdf.pdf - page {ch.page}")
                seen.add(ss)
                if len(picked) >= max_snippets:
                    break
        if len(picked) >= max_snippets:
            break

    if not picked:
        return "", []
    return " ".join(picked), srcs

# ----------------------- LLM Editor (Qwen) -----------------------

def build_editor_prompt(question: str, evidence: str, extractive: str) -> List[dict]:
    system = (
        "أنت محرّر عربي محافظ. مهمتك إعادة صياغة الجواب في فقرة عربية واحدة واضحة "
        "تعتمد فقط على النص المقتبس. لا تُدخل أي معلومة جديدة غير موجودة. "
        "يُمنع إدخال أرقام/نسب/أوقات غير مذكورة حرفيًا في الأدلة. "
        "إن كان النص يقول «لا توجد معلومات كافية»، فأعد صياغته بوضوح بنفس المعنى."
    )
    user = (
        "الأدلة المقتبسة (لا تُضِف غيرها):\n-----\n"
        f"{evidence.strip()}\n-----\n\n"
        "الجواب المستخلص الذي يجب إعادة صياغته دون إضافة حقائق:\n<<<\n"
        f"{extractive.strip()}\n>>>\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]

def load_llm(model_name: str, device: str = "auto", use_4bit: bool = False, use_8bit: bool = False):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    kwargs = {}
    if torch.cuda.is_available():
        # bf16 if possible, else fp16
        if torch.cuda.is_bf16_supported():
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32
        kwargs["device_map"] = "cpu"

    if use_4bit or use_8bit:
        try:
            import bitsandbytes as bnb  # noqa
            if use_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            elif use_8bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except Exception as e:
            LOG.warning("bitsandbytes not available (%s). Loading without quantization.", e)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    return tok, mdl

def llm_rewrite_to_paragraph(tokenizer, model, device, question: str, body: str, evidence_text: str) -> str:
    # Build chat prompt (prefer chat template if available)
    msgs = build_editor_prompt(question, evidence_text, body)
    try:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback to manual prompt
        prompt = msgs[0]["content"] + "\n\n" + msgs[1]["content"]

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen = model.generate(
        **inputs,
        max_new_tokens=160,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    out = tokenizer.decode(gen[0], skip_special_tokens=True)

    # Strip the prompt to keep only the assistant continuation
    # Heuristic: take text after the last '>>>' block
    if ">>>" in out:
        out = out.split(">>>")[-1]
    # Clean tokens like "assistant", "system" if surfaced
    out = re.sub(r'^\s*(assistant|system)\s*[:]?\s*', '', out.strip(), flags=re.I)
    # Keep first paragraph only (avoid trailing sources the model might echo)
    out = out.strip().split("\n")[0].strip()
    return out

# ----------------------- Orchestration -----------------------

def ask_once(index: RET.HybridIndex,
             tokenizer, model,
             question: str,
             pdf_name: str,
             use_llm: bool = True) -> str:
    t0 = time.time()
    intent = RET.classify_intent(question)

    # 1) Extractive answer from your retriever synthesizers
    extractive = RET.answer(question, index, intent, use_rerank_flag=False)
    body, sources = split_sources_block(extractive)
    evidence_text = body  # default evidence is extractive body

    # 2) If the first extractive doesn't satisfy intent, mine other hits
    if not extractive_answers_intent(question, intent, body):
        mined_body, mined_sources = mine_intent_evidence(index, question, intent,
                                                         topk=12, max_snippets=3)
        if mined_body:
            body = mined_body
            sources = mined_sources
            evidence_text = mined_body
        else:
            dt = time.time() - t0
            fail_msg = "لا توجد معلومات كافية في السياق للإجابة على هذا السؤال."
            return f"⏱ {dt:.2f}s | 🤖 {fail_msg}\n{join_sources(sources)}"

    # 3) Optionally rewrite to a single Arabic paragraph (conservative)
    if not use_llm or tokenizer is None or model is None:
        dt = time.time() - t0
        return f"⏱ {dt:.2f}s | 🤖 {body}\n{join_sources(sources)}"

    rewritten = llm_rewrite_to_paragraph(tokenizer, model, model.device, question, body, evidence_text)

    # 4) Numeric guard: if LLM introduced numbers not in evidence, fall back
    final_body = rewritten if all_numbers_in_evidence(rewritten, evidence_text) else body

    dt = time.time() - t0
    return f"⏱ {dt:.2f}s | 🤖 {final_body}\n{join_sources(sources)}"

def run_sanity(index: RET.HybridIndex, tokenizer, model, use_llm: bool):
    print("\n🧪 Sanity run…\n")
    for q in RET.SANITY_PROMPTS:
        print(f"• {q}")
        out = ask_once(index, tokenizer, model, q, "Data_pdf.pdf", use_llm=use_llm)
        print(out)
        print()

# ----------------------- CLI -----------------------

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

    # 0) Build/load index from your retriever
    hier = RET.load_hierarchy(args.hier_index, args.aliases)
    chunks, chunks_hash = RET.load_chunks(path=args.chunks)
    index = RET.HybridIndex(chunks, chunks_hash, hier=hier)

    loaded = False
    if args.load_index:
        loaded = index.load(args.load_index)
    if not loaded:
        LOG.warning("Artifact/chunks mismatch; will rebuild instead of loading.")
        index.build()
        if args.save_index:
            index.save(args.save_index)

    # 1) Optional LLM
    tok = mdl = None
    use_llm = not args.no_llm
    if use_llm:
        tok, mdl = load_llm(args.model, use_4bit=args.use_4bit, use_8bit=args.use_8bit)

    # 2) Modes
    if args.sanity:
        run_sanity(index, tok, mdl, use_llm=use_llm)
        return

    if args.ask:
        out = ask_once(index, tok, mdl, args.ask, "Data_pdf.pdf", use_llm=use_llm)
        print(out)
        return

    # 3) Interactive
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
        ans = ask_once(index, tok, mdl, q, "Data_pdf.pdf", use_llm=use_llm)
        print(ans)

if __name__ == "__main__":
    main()

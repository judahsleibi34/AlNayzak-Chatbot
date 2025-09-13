# -*- coding: utf-8 -*-
"""
RAG Orchestrator (Arabic) — fixed version

- Calls your retriever with rerank=True (better hits)
- Softer intent satisfaction (don’t require times to exist to accept work_hours)
- Two-pass evidence mining: strict (with numbers) then soft (keywords)
- Single-paragraph Arabic answer
- Numeric guard: if LLM injects numbers not in evidence -> fallback to extractive
- Clean 'المصادر:' block (no duplicate 'Sources:')

Usage
-----
# Build index once and cache:
python rag_qwen_editor.py --chunks Data_pdf_clean_chunks.jsonl --save-index .artifact --sanity --model Qwen/Qwen2.5-7B-Instruct

# Load cached index and run sanity:
python rag_qwen_editor.py --load-index .artifact --sanity --model Qwen/Qwen2.5-7B-Instruct

# Ask one question:
python rag_qwen_editor.py --load-index .artifact --ask "ما هي ساعات الدوام الرسمية من وإلى؟" --model Qwen/Qwen2.5-7B-Instruct

# Extractive only (no LLM):
python rag_qwen_editor.py --load-index .artifact --no-llm --ask "ما ساعات العمل في شهر رمضان؟ وهل تتغير؟"
"""

import os, re, time, argparse, logging
import torch
from typing import List, Tuple

# Your retriever module
import retrival_model as RET

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger("rag_qwen_fixed")

# ---------------- Digit/number utils ----------------

AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"٠١٢٣٤٥٦٧٨٩")}
IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"۰۱۲۳۴۵۶۷۸۹")}

def norm_digits(s: str) -> str:
    return (s or "").translate(AR_NUMS).translate(IR_NUMS)

NUM_RE = re.compile(r'(?<![\w/])(?:\d+(?:[.:]\d+)*)(?![\w/])')

def numbers_in(s: str) -> List[str]:
    return NUM_RE.findall(norm_digits(s or ""))

def all_numbers_in_evidence(candidate: str, evidence: str) -> bool:
    cand = set(numbers_in(candidate))
    if not cand:
        return True
    ev = set(numbers_in(evidence))
    return cand.issubset(ev)

# ---------------- Source block helpers ----------------

def join_sources(srcs: List[str]) -> str:
    if not srcs:
        return ""
    return "المصادر:\n" + "\n".join(srcs)

def split_sources_block(extractive: str) -> Tuple[str, List[str]]:
    """
    Split body and sources from your retriever's textual answer.
    Works whether it printed 'Sources:' or 'المصادر:'.
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

# ---------------- Intent checks (relaxed) ----------------

def _has_any(sn: str, words: List[str]) -> bool:
    sn = RET.ar_normalize(sn)
    return any(w in sn for w in words)

def sentence_satisfies_intent_soft(s: str, intent: str) -> bool:
    sn = RET.ar_normalize(s)
    if intent in ("work_hours", "ramadan_hours"):
        # soft: allow answers that mention ساعات/دوام even w/o explicit times
        return _has_any(sn, ["ساعات", "دوام", "العمل"]) or bool(RET.extract_all_ranges(s, intent))
    if intent == "workdays":
        return _has_any(sn, ["ايام", "الدوام", "العمل", "السبت", "الاحد", "الأحد", "الخميس"])
    if intent == "overtime":
        return _has_any(sn, ["اضافي","احتساب","نسبة","أجر","اجر","موافق","موافقة","اعتماد","125"])
    if intent == "leave":
        return _has_any(sn, ["اجاز","إجاز","سنويه","سنوية","مرضية","مرضيه","طارئة","امومه","أمومة","حداد"])
    if intent == "procurement":
        return _has_any(sn, ["عروض","عرض","ثلاث","3","سقف","حد","شيكل","₪","دينار","دولار","شراء","مشتريات"])
    if intent == "per_diem":
        return _has_any(sn, ["مياومات","مياومه","بدل","سفر","نفقات","مصاريف","فواتير","ايصالات","تذاكر","فندق"])
    if intent == "flex":
        return _has_any(sn, ["مرون","تأخير","تاخير","الحضور","الانصراف","خصم","بصمة","بصمه"])
    if intent == "break":
        return _has_any(sn, ["استراح","راحة","بريك","رضاع","ساعه","ساعة","دقائق"])
    return bool(sn)

def extractive_answers_intent(question: str, intent: str, body: str) -> bool:
    if not body.strip():
        return False
    if intent in ("work_hours", "ramadan_hours"):
        # accept either explicit ranges OR a clear statement about ساعات الدوام
        if RET.extract_all_ranges(body, intent):
            return True
        sn = RET.ar_normalize(body)
        return _has_any(sn, ["ساعات", "دوام", "العمل", "رمضان"])
    # generic relaxed check
    return sentence_satisfies_intent_soft(body, intent)

# ---------------- Evidence miner: strict -> soft ----------------

def mine_intent_evidence(index: RET.HybridIndex, question: str, intent: str,
                         topk: int = 12, max_snippets: int = 3) -> Tuple[str, List[str]]:
    """
    Strict pass: sentences that fully satisfy intent with numbers (when applicable).
    Soft pass: relaxed keyword-based satisfaction if strict found nothing.
    """
    hits = RET.retrieve(index, question, rerank=True)
    if not hits:
        return "", []

    picked, srcs, seen = [], [], set()
    # ---- STRICT PASS ----
    for _, i in hits[:topk]:
        ch = index.chunks[i]
        for s in RET.sent_split(ch.text):
            ss = s.strip()
            if not ss or ss in seen: 
                continue
            ok = False
            if intent in ("work_hours", "ramadan_hours"):
                ok = bool(RET.extract_all_ranges(ss, intent))
            elif intent in ("overtime", "leave", "procurement", "per_diem"):
                # Prefer sentences with numbers for these intents
                ok = sentence_satisfies_intent_soft(ss, intent) and bool(numbers_in(ss))
            else:
                ok = sentence_satisfies_intent_soft(ss, intent)
            if ok:
                picked.append(ss)
                srcs.append(f"{len(srcs)+1}. Data_pdf.pdf - page {ch.page}")
                seen.add(ss)
                if len(picked) >= max_snippets:
                    break
        if len(picked) >= max_snippets:
            break

    # ---- SOFT PASS (if nothing strict) ----
    if not picked:
        for _, i in hits[:topk]:
            ch = index.chunks[i]
            for s in RET.sent_split(ch.text):
                ss = s.strip()
                if not ss or ss in seen:
                    continue
                if sentence_satisfies_intent_soft(ss, intent):
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

# ---------------- LLM (Qwen) editor ----------------

def build_editor_prompt(question: str, evidence: str, extractive: str) -> List[dict]:
    system = (
        "أنت محرّر عربي محافظ. أعد صياغة الجواب في فقرة عربية واحدة واضحة،"
        " وتعتمد فقط على النص المقتبس أدناه. يُمنع إضافة أية معلومات جديدة أو تعميمات،"
        " ويُمنع إدخال أرقام/نسب/أوقات غير مذكورة حرفيًا في الأدلة."
        " إذا كان المتاح جزئيًا، فقل المعنى المتاح فقط دون اختلاق."
    )
    user = (
        "الأدلة المقتبسة (لا تُضِف غيرها):\n-----\n"
        f"{evidence.strip()}\n-----\n\n"
        "الجواب المستخلص الذي يجب إعادة صياغته دون إضافة حقائق:\n<<<\n"
        f"{extractive.strip()}\n>>>\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]

def load_llm(model_name: str, use_4bit: bool = False, use_8bit: bool = False):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    kwargs = {}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        kwargs["device_map"] = "cpu"
        kwargs["torch_dtype"] = torch.float32

    if use_4bit or use_8bit:
        try:
            import bitsandbytes as bnb  # noqa
            if use_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            elif use_8bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except Exception as e:
            LOG.warning("bitsandbytes unavailable (%s). Loading model without quantization.", e)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    return tok, mdl

def llm_rewrite_to_paragraph(tokenizer, model, question: str, body: str, evidence_text: str) -> str:
    msgs = build_editor_prompt(question, evidence_text, body)
    try:
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = msgs[0]["content"] + "\n\n" + msgs[1]["content"]

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out_ids = model.generate(
        **inputs,
        max_new_tokens=160,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    # Heuristic: keep only the final assistant continuation line/paragraph
    if ">>>" in text:
        text = text.split(">>>")[-1]
    text = re.sub(r'^\s*(assistant|system)\s*[:]?\s*', '', text, flags=re.I).strip()
    # Keep first paragraph
    text = text.split("\n\n")[0].strip()
    return text

# ---------------- Orchestrator ----------------

def ask_once(index: RET.HybridIndex,
             tokenizer, model,
             question: str,
             use_llm: bool = True) -> str:
    t0 = time.time()
    intent = RET.classify_intent(question)

    # 1) Use your retriever with rerank=True (important!)
    extractive = RET.answer(question, index, intent, use_rerank_flag=True)
    body, sources = split_sources_block(extractive)
    evidence_text = body

    # 2) If the extractive doesn’t satisfy intent, mine strict->soft
    if not extractive_answers_intent(question, intent, body):
        mined_body, mined_sources = mine_intent_evidence(index, question, intent,
                                                         topk=12, max_snippets=3)
        if mined_body:
            body = mined_body
            sources = mined_sources
            evidence_text = mined_body
        else:
            dt = time.time() - t0
            return f"⏱ {dt:.2f}s | 🤖 لا توجد معلومات كافية في السياق للإجابة على هذا السؤال.\n{join_sources(sources)}"

    # 3) Optional: rewrite to a single Arabic paragraph
    if not use_llm or tokenizer is None or model is None:
        dt = time.time() - t0
        return f"⏱ {dt:.2f}s | 🤖 {body}\n{join_sources(sources)}"

    rewritten = llm_rewrite_to_paragraph(tokenizer, model, question, body, evidence_text)

    # 4) Numeric guard
    final_body = rewritten if all_numbers_in_evidence(rewritten, evidence_text) else body
    dt = time.time() - t0
    return f"⏱ {dt:.2f}s | 🤖 {final_body}\n{join_sources(sources)}"

def run_sanity(index: RET.HybridIndex, tokenizer, model, use_llm: bool):
    print("\n🧪 Sanity run…\n")
    for q in RET.SANITY_PROMPTS:
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
        print(ask_once(index, tok, mdl, args.ask, use_llm=use_llm))
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
        print(ask_once(index, tok, mdl, q, use_llm=use_llm))

if __name__ == "__main__":
    main()

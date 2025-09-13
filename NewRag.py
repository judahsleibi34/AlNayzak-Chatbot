# -*- coding: utf-8 -*-
"""
RAG driver on top of your retriever (retrival_model.py), tuned for Arabic.

- Uses your HybridIndex / combine_scores exactly as-is
- Adds Arabic reranking with BAAI/bge-reranker-v2-m3
- Expands time-range lines to ±2 sentences
- Builds compact context (<=1400 chars)
- Prompts Qwen2.5-7B-Instruct via chat template
- Outputs ONE continuous Arabic paragraph (no bullets)

Usage (examples):
  # build/load index, then run sanity questions with Qwen
  python rag_qwen25.py --chunks Data_pdf_clean_chunks.jsonl --load-index .artifact --model Qwen/Qwen2.5-7B-Instruct --sanity

  # interactive chat
  python rag_qwen25.py --chunks Data_pdf_clean_chunks.jsonl --load-index .artifact --model Qwen/Qwen2.5-7B-Instruct

Colab/T4 note:
- 4-bit quantization is enabled (BitsAndBytes). If bnb is missing, it falls back.
"""

import os, re, json, argparse, time, logging
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---- your retriever (keep it unchanged) ----
from retrival_model import (
    HybridIndex, load_chunks, load_hierarchy, combine_scores,
    ar_normalize, sent_split, classify_intent
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger("rag")

# Arabic time-range regex (same spirit as your retriever)
TIME_RE = re.compile(
    r'(?:من\s*)?(\d{1,2}(?::|\.)?\d{0,2})\s*(?:[-–—]|الى|إلى|حتي|حتى)\s*(\d{1,2}(?::|\.)?\d{0,2})'
)

def has_time_range(s: str) -> bool:
    return bool(TIME_RE.search(ar_normalize(s)))

# -------------- Arabic reranker --------------
CE_MODEL = "BAAI/bge-reranker-v2-m3"
_ce = None
def ensure_reranker(device: str):
    global _ce
    if _ce is not None:
        return _ce
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
        _ce = CrossEncoder(CE_MODEL, device=device)
        LOG.info("Loaded Arabic reranker: %s", CE_MODEL)
    except Exception as e:
        LOG.warning("Could not load reranker (%s). Proceeding without.", e)
        _ce = None
    return _ce

# -------------- Retrieval → candidates --------------
def retrieve_candidates(index: HybridIndex, query: str,
                        topk_dense=120, hybrid_topk=30,
                        w_dense=0.70, w_char=0.15, w_word=0.15) -> List[Tuple[float, int]]:
    qn = ar_normalize(query)
    dS, dI = index.dense(qn, topk=topk_dense, restrict_ids=None)
    cS, wS = index.sparse(qn)
    prelim = combine_scores(dS, dI, cS, wS, w_dense=w_dense, w_char=w_char, w_word=w_word, topk=hybrid_topk)
    # prelim = [(score, idx), ...]
    return prelim

def apply_rerank(index: HybridIndex, query: str, prelim: List[Tuple[float, int]],
                 device: str, keep_top=8) -> List[Tuple[float, int]]:
    ce = ensure_reranker(device)
    if ce is None or len(prelim) <= 1:
        return prelim[:keep_top]

    pairs = [(query, index.chunks[i].text) for _, i in prelim]
    try:
        ce_scores = ce.predict(pairs)  # higher is better
        scored = list(zip([float(s) for s in ce_scores], [i for _, i in prelim]))
        scored.sort(key=lambda x: -x[0])
        return scored[:keep_top]
    except Exception as e:
        LOG.warning("Rerank failed: %s", e)
        return prelim[:keep_top]

# -------------- Snippet/window builder --------------
def best_sentence_idx(text: str, query_norm: str) -> Optional[int]:
    sents = sent_split(text)
    if not sents:
        return None
    q_terms = set([w for w in query_norm.split() if len(w) >= 3])
    best_k, best_score = None, -1e9
    for k, s in enumerate(sents):
        sn = ar_normalize(s)
        overlap = len(q_terms & set(sn.split()))
        score = overlap
        if has_time_range(s):  # strong signal for hours/policies
            score += 1.5
        if score > best_score:
            best_score, best_k = score, k
    return best_k

def window_around_sentence(text: str, center_idx: int, win=2) -> str:
    sents = sent_split(text)
    if not sents:
        return text.strip()
    lo = max(0, center_idx - win)
    hi = min(len(sents), center_idx + win + 1)
    return " ".join(sents[lo:hi]).strip()

def build_context(index: HybridIndex, query: str, hits: List[Tuple[float,int]],
                  max_chars=1400, prefer_time: bool = False) -> Tuple[str, List[Tuple[int,int]]]:
    """
    Returns (context, [(rank, chunk_idx), ...]).
    Each context block is prefixed with [المصدر N: filename - صX]
    """
    used = []
    parts: List[str] = []
    total = 0
    qn = ar_normalize(query)

    # Prefer lines with explicit time-ranges for hour-related intents
    ordered = []
    for rank, (_, i) in enumerate(hits, 1):
        txt = index.chunks[i].text
        if prefer_time and has_time_range(txt):
            ordered.append((0, rank, i))  # priority 0
        else:
            ordered.append((1, rank, i))
    ordered.sort()

    for _, rank, i in ordered:
        ch = index.chunks[i]
        # pick the best sentence and expand window
        k = best_sentence_idx(ch.text, qn)
        snippet = window_around_sentence(ch.text, k if k is not None else 0, win=2)

        head = f"[المصدر {len(parts)+1}: Data_pdf.pdf - ص{ch.page}]"
        block = f"{head}\n{snippet}"
        if total + len(block) + 2 > max_chars:
            # allow a small last block if still empty
            if total == 0:
                block = block[:max_chars-3] + "…"
                parts.append(block)
                used.append((rank, i))
            break
        parts.append(block)
        used.append((rank, i))
        total += len(block) + 2

        if total >= max_chars:
            break

    return "\n\n".join(parts), used

# -------------- LLM (Qwen2.5-7B-Instruct) --------------
def load_qwen(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Prefer 4-bit on T4
    quant_cfg = None
    use_bnb = False
    try:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        use_bnb = True
    except Exception as e:
        LOG.warning("BitsAndBytes not available: %s", e)

    kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    if use_bnb:
        kwargs["quantization_config"] = quant_cfg
    else:
        kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    # Avoid the header echo: use chat template
    return tokenizer, model, device

def generate_answer(tokenizer, model, device, query: str, context: str,
                    paragraph_style: bool = True,
                    max_new_tokens=256, temperature=0.3, top_p=0.9) -> str:
    sys_msg = (
        "أنت مساعد عربي يعتمد فقط على السياق المقدم. "
        "اكتب الجواب بالعربية الفصحى كفقرة واحدة متصلة بدون تعداد نقطي أو عناوين. "
        "إن لم تجد الإجابة في السياق، اكتب: لا توجد معلومات كافية في السياق للإجابة."
    )
    user_msg = (
        "السياق المقتبس من الدليل (اعتمد عليه فقط):\n\n"
        f"{context}\n\n"
        f"السؤال: {query}"
    )

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]

    # Chat template ensures the model sees roles properly (no literal 'system/user' in output)
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        )
    except TypeError:
        # Older HF versions
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            prompt_text, return_tensors="pt", padding="longest", truncation=True
        )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Deterministic for consistency; switch to do_sample=True if you want more “creative” phrasing
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    # Cut the prompt off (new tokens only)
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # Clean minimal artifacts
    return text.strip()

# -------------- Sanity prompts (same spirit as your file) --------------
SANITY_QS = [
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
    "ما سياسة الإجازة المرضية؟ وعدد أيامها؟ وهل يلزم تقرير طبي؟",
    "ما هو بدل المواصلات؟ وهل يشمل الذهاب من المنزل للعمل؟ وكيف يُصرف؟",
    "ما سقف الشراء الذي يستلزم ثلاثة عروض أسعار؟",
    "ما ضوابط تضارب المصالح في المشتريات؟",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default="Data_pdf_clean_chunks.jsonl")
    ap.add_argument("--hier-index", type=str, default="heading_inverted_index.json")
    ap.add_argument("--aliases", type=str, default="section_aliases.json")
    ap.add_argument("--load-index", type=str, default=".artifact")
    ap.add_argument("--save-index", type=str, default=None)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--no-rerank", action="store_true", help="Disable CE reranker")
    args = ap.parse_args()

    # 1) Load / build retriever artifacts
    hier = load_hierarchy(args.hier_index, args.aliases)
    chunks, chunks_hash = load_chunks(path=args.chunks)
    index = HybridIndex(chunks, chunks_hash, hier=hier)

    loaded = False
    if args.load_index and os.path.isdir(args.load_index):
        loaded = index.load(args.load_index)
    if not loaded:
        LOG.info("Building dense+TF-IDF indexes (first run)…")
        index.build()
        if args.save_index:
            index.save(args.save_index)

    # 2) Load LLM
    LOG.info("Loading Qwen model…")
    tokenizer, model, device = load_qwen(args.model)
    _ = ensure_reranker(device) if not args.no_rerank else None
    LOG.info("Ready.")

    def ask(q: str) -> str:
        t0 = time.time()
        # intent helps us prefer time-range snippets for hours questions
        intent = classify_intent(q)
        prefer_time = intent in ("work_hours", "ramadan_hours")

        prelim = retrieve_candidates(index, q, topk_dense=120, hybrid_topk=30,
                                     w_dense=0.70, w_char=0.15, w_word=0.15)
        if not args.no_rerank:
            hits = apply_rerank(index, q, prelim, device=device, keep_top=8)
        else:
            hits = prelim[:8]

        # if time intent but no candidates with time, keep as is (fallback)
        ctx, used = build_context(index, q, hits, max_chars=1400, prefer_time=prefer_time)
        if not ctx.strip():
            return "لا توجد معلومات كافية في السياق للإجابة."

        ans = generate_answer(tokenizer, model, device, q, ctx,
                              paragraph_style=True, max_new_tokens=256, temperature=0.3, top_p=0.9)
        dt = time.time() - t0
        print(f"⏱ {dt:.2f}s | 🤖 {ans}\n")
        return ans

    if args.sanity:
        print("\n🧪 Sanity run…\n")
        for q in SANITY_QS:
            print(f"• {q}")
            ask(q)
        return

    # Interactive loop
    print("\n💬 Ready (Arabic). Type 'خروج' to exit.\n")
    while True:
        try:
            q = input("سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nانتهينا."); break
        if not q:
            continue
        if q.lower() in ("خروج", "exit", "quit", "q"):
            print("انتهينا."); break
        ask(q)

if __name__ == "__main__":
    main()

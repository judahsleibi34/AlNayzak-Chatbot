# NewRag_qwen_fixed.py
# Arabic RAG with Qwen2.5-7B-Instruct + your hybrid retriever
# - Single-paragraph answers
# - Reranker toggle (--rerank / --no-rerank, default: ON)
# - Works with .artifact index cache
# - Fixes tensor/dict device move by creating attention_mask when needed

import os, argparse, time, torch
from typing import List, Tuple
from dataclasses import dataclass

# --- your retriever pieces ---
from retrival_model import (
    HybridIndex, load_hierarchy, load_chunks,
    retrieve, classify_intent, best_snippet, ar_normalize,
)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ----------------- Config -----------------
DEFAULT_CHUNKS = "Data_pdf_clean_chunks.jsonl"
DEFAULT_HIER   = "heading_inverted_index.json"
DEFAULT_ALIAS  = "section_aliases.json"
DEFAULT_IDXDIR = ".artifact"
DEFAULT_MODEL  = "Qwen/Qwen2.5-7B-Instruct"

MAX_CTX_CHARS  = 1100   # keep prompts lean for speed/quality
TOPK_DOCS      = 3      # how many snippets to pass to the LLM
MAX_NEW_TOKENS = 320
TEMP           = 0.2
TOP_P          = 0.9
REPETITION_P   = 1.05

# ----------------- LLM -----------------

@dataclass
class QwenBundle:
    tok: AutoTokenizer
    model: AutoModelForCausalLM
    device: str

def load_qwen(model_name: str = DEFAULT_MODEL) -> QwenBundle:
    use_bnb = True
    bnb_cfg = None
    if use_bnb:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # avoid attention_mask warnings

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_cfg,
        low_cpu_mem_usage=True,
    )
    model.eval()
    try:
        if torch.__version__ >= "2.0":
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)  # safe compile
    except Exception:
        pass

    device = str(model.device)
    return QwenBundle(tok, model, device)

# ----------------- Prompting -----------------

def make_messages(question: str, stitched_context: str) -> List[dict]:
    sys_msg = (
        "أنت مساعد عربي يعتمد فقط على السياق المقتبس. "
        "أجب بجملة أو فقرتين متصلتين وبأسلوب واضح ومباشر. "
        "إن لم تجد الإجابة في السياق، قل: لا توجد معلومات كافية في السياق للإجابة."
    )
    user_msg = (
        "السياق المقتبس من الدليل (استخدمه فقط):\n\n"
        f"{stitched_context}\n\n"
        f"السؤال: {question}\n\n"
        "الجواب:"
    )
    return [{"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg}]

def generate_answer(qwen: QwenBundle, question: str, context: str) -> str:
    messages = make_messages(question, context)

    # Qwen chat template -> Tensor (not dict!)
    input_ids = qwen.tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=4096  # guardrail; actual will be shorter
    )

    # --- FIX: wrap tensor + create attention_mask ---
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.to(qwen.device)
        if qwen.tok.pad_token_id is not None:
            attention_mask = (input_ids != qwen.tok.pad_token_id).long()
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    else:
        # (rare) older tokenizers may return dict
        model_inputs = {k: v.to(qwen.device) for k, v in input_ids.items()}

    with torch.no_grad():
        out = qwen.model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMP,
            top_p=TOP_P,
            repetition_penalty=REPETITION_P,
            do_sample=(TEMP > 0),
            eos_token_id=qwen.tok.eos_token_id,
            pad_token_id=qwen.tok.eos_token_id,
            use_cache=True,
        )

    # strip the prompt from the decoded text to keep only the assistant part
    prompt_txt = qwen.tok.decode(model_inputs["input_ids"][0], skip_special_tokens=True)
    gen_txt    = qwen.tok.decode(out[0], skip_special_tokens=True)
    ans = gen_txt[len(prompt_txt):].strip()
    # normalize some stray artifacts
    ans = ans.replace("\n\n", "\n").strip()
    return ans if ans else "لا توجد معلومات كافية في السياق للإجابة."

# ----------------- Context builder -----------------

def build_context(index: HybridIndex, question: str, rerank: bool) -> Tuple[str, str]:
    """Return stitched short context + a sources line."""
    intent = classify_intent(question)
    hits = retrieve(index, question, rerank)
    if not hits:
        return "", ""

    parts = []
    srcs  = []
    used  = 0
    for rank, (_, i) in enumerate(hits, 1):
        if used >= TOPK_DOCS:
            break
        ch = index.chunks[i]
        snip = best_snippet(ch, ar_normalize(question), intent, max_len=420)
        if not snip:
            continue
        # avoid near-duplicates
        if any(snip in p or p in snip for p in parts):
            continue
        parts.append(f"[المصدر {rank}: Data_pdf.pdf - ص{ch.page}]\n{snip}")
        srcs.append(f"{rank}. Data_pdf.pdf - ص{ch.page}")
        used += 1
        if sum(len(p) for p in parts) > MAX_CTX_CHARS:
            break

    stitched = "\n\n".join(parts)
    sources_line = "المصادر:\n" + "\n".join(srcs) if srcs else ""
    return stitched, sources_line

# ----------------- Sanity set -----------------

SANITY = [
    "ما هي ساعات الدوام الرسمية من وإلى؟",
    "هل يوجد مرونة في الحضور والانصراف؟ وكيف تُحسب دقائق التأخير؟",
    "هل توجد استراحة خلال الدوام؟ وكم مدتها؟",
    "ما ساعات العمل في شهر رمضان؟ وهل تتغير؟",
    "ما أيام الدوام الرسمي؟ وهل السبت يوم عمل؟",
    "كيف يُحتسب الأجر عن الساعات الإضافية في الأيام العادية؟",
    "ما التعويض عند العمل في العطل الرسمية؟",
    "هل يحتاج العمل الإضافي لموافقة مسبقة؟ ومن يعتمدها؟",
]

# ----------------- Main -----------------

def run_sanity(index: HybridIndex, qwen: QwenBundle, rerank: bool):
    print("🧪 Sanity run…\n")
    for q in SANITY:
        ctx, srcs = build_context(index, q, rerank)
        print(f"• {q}")
        if not ctx:
            print("لا توجد معلومات كافية في السياق للإجابة.\n")
            continue
        t0 = time.time()
        a = generate_answer(qwen, q, ctx)
        dt = time.time() - t0
        # one clean paragraph + sources
        a = a.replace("\n", " ").strip()
        print(f"⏱ {dt:.2f}s | 🤖 {a}\n{srcs}\n")

def chat_loop(index: HybridIndex, qwen: QwenBundle, rerank: bool):
    print("\n💬 اكتب سؤالك (اكتب exit للخروج):")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q: 
            continue
        if q.lower() in ("exit","quit","q"):
            break
        ctx, srcs = build_context(index, q, rerank)
        if not ctx:
            print("🤖 لا توجد معلومات كافية في السياق للإجابة.\n")
            continue
        a = generate_answer(qwen, q, ctx).replace("\n"," ").strip()
        print(f"🤖 {a}\n{srcs}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default=DEFAULT_CHUNKS)
    ap.add_argument("--hier-index", type=str, default=DEFAULT_HIER)
    ap.add_argument("--aliases", type=str, default=DEFAULT_ALIAS)
    ap.add_argument("--load-index", type=str, default=DEFAULT_IDXDIR)
    ap.add_argument("--save-index", type=str, default=None)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--sanity", action="store_true")
    # rerank toggle (default ON)
    ap.add_argument("--rerank",     dest="rerank", action="store_true")
    ap.add_argument("--no-rerank",  dest="rerank", action="store_false")
    ap.set_defaults(rerank=True)
    args = ap.parse_args()

    # Build/Load index
    hier = load_hierarchy(args.hier_index, args.aliases)
    chunks, chash = load_chunks(path=args.chunks)
    index = HybridIndex(chunks, chash, hier=hier)

    loaded = False
    if args.load_index and os.path.isdir(args.load_index):
        loaded = index.load(args.load_index)
    if not loaded:
        index.build()
        if args.save_index:
            index.save(args.save_index)

    # LLM
    qwen = load_qwen(args.model)

    if args.sanity:
        run_sanity(index, qwen, args.rerank)
    else:
        chat_loop(index, qwen, args.rerank)

if __name__ == "__main__":
    main()

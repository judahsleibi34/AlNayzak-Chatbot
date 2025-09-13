# qwen_rag_runner.py
# -*- coding: utf-8 -*-
"""
RAG runner that reuses your retriever (retrival_model.py) and drives Qwen2.5-7B-Instruct.

Highlights
- Arabic-first prompt -> single paragraph answer + sources.
- Uses your HybridIndex / retrieve(...) / best_snippet(...) / classify_intent(...).
- Optional CE reranker (--rerank).
- Index persistence (--save-index / --load-index).
- Fallback to deterministic synthesizers when the LLM under-answers.

Usage examples
--------------
# First run: build + save artifacts, and quick sanity
python qwen_rag_runner.py --chunks Data_pdf_clean_chunks.jsonl --save-index .artifact --sanity

# Fast runs: load artifacts, chat interactively
python qwen_rag_runner.py --chunks Data_pdf_clean_chunks.jsonl --load-index .artifact

# With reranker (slower but better)
python qwen_rag_runner.py --chunks Data_pdf_clean_chunks.jsonl --load-index .artifact --rerank
"""

import os, time, argparse, textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import retrival_model as rm  # <- your file

# ---------- Model ----------

def load_qwen(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    use_cuda = torch.cuda.is_available()
    quant_cfg = None
    if use_cuda:
        try:
            import bitsandbytes as _bnb  # noqa: F401
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            print("⚙️ Using 4-bit quantization (bitsandbytes).")
        except Exception:
            print("ℹ️ bitsandbytes not available -> loading in float16 on GPU." if use_cuda else "ℹ️ CPU mode.")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # ensure pad exists

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        quantization_config=quant_cfg,
        low_cpu_mem_usage=True,
    )
    model.eval()
    try:
        if torch.__version__ >= "2.0" and use_cuda:
            model = torch.compile(model, mode="reduce-overhead")
            print("🚀 Compiled with torch.compile")
    except Exception as e:
        print(f"compile skipped: {e}")
    device = model.device
    print(f"📊 Model on: {device}")
    return tok, model, device

# ---------- Context building (uses your retriever) ----------

def build_context(index: rm.HybridIndex, question: str, use_rerank: bool,
                  topk: int = 4, max_chars: int = 1400):
    intent = rm.classify_intent(question)
    hits = rm.retrieve(index, question, rerank=use_rerank)
    if not hits:
        return "", intent, []

    # Collect intent-aware snippets with sources
    snippets = []
    sources = []
    used_chars = 0
    for rank, (_, idx) in enumerate(hits[:topk], 1):
        chunk = index.chunks[idx]
        sn = rm.best_snippet(chunk, rm.ar_normalize(question), intent, max_len=300)
        if not sn:
            continue
        piece = f"[المصدر {rank}: Data_pdf.pdf - ص{chunk.page}]\n{sn}\n"
        if used_chars + len(piece) > max_chars:
            break
        snippets.append(piece)
        sources.append(f"{rank}. Data_pdf.pdf - ص{chunk.page}")
        used_chars += len(piece)

    return "\n".join(snippets), intent, sources

# ---------- Prompting ----------

SYS_AR = (
    "أنت مساعد عربي يعتمد فقط على السياق المقتبس. "
    "أجب بدقة وبفِقْرة عربية واحدة متماسكة بأسلوب مهني وواضح. "
    "لا تضف أي معلومة ليست موجودة في السياق. "
    "إذا لم يكفِ السياق، قل حرفيًا: لا توجد معلومات كافية في السياق للإجابة."
)

def format_prompt(context: str, question: str) -> str:
    # Qwen chat template works fine with plain strings too; we craft a strict instruction.
    return (
        f"system\n{SYS_AR}\n\n"
        f"user\nالسياق:\n{context}\n\n"
        f"السؤال: {question}\n"
        f"أجب الآن بفِقْرة واحدة فقط."
    )

def call_llm(tok, model, device, prompt: str, max_new_tokens=320, temperature=0.15, top_p=0.9):
    enc = tok(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    enc = {k: v.to(device) for k, v in enc.items()}  # includes attention_mask => no warnings
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.05,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    txt = tok.decode(out[0], skip_special_tokens=True)
    # Keep only the part after our "user" block if template echoes:
    if "user\n" in prompt and txt.startswith(prompt):
        txt = txt[len(prompt):].strip()
    return txt.strip()

# ---------- Answer orchestration ----------

def answer_once(index: rm.HybridIndex, tok, model, device, question: str, use_rerank: bool):
    t0 = time.time()
    context, intent, sources = build_context(index, question, use_rerank)
    if not context.strip():
        return "لا توجد معلومات كافية في السياق للإجابة.", sources, time.time() - t0

    prompt = format_prompt(context, question)
    llm_out = call_llm(tok, model, device, prompt)

    # Heuristic fallback if the LLM under-answers
    bad = (
        (len(llm_out) < 30)
        or ("لا توجد معلومات كافية" in llm_out)
        or llm_out.strip().lower() in {"", "ok", "?", "null"}
    )
    if bad:
        # Use your deterministic synthesizers as a failsafe
        det = rm.answer(question, index, intent, use_rerank)
        return det, sources, time.time() - t0

    # Ensure single-paragraph style, add sources footer
    body = " ".join(llm_out.splitlines()).strip()
    footer = "\nالمصادر:\n" + "\n".join(sources) if sources else ""
    return body + footer, sources, time.time() - t0

# ---------- CLI ----------

def ensure_index(args):
    hier = rm.load_hierarchy(args.hier_index, args.aliases)
    chunks, chunks_hash = rm.load_chunks(path=args.chunks)
    index = rm.HybridIndex(chunks, chunks_hash, hier=hier)
    loaded = False
    if args.load_index:
        loaded = index.load(args.load_index)
    if not loaded:
        print("🔄 Building dense+TF-IDF indexes …")
        index.build()
        if args.save_index:
            index.save(args.save_index)
    return index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default="Data_pdf_clean_chunks.jsonl")
    ap.add_argument("--hier-index", type=str, default="heading_inverted_index.json")
    ap.add_argument("--aliases", type=str, default="section_aliases.json")
    ap.add_argument("--save-index", type=str, default=None)
    ap.add_argument("--load-index", type=str, default=None)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--rerank", action="store_true", help="enable cross-encoder reranker")
    ap.add_argument("--no-rerank", action="store_true", help="force disable reranker")
    ap.add_argument("--sanity", action="store_true", help="run sanity prompts and exit")
    args = ap.parse_args()
    use_rerank = args.rerank and not args.no_rerank

    # Build / load index
    index = ensure_index(args)

    # Load LLM
    print("🧠 Loading Qwen2.5 …")
    tok, model, device = load_qwen(args.model)

    def ask(q):
        out, sources, dt = answer_once(index, tok, model, device, q, use_rerank)
        print(f"\n• {q}\n⏱ {dt:.2f}s | 🤖 {out}\n")

    if args.sanity:
        print("🧪 Sanity run…")
        for q in rm.SANITY_PROMPTS[:12]:  # 12 to keep it quick; adjust as needed
            ask(q)
        return

    # Interactive
    print("\n💬 Ready. اكتب سؤالك بالعربية (أو 'exit' للخروج).\n")
    while True:
        try:
            q = input("سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye."); break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("Bye."); break
        ask(q)

if __name__ == "__main__":
    main()

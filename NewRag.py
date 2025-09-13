# -*- coding: utf-8 -*-
"""
RAG (Arabic) with Qwen2.5-7B-Instruct, using YOUR retriever (retrival_model.py).

- Loads HybridIndex from your artifacts (fast) or builds/saves if needed
- Uses your `retrieve(...)`, `best_snippet(...)`, `classify_intent(...)`
- Extractive-style context + concise Arabic answer
- Safe defaults: if no --chunks passed, it tries Data_pdf_clean_chunks.jsonl in CWD
- Fixes attention_mask warning, avoids invalid generation flags

Usage examples
--------------
# 1) Fast load from saved index
python rag_qwen25_external.py --chunks Data_pdf_clean_chunks.jsonl --load-index .artifact --rerank --chat

# 2) Build then save artifacts (first run), then sanity test
python rag_qwen25_external.py --chunks Data_pdf_clean_chunks.jsonl --save-index .artifact --sanity --rerank

# 3) Quick benchmark on a few HR questions
python rag_qwen25_external.py --chunks Data_pdf_clean_chunks.jsonl --load-index .artifact --bench
"""

import os, sys, argparse, time, re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional quantization (BitsAndBytes)
_USE_BNB = True
try:
    from transformers import BitsAndBytesConfig
except Exception:
    _USE_BNB = False

# ----------------------------
# Bring in YOUR retriever bits
# ----------------------------
# Make sure this file is in the same folder as retrival_model.py
try:
    from retrival_model import (
        HybridIndex, load_chunks, load_hierarchy,
        retrieve, best_snippet, classify_intent, ar_normalize, SANITY_PROMPTS
    )
except Exception as e:
    print("❌ Could not import retrival_model.py. Make sure it exists next to this script.")
    raise

DEFAULT_CHUNKS = "Data_pdf_clean_chunks.jsonl"
DEFAULT_HIER   = "heading_inverted_index.json"
DEFAULT_ALIAS  = "section_aliases.json"
DEFAULT_ART    = ".artifact"

@dataclass
class Hit:
    score: float
    idx: int
    page: int
    source: str
    text: str
    snippet: str

def _fmt_source(page: int, source: str = "Data_pdf.pdf") -> str:
    return f"{source} - ص{page}"

def _select_snippets(
    question: str,
    intent: str,
    index: HybridIndex,
    prelim: List[Tuple[float,int]],
    top_k: int = 8,
    max_snippet_len: int = 320
) -> List[Hit]:
    """Turn (score, idx) into ready-to-pack context snippets with pages/sources."""
    qn = ar_normalize(question)
    hits: List[Hit] = []
    used_ids = set()
    for sc, i in prelim[:top_k]:
        if i in used_ids:
            continue
        used_ids.add(i)
        ch = index.chunks[i]
        sn = best_snippet(ch, qn, intent, max_len=max_snippet_len)
        if not sn:
            # fallback to trimmed chunk text
            raw = (ch.text or "").strip()
            sn = raw[:max_snippet_len] + ("…" if len(raw) > max_snippet_len else "")
        hits.append(Hit(score=float(sc), idx=int(i), page=int(ch.page), source="Data_pdf.pdf", text=ch.text, snippet=sn))
    return hits

def _build_context_blocks(hits: List[Hit], max_chars: int = 1600) -> Tuple[str, List[Dict]]:
    """Pack snippets into a single string with per-source headers; keep under max_chars."""
    parts: List[str] = []
    meta: List[Dict] = []
    total = 0
    for k, h in enumerate(hits, 1):
        header = f"[المصدر {k}: {h.source} - ص{h.page}]"
        block = f"{header}\n{h.snippet}"
        add = len(block) + 2
        if total + add > max_chars:
            # try a smaller tail if we still can add something meaningful
            room = max_chars - total - len(header) - 2
            if room > 80:
                short = h.snippet[:room] + "…"
                parts.append(f"{header}\n{short}")
                meta.append({"k": k, "page": h.page, "source": h.source, "score": h.score, "truncated": True, "idx": h.idx})
                total = max_chars
            break
        parts.append(block)
        meta.append({"k": k, "page": h.page, "source": h.source, "score": h.score, "truncated": False, "idx": h.idx})
        total += add
    return "\n\n".join(parts), meta

def _make_prompt_ar(context: str, question: str) -> str:
    """
    A compact Arabic-first instruction with strong grounding requirement.
    Uses simple ChatML-style markers that Qwen2.5 understands well.
    """
    sys_msg = (
        "أنت مساعد عربي يعتمد فقط على السياق المقدم. "
        "أجب بدقة وباختصار وبنقاط مرتبة. "
        "إن لم تجد الإجابة في السياق، قل: «لا توجد معلومات كافية في السياق للإجابة»."
    )
    user_msg = (
        f"السياق المقتبس من الدليل (لا تضف معلومات خارج ما يلي):\n\n{context}\n\n"
        f"السؤال: {question}\n\n"
        "التعليمات للإخراج:\n"
        "- جواب موجز ومنظم (نقاط مختصرة).\n"
        "- إن وجدت أرقام/أوقات في السياق فاذكرها كما هي.\n"
        "- لا تضف مصادر داخل النص؛ سأضيفها لاحقًا."
    )
    return (
        "<|im_start|>system\n" + sys_msg + "\n<|im_end|>\n"
        "<|im_start|>user\n" + user_msg + "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

class QwenRAG:
    def __init__(
        self,
        chunks_path: Optional[str] = None,
        hier_index: Optional[str] = DEFAULT_HIER,
        aliases_path: Optional[str] = DEFAULT_ALIAS,
        load_index_dir: Optional[str] = DEFAULT_ART,
        save_index_dir: Optional[str] = None,
        use_rerank: bool = False,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        top_k: int = 8,
        max_ctx_chars: int = 1600,
        max_new_tokens: int = 320,
        temperature: float = 0.2,
    ):
        self.top_k = top_k
        self.max_ctx_chars = max_ctx_chars
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_rerank = use_rerank

        # ---------- 1) Load chunks + hierarchy
        kb = chunks_path or DEFAULT_CHUNKS
        if not os.path.exists(kb):
            print(f"⚠️  Chunks file not found at '{kb}'. I will try default '{DEFAULT_CHUNKS}' in CWD.")
            kb = DEFAULT_CHUNKS
            if not os.path.exists(kb):
                raise FileNotFoundError(
                    f"Chunks file not found. Provide --chunks path or put '{DEFAULT_CHUNKS}' in this folder."
                )

        hier = load_hierarchy(hier_index, aliases_path)
        chunks, chunks_hash = load_chunks(path=kb)
        self.index = HybridIndex(chunks, chunks_hash, hier=hier)

        # ---------- 2) Load or build index artifacts
        loaded = False
        if load_index_dir and os.path.isdir(load_index_dir):
            loaded = self.index.load(load_index_dir)

        if not loaded:
            print("🔄 Building dense+TF-IDF indexes (first run)…")
            self.index.build()
            if save_index_dir:
                self.index.save(save_index_dir)
        else:
            print("✅ Index loaded from artifacts")

        # ---------- 3) Load LLM (quantized if possible)
        self._load_llm(model_name)

    # LLM loader
    def _load_llm(self, model_name: str):
        print("🧠 Loading Qwen2.5 model…")
        bnb_cfg = None
        model_kwargs = dict(
            device_map="auto",
            trust_remote_code=True,
        )

        if torch.cuda.is_available() and _USE_BNB:
            try:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["quantization_config"] = bnb_cfg
                model_kwargs["torch_dtype"] = torch.float16
            except Exception as e:
                print(f"⚠️  4-bit quantization setup failed: {e}. Falling back to fp16 if GPU.")
                model_kwargs["torch_dtype"] = torch.float16
        else:
            # CPU or BnB not available
            model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Qwen warns if pad == eos and no attention_mask; we'll build attention_mask manually.
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.llm.eval()

        # Compile (PyTorch 2+); safe try
        try:
            if hasattr(torch, "compile"):
                self.llm = torch.compile(self.llm, mode="reduce-overhead")
                print("🚀 Model compiled for faster inference")
        except Exception as e:
            print(f"⚠️  torch.compile skipped: {e}")

        device = next(self.llm.parameters()).device
        print(f"📊 Model on: {device}")

    # Core QA
    def generate(self, question: str) -> Dict:
        t0 = time.time()
        intent = classify_intent(question)

        # 1) retrieve via YOUR function
        prelim = retrieve(self.index, question, self.use_rerank)
        if not prelim:
            return {
                "answer": "لا توجد معلومات كافية في السياق للإجابة.",
                "context": "",
                "meta": [],
                "time": time.time() - t0,
            }

        # 2) turn into snippets + context
        hits = _select_snippets(question, intent, self.index, prelim, top_k=self.top_k)
        context, meta = _build_context_blocks(hits, max_chars=self.max_ctx_chars)
        if not context.strip():
            return {
                "answer": "لا توجد معلومات كافية في السياق للإجابة.",
                "context": "",
                "meta": meta,
                "time": time.time() - t0,
            }

        # 3) prompt & generate
        prompt = _make_prompt_ar(context, question)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        # Build explicit attention mask to avoid pad/eos ambiguity warnings
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        enc = {k: v.to(self.llm.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.llm.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature,
                top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                use_cache=True,
            )
        text = self.tok.decode(out[0], skip_special_tokens=True)

        # Extract only the assistant tail after our ChatML tag to avoid echo
        ans = text.split("<|im_start|>assistant")[-1]
        ans = ans.replace("<|im_end|>", "").strip()
        if not ans:
            ans = text.strip()

        # 4) add sources block
        refs = "\n".join([f"{m['k']}. {m['source']} - ص{m['page']}" for m in meta])
        final = ans + ("\n\nالمراجع:\n" + refs if refs else "")

        return {
            "answer": final,
            "context": context,
            "meta": meta,
            "time": time.time() - t0,
        }

    # helpers
    def ask(self, q: str) -> str:
        return self.generate(q)["answer"]

    def sanity(self, subset: Optional[int] = None):
        qs = SANITY_PROMPTS[:subset] if subset else SANITY_PROMPTS
        print("\n🧪 Sanity run…")
        total = 0.0
        for q in qs:
            r = self.generate(q)
            total += r["time"]
            print(f"\n• {q}\n⏱ {r['time']:.2f}s | 🤖 {r['answer'][:500]}{'…' if len(r['answer'])>500 else ''}")
        print(f"\n📊 Avg: {total/len(qs):.2f}s | Total: {total:.2f}s")

    def chat(self):
        print("\n=== Arabic Q&A (Qwen2.5 + your retriever) ===")
        print("أكتب سؤالك. اكتب exit للخروج.\n")
        while True:
            try:
                q = input("سؤالك: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nخروج."); break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                print("خروج."); break
            r = self.generate(q)
            print("\n" + r["answer"] + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default=DEFAULT_CHUNKS, help="Path to chunks JSON/JSONL")
    ap.add_argument("--hier-index", type=str, default=DEFAULT_HIER, help="(optional) heading_inverted_index.json")
    ap.add_argument("--aliases", type=str, default=DEFAULT_ALIAS, help="(optional) section_aliases.json")
    ap.add_argument("--load-index", type=str, default=DEFAULT_ART, help="dir to load saved index")
    ap.add_argument("--save-index", type=str, default=None, help="dir to save built index")
    ap.add_argument("--rerank", action="store_true", help="enable CrossEncoder re-ranking in retriever")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HF model id")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-ctx", type=int, default=1600)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()

    rag = QwenRAG(
        chunks_path=args.chunks,
        hier_index=args.hier_index,
        aliases_path=args.aliases,
        load_index_dir=args.load_index,
        save_index_dir=args.save_index,
        use_rerank=args.rerank,
        model_name=args.model,
        top_k=args.top_k,
        max_ctx_chars=args.max_ctx,
        max_new_tokens=args.max_new,
        temperature=args.temp,
    )

    if args.sanity:
        rag.sanity(subset=None)
        return

    if args.bench:
        qs = [
            "ما هي سياسات التوظيف في مؤسسة النيزك؟",
            "كيف يمكن التقدم للوظائف؟",
            "ما هي المستندات المطلوبة للموظف الجديد؟",
        ]
        print("\n🧪 Benchmark…")
        total = 0.0
        for q in qs:
            r = rag.generate(q)
            total += r["time"]
            print(f"\n• {q}\n⏱ {r['time']:.2f}s | 🤖 {r['answer'][:600]}{'…' if len(r['answer'])>600 else ''}")
        print(f"\n📊 Avg time: {total/len(qs):.2f}s | Total: {total:.2f}s")

    if args.chat:
        rag.chat()

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
NewRag.py — Arabic RAG with hybrid retrieval and Qwen2.5 chat formatting.

What’s inside (fixed & improved):
- Safe KB path resolution:  --kb  >  AR_KB_JSONL env  >  Data_pdf_clean_chunks.jsonl
- Float-index bug fixed (we now use the idx from (score, idx))
- Qwen2.5 chat-template via tokenizer.apply_chat_template(...)
- Tight Arabic system prompt (grounded RAG; no hallucinations)
- Hybrid retrieval: dense (intfloat/multilingual-e5-base) + TF-IDF (char+word), FAISS
- Optional multilingual reranker (Jina v2), fallback to mMiniLM cross-encoder
- Artifacts cache (embeddings/FAISS/TF-IDF) keyed by file fingerprint
- Better sentence picking for coherent snippets

Usage (Colab T4):
  %cd /content/AlNayzak-Chatbot
  !python NewRag.py --llm Qwen/Qwen2.5-7B-Instruct --rerank --top-k 8 --max-chars 2400
"""

import os, re, json, time, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

# ---------------- Optional deps ----------------
try:
    import faiss  # faiss-cpu or faiss-gpu if available
except Exception:
    faiss = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib
except Exception:
    TfidfVectorizer = None
    joblib = None

from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Optional reranker
HAS_RERANK = True
try:
    from sentence_transformers import CrossEncoder
except Exception:
    HAS_RERANK = False

# ---------------- Arabic utils ----------------
AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"٠١٢٣٤٥٦٧٨٩")}
IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"۰۱۲۳۴۵۶۷۸۹")}
SENT_SPLIT_RE = re.compile(r'(?<=[\.\!\؟\?،؛;]|[\n—])\s+')

def ar_normalize(s: str) -> str:
    if not s:
        return ""
    s = s.replace('\u0640', '')         # tatweel
    s = AR_DIAC.sub('', s)              # diacritics
    s = (s.replace('أ','ا').replace('إ','ا').replace('آ','ا')
           .replace('ى','ي').replace('ة','ه'))
    s = s.translate(AR_NUMS).translate(IR_NUMS)
    s = s.replace('،', ',').replace('٫','.')
    s = ' '.join(s.split())
    return s

def sent_split(s: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(s) if p and p.strip()]
    return parts if parts else ([s.strip()] if s and s.strip() else [])

# ---------------- Data model ----------------
@dataclass
class Chunk:
    id: str
    text_display: str
    text_embed: str
    source: str
    page: int

# ---------------- IO helpers ----------------
_TEXT_KEYS = {"text","text_display","content","body","raw","paragraph","para","value","data","clean_text","norm"}
_TEXT_ARRAY_KEYS = {"lines","paragraphs","paras","sentences","chunks","blocks","spans","tokens"}
_PAGE_KEYS = {"page","page_no","page_num","pageNumber","page_index","Page","PageNo"}
_ID_KEYS = {"id","chunk_id","cid","idx","index","Id","ID"}

def _as_text(v):
    if isinstance(v, str):
        v = v.strip()
        return v if v else None
    if isinstance(v, list):
        parts = [str(x).strip() for x in v if isinstance(x, (str,int,float)) and str(x).strip()]
        return "\n".join(parts) if parts else None
    return None

def _get_any(d: dict, keys: set):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    lower = {k.lower(): k for k in d.keys()}
    for k in keys:
        lk = k.lower()
        if lk in lower and d[lower[lk]] not in (None, ""):
            return d[lower[lk]]
    return None

# ---------------- Retriever ----------------
class HybridRetriever:
    def __init__(self, artifact_dir: str = ".artifact",
                 model_name: str = "intfloat/multilingual-e5-base"):
        self.artifacts = Path(artifact_dir)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        print("🔄 Loading embedding model:", model_name)
        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 384
        print(f"✅ Embedding model loaded in {time.time()-t0:.2f}s")
        self.chunks: List[Chunk] = []
        self.emb: Optional[np.ndarray] = None
        self.faiss = None
        self.tf_char = None; self.tf_word = None
        self.char_mat = None; self.word_mat = None

    def _encode(self, texts: List[str], is_query: bool) -> np.ndarray:
        tnorm = [ar_normalize(t) for t in texts]
        if "intfloat/multilingual-e5" in self.model_name:
            pref = "query: " if is_query else "passage: "
            tnorm = [pref + t for t in tnorm]
        return self.model.encode(
            tnorm, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )

    def load_jsonl(self, path: str) -> List[Dict]:
        rows = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line: continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        if i <= 10:
                            print(f"⚠️ JSON error at line {i}: {e}")
        except FileNotFoundError:
            print(f"❌ File not found: {path}")
            return []
        print(f"✅ Loaded {len(rows)} rows from {path}")
        return rows

    def prepare_chunks(self, rows: List[Dict]):
        self.chunks.clear()
        for i, j in enumerate(rows):
            disp = _as_text(_get_any(j, _TEXT_KEYS)) or _as_text(_get_any(j, _TEXT_ARRAY_KEYS)) or ""
            embt = _as_text(_get_any(j, _TEXT_KEYS)) or disp
            page = _get_any(j, _PAGE_KEYS)
            try: page = int(page) if page is not None else 0
            except Exception: page = 0
            cid = str(_get_any(j, _ID_KEYS) or f"row{i}")
            embt = ar_normalize(embt or "")
            if not embt: continue
            self.chunks.append(Chunk(
                id=cid,
                text_display=(disp or embt).strip(),
                text_embed=embt,
                source=str(j.get("source","Data_pdf.pdf")),
                page=page
            ))
        print(f"✅ Prepared {len(self.chunks)} chunks")

    def _kb_fingerprint(self, kb_path: str) -> str:
        p = Path(kb_path)
        st = p.stat()
        h = hashlib.sha256()
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
        for c in self.chunks[:200]:
            h.update(c.id.encode()); h.update(c.text_embed[:256].encode())
        return h.hexdigest()[:24]

    def _p(self, name: str) -> Path:
        return self.artifacts / name

    def build_or_load(self, kb_path: str, use_cache=True):
        if not self.chunks:
            raise ValueError("No chunks loaded")
        fpr = self._kb_fingerprint(kb_path)
        meta = self._p("meta.json"); embp = self._p("embeddings.npy"); fip = self._p("faiss.index")
        tfc = self._p("tf_char.pkl"); tfw = self._p("tf_word.pkl"); cm = self._p("char_mat.pkl"); wm = self._p("word_mat.pkl")

        if use_cache and meta.exists() and embp.exists():
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                if m.get("fp")==fpr and m.get("model")==self.model_name and m.get("n")==len(self.chunks):
                    print("📦 Loading dense artifacts…")
                    self.emb = np.load(str(embp), mmap_mode="r")
                    if faiss is not None and fip.exists():
                        self.faiss = faiss.read_index(str(fip))
                    else:
                        if faiss is not None:
                            d = self.emb.shape[1]
                            self.faiss = faiss.IndexFlatIP(d)
                            self.faiss.add(self.emb.astype("float32"))
                    if TfidfVectorizer and joblib and tfc.exists() and tfw.exists() and cm.exists() and wm.exists():
                        print("📦 Loading TF-IDF artifacts…")
                        self.tf_char = joblib.load(str(tfc)); self.tf_word = joblib.load(str(tfw))
                        self.char_mat = joblib.load(str(cm)); self.word_mat = joblib.load(str(wm))
                    print("✅ Index loaded from artifacts")
                    return
            except Exception as e:
                print("⚠️ Cache load failed, rebuilding…", e)

        print("🔄 Building embeddings + indexes…")
        texts = [c.text_embed for c in self.chunks]
        t0 = time.time()

        self.emb = self._encode(texts, is_query=False)

        if faiss is not None:
            d = self.emb.shape[1]
            if len(self.emb) < 10_000:
                self.faiss = faiss.IndexFlatIP(d)
                self.faiss.add(self.emb.astype("float32"))
            else:
                nlist = min(100, len(self.emb)//100)
                quant = faiss.IndexFlatIP(d)
                self.faiss = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
                self.faiss.train(self.emb.astype("float32"))
                self.faiss.add(self.emb.astype("float32"))

        if TfidfVectorizer:
            print("🧪 Building TF-IDF (char+word)…")
            self.tf_char = TfidfVectorizer(analyzer='char', ngram_range=(2,5), min_df=1)
            self.char_mat = self.tf_char.fit_transform(texts)
            self.tf_word = TfidfVectorizer(analyzer='word', ngram_range=(1,2),
                                           token_pattern=r"(?u)\b\w+\b", min_df=1)
            self.word_mat = self.tf_word.fit_transform(texts)

        # save artifacts
        print("💾 Saving artifacts…")
        np.save(str(embp), self.emb)
        if faiss is not None and self.faiss is not None:
            faiss.write_index(self.faiss, str(fip))
        if TfidfVectorizer and joblib:
            joblib.dump(self.tf_char, str(tfc)); joblib.dump(self.tf_word, str(tfw))
            joblib.dump(self.char_mat, str(cm)); joblib.dump(self.word_mat, str(wm))
        meta.write_text(json.dumps({"fp":fpr,"model":self.model_name,"n":len(self.chunks)}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Artifacts saved to {self.artifacts}")
        print(f"✅ Index build complete in {time.time()-t0:.2f}s")

    def semantic_search(self, query: str, top_k: int = 10, pre_candidates: int = 80) -> List[Tuple[float, int]]:
        """
        Returns list of (score, idx). The higher the score, the better.
        """
        qv = self._encode([query], is_query=True)[0]
        if self.faiss is not None:
            D, I = self.faiss.search(qv.reshape(1,-1).astype("float32"), max(pre_candidates, top_k*6))
            dS, dI = D[0], I[0]
        else:
            sims = self.emb @ qv
            dI = np.argsort(-sims)[:max(pre_candidates, top_k*6)]
            dS = sims[dI]

        # sparse channel
        if self.tf_char is not None and self.tf_word is not None:
            qc = self.tf_char.transform([ar_normalize(query)])
            qw = self.tf_word.transform([ar_normalize(query)])
            cS = (self.char_mat @ qc.T).toarray().ravel()
            wS = (self.word_mat @ qw.T).toarray().ravel()
        else:
            cS = wS = None

        # fuse
        fused: List[Tuple[float,int]] = []
        for s, i in zip(dS, dI):
            i = int(i)
            sc = float(s)*0.55
            if cS is not None and i < len(cS): sc += float(cS[i])*0.25
            if wS is not None and i < len(wS): sc += float(wS[i])*0.20
            fused.append((sc, i))
        fused.sort(key=lambda x: -x[0])
        return fused[:max(pre_candidates, top_k)]

    def rerank(self, query: str, candidates: List[int], max_keep: int = 10) -> List[int]:
        if not HAS_RERANK or not candidates:
            return candidates[:max_keep]
        try:
            ce = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual")
        except Exception:
            try:
                ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
            except Exception:
                print("⚠️ Reranker unavailable; continuing without.")
                return candidates[:max_keep]
        pairs = [(ar_normalize(query), self.chunks[int(i)].text_display) for i in candidates]
        scores = ce.predict(pairs)
        order = np.argsort(-scores)
        return [int(candidates[i]) for i in order[:max_keep]]

    def sentence_pick(self, question: str, text: str) -> List[str]:
        # improved coherence picker
        qset = set([w for w in ar_normalize(question).split() if len(w) >= 3])
        sents = sent_split(text)
        scored = []
        for s in sents:
            sn = ar_normalize(s)
            overlap = len(qset & set(sn.split()))
            has_num = bool(re.search(r'\d', sn))
            has_time = any(t in sn for t in ["من","الى","إلى","حتى",":","."])
            score = 1.2*overlap + 0.8*has_num + 0.6*has_time
            scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return sents[:2]
        top_idx = [sents.index(scored[0][1])]
        if len(scored) > 1:
            top_idx.append(sents.index(scored[1][1]))
        keep = set()
        for i in top_idx:
            for j in [i-1, i, i+1]:
                if 0 <= j < len(sents):
                    keep.add(j)
        ordered = [sents[i] for i in sorted(keep)]
        return ordered[:6]

    def build_context(self, question: str, max_chars: int = 2200, top_k: int = 10, use_rerank: bool = False) -> Tuple[str, List[Dict]]:
        pre = self.semantic_search(question, top_k=top_k, pre_candidates=80)  # [(score, idx)]
        cand = [idx for _, idx in pre]  # use index, not score  ✅

        if use_rerank:
            cand = self.rerank(question, cand, max_keep=top_k)
        else:
            cand = cand[:top_k]

        parts, meta, total = [], [], 0
        for rank, idx in enumerate(cand, 1):
            idx = int(idx)  # safety
            c = self.chunks[idx]
            head = f"[المصدر {rank}: {c.source} - ص{c.page}]"
            picked = self.sentence_pick(question, c.text_display)
            body = " ".join(picked)
            need = len(head) + 1 + len(body) + 2
            if total + need <= max_chars:
                parts.append(f"{head}\n{body}")
                total += need
                meta.append({"rank": rank, "source": c.source, "page": c.page, "id": c.id})
            else:
                remain = max_chars - total - len(head) - 2
                if remain > 120:
                    parts.append(f"{head}\n{body[:remain-20]} …")
                    meta.append({"rank": rank, "source": c.source, "page": c.page, "id": c.id, "truncated": True})
                break
        return "\n\n".join(parts), meta

# ---------------- RAG pipeline ----------------
class RAGPipeline:
    def __init__(self, kb_path: str, artifact_dir: str = ".artifact",
                 use_cache: bool = True, llm_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 use_rerank: bool = False, top_k: int = 10, max_chars: int = 2200):
        print("🚀 Initializing Optimized Arabic RAG System…")
        print("📚 Setting up hybrid retriever…")
        self.use_rerank = use_rerank; self.top_k = top_k; self.max_chars = max_chars

        # retriever
        self.ret = HybridRetriever(artifact_dir=artifact_dir, model_name="intfloat/multilingual-e5-base")
        rows = self.ret.load_jsonl(kb_path)
        if not rows:
            raise ValueError("No data loaded from JSONL file")
        self.ret.prepare_chunks(rows)
        self.ret.build_or_load(kb_path, use_cache=use_cache)

        # LLM
        print("🧠 Loading LLM…")
        self.tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True,
            dtype=(torch.float16 if torch.cuda.is_available() else torch.bfloat16),
            quantization_config=quant
        )
        self.model.eval()
        try:
            if hasattr(torch, "compile") and torch.__version__ >= "2.0":
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("🚀 Model compiled for faster inference")
        except Exception as e:
            print("ℹ️ torch.compile skipped:", e)
        print(f"📊 Model on: {self.model.device}")
        print("✅ RAG System initialized")

    # ---- Qwen2.5 chat messages ----
    def _chat_messages(self, question: str, context: str):
        system = (
            "أنت مساعد عربي دقيق يعمل بأسلوب RAG. "
            "أجب فقط من (السياق) المرفق. لا تُخمن ولا تضف معلومات خارج السياق. "
            "ضع إشارة مرجعية [المصدر n] بعد كل جملة/حقيقة مستندة إلى النص. "
            "إن لم تجد الإجابة في السياق فقل: غير متوفر في السياق."
        )
        user = (
            f"السياق:\n{context}\n\n"
            f"السؤال: {question}\n\n"
            "أجب بلغة عربية واضحة ومختصرة، مع وضع [المصدر n] بعد كل حقيقة."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate(self, question: str, max_new_tokens: int = 280, temperature: float = 0.15) -> Dict[str, Any]:
        t0 = time.time()
        context, meta = self.ret.build_context(
            question, max_chars=self.max_chars, top_k=self.top_k, use_rerank=self.use_rerank
        )
        if not context.strip():
            return {"answer":"عذراً، لا يوجد سياق ذي صلة.","context":"","meta":meta,"time":time.time()-t0,"confidence":0.0}

        # build chat template for Qwen
        messages = self._chat_messages(question, context)
        prompt_ids = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=0.85,
                repetition_penalty=1.18,
                no_repeat_ngram_size=4,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                use_cache=True
            )
        # decode only the new tokens
        gen = out[0][prompt_ids.shape[-1]:]
        answer = self.tok.decode(gen, skip_special_tokens=True).strip()
        # small cleanup for common drift
        answer = answer.replace("المصدار", "المصدر").replace("المصدَر", "المصدر")

        return {"answer": answer, "context": context, "meta": meta, "time": time.time()-t0, "confidence": 0.85}

    def chat(self):
        print("\n🤖 جاهز. اكتب سؤالك (أو 'خروج').\n")
        while True:
            try:
                q = input("🙋 سؤالك: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q: continue
            if q.lower() in ["خروج","quit","exit"]: break
            r = self.generate(q)
            print("\n🤖", r["answer"], "\n")

# ---------------- main ----------------
def resolve_kb_path(cli_kb: Optional[str]) -> str:
    """
    Priority: CLI --kb > AR_KB_JSONL env > Data_pdf_clean_chunks.jsonl
    """
    if cli_kb:
        return cli_kb
    env = os.environ.get("AR_KB_JSONL")
    if env and Path(env).exists():
        return env
    default = "Data_pdf_clean_chunks.jsonl"
    if Path(default).exists():
        return default
    raise FileNotFoundError(
        "Knowledge base JSONL not found. "
        "Set --kb, or AR_KB_JSONL, or place Data_pdf_clean_chunks.jsonl in the working directory."
    )

if __name__ == "__main__":
    import argparse
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")

    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=str, default=None, help="Path to KB .jsonl")
    ap.add_argument("--artifact", type=str, default=".artifact", help="Artifacts dir")
    ap.add_argument("--no-cache", action="store_true", help="Disable retriever cache")
    ap.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranker")
    ap.add_argument("--top-k", type=int, default=10, help="Top K chunks for context")
    ap.add_argument("--max-chars", type=int, default=2200, help="Context character budget")
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="LLM id, e.g. 'Qwen/Qwen2.5-7B-Instruct' or 'sambanovasystems/SambaLingo-Arabic-Chat'")
    args = ap.parse_args()

    try:
        kb_path = resolve_kb_path(args.kb)
        print(f"📂 Using KB: {kb_path}")
        print(f"🗂  Using artifact dir: {args.artifact}")

        rag = RAGPipeline(
            kb_path=kb_path,
            artifact_dir=args.artifact,
            use_cache=not args.no_cache,
            llm_name=args.llm,
            use_rerank=args.rerank,
            top_k=args.top_k,
            max_chars=args.max_chars
        )

        # quick smoke benchmark
        print("\n🧪 Benchmark…")
        tests = [
            "ما هي سياسات التوظيف في مؤسسة النيزك؟",
            "كيف يمكن التقدم للوظائف؟",
            "ما هي المستندات المطلوبة للموظف الجديد؟",
        ]
        for q in tests:
            r = rag.generate(q)
            print(f"\n• {q}")
            print(f"⏱ {r['time']:.2f}s | 🤖 {r['answer'][:220]}…")

        print("\n💬 Starting chat…")
        # rag.chat()  # uncomment for interactive mode

    except Exception as e:
        print(f"❌ خطأ في النظام: {e}")
        raise

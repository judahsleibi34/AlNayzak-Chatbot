# -*- coding: utf-8 -*-
"""
NewRag.py — Arabic RAG with hybrid retrieval and configurable LLM.

✅ Fix included:
    - Knowledge base path now uses a safe default.
      Priority: --kb CLI  >  AR_KB_JSONL env  >  Data_pdf_clean_chunks.jsonl

Features:
- Robust JSONL loader (tolerant to field naming)
- Arabic normalization + sentence splitting
- Hybrid retrieval (dense intfloat/multilingual-e5-base + TF-IDF char/word)
- Optional Cross-Encoder reranker (Jina multilingual or mMiniLM)
- Artifact caching (embeddings, FAISS, TF-IDF) keyed to file fingerprint
- Grounded Arabic prompt with explicit citation style
- LLM configurable (default Qwen2.5-7B-Instruct; SambaLingo also supported)

Run examples (Colab):
  %cd /content/AlNayzak-Chatbot
  # If your file is Data_pdf_clean_chunks.jsonl (default):
  !python NewRag.py --rerank
  # Or select explicitly:
  !python NewRag.py --kb Data_pdf_clean_chunks.jsonl --rerank
  # Switch LLM to SambaLingo:
  !python NewRag.py --llm sambanovasystems/SambaLingo-Arabic-Chat --rerank
"""

import os, re, json, time, hashlib, sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

# -------- optional deps ----------
try:
    import faiss  # faiss-cpu
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
CE_MODEL_CANDIDATES = [
    "jinaai/jina-reranker-v2-base-multilingual",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
]
try:
    from sentence_transformers import CrossEncoder
except Exception:
    HAS_RERANK = False

# ---------------- Arabic utils ----------------
AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"٠١٢٣٤٥٦٧٨٩")}
IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"۰۱۲۳۴۵۶۷۸۹")}
SENT_SPLIT_RE = re.compile(r'(?<=[\.\!\؟\?،；;]|[\n—])\s+')

def ar_normalize(s: str) -> str:
    if not s: return ""
    s = s.replace('\u0640','')  # tatweel
    s = AR_DIAC.sub('', s)
    s = (s.replace('أ','ا').replace('إ','ا').replace('آ','ا')
           .replace('ى','ي').replace('ة','ه'))
    s = s.translate(AR_NUMS).translate(IR_NUMS)
    s = s.replace('،', ',').replace('٫','.')
    s = ' '.join(s.split())
    return s

def sent_split(s: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(s) if p and p.strip()]
    return parts if parts else ([s.strip()] if s.strip() else [])

# ---------------- data model ----------------
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

# ---------------- Retriever (e5-base + TF-IDF hybrid) ----------------
class HybridRetriever:
    def __init__(self, artifact_dir: str = ".artifact",
                 model_name: str = "intfloat/multilingual-e5-base"):
        self.artifacts = Path(artifact_dir); self.artifacts.mkdir(parents=True, exist_ok=True)
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
        return self.model.encode(tnorm, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

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
                        if i <= 10: print(f"⚠️ JSON error at line {i}: {e}")
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
                id=cid, text_display=(disp or embt).strip(),
                text_embed=embt, source=str(j.get("source","Data_pdf.pdf")), page=page
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
        if not self.chunks: raise ValueError("No chunks loaded")
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
                            d = self.emb.shape[1]; self.faiss = faiss.IndexFlatIP(d); self.faiss.add(self.emb.astype("float32"))
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
                self.faiss = faiss.IndexFlatIP(d); self.faiss.add(self.emb.astype("float32"))
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
            self.tf_word = TfidfVectorizer(analyzer='word', ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b", min_df=1)
            self.word_mat = self.tf_word.fit_transform(texts)

        # save
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

    def semantic_search(self, query: str, top_k: int = 10, pre_candidates: int = 80) -> List[Tuple[int, float]]:
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
        fused = []
        for s, i in zip(dS, dI):
            sc = float(s)*0.55
            if cS is not None and i < len(cS): sc += float(cS[i])*0.25
            if wS is not None and i < len(wS): sc += float(wS[i])*0.20
            fused.append((sc, int(i)))
        fused.sort(key=lambda x: -x[0])
        return fused[:max(pre_candidates, top_k)]

    def rerank(self, query: str, candidates: List[int], max_keep: int = 10) -> List[int]:
        if not HAS_RERANK or not candidates:
            return candidates[:max_keep]
        ce = None
        last_err = None
        for name in CE_MODEL_CANDIDATES:
            try:
                print(f"🔁 Loading reranker: {name}")
                ce = CrossEncoder(name)
                break
            except Exception as e:
                last_err = e; ce = None
                continue
        if ce is None:
            print(f"⚠️ Reranker unavailable ({last_err}); continuing without.")
            return candidates[:max_keep]

        pairs = [(ar_normalize(query), self.chunks[i].text_display) for i in candidates]
        scores = ce.predict(pairs)
        order = np.argsort(-scores)
        reranked = [candidates[i] for i in order[:max_keep]]
        return reranked

    def sentence_pick(self, question: str, text: str) -> List[str]:
        qnorm = set([w for w in ar_normalize(question).split() if len(w)>=3])
        out = []
        for s in sent_split(text):
            sn = ar_normalize(s)
            has_overlap = len(qnorm & set(sn.split())) > 0
            has_numbers = bool(re.search(r'\d', sn))
            has_time_hint = any(t in sn for t in ["من","الى","إلى","حتى",":","."])
            if has_overlap or has_numbers or has_time_hint:
                out.append(s.strip())
        return out[:6] if out else sent_split(text)[:2]

    def build_context(self, question: str, max_chars: int = 2200, top_k: int = 10, use_rerank: bool = False) -> Tuple[str, List[Dict]]:
        pre = self.semantic_search(question, top_k=top_k, pre_candidates=80)
        cand = [i for i,_ in pre]
        if use_rerank:
            cand = self.rerank(question, cand, max_keep=top_k)
        else:
            cand = cand[:top_k]

        parts, meta, total = [], [], 0
        for rank, idx in enumerate(cand, 1):
            c = self.chunks[idx]
            head = f"[المصدر {rank}: {c.source} - ص{c.page}]"
            picked = self.sentence_pick(question, c.text_display)
            body = " ".join(picked)
            need = len(head)+1+len(body)+2
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

# ---------------- LLM wrapper ----------------
class RAGPipeline:
    def __init__(self, kb_path: str, artifact_dir: str = ".artifact",
                 use_cache: bool = True, llm_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 use_rerank: bool = False, top_k: int = 10, max_chars: int = 2200):
        print("🚀 Initializing Optimized Arabic RAG System…")
        print("📚 Setting up hybrid retriever…")
        self.use_rerank = use_rerank; self.top_k = top_k; self.max_chars = max_chars

        self.ret = HybridRetriever(artifact_dir=artifact_dir, model_name="intfloat/multilingual-e5-base")
        rows = self.ret.load_jsonl(kb_path)
        if not rows:
            raise ValueError("No data loaded from JSONL file")
        self.ret.prepare_chunks(rows)
        self.ret.build_or_load(kb_path, use_cache=use_cache)

        print("🧠 Loading LLM…")
        self.tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
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
            print("ℹ️ compile skipped:", e)
        print(f"📊 Model on: {self.model.device}")
        print("✅ RAG System initialized")

    def _prompt(self, question: str, context: str) -> str:
        return (
            "أنت مساعد عربي دقيق. أجب فقط من النصوص الواردة في (السياق) أدناه. "
            "ضع إشارة مصدر بعد كل حقيقة بالشكل [المصدر n]. "
            "إن لم تجد الإجابة في السياق فقل: \"غير متوفر في السياق\" دون اختلاق.\n\n"
            f"السياق:\n{context}\n\n"
            f"السؤال: {question}\n\n"
            "الإجابة:"
        )

    def generate(self, question: str, max_new_tokens: int = 320, temperature: float = 0.2) -> Dict[str, Any]:
        t0 = time.time()
        context, meta = self.ret.build_context(
            question, max_chars=self.max_chars, top_k=self.top_k, use_rerank=self.use_rerank
        )
        if not context:
            return {"answer":"عذراً، لا يوجد سياق ذي صلة.","context":"","meta":meta,"time":time.time()-t0,"confidence":0.0}

        prompt = self._prompt(question, context)
        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1800, padding=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature>0,
                top_p=0.85,
                repetition_penalty=1.12,
                no_repeat_ngram_size=3,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                use_cache=True
            )
        txt = self.tok.decode(out[0], skip_special_tokens=True)
        ans = txt[len(prompt):].strip()
        return {"answer": ans, "context": context, "meta": meta, "time": time.time()-t0, "confidence": 0.8}

    def chat(self):
        print("\n🤖 جاهز. اكتب سؤالك (أو 'خروج').\n")
        while True:
            q = input("🙋 سؤالك: ").strip()
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
    ap.add_argument("--kb", type=str, default=None, help="Path to KB jsonl")
    ap.add_argument("--artifact", type=str, default=".artifact", help="Artifacts dir")
    ap.add_argument("--no-cache", action="store_true", help="Disable retriever cache")
    ap.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranker")
    ap.add_argument("--top-k", type=int, default=10, help="Top K chunks for context")
    ap.add_argument("--max-chars", type=int, default=2200, help="Context character budget")
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="LLM, e.g., 'Qwen/Qwen2.5-7B-Instruct' or 'sambanovasystems/SambaLingo-Arabic-Chat'")
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

        # Quick smoke test
        print("\n🧪 Benchmark…")
        tests = [
            "ما هي سياسات التوظيف في مؤسسة النيزك؟",
            "كيف يمكن التقدم للوظائف؟",
            "ما هي المستندات المطلوبة للموظف الجديد؟",
        ]
        for q in tests:
            print(f"\n• {q}")
            r = rag.generate(q)
            print(f"⏱ {r['time']:.2f}s | 🤖 {r['answer'][:220]}…")

        # Start chat (comment out in notebooks if undesired)
        print("\n💬 Starting chat…")
        # rag.chat()

    except Exception as e:
        print(f"❌ خطأ في النظام: {e}")
        raise

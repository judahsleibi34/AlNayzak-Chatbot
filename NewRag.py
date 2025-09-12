# ==================== PATCH FOR BITSANDBYTES METADATA ISSUE ====================
import importlib.metadata
_original_version = importlib.metadata.version
def patched_version(package_name: str) -> str:
    if package_name == "bitsandbytes":
        return "0.47.0"  # set to your installed version
    return _original_version(package_name)
importlib.metadata.version = patched_version
# ==============================================================================

import os, re, json, time, pickle, warnings, hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set
from pathlib import Path

import numpy as np
warnings.filterwarnings("ignore")

# ---------------- Optional deps ----------------
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
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# ======================= Arabic utilities =======================
AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"٠١٢٣٤٥٦٧٨٩")}
IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"۰۱۲۳۴۵۶۷۸۹")}
SENT_SPLIT_RE = re.compile(r'(?<=[\.\!\؟\?،؛]|[\n—])\s+')

def ar_normalize(s: str) -> str:
    if not s: return ""
    s = s.replace('\u0640','')                       # tatweel
    s = AR_DIAC.sub('', s)                           # diacritics
    s = (s.replace('أ','ا').replace('إ','ا').replace('آ','ا')
           .replace('ى','ي').replace('ة','ه'))       # unify alif/ya/ta marbuta
    s = s.translate(AR_NUMS).translate(IR_NUMS)      # Arabic/Indic digits -> ASCII
    s = s.replace('،', ',').replace('٫','.')
    s = ' '.join(s.split())
    return s

def rtl_wrap(t: str) -> str:
    return '\u202B' + t + '\u202C'

def sent_split(s: str) -> List[str]:
    if not s: return []
    parts = [p.strip() for p in SENT_SPLIT_RE.split(s) if p and p.strip()]
    out = []
    for p in parts:
        pn = ar_normalize(p)
        if len(pn) < 6:  # drop tiny
            continue
        letters = sum(ch.isalpha() for ch in pn)
        total = len(pn.replace(" ", ""))
        if total > 0 and letters/total < 0.5:  # drop noisy OCR-ish lines
            continue
        out.append(p)
    return out if out else ([s.strip()] if s.strip() else [])

# ======================= Data model =======================
@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    id: str
    text_display: str
    text_embed: str
    language: str
    source: str
    page: int
    chunk_no: int
    embedding: Optional[np.ndarray] = None

# ======================= Hybrid Arabic Retriever =======================
class HybridArabicRetriever:
    """
    - Dense: SentenceTransformer + (FAISS IP) or NumPy fallback
    - Sparse: TF-IDF (char 2–5 + word 1–2), automatically disabled if sklearn not available
    - Index persistence: embeddings.npy, faiss.index, tfidf pickles + meta.json in artifact_dir
    - Input format: JSONL rows containing at least: id, text_display or text_embed, language, source, page, chunk_no
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        artifact_dir: str = "./artifacts"
    ):
        self.model_name = model_name
        print("🔄 Loading embedding model…")
        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 256
        print(f"✅ Embedding model loaded in {time.time()-t0:.2f}s")

        self.chunks: List[DocumentChunk] = []
        self.emb: Optional[np.ndarray] = None
        self.faiss_index = None

        self.tf_char = None; self.tf_word = None
        self.char_mat = None; self.word_mat = None

        self.artifacts = Path(artifact_dir)
        self.artifacts.mkdir(parents=True, exist_ok=True)

    # ---------- dataset I/O ----------
    def load_jsonl(self, path: str) -> List[Dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    if i <= 10:
                        print(f"⚠️ JSON error at line {i}: {e}")
        print(f"✅ Loaded {len(data)} raw rows from {path}")
        return data

    def load_documents(self, rows: List[Dict]) -> None:
        """Use provided chunk fields; normalize text_embed if missing."""
        self.chunks.clear()
        for i, item in enumerate(rows):
            disp = item.get("text_display") or item.get("text") or ""
            embtxt = item.get("text_embed") or disp
            embtxt = ar_normalize(embtxt)
            if not embtxt: continue
            self.chunks.append(
                DocumentChunk(
                    id=str(item.get("id", f"row{i}")),
                    text_display=(disp or embtxt).strip(),
                    text_embed=embtxt,
                    language=item.get("language","ar"),
                    source=item.get("source","unknown"),
                    page=int(item.get("page",0)),
                    chunk_no=int(item.get("chunk_no",0))
                )
            )
        print(f"✅ Prepared {len(self.chunks)} chunks")

    # ---------- persistence helpers ----------
    def _dataset_hash(self) -> str:
        h = hashlib.sha256()
        for c in self.chunks[:200]:
            h.update(c.id.encode("utf-8"))
            h.update(c.text_embed[:256].encode("utf-8"))
        return h.hexdigest()[:16]

    def _p(self, name: str) -> Path:
        return self.artifacts / name

    def build_or_load(self, use_cache: bool = True, batch_size: int = 128) -> None:
        if not self.chunks:
            raise ValueError("No documents loaded.")
        ds_hash = self._dataset_hash()
        meta_path = self._p("meta.json")
        emb_path  = self._p("embeddings.npy")
        faiss_path= self._p("faiss.index")
        tfc_path  = self._p("tf_char.pkl")
        tfw_path  = self._p("tf_word.pkl")
        cm_path   = self._p("char_mat.pkl")
        wm_path   = self._p("word_mat.pkl")

        # try load
        if use_cache and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                cond = (meta.get("ds_hash")==ds_hash and
                        meta.get("model_name")==self.model_name and
                        meta.get("n_chunks")==len(self.chunks))
                if cond and emb_path.exists():
                    print("📦 Loading dense artifacts…")
                    self.emb = np.load(str(emb_path), mmap_mode="r")
                    if faiss is not None and faiss_path.exists():
                        self.faiss_index = faiss.read_index(str(faiss_path))
                    elif faiss is not None:
                        d = self.emb.shape[1]
                        self.faiss_index = faiss.IndexFlatIP(d); self.faiss_index.add(self.emb.astype("float32"))
                    if TfidfVectorizer and joblib and tfc_path.exists() and tfw_path.exists() and cm_path.exists() and wm_path.exists():
                        print("📦 Loading TF-IDF artifacts…")
                        self.tf_char = joblib.load(str(tfc_path))
                        self.tf_word = joblib.load(str(tfw_path))
                        self.char_mat = joblib.load(str(cm_path))
                        self.word_mat = joblib.load(str(wm_path))
                    print("✅ Index loaded from artifacts")
                    return
            except Exception as e:
                print(f"⚠️ Artifact load failed, rebuilding… ({e})")

        # build
        print("🔄 Building embeddings + indexes…")
        t0 = time.time()
        texts = [c.text_embed for c in self.chunks]
        self.emb = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

        if faiss is not None:
            d = self.emb.shape[1]
            if len(self.emb) < 10_000:
                self.faiss_index = faiss.IndexFlatIP(d)
                self.faiss_index.add(self.emb.astype("float32"))
            else:
                nlist = min(100, len(self.emb) // 100)
                quantizer = faiss.IndexFlatIP(d)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                self.faiss_index.train(self.emb.astype("float32"))
                self.faiss_index.add(self.emb.astype("float32"))
        else:
            self.faiss_index = None  # fallback to NumPy

        if TfidfVectorizer:
            print("🧪 Building TF-IDF (char+word)…")
            self.tf_char = TfidfVectorizer(analyzer="char", ngram_range=(2,5), min_df=1)
            self.char_mat = self.tf_char.fit_transform(texts)
            self.tf_word = TfidfVectorizer(analyzer="word", ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b", min_df=1)
            self.word_mat = self.tf_word.fit_transform(texts)
        else:
            print("ℹ️ scikit-learn not available — sparse channel disabled")

        # persist
        print("💾 Saving artifacts…")
        try:
            np.save(str(emb_path), self.emb)
            if faiss is not None and self.faiss_index is not None:
                faiss.write_index(self.faiss_index, str(faiss_path))
            if TfidfVectorizer and joblib:
                joblib.dump(self.tf_char, str(tfc_path)); joblib.dump(self.tf_word, str(tfw_path))
                joblib.dump(self.char_mat, str(cm_path)); joblib.dump(self.word_mat, str(wm_path))
            meta = {"ds_hash": ds_hash, "model_name": self.model_name, "n_chunks": len(self.chunks)}
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✅ Artifacts saved to {self.artifacts}")
        except Exception as e:
            print(f"⚠️ Failed to save artifacts: {e}")

        print(f"✅ Index build complete in {time.time()-t0:.2f}s")

    # ---------- retrieval ----------
    def _dense_search(self, q: str, topk: int = 20):
        qn = ar_normalize(q)
        qv = self.model.encode([qn], convert_to_numpy=True, normalize_embeddings=True)
        if self.faiss_index is not None:
            D, I = self.faiss_index.search(qv.astype("float32"), topk)
            return D[0], I[0]
        # NumPy fallback
        sims = self.emb @ qv[0]
        idxs = np.argsort(-sims)[:topk]
        return sims[idxs], idxs

    def _sparse_scores(self, q: str):
        if self.tf_char is None or self.tf_word is None:
            return None, None
        qc = self.tf_char.transform([ar_normalize(q)])
        qw = self.tf_word.transform([ar_normalize(q)])
        c_scores = (self.char_mat @ qc.T).toarray().ravel()
        w_scores = (self.word_mat @ qw.T).toarray().ravel()
        return c_scores, w_scores

    def _combine(self, dense_scores, dense_idx, c_scores, w_scores,
                 w_dense=0.65, w_char=0.20, w_word=0.15, topk=10):
        out = []
        for s, i in zip(dense_scores, dense_idx):
            sc = float(s) * w_dense
            if c_scores is not None and len(c_scores) > int(i): sc += float(c_scores[i]) * w_char
            if w_scores is not None and len(w_scores) > int(i): sc += float(w_scores[i]) * w_word
            out.append((sc, int(i)))
        out.sort(key=lambda x: -x[0])
        return out[:topk]

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        dS, dI = self._dense_search(query, topk=max(top_k*4, 20))
        cS, wS = self._sparse_scores(query)
        combined = self._combine(dS, dI, cS, wS, topk=top_k)
        results = []
        for sc, idx in combined:
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(sc)))
        return results

    def get_relevant_context(self, query: str, max_context_length: int = 1500, top_k: int = 5) -> Tuple[str, List[Dict]]:
        hits = self.semantic_search(query, top_k=top_k)
        if not hits: return "", []
        parts, meta, total = [], [], 0
        for rank, (chunk, score) in enumerate(hits, 1):
            head = f"[المصدر {rank}: {chunk.source} - ص{chunk.page}]"
            body = chunk.text_display.strip()
            needed = len(head)+1+len(body)+2
            if total + needed <= max_context_length:
                parts.append(f"{head}\n{body}")
                total += needed
                meta.append({"source": chunk.source, "page": chunk.page, "score": score, "chunk_id": chunk.id})
            else:
                remain = max_context_length - total - len(head) - 2
                if remain > 120:
                    trunc = body[:remain-20] + " …"
                    parts.append(f"{head}\n{trunc}")
                    meta.append({"source": chunk.source, "page": chunk.page, "score": score, "chunk_id": chunk.id, "truncated": True})
                break
        return "\n\n".join(parts), meta

# ======================= Optimized SambaLingo RAG =======================
class OptimizedSambaLingoRAG:
    """
    Fast, Arabic-first RAG:
      - Hybrid retriever (dense + TF-IDF) with caching
      - SambaLingo Arabic chat model (4-bit NF4 by default)
      - Short, focused prompt (keeps latency low)
    """

    def __init__(self, jsonl_filepath: str, artifact_dir: str = "./artifacts", use_cache: bool = True):
        print("🚀 Initializing Optimized Arabic RAG System…")
        t0 = time.time()

        print("📚 Setting up hybrid retriever…")
        self.retriever = HybridArabicRetriever(artifact_dir=artifact_dir)
        rows = self._load_jsonl_fast(jsonl_filepath)
        if not rows:
            raise ValueError("No data loaded from JSONL file")
        self.retriever.load_documents(rows)
        self.retriever.build_or_load(use_cache=use_cache, batch_size=128)

        print("🧠 Loading SambaLingo Arabic model…")
        self.tokenizer, self.model = self._setup_sambanova_model()

        print(f"✅ RAG System initialized in {time.time()-t0:.2f}s")

    # ---------- data loader ----------
    def _load_jsonl_fast(self, filepath: str) -> List[Dict]:
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"📖 Processing {len(lines)} lines…")
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line: continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    if i <= 10:
                        print(f"⚠️ Error parsing line {i}: {e}")
                if i % 5000 == 0:
                    print(f"  Processed {i}/{len(lines)}")
            print(f"✅ Loaded {len(data)} valid chunks")
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
        return data

    # ---------- model ----------
    def _setup_sambanova_model(self):
        model_name = "sambanovasystems/SambaLingo-Arabic-Chat"
        # quant config (good on RTX 4060+)
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True
            )
        except Exception:
            bnb_config = None

        tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        kwargs = dict(
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True
        )
        if bnb_config is not None:
            kwargs["quantization_config"] = bnb_config
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.eval()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        try:
            if torch.__version__ >= "2.0":
                model = torch.compile(model, mode="reduce-overhead")
                print("🚀 Model compiled for faster inference")
        except Exception as e:
            print(f"ℹ️ Compile skipped: {e}")

        print(f"📊 Model on: {model.device}")
        return tok, model

    # ---------- prompt ----------
    def _make_prompt(self, question: str, context: str) -> str:
        return f"""<|start_header_id|>system<|end_header_id|>
أجب على السؤال باللغة العربية اعتماداً على السياق التالي. كن مختصراً ودقيقاً، وإن غاب الدليل فاذكر ذلك صراحةً.<|end_header_id|>

<|start_header_id|>user<|end_header_id|>
السياق:
{context}

السؤال: {question}<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>"""

    # ---------- generate ----------
    def generate_answer(self, question: str, max_new_tokens: int = 256, temperature: float = 0.3) -> Dict:
        t0 = time.time()
        print("🔍 Retrieving context…")
        context, meta = self.retriever.get_relevant_context(question, max_context_length=1200, top_k=5)
        if not context.strip():
            return {
                "answer": "عذراً، لم أجد معلومات ذات صلة في قاعدة المعرفة.",
                "context": "",
                "metadata": [],
                "confidence": 0.0,
                "time_taken": time.time()-t0
            }

        prompt = self._make_prompt(question, context)
        print("🧠 Generating answer…")
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1500,
                padding=False
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.85,
                    repetition_penalty=1.08,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                    use_cache=True
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # extract assistant tail
            if "<|start_header_id|>assistant<|end_header_id|>" in text:
                answer = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            else:
                answer = text[len(prompt):].strip()
            answer = answer.replace("<|end_header_id|>", "").strip()

            return {
                "answer": answer,
                "context": context,
                "metadata": meta,
                "confidence": 0.8,   # simple heuristic baseline
                "time_taken": time.time()-t0
            }
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return {
                "answer": "حدث خطأ أثناء توليد الإجابة.",
                "context": context,
                "metadata": meta,
                "confidence": 0.0,
                "time_taken": time.time()-t0
            }

    # ---------- helpers ----------
    def ask(self, question: str) -> str:
        return self.generate_answer(question)["answer"]

    def benchmark(self, questions: List[str]) -> None:
        print("\n🧪 Benchmark…")
        total = 0.0
        for i, q in enumerate(questions, 1):
            print(f"\n{i}. {q}")
            r = self.generate_answer(q)
            print(f"⏱ {r['time_taken']:.2f}s | 🤖 {r['answer'][:200]}{'…' if len(r['answer'])>200 else ''}")
            total += r["time_taken"]
        n = max(len(questions), 1)
        print(f"\n📊 Avg time: {total/n:.2f}s | Total: {total:.2f}s")

    def chat_loop(self):
        print("\n" + "="*60)
        print("🤖 مرحباً! أنا مساعدك الذكي المحسّن (SambaLingo + Hybrid RAG)")
        print("💡 اكتب 'خروج' للانتهاء")
        print("="*60 + "\n")
        while True:
            try:
                q = input("🙋 سؤالك: ").strip()
                if not q: continue
                if q.lower() in ["خروج", "exit", "quit"]:
                    print("🙏 شكراً!")
                    break
                r = self.generate_answer(q)
                print(f"\n🤖 {r['answer']}\n")
            except KeyboardInterrupt:
                print("\n👋 تم الإيقاف"); break
            except Exception as e:
                print(f"❌ خطأ: {e}")

    def cleanup(self):
        print("🧹 تنظيف الذاكرة…")
        if hasattr(self, "model"): del self.model
        if hasattr(self, "tokenizer"): del self.tokenizer
        if hasattr(self, "retriever"): del self.retriever
        torch.cuda.empty_cache()
        print("✅ تم")

# ======================= Main =======================
if __name__ == "__main__":
    try:
        kb_path = os.environ.get("AR_KB_JSONL", "arabic_chatbot_knowledge.jsonl")
        rag = OptimizedSambaLingoRAG(kb_path, artifact_dir="./artifacts", use_cache=True)

        # quick bench
        test_qs = [
            "ما هي سياسات التوظيف في مؤسسة النيزك؟",
            "كيف يمكن التقدم للوظائف؟",
            "ما هي المستندات المطلوبة للموظف الجديد؟"
        ]
        rag.benchmark(test_qs)

        # debug: show top hits example
        print("\n🔎 Debug search: 'التوظيف'")
        hits = rag.retriever.semantic_search("التوظيف", top_k=3)
        for i, (chunk, score) in enumerate(hits, 1):
            print(f"  {i}. score={score:.3f} | src={chunk.source} | p={chunk.page} | text={chunk.text_display[:80]}…")

        # start chat
        print("\n💬 Starting chat…")
        rag.chat_loop()

    except KeyboardInterrupt:
        print("\n👋 تم إيقاف النظام")
    except Exception as e:
        print(f"❌ خطأ في النظام: {e}")
        import traceback; traceback.print_exc()
    finally:
        if 'rag' in locals():
            rag.cleanup()

# -*- coding: utf-8 -*-
"""
NewRag_Qwen_extract.py

Arabic-first RAG with extractive, citation-anchored answers using Qwen2.5-7B-Instruct.

Key fixes vs older versions
---------------------------
- Safe default KB path (prefers Data_pdf_clean_chunks.jsonl if present)
- Robust JSONL loader + progress logs
- Hybrid retrieval (dense SBERT + TF‑IDF char/word) with artifact caching
- Deterministic generation (no sampling) to cut hallucinations
- Extractive prompt: copy sentences verbatim, every bullet ends with [المصدر n]
- Output sanitation for Arabic-only text and citation normalization
- Fallback: if the LLM fails to cite, assemble bullets directly from retrieved chunks
- Bug fix: ensure list indices are ints when indexing chunks (no float indices)

Tested on Google Colab T4 with 4‑bit quantization.
"""

import os
import re
import json
import argparse
import logging
import pickle
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np

# Optional deps
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib
except Exception:
    TfidfVectorizer = None
    joblib = None

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger("NewRag")

# ---------------- Arabic utils ----------------
AR_DIAC = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
AR_TATWEEL = "\u0640"

AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate("٠١٢٣٤٥٦٧٨٩")}
IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate("۰۱۲۳۴۵۶۷۸۹")}

SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\؟\?،]|[\n])\s+")


def ar_normalize(s: str) -> str:
    if not s:
        return ""
    s = s.replace(AR_TATWEEL, "")
    s = AR_DIAC.sub('', s)
    s = (s.replace('أ','ا').replace('إ','ا').replace('آ','ا')
           .replace('ى','ي').replace('ة','ه'))
    s = s.translate(AR_NUMS).translate(IR_NUMS)
    s = s.replace('،', ',').replace('٫','.')
    s = ' '.join(s.split())
    return s


def sent_split(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p and p.strip()]
    out = []
    for p in parts:
        pn = ar_normalize(p)
        if len(pn) < 6:
            continue
        letters = sum(ch.isalpha() for ch in pn)
        total = len(pn.replace(" ", ""))
        if total == 0 or letters/total < 0.5:
            continue
        out.append(p)
    return out if out else ([text.strip()] if text.strip() else [])


# ---------------- Data structures ----------------
@dataclass
class Chunk:
    id: int
    page: int
    source: str
    text_display: str
    text_embed: str


# ---------------- IO ----------------

def load_jsonl_fast(path: str) -> List[Dict]:
    if not os.path.exists(path):
        LOG.error("File not found: %s", path)
        return []
    data: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    LOG.info("Processing %d lines from %s", len(lines), path)
    for ln, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            if ln <= 10:
                LOG.warning("JSON error at line %d: %s", ln, e)
    LOG.info("Loaded %d rows", len(data))
    return data


def to_chunks(rows: List[Dict]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for i, r in enumerate(rows):
        td = r.get('text_display') or r.get('text') or r.get('content') or ''
        te = r.get('text_embed') or ar_normalize(td)
        if not te.strip():
            continue
        try:
            cid = int(r.get('id', i))
        except Exception:
            cid = i
        page = int(r.get('page', -1))
        src = str(r.get('source', 'Data_pdf.pdf'))
        chunks.append(Chunk(id=cid, page=page, source=src, text_display=td, text_embed=te))
    LOG.info("Prepared %d chunks", len(chunks))
    return chunks


# ---------------- Hybrid Retriever with persistence ----------------
class HybridRetriever:
    def __init__(self, chunks: List[Chunk], artifact_dir: str = ".artifact"):
        self.chunks = chunks
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

        self.sbert_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        LOG.info("Loading SBERT embeddings model: %s", self.sbert_name)
        self.sbert = SentenceTransformer(self.sbert_name)
        self.sbert.max_seq_length = 256

        self.emb: Optional[np.ndarray] = None
        self.faiss_index = None
        self.tf_char = None
        self.tf_word = None
        self.char_mat = None
        self.word_mat = None

    # ---------- Artifact paths
    def _p(self, name: str) -> str:
        return os.path.join(self.artifact_dir, name)

    # ---------- Build / Load
    def load_artifacts(self) -> bool:
        try:
            emb_p = self._p("embeddings.npy")
            if not os.path.exists(emb_p):
                return False
            self.emb = np.load(emb_p)
            if faiss is not None and os.path.exists(self._p("faiss.index")):
                self.faiss_index = faiss.read_index(self._p("faiss.index"))
            if joblib and TfidfVectorizer is not None:
                tpc, tpw = self._p("tf_char.pkl"), self._p("tf_word.pkl")
                cm, wm = self._p("char_mat.pkl"), self._p("word_mat.pkl")
                if all(os.path.exists(p) for p in [tpc, tpw, cm, wm]):
                    self.tf_char = joblib.load(tpc)
                    self.tf_word = joblib.load(tpw)
                    self.char_mat = joblib.load(cm)
                    self.word_mat = joblib.load(wm)
            LOG.info("Index loaded from artifacts")
            return True
        except Exception as e:
            LOG.warning("Failed to load artifacts: %s", e)
            return False

    def save_artifacts(self):
        try:
            if self.emb is not None:
                np.save(self._p("embeddings.npy"), self.emb)
            if faiss is not None and self.faiss_index is not None:
                faiss.write_index(self.faiss_index, self._p("faiss.index"))
            if joblib and TfidfVectorizer is not None and self.tf_char is not None:
                joblib.dump(self.tf_char, self._p("tf_char.pkl"))
                joblib.dump(self.tf_word, self._p("tf_word.pkl"))
                joblib.dump(self.char_mat, self._p("char_mat.pkl"))
                joblib.dump(self.word_mat, self._p("word_mat.pkl"))
            LOG.info("Artifacts saved to %s", self.artifact_dir)
        except Exception as e:
            LOG.warning("Failed to save artifacts: %s", e)

    def build(self):
        LOG.info("Building embeddings + indexes…")
        texts = [c.text_embed for c in self.chunks]
        # Encode in batches with a progress-ish print
        batch = 128
        all_vecs = []
        for i in range(0, len(texts), batch):
            vecs = self.sbert.encode(texts[i:i+batch], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            all_vecs.append(vecs)
        self.emb = np.vstack(all_vecs) if all_vecs else np.zeros((0, 384), dtype=np.float32)

        if faiss is not None:
            d = self.emb.shape[1]
            self.faiss_index = faiss.IndexFlatIP(d)
            self.faiss_index.add(self.emb.astype('float32'))
        else:
            self.faiss_index = None

        if TfidfVectorizer is not None:
            self.tf_char = TfidfVectorizer(analyzer='char', ngram_range=(2,5), min_df=1)
            self.char_mat = self.tf_char.fit_transform(texts)
            self.tf_word = TfidfVectorizer(analyzer='word', ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b", min_df=1)
            self.word_mat = self.tf_word.fit_transform(texts)
        else:
            LOG.warning("scikit-learn not available; skipping TF-IDF")

        self.save_artifacts()

    # ---------- Retrieval
    def _dense(self, qn: str, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        qv = self.sbert.encode([qn], convert_to_numpy=True, normalize_embeddings=True)
        if self.faiss_index is not None:
            D, I = self.faiss_index.search(qv.astype('float32'), max(topk, 60))
            return D[0], I[0]
        # fallback numpy
        sims = self.emb @ qv[0]
        idxs = np.argsort(-sims)[:max(topk, 60)]
        return sims[idxs], idxs

    def _sparse(self, qn: str):
        if self.tf_char is None or self.tf_word is None:
            return None, None
        qc = self.tf_char.transform([qn])
        qw = self.tf_word.transform([qn])
        c_scores = (self.char_mat @ qc.T).toarray().ravel()
        w_scores = (self.word_mat @ qw.T).toarray().ravel()
        return c_scores, w_scores

    def _combine(self, dS, dI, cS, wS, w_dense=0.65, w_char=0.2, w_word=0.15, topk=20):
        out: List[Tuple[float,int]] = []
        for s, i in zip(dS, dI):
            i = int(i)
            sc = float(s) * w_dense
            if cS is not None and len(cS) > i:
                sc += float(cS[i]) * w_char
            if wS is not None and len(wS) > i:
                sc += float(wS[i]) * w_word
            out.append((sc, i))
        out.sort(key=lambda x: -x[0])
        return out[:topk]

    def sentence_pick(self, query: str, para: str) -> List[str]:
        qn = set([w for w in ar_normalize(query).split() if len(w) >= 3])
        best: List[Tuple[float,str]] = []
        for s in sent_split(para):
            sn = ar_normalize(s)
            if not sn:
                continue
            overlap = len(qn & set(sn.split()))
            score = overlap
            if re.search(r"\d", sn):
                score += 0.3
            if any(tok in sn for tok in ["من","الى","حتى",":","."]):
                score += 0.2
            best.append((score, s.strip()))
        best.sort(key=lambda x: -x[0])
        return [b for _, b in best[:2]]

    def semantic_search(self, query: str, top_k: int = 8, pre_candidates: int = 60) -> List[Tuple[int, float]]:
        qn = ar_normalize(query)
        dS, dI = self._dense(qn, pre_candidates)
        cS, wS = self._sparse(qn)
        comb = self._combine(dS, dI, cS, wS, topk=top_k)
        return comb  # list of (score, idx)

    def build_context(self, query: str, top_k: int = 8, max_chars: int = 2200) -> Tuple[str, List[Dict]]:
        hits = self.semantic_search(query, top_k=top_k)
        if not hits:
            return "", []
        parts: List[str] = []
        meta: List[Dict] = []
        total = 0
        for rank, (sc, idx) in enumerate(hits, start=1):
            idx = int(idx)  # ensure integer index
            c = self.chunks[idx]
            head = f"[المصدر {rank}: {c.source} - ص{c.page}]"
            body = c.text_display.strip()
            need = len(head) + 1 + len(body) + 2
            if total + need <= max_chars:
                parts.append(head)
                parts.append(body)
                total += need
                meta.append({"rank": rank, "source": c.source, "page": c.page, "id": c.id, "idx": idx, "score": float(sc)})
            else:
                remain = max_chars - total - len(head) - 1
                if remain > 150:
                    parts.append(head)
                    parts.append(body[:remain-10] + "…")
                    total = max_chars
                    meta.append({"rank": rank, "source": c.source, "page": c.page, "id": c.id, "idx": idx, "score": float(sc), "truncated": True})
                break
        context = "\n\n".join(parts)
        return context, meta


# ---------------- Output sanitizers ----------------

def sanitize_arabic_output(text: str) -> str:
    # Drop role echoes
    text = re.sub(r'^\s*(user|assistant)\s*:?[\s\n]+', '', text, flags=re.I|re.M)
    # Remove CJK
    text = re.sub(r'[\u3400-\u4DBF\u4E00-\u9FFF]+', '', text)
    # Remove single Latin letters between Arabic letters (e.g., "الزk")
    text = re.sub(r'(?<=[\u0600-\u06FF])[A-Za-z](?=[\u0600-\u06FF])', '', text)
    # Remove remaining Latin runs
    text = re.sub(r'[A-Za-z]+', '', text)
    # Normalize common typos
    text = (text
            .replace("المصدار", "المصدر")
            .replace("المصدـر", "المصدر")
            .replace("المصد~r", "المصدر"))
    # Collapse spaces/blank lines
    text = re.sub(r'[ \t]{2,}', ' ', text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def fix_citations(answer: str, meta: List[Dict]) -> str:
    valid = {m.get("rank") for m in meta if m.get("rank") is not None}
    valid = {int(v) for v in valid if isinstance(v, (int, np.integer)) or (isinstance(v, str) and v.isdigit())}
    if not valid:
        return answer
    lowest = min(valid)

    def repl(m):
        try:
            n = int(m.group(1))
            return f"[المصدر {n}]" if n in valid else f"[المصدر {lowest}]"
        except Exception:
            return f"[المصدر {lowest}]"

    return re.sub(r'\[المصدر\s+(\d+)\]', repl, answer)


# ---------------- LLM wrapper (Qwen2.5 default) ----------------
class QwenExtractive:
    def __init__(self, llm_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.llm_name = llm_name
        LOG.info("Loading LLM: %s", self.llm_name)

        # 4-bit quant for T4
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.tok = AutoTokenizer.from_pretrained(self.llm_name, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_name,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=True,
        )
        self.model.eval()
        try:
            if hasattr(torch, "compile"):
                self.model = torch.compile(self.model, mode="reduce-overhead")
                LOG.info("Model compiled for faster inference")
        except Exception as e:
            LOG.warning("torch.compile failed: %s", e)

    def _messages(self, question: str, context: str) -> List[Dict[str,str]]:
        system = (
            "أنت مساعد عربي يعتمد على RAG. "
            "أجب بالعربية الفصحى فقط. "
            "انسخ الجُمل ذات الصلة حرفيًا من (السياق) كما هي دون إعادة صياغة أو إضافة معلومات خارجية. "
            "ضع [المصدر n] في نهاية كل نقطة مستشهدة، حيث n هو رقم المصدر الظاهر في قسم السياق. "
            "إن لم تجد إجابة صريحة في السياق فقل: غير متوفر في السياق."
        )
        user = (
            f"السياق:\n{context}\n\n"
            f"السؤال: {question}\n\n"
            "أخرج نقاطًا موجزة (شرطات) منسوخة حرفيًا من السياق، كل نقطة بسطر وتنتهي بـ [المصدر n]."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def answer(self, question: str, context: str, meta: List[Dict], max_new_tokens: int = 300) -> str:
        messages = self._messages(question, context)
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=4096 - max_new_tokens)
        input_ids = enc.input_ids.to(self.model.device)
        attn_mask = enc.attention_mask.to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # deterministic
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.12,
                no_repeat_ngram_size=8,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                use_cache=True,
            )
        gen = out[0][input_ids.shape[-1]:]
        raw = self.tok.decode(gen, skip_special_tokens=True).strip()
        ans = sanitize_arabic_output(raw)
        ans = fix_citations(ans, meta)

        # Fallback: build bullets from retrieved sentences if citations missing
        if "[المصدر" not in ans:
            bullets: List[str] = []
            for m in meta[:4]:
                idx = int(m.get("idx", -1))
                if 0 <= idx < len(rag.ret.chunks):
                    picked = rag.ret.sentence_pick(question, rag.ret.chunks[idx].text_display)
                    if picked:
                        bullets.append(f"- {picked[0]} [المصدر {m['rank']}]")
            if bullets:
                ans = "\n".join(bullets)
        return ans


# ---------------- RAG Orchestrator ----------------
class NewRag:
    def __init__(self, kb_path: str, artifact_dir: str = ".artifact", force_build: bool = False, llm_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.kb_path = kb_path
        self.artifact_dir = artifact_dir
        LOG.info("Using KB: %s", self.kb_path)
        LOG.info("Using artifact dir: %s", self.artifact_dir)

        rows = load_jsonl_fast(self.kb_path)
        if not rows:
            raise ValueError("No data loaded from JSONL file")
        self.chunks = to_chunks(rows)

        self.ret = HybridRetriever(self.chunks, artifact_dir=self.artifact_dir)
        loaded = (not force_build) and self.ret.load_artifacts()
        if not loaded:
            self.ret.build()

        self.llm = QwenExtractive(llm_name)

    def generate(self, question: str, top_k: int = 12, max_chars: int = 3000, max_new_tokens: int = 300) -> str:
        context, meta = self.ret.build_context(question, top_k=top_k, max_chars=max_chars)
        if not context:
            return "غير متوفر في السياق."
        ans = self.llm.answer(question, context, meta, max_new_tokens=max_new_tokens)
        return ans


# ---------------- CLI ----------------
SANITY = [
    "ما هي سياسات التوظيف في مؤسسة النيزك؟",
    "كيف يمكن التقدم للوظائف؟",
    "ما هي المستندات المطلوبة للموظف الجديد؟",
]


def pick_default_kb() -> str:
    # Prefer cleaned chunks if present
    if os.path.exists("Data_pdf_clean_chunks.jsonl"):
        return "Data_pdf_clean_chunks.jsonl"
    # fallback legacy name
    if os.path.exists("arabic_chatbot_knowledge.jsonl"):
        return "arabic_chatbot_knowledge.jsonl"
    # last resort: try in current dir
    raise FileNotFoundError("Neither Data_pdf_clean_chunks.jsonl nor arabic_chatbot_knowledge.jsonl found.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=str, default=None, help="Path to knowledge JSONL")
    ap.add_argument("--artifacts", type=str, default=".artifact")
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--force-build", action="store_true", help="Rebuild indexes even if artifacts exist")
    ap.add_argument("--bench", action="store_true", help="Run a short benchmark and exit")
    ap.add_argument("--chat", action="store_true", help="Start an interactive chat loop")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--max-new", type=int, default=300)
    args = ap.parse_args()

    kb_path = args.kb or None
    if kb_path is None:
        try:
            kb_path = pick_default_kb()
        except Exception as e:
            LOG.error("%s", e)
            raise

    global rag
    rag = NewRag(kb_path, artifact_dir=args.artifacts, force_build=args.force_build, llm_name=args.llm)

    if args.bench:
        LOG.info("\n🧪 Benchmark…\n")
        for q in SANITY:
            t0 = time.time()
            a = rag.generate(q, top_k=args.top_k, max_chars=args.max_chars, max_new_tokens=args.max_new)
            dt = time.time() - t0
            print(f"• {q}\n⏱ {dt:.2f}s | 🤖 {a[:600]}\n")
        return

    if args.chat or not args.bench:
        print("\n💬 Starting chat… (Ctrl+C to exit)\n")
        try:
            while True:
                q = input("🙋 سؤالك: ").strip()
                if not q:
                    continue
                t0 = time.time()
                a = rag.generate(q, top_k=args.top_k, max_chars=args.max_chars, max_new_tokens=args.max_new)
                dt = time.time() - t0
                print(f"⏱ {dt:.2f}s | 🤖 {a}\n")
        except KeyboardInterrupt:
            print("\n👋 تم الإنهاء")


if __name__ == "__main__":
    main()

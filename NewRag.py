# -*- coding: utf-8 -*-
"""
Arabic-first EXTRACTIVE retriever (PDF-only, no fabrication) — generalized & RTL-clean.

Key properties
--------------
- Extractive-only: returns verbatim sentences from your chunks (no LLM paraphrase).
- PDF-locked: every answer includes page citations from the PDF you specify.
- Generalized: no domain-specific intent bias; scoring derives from the question itself.
- RTL-stable output: proper bidi handling; optional Arabic-Indic digits.
- Confidence gating: refuses when evidence is weak or mismatched (prevents wrong answers).
- Optional hierarchy/alias gating: constrain search to sections if you provide indices.
- Optional TF–IDF fusion with dense embeddings (if scikit-learn is installed).
- Index persistence (embeddings+FAISS+TF–IDF) for fast reloads.

Usage
-----
Build from scratch and save artifacts:
  python retrival_model.py --chunks Data_pdf_clean_chunks.jsonl --save-index .artifact --pdf-name Data_pdf.pdf

Load saved artifacts and run interactive QA:
  python retrival_model.py --chunks Data_pdf_clean_chunks.jsonl --load-index .artifact --pdf-name Data_pdf.pdf

Run the bundled 30-question sanity suite (prints each Q/A with pass flags):
  python retrival_model.py --chunks Data_pdf_clean_chunks.jsonl --sanity --pdf-name Data_pdf.pdf

Arabic formatting choices:
  python retrival_model.py --chunks ... --sanity --pdf-name Data_pdf.pdf --rtl force --digits arabic
"""

import os, re, sys, json, argparse, logging, hashlib, pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Set

import numpy as np

# -------- Optional deps
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

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("Please install: sentence-transformers (pip install sentence-transformers)")
    raise

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger(__name__)

# =========================
# Global toggles & limits
# =========================
STRICT_MODE = True                      # keep True: refuse when evidence is weak
MAX_SNIPPET_CHARS = 260
MAX_CITATIONS_DEFAULT = 3
TOPK_DENSE = 60
TOPK_FINAL = 12
WINDOW_CHARS = 48                       # keyword/number proximity
PDF_DEFAULT_NAME = "Data_pdf.pdf"

# =========================
# Arabic utils
# =========================
AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"٠١٢٣٤٥٦٧٨٩")}
IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"۰۱۲۳۴۵۶۷۸۹")}

def ar_normalize(s: str) -> str:
    """Normalization for retrieval/scoring (keeps ة intact)."""
    if not s: return ""
    s = s.replace('\u0640','')         # tatweel
    s = AR_DIAC.sub('', s)             # remove diacritics
    s = (s.replace('أ','ا')
           .replace('إ','ا')
           .replace('آ','ا')
           .replace('ى','ي'))
    s = s.translate(AR_NUMS).translate(IR_NUMS)   # Arabic/Indic -> ASCII digits
    s = s.replace('،', ',').replace('٫','.')
    s = ' '.join(s.split())
    return s

# ---------- sentence split (OCR-aware)
SENT_SPLIT_RE = re.compile(r'(?<=[\.\!\؟\?،]|[\n])\s+')

def sent_split(s: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(s or "") if p and p.strip()]
    out = []
    for p in parts:
        pn = ar_normalize(p)
        if len(pn) < 6:                      # drop tiny fragments
            continue
        letters = sum(ch.isalpha() for ch in pn)
        total = len(pn.replace(" ", ""))
        if total == 0 or letters/total < 0.5:
            continue
        out.append(p)
    return out if out else ([s.strip()] if (s and s.strip()) else [])

# =========================
# RTL formatting helpers
# =========================
BIDI_RLE = "\u202B"  # Right-to-Left Embedding
BIDI_PDF = "\u202C"  # Pop Directional Formatting
LRM      = "\u200E"  # Left-to-Right Mark

_ARABIC_DIGITS_MAP = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

def _strip_bidi_controls(s: str) -> str:
    return re.sub(r"[\u202A-\u202E\u2066-\u2069\u200E\u200F]", "", s or "")

def _to_arabic_digits(s: str) -> str:
    return re.sub(r"\d+", lambda m: m.group(0).translate(_ARABIC_DIGITS_MAP), s or "")

def _rtl_line(text: str, rtl_mode: str = "auto", digits: str = "ascii") -> str:
    """
    rtl_mode: 'auto' (default), 'force', 'off'
    digits: 'ascii' (default) or 'arabic'
    """
    t = _strip_bidi_controls(text or "")
    if digits == "arabic":
        t = _to_arabic_digits(t)
    if rtl_mode == "off":
        return t
    # stabilize punctuation and numbers
    t = (t.replace(":", ":" + LRM)
           .replace("-", "-" + LRM)
           .replace("–", "–" + LRM)
           .replace("/", "/" + LRM))
    return f"{BIDI_RLE}{t}{BIDI_PDF}"

def _format_citations(pages: List[int], pdf_name: str, limit: int) -> str:
    uniq = []
    for p in pages:
        if isinstance(p, int) and p >= 0 and p not in uniq:
            uniq.append(p)
        if len(uniq) >= limit:
            break
    if not uniq:
        return ""
    lines = [f"{LRM}{i}. {pdf_name} - page {p}" for i, p in enumerate(uniq, 1)]
    return "Sources:\n" + "\n".join(lines)

def _final_answer(body_text: str,
                  cite_pages: List[int],
                  pdf_name: str,
                  rtl_mode: str = "auto",
                  digits: str = "ascii",
                  max_cites: int = MAX_CITATIONS_DEFAULT) -> str:
    body = _rtl_line(body_text.strip(), rtl_mode=rtl_mode, digits=digits)
    cites = _format_citations(cite_pages, pdf_name=pdf_name, limit=max_cites)
    return f"{body}\n{cites}" if cites else body

# =========================
# Data IO (robust JSON/JSONL)
# =========================
@dataclass
class Chunk:
    id: int
    page: int
    text: str
    norm: str

_TEXT_KEYS = {"text","text_display","content","body","raw","paragraph","para","line","value","data","clean_text","norm"}
_TEXT_ARRAY_KEYS = {"lines","paragraphs","paras","sentences","chunks","blocks","spans","tokens"}
_PAGE_KEYS = {"page","page_no","page_num","pageNumber","page_index","Page","PageNo"}
_ID_KEYS = {"id","chunk_id","cid","idx","index","Id","ID"}

def _as_text(v):
    if isinstance(v, str):
        return v.strip() if v.strip() else None
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

def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()

def load_chunks(path: str) -> Tuple[List[Chunk], str]:
    if not os.path.exists(path):
        LOG.error("Chunks file not found: %s", path); sys.exit(1)
    LOG.info("Loading chunks from %s ...", path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            rows = json.load(f)
        else:
            for line in f:
                line = line.strip().rstrip(",")
                if not line: continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    if line.startswith('"') and line.endswith('"'):
                        rows.append(line.strip('"'))
                    continue
    chunks: List[Chunk] = []
    for idx, j in enumerate(rows):
        if isinstance(j, str):
            t = j; page = -1; cid = idx
        elif isinstance(j, dict):
            t = _as_text(_get_any(j, _TEXT_KEYS)) or _as_text(_get_any(j, _TEXT_ARRAY_KEYS))
            if not t: continue
            page = _get_any(j, _PAGE_KEYS)
            try: page = int(page) if page is not None else -1
            except Exception: page = -1
            cid = _get_any(j, _ID_KEYS)
            try: cid = int(cid)
            except Exception: cid = idx
        else:
            continue
        t = t.strip()
        if not t: continue
        chunks.append(Chunk(id=int(cid), page=int(page), text=t, norm=ar_normalize(t)))
    if not chunks:
        LOG.error("No chunks parsed. Ensure your file has textual fields.")
        sys.exit(1)
    LOG.info("Loaded %d chunks", len(chunks))
    return chunks, _file_hash(path)

# =========================
# Hierarchy / aliases (optional)
# =========================
@dataclass
class HierData:
    inverted: Dict[str, List[int]]
    aliases: Dict[str, List[str]]

def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def load_hierarchy(hier_index_path: Optional[str], aliases_path: Optional[str]) -> Optional[HierData]:
    inv = _load_json(hier_index_path)
    aliases = _load_json(aliases_path)
    if not inv:
        LOG.info("No hierarchy index provided/loaded (section gating disabled).")
        return None

    def _n(s: str) -> str: return ar_normalize(s).lower()

    inv_n: Dict[str, List[int]] = {}
    for k, v in inv.items():
        if not isinstance(v, list): continue
        cleaned: List[int] = []
        for x in v:
            try:
                cleaned.append(int(x))
            except Exception:
                m = re.search(r'(\d+)$', str(x))
                if m:
                    try: cleaned.append(int(m.group(1)))
                    except Exception: pass
        inv_n[_n(k)] = cleaned

    aliases_n = {_n(k): [_n(a) for a in v] for k, v in (aliases or {}).items()}
    LOG.info("Loaded hierarchy: %d keys, %d alias sets", len(inv_n), len(aliases_n))
    return HierData(inverted=inv_n, aliases=aliases_n)

def _hier_candidates(query: str, hd: HierData) -> Set[int]:
    qn = ar_normalize(query).lower()
    toks = [t for t in re.split(r"[\s\|\:/,;]+", qn) if t]
    keys = set(toks)
    for canon, alist in hd.aliases.items():
        if any(a in qn for a in alist+[canon]):
            keys.add(canon)
    cand: Set[int] = set()
    for k in keys:
        if k in hd.inverted:
            cand.update(hd.inverted[k])
    return cand

# =========================
# Index (dense + optional sparse) with persistence
# =========================
class HybridIndex:
    def __init__(self, chunks: List[Chunk], chunks_hash: str, hier: Optional[HierData] = None,
                 model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.chunks = chunks
        self.chunks_hash = chunks_hash
        self.hier = hier
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.emb = None
        self.faiss = None
        self.tf_char = None
        self.tf_word = None
        self.char_mat = None
        self.word_mat = None

    # ---------- persistence
    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            "model_name": self.model_name,
            "chunks_hash": self.chunks_hash,
            "n_chunks": len(self.chunks),
            "ids": [c.id for c in self.chunks],
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if self.emb is not None:
            np.save(os.path.join(out_dir, "embeddings.npy"), self.emb)

        if faiss is not None and self.faiss is not None:
            faiss.write_index(self.faiss, os.path.join(out_dir, "faiss.index"))

        if joblib and TfidfVectorizer is not None:
            if self.tf_char is not None:
                joblib.dump(self.tf_char, os.path.join(out_dir, "tf_char.pkl"))
            if self.tf_word is not None:
                joblib.dump(self.tf_word, os.path.join(out_dir, "tf_word.pkl"))
            if self.char_mat is not None:
                joblib.dump(self.char_mat, os.path.join(out_dir, "char_mat.pkl"))
            if self.word_mat is not None:
                joblib.dump(self.word_mat, os.path.join(out_dir, "word_mat.pkl"))

        LOG.info("Saved index artifacts to %s", out_dir)

    def load(self, in_dir: str) -> bool:
        try:
            with open(os.path.join(in_dir, "meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("chunks_hash") != self.chunks_hash or meta.get("n_chunks") != len(self.chunks):
                LOG.warning("Artifact/chunks mismatch; will rebuild instead of loading.")
                return False

            emb_path = os.path.join(in_dir, "embeddings.npy")
            if os.path.exists(emb_path):
                self.emb = np.load(emb_path)
            else:
                LOG.warning("embeddings.npy missing; cannot load."); return False

            if faiss is not None:
                faiss_path = os.path.join(in_dir, "faiss.index")
                if os.path.exists(faiss_path):
                    self.faiss = faiss.read_index(faiss_path)
                else:
                    d = self.emb.shape[1]
                    self.faiss = faiss.IndexFlatIP(d)
                    self.faiss.add(self.emb.astype('float32'))
            else:
                self.faiss = None

            if joblib and TfidfVectorizer is not None:
                tp = os.path.join(in_dir, "tf_char.pkl"); wp = os.path.join(in_dir, "tf_word.pkl")
                cmp = os.path.join(in_dir, "char_mat.pkl"); wmp = os.path.join(in_dir, "word_mat.pkl")
                if all(os.path.exists(p) for p in [tp, wp, cmp, wmp]):
                    self.tf_char = joblib.load(tp)
                    self.tf_word = joblib.load(wp)
                    self.char_mat = joblib.load(cmp)
                    self.word_mat = joblib.load(wmp)
            LOG.info("Loaded index artifacts from %s", in_dir)
            return True
        except Exception as e:
            LOG.warning("Failed to load artifacts from %s: %s", in_dir, e)
            return False

    def build(self):
        LOG.info("Building embeddings...")
        texts = [c.norm for c in self.chunks]
        self.emb = self.model.encode(texts, batch_size=128, convert_to_numpy=True,
                                     show_progress_bar=True, normalize_embeddings=True)
        if faiss is not None:
            d = self.emb.shape[1]
            self.faiss = faiss.IndexFlatIP(d)
            self.faiss.add(self.emb.astype('float32'))
            LOG.info("Built FAISS index")
        else:
            LOG.warning("faiss not available; dense search will use numpy.")

        if TfidfVectorizer is not None:
            self.tf_char = TfidfVectorizer(analyzer='char', ngram_range=(2,5), min_df=1)
            self.char_mat = self.tf_char.fit_transform(texts)
            self.tf_word = TfidfVectorizer(analyzer='word', ngram_range=(1,2),
                                           token_pattern=r"(?u)\b\w+\b", min_df=1)
            self.word_mat = self.tf_word.fit_transform(texts)
            LOG.info("Built TF-IDF (char+word) indexes")
        else:
            LOG.info("scikit-learn not installed; skipping TF-IDF.")

        LOG.info("Built embeddings for %d chunks", len(self.chunks))

    # ---------- retrieval
    def dense(self, q: str, topk=TOPK_DENSE, restrict_ids: Optional[Set[int]] = None):
        qv = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        if self.faiss is not None:
            D, I = self.faiss.search(qv.astype('float32'), topk)
            scores, idxs = D[0], I[0]
        else:
            sims = self.emb @ qv[0]
            idxs = np.argsort(-sims)[:topk]
            scores = sims[idxs]
        if restrict_ids is None:
            return scores, idxs
        # STRICT gating: return only restricted; if empty -> empty
        rset = set(int(x) for x in restrict_ids)
        filtS, filtI = [], []
        for s, i in zip(scores, idxs):
            if int(i) in rset:
                filtS.append(float(s)); filtI.append(int(i))
            if len(filtI) >= topk: break
        return np.array(filtS), np.array(filtI)

    def sparse(self, q: str):
        if self.tf_char is None or self.tf_word is None:
            return None, None
        qc = self.tf_char.transform([q]); qw = self.tf_word.transform([q])
        c_scores = (self.char_mat @ qc.T).toarray().ravel()
        w_scores = (self.word_mat @ qw.T).toarray().ravel()
        return c_scores, w_scores

# =========================
# Question analysis (general, not domain-biased)
# =========================
TIME_HINTS = ("من", "الى", "إلى", "حتى", ":", ".", "ص", "م")
NUM_UNITS  = ("ساعة","ساعه","دقيقة","دقائق","يوم","أيام","٪","%")

def expects_numeric_or_time(q: str) -> bool:
    qn = ar_normalize(q)
    if re.search(r'\d', qn): return True
    # generic arabic cues without domain bias
    cues = ("كم","مدة","مده","عدد","نسبة","نسبه","ساعات","وقت","من","الى","إلى","حتى","%","٪")
    return any(c in qn for c in cues)

def _question_keywords(q: str) -> List[str]:
    """Lightweight tokenization: keep tokens len>=2 for overlap/proximity."""
    qn = ar_normalize(q)
    toks = [t for t in re.split(r"[^\w%٪]+", qn) if len(t) >= 2]
    return list(dict.fromkeys(toks))[:24]

# =========================
# Numeric/time extraction (for validation, not rewriting)
# =========================
TIME_RE = re.compile(
    r'(?:من\s*)?'
    r'(\d{1,2}(?::|\.)?\d{0,2})\s*'
    r'(?:[-–—]|الى|إلى|حتي|حتى)\s*'
    r'(\d{1,2}(?::|\.)?\d{0,2})'
)

def _normalize_hhmm(tok: str) -> str:
    tok = tok.replace('.', ':')
    if ':' not in tok:
        return f"{int(tok):d}:00"
    h, m = tok.split(':', 1)
    if m == "": m = "00"
    return f"{int(h):d}:{int(m):02d}"

def _to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(':'); return int(h)*60 + int(m)

def _plausible_work_window(a: int, b: int) -> bool:
    return 6*60 <= a <= 20*60+30 and 6*60 <= b <= 20*60+30 and b > a

def has_time_like(s: str) -> bool:
    n = ar_normalize(s)
    if TIME_RE.search(n): return True
    return any(h in n for h in TIME_HINTS)

def has_numeric_with_units(s: str) -> bool:
    n = ar_normalize(s)
    if re.search(r'\d', n): return True
    return any(u in n for u in NUM_UNITS)

# =========================
# Proximity helpers
# =========================
def _prox_numeric_with_kws(text: str, kws: List[str], window=WINDOW_CHARS) -> bool:
    T = ar_normalize(text)
    for m in re.finditer(r'\d+', T):
        start = max(0, m.start()-window); end = m.end()+window
        ctx = T[start:end]
        if any(kw in ctx for kw in kws):
            return True
    return False

# =========================
# Retrieval pipeline
# =========================
def combine_scores(dense_scores, dense_idx, c_scores, w_scores,
                   w_dense=0.65, w_char=0.20, w_word=0.15, topk=TOPK_FINAL):
    pairs = []
    for s, i in zip(dense_scores, dense_idx):
        sc = float(s) * w_dense
        if c_scores is not None and len(c_scores) > int(i): sc += float(c_scores[i]) * w_char
        if w_scores is not None and len(w_scores) > int(i): sc += float(w_scores[i]) * w_word
        pairs.append((sc, int(i)))
    pairs.sort(key=lambda x: -x[0])
    return pairs[:topk]

def retrieve(index: HybridIndex, q: str, hier: Optional[HierData]) -> List[Tuple[float,int]]:
    qn = ar_normalize(q)
    restrict_ids: Optional[Set[int]] = None
    if hier is not None:
        cand = _hier_candidates(qn, hier)
        if cand:
            restrict_ids = cand
    dS, dI = index.dense(qn, topk=TOPK_DENSE, restrict_ids=restrict_ids)
    if restrict_ids is not None and (dI is None or len(dI) == 0):
        # strict: do not fallback globally; we will refuse later
        return []
    cS, wS = index.sparse(qn) if TfidfVectorizer is not None else (None, None)
    prelim = combine_scores(dS, dI, cS, wS, topk=TOPK_FINAL)
    return prelim

# =========================
# Snippet scoring (generalized)
# =========================
def _looks_junky(sn: str) -> bool:
    snn = ar_normalize(sn)
    if len(snn) < 6: return True
    letters = sum(ch.isalpha() for ch in snn)
    total = len(snn.replace(" ", ""))
    if total > 0 and letters/total < 0.5:
        return True
    return False

def best_snippet(chunk: Chunk, qnorm: str, expect_numeric_time: bool) -> Optional[str]:
    sents = sent_split(chunk.text)
    if not sents:
        return (chunk.text[:MAX_SNIPPET_CHARS] if chunk.text else None)
    q_terms = set([w for w in qnorm.split() if len(w) >= 2])

    best, best_score = None, -1e9
    for s in sents:
        if _looks_junky(s): 
            continue
        sn = ar_normalize(s)
        overlap = len(q_terms & set(sn.split()))
        has_time = has_time_like(s)
        has_numu = has_numeric_with_units(s)
        prox = _prox_numeric_with_kws(s, list(q_terms), window=WINDOW_CHARS)

        score = 1.2 * overlap + (0.8 if has_time else 0.0) + (0.6 if has_numu else 0.0) + (0.4 if prox else 0.0)

        # gating for numeric/time questions
        if expect_numeric_time and not (has_time or has_numu or prox or re.search(r"\d", sn)):
            score -= 1.5

        # mild length preference (shorter is clearer)
        L = len(sn)
        score -= 0.001 * max(0, L - 140)

        if score > best_score:
            best, best_score = s, score

    if not best:
        # fallback: least junky sentence
        for s in sents:
            if not _looks_junky(s):
                best = s; break
    if not best:
        return None

    txt = best.strip()
    if len(txt) > MAX_SNIPPET_CHARS:
        txt = txt[:MAX_SNIPPET_CHARS].rstrip() + "…"
    return txt

# =========================
# Answer (extractive only)
# =========================
def answer(q: str,
           index: HybridIndex,
           hier: Optional[HierData],
           pdf_name: str,
           rtl_mode: str = "auto",
           digits: str = "ascii",
           max_cites: int = MAX_CITATIONS_DEFAULT) -> str:

    hits = retrieve(index, q, hier)
    if not hits:
        return _final_answer("لم يرد نص صريح حول ذلك في الدليل المرفق.",
                             [], pdf_name, rtl_mode, digits, max_cites)

    expect_num = expects_numeric_or_time(q)
    qn = ar_normalize(q)

    # try best sentence from top hits
    for _, i in hits:
        ch = index.chunks[i]
        sn = best_snippet(ch, qn, expect_num)
        if sn:
            # extra confidence check for numeric/time questions
            if expect_num:
                cond = (has_time_like(sn) or has_numeric_with_units(sn)
                        or _prox_numeric_with_kws(sn, _question_keywords(q)))
                if not cond:
                    continue
            return _final_answer(sn, [ch.page], pdf_name, rtl_mode, digits, max_cites)

    # if we got here with no acceptable sentence:
    if STRICT_MODE:
        return _final_answer("لم يرد نص صريح حول ذلك في الدليل المرفق.",
                             [], pdf_name, rtl_mode, digits, max_cites)

    # non-strict fallback (disabled by default)
    ch = index.chunks[hits[0][1]]
    sn = (ch.text or "").strip()
    sn = sn[:MAX_SNIPPET_CHARS] + ("…" if len(sn) > MAX_SNIPPET_CHARS else "")
    return _final_answer(sn, [ch.page], pdf_name, rtl_mode, digits, max_cites)

# =========================
# Sanity prompts (30)
# =========================
SANITY_PROMPTS = [
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
    "ما سياسة الإجازة الطارئة؟ وكيف أطلبها؟",
    "ما سياسة الإجازة المرضية؟ وعدد أيامها؟ وهل يلزم تقرير طبي؟",
    "كم مدة إجازة الأمومة؟ وهل يمكن أخذ جزء قبل الولادة؟",
    "ما هي إجازة الحداد؟ لمن تُمنح وكم مدتها؟",
    "متى يتم صرف الرواتب شهريًا؟",
    "ما هو بدل المواصلات؟ وهل يشمل الذهاب من المنزل للعمل؟ وكيف يُصرف؟",
    "هل توجد سلف على الراتب؟ وما شروطها؟",
    "ما الحد الأقصى للنثريات اليومية؟ وكيف تتم التسوية والمستندات المطلوبة؟",
    "ما سقف الشراء الذي يستلزم ثلاثة عروض أسعار؟",
    "ما ضوابط تضارب المصالح في المشتريات؟",
    "ما حدود قبول الهدايا والضيافة؟ ومتى يجب الإبلاغ؟",
    "كيف أستلم عهدة جديدة؟ وما النموذج المطلوب؟",
    "كيف أسلّم العهدة عند الاستقالة أو الانتقال؟",
    "ما سياسة العمل عن بُعد/من المنزل؟ وكيف يتم اعتماده؟",
    "كيف أقدّم إذن مغادرة ساعية؟ وما الحد الأقصى الشهري؟",
    "متى يتم تقييم الأداء السنوي؟ وما معاييره الأساسية؟",
    "ما إجراءات الإنذار والتدرّج التأديبي للمخالفات؟",
    "ما سياسة السرية وحماية المعلومات؟",
    "ما سياسة السلوك المهني ومكافحة التحرش؟",
    "هل توجد مياومات/بدل سفر؟ وكيف تُصرف",
]

def _pass_loose(ans_text: str) -> bool:
    return "Sources:" in ans_text and "لم يرد نص" not in ans_text

def _pass_strict(q: str, body_only: str) -> bool:
    # If question expects numeric/time, make sure body shows at least some numeric/time cues
    needs = expects_numeric_or_time(q)
    if not needs:
        return len(ar_normalize(body_only)) >= 6
    return (has_time_like(body_only) or has_numeric_with_units(body_only) or re.search(r"\d", ar_normalize(body_only)) is not None)

def run_sanity(index: HybridIndex, hier: Optional[HierData], pdf_name: str, rtl_mode: str, digits: str, max_cites: int):
    LOG.info(f"🧪 Running sanity prompts ({len(SANITY_PROMPTS)}) …")
    print("="*80)
    pass_loose = 0
    pass_strict = 0
    for i, q in enumerate(SANITY_PROMPTS, 1):
        print(f"\n📝 Test {i}/{len(SANITY_PROMPTS)}: {q}")
        print("-"*60)
        out = answer(q, index, hier, pdf_name, rtl_mode, digits, max_cites)
        # split body and sources visually
        parts = out.split("\nSources:")
        body = parts[0]
        print(body)
        if len(parts) > 1:
            print("Sources:" + parts[1])
        okL = _pass_loose(out); okS = _pass_strict(q, body)
        pass_loose += int(okL); pass_strict += int(okS)
        print("✅ PASS_LOOSE" if okL else "⚪ FAIL_LOOSE")
        print("✅ PASS_STRICT" if okS else "⚪ FAIL_STRICT")
        print("="*80)
    print(f"Summary: PASS_LOOSE {pass_loose}/{len(SANITY_PROMPTS)} | PASS_STRICT {pass_strict}/{len(SANITY_PROMPTS)}")

# =========================
# Interactive loop
# =========================
def interactive_loop(index: HybridIndex, hier: Optional[HierData], pdf_name: str, rtl_mode: str, digits: str, max_cites: int):
    print("جاهز.")
    print("اسأل عن بنود الدليل (إجابات مقتبسة حرفياً من الـPDF). اكتب 'exit' للخروج.\n")
    while True:
        try:
            q = input("سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nخروج."); break
        if not q: continue
        if q.lower() in ("exit","quit","q"): print("خروج."); break
        print(answer(q, index, hier, pdf_name, rtl_mode, digits, max_cites))
        print("-"*66)

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, required=True, help="Path to chunks (JSONL/JSON)")
    ap.add_argument("--hier-index", type=str, default=None, help="Optional hierarchical inverted index path")
    ap.add_argument("--aliases", type=str, default=None, help="Optional aliases for headings")
    ap.add_argument("--save-index", type=str, default=None, help="Directory to save index artifacts")
    ap.add_argument("--load-index", type=str, default=None, help="Directory to load index artifacts from")
    ap.add_argument("--sanity", action="store_true", help="Run sanity prompts and exit")
    ap.add_argument("--pdf-name", type=str, default=PDF_DEFAULT_NAME, help="File name used in citations (display only)")
    ap.add_argument("--rtl", choices=["auto","off","force"], default="auto", help="RTL formatting for Arabic answers")
    ap.add_argument("--digits", choices=["ascii","arabic"], default="ascii", help="Digit style in answers")
    ap.add_argument("--max-citations", type=int, default=MAX_CITATIONS_DEFAULT, help="Max citation lines in output (default 3)")
    ap.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    help="SentenceTransformer model id")
    args = ap.parse_args()

    hier = load_hierarchy(args.hier_index, args.aliases)
    chunks, chunks_hash = load_chunks(path=args.chunks)
    index = HybridIndex(chunks, chunks_hash, hier=hier, model_name=args.model)

    loaded = False
    if args.load_index:
        loaded = index.load(args.load_index)

    if not loaded:
        index.build()
        if args.save_index:
            index.save(args.save_index)

    LOG.info("Ready (STRICT_MODE=%s).", STRICT_MODE)

    if args.sanity:
        run_sanity(index, hier, args.pdf_name, args.rtl, args.digits, args.max_citations)
        return

    interactive_loop(index, hier, args.pdf_name, args.rtl, args.digits, args.max_citations)

if __name__ == "__main__":
    main()

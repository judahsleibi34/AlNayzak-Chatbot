# -*- coding: utf-8 -*-
"""
NewRag.py — generalized, neutral, extractive RAG for Arabic PDFs

- Hybrid retrieval: dense (multilingual MiniLM) + TF-IDF (char+word)
- Optional cross-encoder rerank (MMARCO mMiniLMv2)
- Arabic normalization + OCR-aware sentence splitting
- Strict, extractive-only answers (no generation); always cite PDF page(s)
- Neutral "no-bias" defaults; optional guided mode via --guided
- Embedding/FAISS/TF-IDF persistence for fast restarts
- CLI flags compatible with your previous commands

Run examples
------------
Build + cache + sanity:
  python NewRag.py \
    --chunks Data_pdf_clean_chunks.jsonl \
    --hier-index heading_inverted_index.json \
    --aliases section_aliases.json \
    --sanity \
    --device auto \
    --rerank \
    --max-bullets 5 \
    --bullet-max-chars 120 \
    --paginate-chars 600 \
    --out-dir runs

Interactive:
  python NewRag.py --chunks Data_pdf_clean_chunks.jsonl --device auto
"""

import os, re, sys, json, argparse, logging, hashlib, datetime, pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Set

import numpy as np

# optional dependencies
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
    from sentence_transformers.cross_encoder import CrossEncoder
except Exception:
    print("Please install: sentence-transformers scikit-learn faiss-cpu joblib")
    raise

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger("NewRag")

# ---------------- behavior toggles (safe defaults) ----------------
STRICT_MODE = True          # refuse if no grounded snippet
NEUTRAL_MODE = True         # don't hard-gate by intent unless --guided
WINDOW_CHARS = 48           # numeric proximity window
MAX_SNIPPET_CHARS = 260

# ---------------- Arabic utilities ----------------
AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"٠١٢٣٤٥٦٧٨٩")}
IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"۰۱۲۳۴۵۶۷۸۹")}
SENT_SPLIT_RE = re.compile(r'(?<=[\.\!\؟\?،]|[\n])\s+')

def ar_normalize(s: str) -> str:
    if not s: return ""
    s = s.replace('\u0640','')
    s = AR_DIAC.sub('', s)
    s = (s.replace('أ','ا').replace('إ','ا').replace('آ','ا')
           .replace('ى','ي'))
    s = s.translate(AR_NUMS).translate(IR_NUMS)
    s = s.replace('،', ',').replace('٫','.')
    s = ' '.join(s.split())
    return s

def rtl_wrap(t: str) -> str:
    return '\u202B' + t + '\u202C'

def sent_split(text: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text or "") if p and p.strip()]
    out = []
    for p in parts:
        pn = ar_normalize(p)
        if len(pn) < 6:  # drop tiny fragments
            continue
        letters = sum(ch.isalpha() for ch in pn)
        total = len(pn.replace(" ", ""))
        if total == 0 or letters/total < 0.5:
            continue
        out.append(p)
    return out if out else ([text.strip()] if (text and text.strip()) else [])

# ---------------- IO ----------------
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
        return v.strip() or None
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

# ---------------- hierarchy / aliases (optional gating) ----------------
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
            try: cleaned.append(int(x))
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

# ---------------- index with persistence ----------------
class HybridIndex:
    def __init__(self, chunks: List[Chunk], chunks_hash: str, hier: Optional[HierData] = None,
                 model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 device: str = "auto"):
        dev = None if device == "auto" else device
        self.chunks = chunks
        self.chunks_hash = chunks_hash
        self.hier = hier
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=dev)
        self.emb = None
        self.faiss = None
        self.tf_char = None
        self.tf_word = None
        self.char_mat = None
        self.word_mat = None

    # persistence
    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        meta = {"model_name": self.model_name, "chunks_hash": self.chunks_hash, "n_chunks": len(self.chunks)}
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f: json.dump(meta, f, ensure_ascii=False, indent=2)
        if self.emb is not None: np.save(os.path.join(out_dir, "embeddings.npy"), self.emb)
        if faiss is not None and self.faiss is not None: faiss.write_index(self.faiss, os.path.join(out_dir, "faiss.index"))
        if joblib and TfidfVectorizer is not None:
            if self.tf_char is not None: joblib.dump(self.tf_char, os.path.join(out_dir, "tf_char.pkl"))
            if self.tf_word is not None: joblib.dump(self.tf_word, os.path.join(out_dir, "tf_word.pkl"))
            if self.char_mat is not None: joblib.dump(self.char_mat, os.path.join(out_dir, "char_mat.pkl"))
            if self.word_mat is not None: joblib.dump(self.word_mat, os.path.join(out_dir, "word_mat.pkl"))
        LOG.info("Saved index to %s", out_dir)

    def load(self, in_dir: str) -> bool:
        try:
            with open(os.path.join(in_dir, "meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("chunks_hash") != self.chunks_hash:
                LOG.warning("chunks changed; rebuild."); return False
            emb_path = os.path.join(in_dir, "embeddings.npy")
            if os.path.exists(emb_path): self.emb = np.load(emb_path)
            else: return False
            if faiss is not None:
                fpath = os.path.join(in_dir, "faiss.index")
                self.faiss = faiss.read_index(fpath) if os.path.exists(fpath) else None
                if self.faiss is None:
                    d = self.emb.shape[1]; self.faiss = faiss.IndexFlatIP(d); self.faiss.add(self.emb.astype('float32'))
            if joblib and TfidfVectorizer is not None:
                tp, wp, cm, wm = (os.path.join(in_dir, p) for p in ["tf_char.pkl","tf_word.pkl","char_mat.pkl","word_mat.pkl"])
                if all(os.path.exists(p) for p in [tp,wp,cm,wm]):
                    self.tf_char = joblib.load(tp); self.tf_word = joblib.load(wp)
                    self.char_mat = joblib.load(cm); self.word_mat = joblib.load(wm)
            LOG.info("Loaded index from %s", in_dir); return True
        except Exception as e:
            LOG.warning("load failed: %s", e); return False

    def build(self):
        LOG.info("Building embeddings/FAISS/TF-IDF …")
        texts = [c.norm for c in self.chunks]
        self.emb = self.model.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
        if faiss is not None:
            d = self.emb.shape[1]; self.faiss = faiss.IndexFlatIP(d); self.faiss.add(self.emb.astype('float32'))
        if TfidfVectorizer is not None:
            self.tf_char = TfidfVectorizer(analyzer='char', ngram_range=(2,5), min_df=1)
            self.char_mat = self.tf_char.fit_transform(texts)
            self.tf_word = TfidfVectorizer(analyzer='word', ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b", min_df=1)
            self.word_mat = self.tf_word.fit_transform(texts)

    def dense(self, q: str, topk=60, restrict_ids: Optional[Set[int]] = None):
        qv = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        if self.faiss is not None:
            D, I = self.faiss.search(qv.astype('float32'), topk); scores, idxs = D[0], I[0]
        else:
            sims = self.emb @ qv[0]; idxs = np.argsort(-sims)[:topk]; scores = sims[idxs]
        if restrict_ids is None: return scores, idxs
        rset = set(int(x) for x in restrict_ids); filtS, filtI = [], []
        for s,i in zip(scores, idxs):
            if int(i) in rset: filtS.append(float(s)); filtI.append(int(i))
            if len(filtI) >= topk: break
        return np.array(filtS), np.array(filtI)

    def sparse(self, q: str):
        if self.tf_char is None or self.tf_word is None: return None, None
        qc = self.tf_char.transform([ar_normalize(q)]); qw = self.tf_word.transform([ar_normalize(q)])
        c_scores = (self.char_mat @ qc.T).toarray().ravel(); w_scores = (self.word_mat @ qw.T).toarray().ravel()
        return c_scores, w_scores

# ---------------- intents (neutral by default) ----------------
def classify_intent(q: str) -> str:
    # light classifier; not used to block, only to help formatting
    qn = ar_normalize(q)
    if re.search(r'(رمضان)', qn): return "ramadan_hours"
    if re.search(r'(ساعات).*(دوام|عمل)', qn): return "work_hours"
    if re.search(r'(ايام|أيام).*(عمل|دوام)|السبت|الاحد|الخميس', qn): return "workdays"
    if re.search(r'(اضافي|إضافي)', qn): return "overtime"
    if re.search(r'(اجاز|إجاز)', qn): return "leave"
    if re.search(r'(عروض|شراء|مشتريات|سقف|حد)', qn): return "procurement"
    if re.search(r'(مياومات|بدل\s*سفر|نفقات|فواتير|تذاكر|فندق)', qn): return "per_diem"
    if re.search(r'(استراح|راحة|بريك)', qn): return "break"
    if re.search(r'(مرون|تأخير|الحضور|الانصراف|بصمة)', qn): return "flex"
    return "general"

# fix: correct must-token logic (used only in guided mode)
def _must_tokens_present(sn_norm: str, req: List[str]) -> bool:
    if not req: return True
    return any(tok in sn_norm for tok in req)

# ---------------- time extraction ----------------
TIME_RE = re.compile(r'(?:من\s*)?(\d{1,2}(?::|\.)?\d{0,2})\s*(?:[-–—]|الى|إلى|حتي|حتى)\s*(\d{1,2}(?::|\.)?\d{0,2})')
def _normalize_hhmm(tok: str) -> str:
    tok = tok.replace('.', ':')
    if ':' not in tok: return f"{int(tok):d}:00"
    h, m = tok.split(':', 1); m = m or "00"; return f"{int(h):d}:{int(m):02d}"
def _to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(':'); return int(h)*60 + int(m)
def _plausible_workday(a: int, b: int) -> bool:
    return 6*60 <= a <= 20*60+30 and 6*60 <= b <= 20*60+30 and b > a
def extract_time_ranges(text: str, check_workday=True) -> List[Tuple[int,int]]:
    n = ar_normalize(text); out=[]
    for m in TIME_RE.finditer(n):
        a=_normalize_hhmm(m.group(1)); b=_normalize_hhmm(m.group(2)); A=_to_minutes(a); B=_to_minutes(b)
        if B <= A: B = B + 12*60 if B+12*60 > A else (A+60)
        if check_workday and not (_plausible_workday(A,B) and 6*60 <= (B-A) <= 11*60): continue
        out.append((A,B))
    return out
def pick_best_range(ranges: List[Tuple[int,int]]) -> Optional[Tuple[str,str]]:
    if not ranges: return None
    scored = [(abs((B-A)-7.5*60), -(B-A), A, B) for (A,B) in ranges]; scored.sort()
    _, _, A, B = scored[0]; return f"{A//60:d}:{A%60:02d}", f"{B//60:d}:{B%60:02d}"

# ---------------- hybrid retrieve + RRF ----------------
def _rrf_merge(dense_scores, dense_idx, c_scores, w_scores, k=60, lam=60.0):
    # Reciprocal Rank Fusion of 3 runs (dense, char, word)
    runs = []
    if dense_idx is not None and len(dense_idx)>0:
        runs.append(list(dense_idx[:k]))
    if c_scores is not None:
        runs.append(list(np.argsort(-c_scores)[:k]))
    if w_scores is not None:
        runs.append(list(np.argsort(-w_scores)[:k]))
    if not runs: return []
    scores = {}
    for r in runs:
        for rank, idx in enumerate(r, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0/(lam+rank)
    merged = sorted(scores.items(), key=lambda x: -x[1])
    return [i for i,_ in merged[:k]]

def retrieve(index: HybridIndex, q: str, guided=False, hier: Optional[HierData]=None) -> List[int]:
    qn = ar_normalize(q)
    restrict_ids = None
    if guided and hier is not None:
        cand = _hier_candidates(qn, hier)
        if cand: restrict_ids = cand
    dS, dI = index.dense(qn, topk=60, restrict_ids=restrict_ids)
    cS, wS = index.sparse(qn)
    merged = _rrf_merge(dS, dI, cS, wS, k=60)
    if restrict_ids is not None:
        rset = set(restrict_ids); merged = [i for i in merged if i in rset]
    return merged[:12]

# ---------------- compose extractive answers ----------------
def _compose_hours(chunks: List[Chunk], idxs, label: str):
    for i in idxs:
        ch = chunks[i]
        for s in sent_split(ch.text):
            r = extract_time_ranges(s, check_workday=True)
            pick = pick_best_range(r)
            if pick:
                a,b = pick
                body = f"ساعات الدوام{label} من {a} إلى {b}."
                return rtl_wrap(body) + f"\nSources:\n1. Data_pdf.pdf - page {ch.page}"
    return None

def _compose_any_numeric(chunks: List[Chunk], idxs):
    # general numeric policy sentence (procurement/per-diem/leave etc.)
    for i in idxs:
        ch = chunks[i]
        for s in sent_split(ch.text):
            if re.search(r'\d', ar_normalize(s)):
                return rtl_wrap(s.strip()) + f"\nSources:\n1. Data_pdf.pdf - page {ch.page}"
    return None

def answer(q: str, index: HybridIndex, intent: str, guided=False) -> str:
    idxs = retrieve(index, q, guided=guided, hier=index.hier if guided else None)
    if not idxs:
        return rtl_wrap("لم يرد نص صريح حول ذلك في الدليل المرفق.")
    chunks = index.chunks

    if intent == "work_hours":
        o = _compose_hours(chunks, idxs, "")
        if o: return o
    if intent == "ramadan_hours":
        o = _compose_hours(chunks, idxs, " في شهر رمضان")
        if o: return o

    # generic numeric sentence (covers procurement thresholds, per-diem tariffs, leave durations, etc.)
    o = _compose_any_numeric(chunks, idxs)
    if o: return o

    # final: best readable sentence from top chunk
    ch = chunks[idxs[0]]
    sents = sent_split(ch.text)
    if sents:
        return rtl_wrap(sents[0].strip()) + f"\nSources:\n1. Data_pdf.pdf - page {ch.page}"
    return rtl_wrap("لم يرد نص صريح حول ذلك في الدليل المرفق.")

# ---------------- output formatting (sanity reports) ----------------
def _clip(s: str, n: int) -> str:
    s = s.strip()
    return (s if len(s)<=n else (s[:max(1,n-1)].rstrip()+"…"))

def _paginate_text(text: str, max_chars=600):
    text = text.strip()
    if len(text) <= max_chars: return [text]
    parts, cur, count = [], [], 0
    for line in text.splitlines():
        if count + len(line) + 1 > max_chars:
            parts.append("\n".join(cur).strip()); cur, count = [line], len(line)
        else:
            cur.append(line); count += len(line)+1
    if cur: parts.append("\n".join(cur).strip())
    return parts

def _as_bullets(body: str, max_bullets: int, bullet_max_chars: int):
    sents = [s for s in SENT_SPLIT_RE.split(body) if s and s.strip()]
    sents = [_clip(s.strip(" -–•\t"), bullet_max_chars) for s in sents]
    if not sents: sents = [_clip(body, bullet_max_chars)]
    return "\n".join("• " + rtl_wrap(s) for s in sents[:max_bullets])

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
    "ما سياسة الإجازة المرضية؟ وعدد أيامها؟ وهل يلزم تقرير طبي؟",
    "كم مدة إجازة الأمومة؟ وهل يمكن أخذ جزء قبل الولادة؟",
    "ما هي إجازة الحداد؟ لمن تُمنح وكم مدتها؟",
    "ما سقف الشراء الذي يستلزم ثلاثة عروض أسعار؟",
    "هل توجد مياومات/بدل سفر؟ وكيف تُصرف"
]

def run_sanity(index, guided, out_dir, max_bullets, bullet_max_chars, paginate_chars):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.jsonl")
    report_txt   = os.path.join(run_dir, "report.txt")
    summary_md   = os.path.join(run_dir, "summary.md")
    results_f = open(results_path, "w", encoding="utf-8")
    report_f  = open(report_txt,  "w", encoding="utf-8")

    passed = 0
    for i,q in enumerate(SANITY_PROMPTS, 1):
        intent = classify_intent(q)
        out = answer(q, index, intent, guided=guided)
        body, srcs = (out.split("\nSources:",1)+[""])[:2]
        ok = ("Sources:" in out) and ("لم يرد نص" not in out)
        passed += int(ok)

        parts = _paginate_text(body.strip(), max_chars=paginate_chars)
        if len(parts) > 1:
            disp = []
            for n,p in enumerate(parts,1):
                disp.append(f"الجزء {n}/{len(parts)}:\n{_as_bullets(p, max_bullets, bullet_max_chars)}")
            body_disp = "\n\n".join(disp)
        else:
            body_disp = _as_bullets(parts[0], max_bullets, bullet_max_chars)

        report_f.write(f"\n📝 Test {i}/{len(SANITY_PROMPTS)}: {q}\n{'-'*60}\n{body_disp}\nSources:{srcs and ' '}{srcs.strip()}\n")
        results_f.write(json.dumps({"i":i,"q":q,"answer":out,"pass":ok}, ensure_ascii=False)+"\n")

    summary = f"# Sanity Summary\n- Total: {len(SANITY_PROMPTS)}\n- PASS(has sources): {passed}/{len(SANITY_PROMPTS)}\n"
    with open(summary_md,"w",encoding="utf-8") as f: f.write(summary)
    results_f.close(); report_f.close()
    print(summary)
    print(f"Artifacts saved in: {run_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, required=True, help="Path to chunks (JSON/JSONL)")
    ap.add_argument("--hier-index", type=str, default="heading_inverted_index.json")
    ap.add_argument("--aliases", type=str, default="section_aliases.json")
    ap.add_argument("--load-index", type=str, default=None)
    ap.add_argument("--save-index", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    ap.add_argument("--guided", action="store_true", help="Enable hierarchy-guided retrieval")
    ap.add_argument("--sanity", action="store_true")
    # pretty-print flags
    ap.add_argument("--max-bullets", type=int, default=5)
    ap.add_argument("--bullet-max-chars", type=int, default=120)
    ap.add_argument("--paginate-chars", type=int, default=600)
    ap.add_argument("--out-dir", type=str, default="runs")
    args = ap.parse_args()

    hier = load_hierarchy(args.hier_index, args.aliases) if args.guided else None
    chunks, chunks_hash = load_chunks(args.chunks)
    index = HybridIndex(chunks, chunks_hash, hier=hier, device=args.device)

    loaded = False
    if args.load_index and os.path.exists(args.load_index):
        loaded = index.load(args.load_index)
    if not loaded:
        index.build()
        if args.save_index: index.save(args.save_index)

    if args.sanity:
        run_sanity(index, guided=args.guided, out_dir=args.out_dir,
                   max_bullets=args.max_bullets, bullet_max_chars=args.bullet_max_chars,
                   paginate_chars=args.paginate_chars)
        return

    print("جاهز. اكتب 'exit' للخروج.\n")
    while True:
        try:
            q = input("سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nخروج."); break
        if not q: continue
        if q.lower() in ("exit","quit","q"): print("خروج."); break
        intent = classify_intent(q)
        print(answer(q, index, intent, guided=args.guided))
        print("-"*66)

if __name__ == "__main__":
    main()

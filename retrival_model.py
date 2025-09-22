# -*- coding: utf-8 -*-
"""
Arabic-first hybrid retriever with intent-aware composing for RAG.

What's new in this revision
---------------------------
- Hours chooser now favors start 07:00–10:00 and end 13:00–18:00; never prints
  the inverted "3:00 → 8:30" case for work hours.
- New composers:
    * _compose_break_answer()         -> استراحة/راحة/بريك/رضاعة + مدة
    * _compose_hourly_leave()         -> إذن/مغادرة ساعية + حدود/أرقام
    * _compose_leave_carryover()      -> الإجازة السنوية غير المستخدمة/الترحيل
    * _compose_emergency_leave()      -> الإجازة الطارئة (آلية الطلب/المدة)
    * _compose_performance_review()   -> تقييم الأداء السنوي + معايير
- STRICT-friendly selection: fallbacks refuse unrelated snippets (e.g., الاستقالة) for
  hourly leave / break / carryover / performance.
- Display sanitizer strips OCR tokens like "uni06BE" before returning answers.
"""

import os, re, sys, json, argparse, logging, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Set

import numpy as np

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
    print("Please install: pip install sentence-transformers scikit-learn faiss-cpu joblib")
    raise

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger(__name__)

# ---------------- Arabic utils ----------------

AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"٠١٢٣٤٥٦٧٨٩")}
IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"۰۱۲۳۴۵۶۷۸۹")}
NOISY_UNI = re.compile(r'/?\buni[0-9A-Fa-f]{3,6}\b')
RIGHTS_LINE = re.compile(r'جميع الحقوق محفوظة|التعليم المساند والإبداع العلمي')
SECTION_NUM = re.compile(r'\b\d+\.\d+\b')

def ar_normalize(s: str) -> str:
    if not s: return ""
    s = s.replace('\u0640','')
    s = AR_DIAC.sub('', s)
    s = (s.replace('أ','ا').replace('إ','ا').replace('آ','ا')
           .replace('ى','ي').replace('ة','ه'))
    s = s.translate(AR_NUMS).translate(IR_NUMS)
    s = s.replace('،', ',').replace('٫','.')
    s = NOISY_UNI.sub('', s)
    s = ' '.join(s.split())
    return s

def clean_display_text(s: str) -> str:
    """Remove OCR garbage and tidy punctuation/whitespace for user-visible text."""
    if not s: return s
    s = NOISY_UNI.sub('', s)
    s = re.sub(r'\s+([،,:;\.])', r'\1', s)
    s = re.sub(r'\s{2,}', ' ', s)
    s = s.replace('‐', '-').replace('-', '-').replace('–', '-').replace('—', '-')
    return s.strip()

def rtl_wrap(t: str) -> str:
    return '\u202B' + t + '\u202C'

SENT_SPLIT_RE = re.compile(r'(?<=[\.\!\؟\?،]|[\n])\s+')

def sent_split(s: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(s) if p and p.strip()]
    out = []
    for p in parts:
        pn = ar_normalize(p)
        if len(pn) < 6:                 # very short/junk
            continue
        if RIGHTS_LINE.search(p):       # boilerplate
            continue
        letters = sum(ch.isalpha() for ch in pn)
        total = len(pn.replace(" ", ""))
        if total > 0 and letters/total < 0.5:
            continue
        out.append(p)
    return out if out else ([s.strip()] if s.strip() else [])

# ---------------- Data IO ----------------

CHUNKS_PATH = "Data_pdf_clean_chunks.jsonl"
PDF_PATH    = "Data_pdf.pdf"

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
        while True:
            b = f.read(1<<20)
            if not b: break
            h.update(b)
    return h.hexdigest()

def load_chunks(path=CHUNKS_PATH):
    if not os.path.exists(path):
        LOG.error("Chunks file not found: %s", path); sys.exit(1)
    LOG.info("Loading documents from %s ...", path)

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
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

    LOG.info("Loaded %d chunks", len(chunks))
    if len(chunks) == 0:
        LOG.error("No chunks parsed. Check that your file contains textual fields.")
        sys.exit(1)
    return chunks, _file_hash(path)

# ---------------- Hierarchy / aliases ----------------

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
        LOG.info("No hierarchy index provided/loaded.")
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
                    try:
                        cleaned.append(int(m.group(1)))
                    except Exception:
                        pass
        inv_n[_n(k)] = cleaned

    aliases_n = {_n(k): [_n(a) for a in v] for k, v in (aliases or {}).items()}
    LOG.info("Loaded hierarchy: %d index keys, %d alias sets", len(inv_n), len(aliases_n))
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

# ---------------- Index with persistence ----------------

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
                LOG.warning("embeddings.npy missing; cannot load.")
                return False

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
                else:
                    LOG.warning("TF-IDF artifacts missing; skipping sparse load.")

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
            LOG.warning("faiss not available; dense retrieval will be slower.")

        if TfidfVectorizer is None:
            LOG.warning("sklearn not available; skipping TF-IDF indexes.")
        else:
            self.tf_char = TfidfVectorizer(analyzer='char', ngram_range=(2,5), min_df=1)
            self.char_mat = self.tf_char.fit_transform(texts)
            self.tf_word = TfidfVectorizer(analyzer='word', ngram_range=(1,2),
                                           token_pattern=r"(?u)\b\w+\b", min_df=1)
            self.word_mat = self.tf_word.fit_transform(texts)
            LOG.info("Built TF-IDF (char+word) indexes")

        LOG.info("Built embeddings for %d chunks", len(self.chunks))

    def dense(self, q: str, topk=60, restrict_ids: Optional[Set[int]] = None):
        qv = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        if self.faiss is not None:
            D, I = self.faiss.search(qv.astype('float32'), max(topk, 60))
            scores, idxs = D[0], I[0]
        else:
            sims = self.emb @ qv[0]
            idxs = np.argsort(-sims)[:max(topk, 60)]
            scores = sims[idxs]
        if restrict_ids is None:
            return scores[:topk], idxs[:topk]
        filtS, filtI = [], []
        rset = set(int(x) for x in restrict_ids)
        for s, i in zip(scores, idxs):
            if int(i) in rset:
                filtS.append(float(s)); filtI.append(int(i))
            if len(filtI) >= topk: break
        return np.array(filtS), np.array(filtI)

    def sparse(self, q: str):
        if self.tf_char is None or self.tf_word is None:
            return None, None
        qc = self.tf_char.transform([q])
        qw = self.tf_word.transform([q])
        c_scores = (self.char_mat @ qc.T).toarray().ravel()
        w_scores = (self.word_mat @ qw.T).toarray().ravel()
        return c_scores, w_scores

# ---------------- Intents ----------------

KW_HOURS = re.compile(r'(ساعات)\s+(الدوام|العمل)')
KW_RAMADAN = re.compile(r'رمضان')
KW_LEAVE = re.compile(r'(إجاز|اجاز)')
KW_OVERTIME = re.compile(r'(إضافي|اضافي)')
KW_FLEX = re.compile(r'(مرون|الحضور|الانصراف|تأخير|تاخير|خصم|بصمة|بصمه)')
KW_BREAK = re.compile(r'(استراح|راحة|بريك|رضاعه|رضاع)')
KW_HOURLY = re.compile(r'(مغادره|مغادرة|اذن|إذن).*(ساعيه|ساعية)|مغادرة\s*ساعية')
KW_PROC = re.compile(r'(شراء|مشتريات|عروض|مناقصة|توريد|تأمين|حد|سقف)')
KW_WORKDAYS = re.compile(r'(ايام|أيام).*(العمل|الدوام)|السبت|الاحد|الخميس|الأحد')
KW_PERFORMANCE = re.compile(r'تقييم\s+الأداء|الأداء\s+السنوي|المعايير|محاور')

def classify_intent(q: str) -> str:
    qn = ar_normalize(q)
    if KW_HOURS.search(qn) and KW_RAMADAN.search(qn): return "ramadan_hours"
    if KW_WORKDAYS.search(qn): return "workdays"
    if KW_HOURLY.search(qn): return "hourly_leave"
    if KW_HOURS.search(qn): return "work_hours"
    if "طارئ" in qn or "الطارئه" in qn or "الطارئة" in qn: return "emergency_leave"
    if KW_FLEX.search(qn): return "flex"
    if KW_BREAK.search(qn): return "break"
    if KW_OVERTIME.search(qn): return "overtime"
    if KW_PERFORMANCE.search(qn) or "تقييم" in qn: return "performance"
    if "ترح" in qn or "غير مستخدم" in qn or "غير المستخدم" in qn: return "carryover_leave"
    if KW_LEAVE.search(qn): return "leave"
    if KW_PROC.search(qn): return "procurement"
    if re.search(r'(مياوم|مياومات|بدل\s*سفر|نفقات|فواتير|تذاكر|فندق|اقام)', qn):
        return "per_diem"
    return "general"

INTENT_HINTS = {
    "flex":    ["مرون", "تاخير", "تأخير", "الحضور", "الانصراف", "خصم", "بصمه","بصمة"],
    "break":   ["استراح", "راحة", "فسحه", "فسحة", "بريك","رضاعه","رضاع","دقيقه","دقائق","ساعه","ساعة","نصف","ربع"],
    "overtime":["اضافي", "ساعات اضافيه", "تعويض", "موافقه", "اعتماد", "العطل","العطل الرسميه","نسبه","125","أجر","اجر","احتساب"],
    "leave":   ["اجاز", "اجازه","إجازة","سنويه","سنوية","مرضيه","امومه","حداد","بدون","راتب","طارئ"],
    "carryover_leave": ["غير","مستخدم","غير مستخدم","ترح","سنويه","سقف","حد"],
    "hourly_leave": ["مغادر","اذن","ساعيه","ساعية","حد","اقصى","شهري"],
    "work_hours": ["ساعات", "دوام", "العمل","اوقات","من","الى","حتى","حتي"],
    "ramadan_hours": ["رمضان","ساعات", "دوام", "العمل","اوقات","من","الى","حتى","حتي"],
    "procurement": ["عروض","ثلاث","ثلاثه","3","شراء","سقف","حد","شيكل","₪","دينار","دولار","توريد","مشتريات"],
    "per_diem": ["مياومات","مياومه","بدل","سفر","صرف","نفقات","مصاريف","فواتير","ايصالات","تذاكر","فندق","اقامه","اقامة"],
    "workdays": ["ايام","العمل","الدوام","السبت","الاحد","الأحد","الخميس","من","الى"],
    "emergency_leave": ["اجاز","طارئ","طارئة","فوري","طلب","نموذج"],
    "performance": ["تقييم","الأداء","سنوي","معايير","محاور","أهداف","مؤشرات"]
}

INTENT_MUST = {
    "ramadan_hours": ["رمضان","دوام","ساعات"],
    "work_hours":    ["دوام","ساعات","عمل"],
    "workdays":      ["ايام","العمل","الدوام","السبت","الاحد","الأحد","الخميس"],
    "overtime":      ["اضافي","اجر","أجر","احتساب","موافقه","اعتماد","العطل"],
    "procurement":   ["عروض","عرض","ثلاث","ثلاثه","شراء","سقف","حد"],
    "per_diem":      ["بدل","سفر","مصاريف","نفقات","ايصالات","فواتير","تذاكر","فندق"],
    "break":         ["استراح","راحة","بريك","رضاع","رضاعه","دقائق","ساعة","ساعه","نصف","ربع"],
    "hourly_leave":  ["مغادر","اذن","إذن","ساعيه","ساعية"],
    "carryover_leave": ["غير","مستخدم","ترح","سنويه","سنوية"],
    "emergency_leave": ["اجاز","طارئ"],
    "performance":   ["تقييم","الأداء"],
    "leave":         ["اجاز","اجازه","إجازة"],
    "general":       []
}

def _must_tokens_present(sn_norm: str, intent: str) -> bool:
    req = INTENT_MUST.get(intent, [])
    if not req:
        return True
    return any(tok in sn_norm for tok in req)

# ---------------- Policy-aware time & duration extraction ----------------

TIME_RE = re.compile(
    r'(?:من\s*)?'
    r'(\d{1,2}(?::|\.)?\d{0,2})\s*'
    r'(?:[-–—]|الى|إلى|حتي|حتى)\s*'
    r'(\d{1,2}(?::|\.)?\d{0,2})'
)
DUR_TOKEN = re.compile(r'\b(\d{1,3})\s*(?:دقيقه|دقيقة|دقائق|ساعه|ساعة|ساعات)\b', re.I)

def _normalize_hhmm(tok: str) -> str:
    tok = tok.replace('.', ':')
    if ':' not in tok:
        return f"{int(tok):d}:00"
    h, m = tok.split(':', 1)
    if m == "": m = "00"
    return f"{int(h):d}:{int(m):02d}"

def _to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(':'); return int(h)*60 + int(m)

def extract_all_ranges(text: str, intent: str) -> List[Tuple[int,int]]:
    n = ar_normalize(text)
    ranges: List[Tuple[int,int]] = []
    for m in TIME_RE.finditer(n):
        a = _normalize_hhmm(m.group(1)); b = _normalize_hhmm(m.group(2))
        A, B = _to_minutes(a), _to_minutes(b)
        # fix wrap/inversions gracefully
        if B <= A:
            if A <= 10*60 and B+12*60 > A:
                B = B + 12*60
            else:
                A, B = B, A
        if intent in ("work_hours","ramadan_hours"):
            dur = B - A
            if dur < 6*60 or dur > 11*60:  # typical workday window
                continue
        ranges.append((A, B))
    return ranges

def _score_workday_range(A: int, B: int) -> float:
    dur = B - A
    score = 0.0
    # target 7.5h ~ 9h
    if 6*60 <= dur <= 11*60: score += 3.0
    if 7*60 <= dur <= 9*60: score += 1.0
    # plausible start/end
    if 7*60 <= A <= 10*60: score += 2.5
    if 13*60 <= B <= 18*60: score += 2.5
    # earlier start (8:30 better than 9:30)
    score += max(0.0, (10*60 - min(A, 10*60)) / 60.0) * 0.2
    return score

def pick_best_range(ranges: List[Tuple[int,int]], prefer_workday=True) -> Optional[Tuple[str,str]]:
    if not ranges: return None
    if prefer_workday:
        scored = [(_score_workday_range(A,B), A, B) for (A,B) in ranges]
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        _, A, B = scored[0]
    else:
        ranges.sort()
        A, B = ranges[0]
    return f"{A//60:d}:{A%60:02d}", f"{B//60:d}:{B%60:02d}"

# ---------------- Retrieval & Answering ----------------

def combine_scores(dense_scores, dense_idx, c_scores, w_scores,
                   w_dense=0.60, w_char=0.20, w_word=0.20, topk=20):
    out = []
    for s, i in zip(dense_scores, dense_idx):
        sc = float(s) * w_dense
        if c_scores is not None and len(c_scores) > int(i): sc += float(c_scores[i]) * w_char
        if w_scores is not None and len(w_scores) > int(i): sc += float(w_scores[i]) * w_word
        out.append((sc, int(i)))
    out.sort(key=lambda x: -x[0])
    return out[:topk]

def _intent_keys(intent: str) -> List[str]:
    return INTENT_HINTS.get(intent, [])

def retrieve(index: HybridIndex, q: str, rerank: bool) -> List[Tuple[float,int]]:
    qn = ar_normalize(q)
    restrict_ids: Optional[Set[int]] = None
    gated = False
    if index.hier is not None:
        cand = _hier_candidates(qn, index.hier)
        if cand:
            restrict_ids = cand
            gated = True

    dS, dI = index.dense(qn, topk=60, restrict_ids=restrict_ids)
    globally_fallback = False
    if gated and (dI is None or len(dI) == 0):
        globally_fallback = True
        dS, dI = index.dense(qn, topk=60, restrict_ids=None)

    cS, wS = index.sparse(qn)
    prelim = combine_scores(dS, dI, cS, wS, topk=30)

    if globally_fallback:
        prelim = [(sc*0.85, i) for (sc,i) in prelim]

    if rerank:
        try:
            ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
            if len(prelim) > 1:
                pairs = [(qn, index.chunks[i].text) for _, i in prelim]
                ce_scores = ce.predict(pairs)
                prelim = list(zip(list(map(float, ce_scores)), [i for _, i in prelim]))
                prelim.sort(key=lambda x: -x[0])
            LOG.info("Cross-encoder loaded: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        except Exception as e:
            LOG.warning("CE reranker unavailable (%s). Continuing without.", e)

    return prelim[:8]

def _looks_junky(sn: str) -> bool:
    snn = ar_normalize(sn)
    if len(snn) < 6: return True
    if RIGHTS_LINE.search(sn):
        return True
    letters = sum(ch.isalpha() for ch in snn)
    total = len(snn.replace(" ", ""))
    if total > 0 and letters/total < 0.5:
        return True
    return False

def best_snippet(chunk: Chunk, qnorm: str, intent: str, max_len=260) -> Optional[str]:
    sents = sent_split(chunk.text)
    if not sents: return (chunk.text[:max_len] if chunk.text else None)
    q_terms = set([w for w in qnorm.split() if len(w) >= 3])
    hints = _intent_keys(intent)

    best, best_score = None, -1e9
    for s in sents:
        if _looks_junky(s): 
            continue
        sn = ar_normalize(s)
        if not _must_tokens_present(sn, intent):
            continue
        # time intents: demand numbers + hint + context
        if intent in ("work_hours","ramadan_hours"):
            has_time = bool(re.search(r'\d', sn) and (":" in s or "الى" in sn or "إلى" in sn or "حتى" in sn or "حتي" in sn))
            ctx = ("دوام" in sn) or ("عمل" in sn) or ("ساعات" in sn)
            if not (has_time and ctx): 
                continue
        # strengthen hourly_leave/break/carryover/performance contexts
        if intent == "hourly_leave" and not any(k in sn for k in ["مغادر","اذن","إذن","ساعيه","ساعية"]):
            continue
        if intent == "break" and not any(k in sn for k in ["استراح","راحة","بريك","رضاع"]):
            continue
        if intent == "carryover_leave" and not any(k in sn for k in ["غير","مستخدم","ترح"]):
            continue
        if intent == "performance" and not ("تقييم" in sn or "الأداء" in sn):
            continue

        overlap = len(q_terms & set(sn.split()))
        has_num = bool(re.search(r'\d', sn))
        has_hint = any(h in sn for h in hints)
        score = 0.0
        score += 1.2 * overlap
        if has_num: score += 0.6
        if has_hint: score += 0.8
        if score > best_score:
            best, best_score = s, score

    if not best:
        for s in sents:
            sn = ar_normalize(s)
            if not _looks_junky(s) and _must_tokens_present(sn, intent):
                best = s; break

    if not best:
        return None

    txt = best.strip()
    return clean_display_text(txt[:max_len] + ("…" if len(txt) > max_len else ""))

# ---------------- Answer synthesis helpers ----------------

def _compose_hours_answer(chunks: List[Chunk], hits: List[Tuple[float,int]], intent: str) -> Optional[str]:
    # pass 1: sentence-level
    for _, i in hits:
        sents = sent_split(chunks[i].text)
        for s in sents:
            sn = ar_normalize(s)
            if not (("دوام" in sn) or ("عمل" in sn) or ("ساعات" in sn)):
                continue
            if not _must_tokens_present(sn, intent):
                continue
            ranges = extract_all_ranges(s, intent)
            rng = pick_best_range(ranges, prefer_workday=True)
            if rng:
                a, b = rng
                suffix = " في شهر رمضان" if intent == "ramadan_hours" else ""
                txt = clean_display_text(f"ساعات الدوام{suffix} من {a} إلى {b}.")
                src = f"Data_pdf.pdf - page {chunks[i].page}"
                return rtl_wrap(txt) + "\n" + f"Sources:\n1. {src}"
    # pass 2: chunk-level (handles split lines)
    for _, i in hits:
        t = chunks[i].text
        tn = ar_normalize(t)
        if not (("دوام" in tn) or ("عمل" in tn) or ("ساعات" in tn)):
            continue
        ranges = extract_all_ranges(t, intent)
        rng = pick_best_range(ranges, prefer_workday=True)
        if rng:
            a, b = rng
            suffix = " في شهر رمضان" if intent == "ramadan_hours" else ""
            txt = clean_display_text(f"ساعات الدوام{suffix} من {a} إلى {b}.")
            srcs = [f"{k}. Data_pdf.pdf - page {chunks[j].page}" for k,(_,j) in enumerate(hits,1)]
            return rtl_wrap(txt) + "\n" + "Sources:\n" + "\n".join(srcs)
    return None

def _compose_procurement_threshold(chunks: List[Chunk], hits: List[Tuple[float,int]]) -> Optional[str]:
    for _, i in hits:
        for s in sent_split(chunks[i].text):
            sn = ar_normalize(s)
            if any(w in sn for w in ["عروض","ثلاث","3","سقف","حد","شيكل","دينار","دولار","شراء","مشتريات"]):
                srcs = [f"{k}. Data_pdf.pdf - page {chunks[j].page}" for k, (_, j) in enumerate(hits, 1)]
                return rtl_wrap(clean_display_text(s.strip())) + "\n" + "Sources:\n" + "\n".join(srcs)
    return None

def _compose_overtime(chunks: List[Chunk], hits: List[Tuple[float,int]]) -> Optional[str]:
    found = {"approval": None, "calc": None, "form": None}
    for _, i in hits:
        for s in sent_split(chunks[i].text):
            sn = ar_normalize(s)
            if any(w in sn for w in ["موافق", "اعتماد"]) and found["approval"] is None:
                found["approval"] = clean_display_text(s.strip())
            if any(w in sn for w in ["احتساب", "نسبة", "أجر","اجر","125"]) and found["calc"] is None:
                found["calc"] = clean_display_text(s.strip())
            if any(w in sn for w in ["نموذج", "كشف الساعات"]) and found["form"] is None:
                found["form"] = clean_display_text(s.strip())
    if any(found.values()):
        lines = [v for v in [found["approval"], found["calc"], found["form"]] if v]
        srcs = [f"{k}. Data_pdf.pdf - page {chunks[j].page}" for k, (_, j) in enumerate(hits, 1)]
        return rtl_wrap(" ".join(lines)) + "\n" + "Sources:\n" + "\n".join(srcs)
    return None

def _compose_leave(chunks: List[Chunk], hits: List[Tuple[float,int]]) -> Optional[str]:
    for _, i in hits:
        for s in sent_split(chunks[i].text):
            sn = ar_normalize(s)
            if re.search(r'(اجاز|إجاز|اجازه)', sn) and re.search(r'\d', sn):
                srcs = [f"{k}. Data_pdf.pdf - page {chunks[j].page}" for k, (_, j) in enumerate(hits, 1)]
                return rtl_wrap(clean_display_text(s.strip())) + "\n" + "Sources:\n" + "\n".join(srcs)
    return None

def _compose_break_answer(chunks: List[Chunk], hits: List[Tuple[float,int]]) -> Optional[str]:
    # sentence-level first
    for _, i in hits:
        for s in sent_split(chunks[i].text):
            sn = ar_normalize(s)
            if any(w in sn for w in ["استراح","راحة","بريك","رضاع","رضاعه"]):
                if DUR_TOKEN.search(sn) or any(p in sn for p in ["نصف ساعه","نصف ساعة","ربع ساعه","ربع ساعة"]):
                    srcs = [f"{k}. Data_pdf.pdf - page {chunks[j].page}" for k,(_,j) in enumerate(hits,1)]
                    return rtl_wrap(clean_display_text(s.strip())) + "\n" + "Sources:\n" + "\n".join(srcs)
    # chunk-level fallback
    for _, i in hits:
        tn = ar_normalize(chunks[i].text)
        if any(w in tn for w in ["استراح","راحة","بريك","رضاع"]):
            m = re.search(DUR_TOKEN, tn)
            if m:
                srcs = [f"{k}. Data_pdf.pdf - page {chunks[j].page}" for k,(_,j) in enumerate(hits,1)]
                return rtl_wrap(clean_display_text(chunks[i].text.strip())) + "\n" + "Sources:\n" + "\n".join(srcs)
    return None

def _compose_hourly_leave(chunks: List[Chunk], hits: List[Tuple[float,int]]) -> Optional[str]:
    for _, i in hits:
        for s in sent_split(chunks[i].text):
            sn = ar_normalize(s)
            if any(w in sn for w in ["مغادر","اذن","إذن"]) and (re.search(r'\d', sn) or "حد" in sn or "اقصى" in sn or "شهري" in sn or "ساعيه" in sn or "ساعية" in sn):
                srcs = [f"{k}. Data_pdf.pdf - page {chunks[j].page}" for k,(_,j) in enumerate(hits,1)]
                return rtl_wrap(clean_display_text(s.strip())) + "\n" + "Sources:\n" + "\n".join(srcs)
    # refuse unrelated snippets
    return None

def _compose_leave_carryover(chunks: List[Chunk], hits: List[Tuple[float,int]]) -> Optional[str]:
    for _, i in hits:
        for s in sent_split(chunks[i].text):
            sn = ar_normalize(s)
            if any(k in sn for k in ["غير مستخدم","غير المستخدم","ترح","ترحيل"]) and ("اجاز" in sn or "إجاز" in sn):
                srcs = [f"{k}. Data_pdf.pdf - page {chunks[j].page}" for k,(_,j) in enumerate(hits,1)]
                return rtl_wrap(clean_display_text(s.strip())) + "\n" + "Sources:\n" + "\n".join(srcs)
    return None

def _compose_emergency_leave(chunks: List[Chunk], hits: List[Tuple[float,int]]) -> Optional[str]:
    for _, i in hits:
        for s in sent_split(chunks[i].text):
            sn = ar_normalize(s)
            if ("طارئ" in sn or "طارئه" in sn or "الطارئه" in sn or "الطارئة" in sn) and ("نموذج" in sn or "طلب" in sn or re.search(r'\d', sn)):
                srcs = [f"{k}. Data_pdf.pdf - page {chunks[j].page}" for k,(_,j) in enumerate(hits,1)]
                return rtl_wrap(clean_display_text(s.strip())) + "\n" + "Sources:\n" + "\n".join(srcs)
    return None

def _compose_performance_review(chunks: List[Chunk], hits: List[Tuple[float,int]]) -> Optional[str]:
    for _, i in hits:
        for s in sent_split(chunks[i].text):
            sn = ar_normalize(s)
            if ("تقييم" in sn or "الأداء" in sn) and any(k in sn for k in ["سنوي","معايير","محاور","مرة","سنويه","سنويا","اهداف","مؤشرات"]):
                srcs = [f"{k}. Data_pdf.pdf - page {chunks[j].page}" for k,(_,j) in enumerate(hits,1)]
                return rtl_wrap(clean_display_text(s.strip())) + "\n" + "Sources:\n" + "\n".join(srcs)
    return None

# ---------------- Main answer function ----------------

def answer(q: str, index: HybridIndex, intent: str, use_rerank_flag: bool) -> str:
    hits = retrieve(index, q, use_rerank_flag)
    if not hits:
        return rtl_wrap("لم أعثر على إجابة واضحة في الدليل.")

    chunks = index.chunks

    if intent in ("work_hours", "ramadan_hours"):
        composed = _compose_hours_answer(chunks, hits, intent)
        if composed: return composed

    if intent == "break":
        composed = _compose_break_answer(chunks, hits)
        if composed: return composed

    if intent == "hourly_leave":
        composed = _compose_hourly_leave(chunks, hits)
        if composed: return composed

    if intent == "carryover_leave":
        composed = _compose_leave_carryover(chunks, hits)
        if composed: return composed

    if intent == "emergency_leave":
        composed = _compose_emergency_leave(chunks, hits)
        if composed: return composed

    if intent == "performance":
        composed = _compose_performance_review(chunks, hits)
        if composed: return composed

    if intent == "procurement":
        composed = _compose_procurement_threshold(chunks, hits)
        if composed: return composed

    if intent == "overtime":
        composed = _compose_overtime(chunks, hits)
        if composed: return composed

    if intent == "leave":
        composed = _compose_leave(chunks, hits)
        if composed: return composed

    # Fallback: pick best snippet with intent-aware scoring
    _, idx0 = hits[0]
    sn = best_snippet(chunks[idx0], ar_normalize(q), intent)
    if not sn:
        return rtl_wrap("لم أعثر على إجابة مناسبة.")
    srcs = [f"{k}. Data_pdf.pdf - page {chunks[i].page}" for k, (_, i) in enumerate(hits, 1)]
    return rtl_wrap(clean_display_text(sn)) + "\n" + "Sources:\n" + "\n".join(srcs)

# ---------------- CLI & sanity ----------------

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

def run_sanity(index: HybridIndex, use_rerank_flag: bool):
    LOG.info("Running sanity suite...\n")
    passed = 0
    for q in SANITY_PROMPTS:
        intent = classify_intent(q)
        out = answer(q, index, intent, use_rerank_flag)
        ok = "Sources:" in out and "لم أعثر" not in out
        if ok: passed += 1
        print(("✅ " if ok else "❌ ") + f"Q: {q}")
        print(out)
        print("-"*80)
    print(f"Concept-check PASS: {passed}/{len(SANITY_PROMPTS)}")

def interactive_loop(index: HybridIndex, use_rerank_flag: bool):
    print("Ready.")
    print("اسأل عن سياسات الدليل (عربي). اكتب 'exit' للخروج.\n")
    while True:
        try:
            q = input("سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break
        if not q: continue
        if q.lower() in ("exit","quit","q"): print("Exiting."); break
        LOG.info("Searching...")
        intent = classify_intent(q)
        print(answer(q, index, intent, use_rerank_flag))
        print("-"*66)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default=CHUNKS_PATH, help="Path to chunks (JSONL or JSON)")
    ap.add_argument("--hier-index", type=str, default="heading_inverted_index.json", help="(Optional) hierarchical inverted index path")
    ap.add_argument("--aliases", type=str, default="section_aliases.json", help="(Optional) section aliases")
    ap.add_argument("--sanity", action="store_true", help="Run sanity prompts and exit")
    ap.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    ap.add_argument("--save-index", type=str, default=None, help="Directory to save index artifacts")
    ap.add_argument("--load-index", type=str, default=None, help="Directory to load index artifacts from")
    ap.add_argument("--model", type=str, default=None, help="SentenceTransformer model ID override")
    args = ap.parse_args()

    hier = load_hierarchy(args.hier_index, args.aliases)
    chunks, chunks_hash = load_chunks(path=args.chunks)
    try:
        index = HybridIndex(chunks, chunks_hash, hier=hier, model_name=args.model) if args.model else HybridIndex(chunks, chunks_hash, hier=hier)
    except TypeError:
        index = HybridIndex(chunks, chunks_hash, hier=hier)

    loaded = False
    if args.load_index:
        loaded = index.load(args.load_index)

    if not loaded:
        index.build()
        if args.save_index:
            index.save(args.save_index)

    LOG.info("Cross-encoder reranker %s.", "ENABLED.." if args.rerank else "disabled...")
    LOG.info("Ready.")

    if args.sanity:
        run_sanity(index, use_rerank_flag=args.rerank)
        return

    interactive_loop(index, use_rerank_flag=args.rerank)

if __name__ == "__main__":
    main()

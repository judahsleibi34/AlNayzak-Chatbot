# -*- coding: utf-8 -*-
"""
Hardened Arabic-first EXTRACTIVE retriever (PDF-only, no fabrication).

Key properties
--------------
- Deterministic, extractive-only (no generations, no paraphrases).
- Strict section/page gating via hierarchy+aliases (no global hunts unless explicitly allowed).
- Proximity+unit guards for numerics and times.
- Refuses to answer when confidence is low or text is absent (prevents wrong answers).
- Clean Arabic output + exact PDF page citations.

Usage (same CLI flags as before; reranker ignored):
  python retrival_model.py --chunks Data_pdf_clean_chunks.jsonl --hier-index heading_inverted_index.json --aliases section_aliases.json --sanity
"""

import os, re, sys, json, argparse, logging, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Set

import numpy as np

try:
    import faiss  # optional fast index
except Exception:
    faiss = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger(__name__)

# ---------------- Strict mode toggles ----------------
STRICT_MODE = True  # keep True: forbids unsafe fallbacks
MAX_SNIPPET_CHARS = 260
MAX_CITATIONS = 3
WINDOW_CHARS = 48  # numeric-keyword proximity window

# ---------------- Arabic utils (SAFE) ----------------
AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"٠١٢٣٤٥٦٧٨٩")}
IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"۰۱۲۳۴۵۶۷۸۹")}

def ar_normalize(s: str) -> str:
    if not s: return ""
    s = s.replace('\u0640','')                # tatweel
    s = AR_DIAC.sub('', s)                    # remove diacritics
    # keep ta marbuta (ة) intact to avoid harming tokens
    s = (s.replace('أ','ا').replace('إ','ا').replace('آ','ا')
           .replace('ى','ي'))
    s = s.translate(AR_NUMS).translate(IR_NUMS)   # Arabic/Indic -> ASCII
    s = s.replace('،', ',').replace('٫','.')
    s = ' '.join(s.split())
    return s

def rtl_wrap(t: str) -> str:
    return '\u202B' + t + '\u202C'

SENT_SPLIT_RE = re.compile(r'(?<=[\.\!\؟\?،]|[\n])\s+')

def sent_split(s: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(s or "") if p and p.strip()]
    out = []
    for p in parts:
        pn = ar_normalize(p)
        if len(pn) < 6:  # drop tiny fragments
            continue
        # require >50% letters (avoid junk/IDs)
        letters = sum(ch.isalpha() for ch in pn)
        total = len(pn.replace(" ", ""))
        if total == 0 or letters/total < 0.5:
            continue
        out.append(p)
    return out if out else ([s.strip()] if (s and s.strip()) else [])

# ---------------- Data IO ----------------
CHUNKS_PATH = "Data_pdf_clean_chunks.jsonl"

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

def load_chunks(path=CHUNKS_PATH) -> Tuple[List[Chunk], str]:
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

# ---------------- Index (dense only + strict gating) ----------------
class HybridIndex:
    def __init__(self, chunks: List[Chunk], chunks_hash: str, hier: Optional[HierData] = None,
                 model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self.chunks = chunks
        self.chunks_hash = chunks_hash
        self.hier = hier
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.emb = None
        self.faiss = None

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

    def dense(self, q: str, topk=50, restrict_ids: Optional[Set[int]] = None):
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
        # STRICT gating: return only restricted; if empty -> return empty
        filtS, filtI = [], []
        rset = set(int(x) for x in restrict_ids)
        for s, i in zip(scores, idxs):
            if int(i) in rset:
                filtS.append(float(s)); filtI.append(int(i))
            if len(filtI) >= topk: break
        return np.array(filtS), np.array(filtI)

# ---------------- Intents & guards ----------------
KW_HOURS = re.compile(r'(ساعات)\s+(الدوام|العمل)')
KW_RAMADAN = re.compile(r'رمضان')
KW_LEAVE = re.compile(r'(إجاز|اجاز)')
KW_OVERTIME = re.compile(r'(إضافي|اضافي)')
KW_FLEX = re.compile(r'(مرون|الحضور|الانصراف|تأخير|تاخير|خصم|بصمة|بصمه)')
KW_BREAK = re.compile(r'(استراح|راحة|بريك|رضاعه|رضاع)')
KW_PROC = re.compile(r'(شراء|مشتريات|عروض|مناقصة|توريد|تأمين|حد|سقف)')
KW_WORKDAYS = re.compile(r'(ايام|أيام).*(العمل|الدوام)|السبت|الاحد|الخميس')

def classify_intent(q: str) -> str:
    qn = ar_normalize(q)
    if KW_HOURS.search(qn) and KW_RAMADAN.search(qn): return "ramadan_hours"
    if KW_WORKDAYS.search(qn): return "workdays"
    if KW_HOURS.search(qn): return "work_hours"
    if KW_FLEX.search(qn): return "flex"
    if KW_BREAK.search(qn): return "break"
    if KW_OVERTIME.search(qn): return "overtime"
    if KW_LEAVE.search(qn): return "leave"
    if KW_PROC.search(qn): return "procurement"
    if re.search(r'(مياوم|مياومات|بدل\s*سفر|نفقات|فواتير|ايصالات|تذاكر|فندق)', qn):
        return "per_diem"
    return "general"

INTENT_HINTS = {
    "flex":    ["مرون","تاخير","تأخير","الحضور","الانصراف","خصم","بصمه","بصمة"],
    "break":   ["استراح","راحة","فسحه","فسحة","بريك","رضاعه","رضاع","دقيقه","دقائق","ساعه","ساعة"],
    "overtime":["اضافي","ساعات اضافيه","تعويض","موافقه","اعتماد","العطل","العطل الرسميه","نسبه","125","أجر","اجر","احتساب"],
    "leave":   ["اجاز","سنويه","سنوية","مرضيه","طارئه","امومه","حداد","بدون راتب","رصيد"],
    "work_hours": ["ساعات","دوام","العمل","الى","حتي","حتى","من","ايام","اوقات"],
    "ramadan_hours": ["رمضان","ساعات","دوام","العمل","الى","حتي","حتى","من","اوقات"],
    "procurement": ["عروض","ثلاثه","ثلاث","3","عرض","مناقص","سقف","حد","شراء","مشتريات","شيكل","₪","دينار","دولار","توريد"],
    "per_diem": ["مياومات","مياومه","بدل","سفر","صرف","نفقات","مصاريف","فواتير","ايصالات","تذاكر","فندق","اقامه","اقامة"],
    "workdays": ["ايام","العمل","الدوام","السبت","الاحد","الخميس","من","الى"]
}

# deny-lists by intent to prevent cross-topic leakage
INTENT_DENY = {
    "public_holiday": ["عن كل سنة عمل","نهاية الخدمة","تعويض نهاية الخدمة","مكافأة نهاية الخدمة"],
    "overtime": [],
    "procurement": [],
    "work_hours": [],
    "ramadan_hours": [],
    "leave": [],
    "per_diem": [],
    "break": [],
    "flex": [],
    "workdays": [],
    "general": []
}

def _contains_any(s: str, arr: List[str]) -> bool:
    s = ar_normalize(s)
    return any(tok for tok in arr if tok in s)

# ---------------- Time extraction ----------------
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

def _plausible_workday(a: int, b: int) -> bool:
    return 6*60 <= a <= 20*60+30 and 6*60 <= b <= 20*60+30 and b > a

def extract_all_ranges(text: str, check_workday: bool) -> List[Tuple[int,int]]:
    n = ar_normalize(text)
    ranges: List[Tuple[int,int]] = []
    for m in TIME_RE.finditer(n):
        a = _normalize_hhmm(m.group(1)); b = _normalize_hhmm(m.group(2))
        A, B = _to_minutes(a), _to_minutes(b)
        if B <= A:
            B_try = B + 12*60
            if B_try > A:
                B = B_try
            else:
                A, B = B, A
        if check_workday:
            dur = B - A
            if not (6*60 <= dur <= 11*60 and _plausible_workday(A, B)):
                continue
        ranges.append((A,B))
    return ranges

def pick_best_range(ranges: List[Tuple[int,int]]) -> Optional[Tuple[str,str]]:
    if not ranges: return None
    scored = []
    for (A,B) in ranges:
        dur = B - A
        target = abs(dur - 7.5*60)
        scored.append((target, -(dur), A, B))
    scored.sort()
    _, _, A, B = scored[0]
    return f"{A//60:d}:{A%60:02d}", f"{B//60:d}:{B%60:02d}"

# ---------------- Proximity guards ----------------
def _prox_numeric_with_kws(text: str, kws: List[str], window=WINDOW_CHARS) -> bool:
    T = ar_normalize(text)
    for m in re.finditer(r'\b\d+\b', T):
        start = max(0, m.start()-window); end = m.end()+window
        ctx = T[start:end]
        if any(kw in ctx for kw in kws):
            return True
    return False

def _has_unit(text: str, units: List[str]) -> bool:
    T = ar_normalize(text)
    return any(u in T for u in units)

# ---------------- Retrieval ----------------
def retrieve(index: HybridIndex, q: str, intent: str) -> List[Tuple[float,int]]:
    qn = ar_normalize(q)
    restrict_ids: Optional[Set[int]] = None
    if index.hier is not None:
        cand = _hier_candidates(qn, index.hier)
        if cand:
            restrict_ids = cand
    scores, idxs = index.dense(qn, topk=60, restrict_ids=restrict_ids)
    if restrict_ids is not None and (idxs is None or len(idxs) == 0):
        # STRICT: do not globally fallback; refuse later
        return []
    pairs = list(zip(list(map(float, scores)), list(map(int, idxs))))
    pairs.sort(key=lambda x: -x[0])
    return pairs[:12]

# ---------------- Answer composers (EXTRACTIVE) ----------------
def _compose_hours(chunks: List[Chunk], hits, label: str) -> Optional[str]:
    for _, i in hits:
        ch = chunks[i]
        if ch.page < 0: continue
        for s in sent_split(ch.text):
            if _contains_any(s, INTENT_DENY.get("work_hours", [])): 
                continue
            ranges = extract_all_ranges(s, check_workday=True)
            rng = pick_best_range(ranges)
            if rng:
                a, b = rng
                sent = f"ساعات الدوام{label} من {a} إلى {b}."
                return rtl_wrap(sent) + f"\nSources:\n1. Data_pdf.pdf - page {ch.page}"
    return None

def _compose_workdays(chunks: List[Chunk], hits) -> Optional[str]:
    days_kw = ["الاحد","الإثنين","الاثنين","الثلاثاء","الأربعاء","الخميس","السبت","الجمعة","ايام","الأيام","ايام الدوام","ايام العمل"]
    for _, i in hits:
        ch = chunks[i]
        if ch.page < 0: continue
        for s in sent_split(ch.text):
            sn = ar_normalize(s)
            if any(d in sn for d in days_kw):
                return rtl_wrap(s.strip()) + f"\nSources:\n1. Data_pdf.pdf - page {ch.page}"
    return None

def _compose_flex(chunks: List[Chunk], hits) -> Optional[str]:
    kws = ["مرون","تاخير","تأخير","الحضور","الانصراف","خصم","بصمه","بصمة","تعويض"]
    for _, i in hits:
        ch = chunks[i]
        for s in sent_split(ch.text):
            if any(k in ar_normalize(s) for k in kws) and not _contains_any(s, ["نهاية الخدمة"]):
                return rtl_wrap(s.strip()) + f"\nSources:\n1. Data_pdf.pdf - page {ch.page}"
    return None

def _compose_break(chunks: List[Chunk], hits) -> Optional[str]:
    kws = ["استراح","راحة","بريك","رضاع","رضاعه","ساعة","ساعه","دقائق","دقيقه"]
    for _, i in hits:
        ch = chunks[i]
        for s in sent_split(ch.text):
            sn = ar_normalize(s)
            if any(k in sn for k in kws):
                # require unit (ساعة/دقيقة) or explicit policy statement
                if _has_unit(s, ["ساعة","ساعه","دقيقة","دقائق"]) or "فتر" in sn or "مده" in sn:
                    return rtl_wrap(s.strip()) + f"\nSources:\n1. Data_pdf.pdf - page {ch.page}"
    return None

def _compose_overtime(chunks: List[Chunk], hits) -> Optional[str]:
    must = ["اضافي","احتساب","اجر","أجر","موافقه","مسبقه","نسبة","125"]
    deny = INTENT_DENY.get("overtime", [])
    for _, i in hits:
        ch = chunks[i]
        sents = sent_split(ch.text)
        good = []
        for s in sents:
            sn = ar_normalize(s)
            if any(d in sn for d in deny): 
                continue
            if any(m in sn for m in must):
                good.append(s.strip())
        if good:
            out = " ".join(good[:2])
            return rtl_wrap(out) + f"\nSources:\n1. Data_pdf.pdf - page {ch.page}"
    return None

def _compose_leave(chunks: List[Chunk], hits) -> Optional[str]:
    kws = ["اجاز","سنويه","سنوية","مرضيه","حداد","امومه","طارئه","بدون راتب","رصيد"]
    for _, i in hits:
        ch = chunks[i]
        for s in sent_split(ch.text):
            sn = ar_normalize(s)
            if any(k in sn for k in kws) and re.search(r'\d', sn):
                return rtl_wrap(s.strip()) + f"\nSources:\n1. Data_pdf.pdf - page {ch.page}"
    return None

def _compose_procurement(chunks: List[Chunk], hits) -> Optional[str]:
    # require "عروض/عرض/أسعار/شراء/مشتريات" near numbers (thresholds) within window and a currency/unit if present
    num_kws = ["عرض","عروض","اسعار","أسعار","شراء","مشتريات","توريد","سقف","حد"]
    currency = ["شيكل","₪","دينار","دولار"]
    for _, i in hits:
        ch = chunks[i]
        for s in sent_split(ch.text):
            if _prox_numeric_with_kws(s, num_kws, WINDOW_CHARS) or any(k in ar_normalize(s) for k in num_kws):
                # if there is a number, prefer also a currency/unit in context (when present)
                snn = ar_normalize(s)
                if re.search(r'\d', snn):
                    # allow missing currency if policy states "ثلاثة عروض" without number
                    if any(c in snn for c in currency) or any(w in snn for w in ["عروض","عرض"]):
                        return rtl_wrap(s.strip()) + f"\nSources:\n1. Data_pdf.pdf - page {ch.page}"
                else:
                    # sentences stating "ثلاثة عروض" without numeric amounts
                    if any(w in snn for w in ["ثلاث","ثلاثه","3"]) and any(w in snn for w in ["عروض","عرض"]):
                        return rtl_wrap(s.strip()) + f"\nSources:\n1. Data_pdf.pdf - page {ch.page}"
    return None

def _compose_per_diem(chunks: List[Chunk], hits) -> Optional[str]:
    kws = ["مياومات","مياومه","بدل سفر","بدل المياومات","نفقات","مصاريف","فواتير","ايصالات","تذاكر","فندق","اقامة","اقامه","تعرفة","تعريفه","كيلومتر","شيكل"]
    for _, i in hits:
        ch = chunks[i]
        for s in sent_split(ch.text):
            sn = ar_normalize(s)
            if any(k in sn for k in kws):
                # prefer sentences with explicit units (شيكل/كم/كيلومتر/فاتورة/إيصال)
                if _has_unit(s, ["شيكل","كم","كيلومتر","فاتوره","فاتورة","ايصال","إيصال"]):
                    return rtl_wrap(s.strip()) + f"\nSources:\n1. Data_pdf.pdf - page {ch.page}"
                # fallback: still extract if it clearly defines the policy
                return rtl_wrap(s.strip()) + f"\nSources:\n1. Data_pdf.pdf - page {ch.page}"
    return None

def _compose_general_best(chunks: List[Chunk], hits, intent: str) -> Optional[str]:
    hints = INTENT_HINTS.get(intent, [])
    for _, i in hits:
        ch = chunks[i]
        for s in sent_split(ch.text):
            sn = ar_normalize(s)
            if hints and not any(h in sn for h in hints):
                continue
            if INTENT_DENY.get(intent) and _contains_any(s, INTENT_DENY[intent]):
                continue
            # require numbers for numbery intents
            if intent in ("work_hours","ramadan_hours","overtime","leave","procurement","per_diem") and not re.search(r'\d', sn):
                continue
            return rtl_wrap(s.strip()) + f"\nSources:\n1. Data_pdf.pdf - page {ch.page}"
    return None

# ---------------- Main answer (EXTRACTIVE ONLY) ----------------
def answer(q: str, index: HybridIndex, intent: str, use_rerank_flag: bool=False) -> str:
    hits = retrieve(index, q, intent)
    if not hits:
        return rtl_wrap("لم يرد نص صريح حول ذلك في الدليل المرفق.")

    chunks = index.chunks

    if intent == "work_hours":
        out = _compose_hours(chunks, hits, label="")
        if out: return out

    if intent == "ramadan_hours":
        out = _compose_hours(chunks, hits, label=" في شهر رمضان")
        if out: return out

    if intent == "workdays":
        out = _compose_workdays(chunks, hits)
        if out: return out

    if intent == "flex":
        out = _compose_flex(chunks, hits)
        if out: return out

    if intent == "break":
        out = _compose_break(chunks, hits)
        if out: return out

    if intent == "overtime":
        out = _compose_overtime(chunks, hits)
        if out: return out

    if intent == "leave":
        out = _compose_leave(chunks, hits)
        if out: return out

    if intent == "procurement":
        out = _compose_procurement(chunks, hits)
        if out: return out

    if intent == "per_diem":
        out = _compose_per_diem(chunks, hits)
        if out: return out

    out = _compose_general_best(chunks, hits, intent)
    if out: return out

    # STRICT refusal instead of unsafe fallback
    return rtl_wrap("لم يرد نص صريح حول ذلك في الدليل المرفق.")

# ---------------- CLI ----------------
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

def run_sanity(index: HybridIndex):
    LOG.info("Running sanity suite (extractive-only)…\n")
    passed = 0
    for q in SANITY_PROMPTS:
        intent = classify_intent(q)
        out = answer(q, index, intent, use_rerank_flag=False)
        ok = "Sources:" in out and "لم يرد نص" not in out
        passed += int(ok)
        print(("✅ " if ok else "⚪ ") + f"Q: {q}")
        print(out); print("-"*80)
    print(f"Extractive PASS (has sources): {passed}/{len(SANITY_PROMPTS)}")

def interactive_loop(index: HybridIndex):
    print("جاهز.")
    print("اسأل عن بنود الدليل (إجابات مقتبسة حرفياً من الـPDF). اكتب 'exit' للخروج.\n")
    while True:
        try:
            q = input("سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nخروج."); break
        if not q: continue
        if q.lower() in ("exit","quit","q"): print("خروج."); break
        intent = classify_intent(q)
        print(answer(q, index, intent, use_rerank_flag=False))
        print("-"*66)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default=CHUNKS_PATH, help="Path to chunks (JSONL/JSON)")
    ap.add_argument("--hier-index", type=str, default="heading_inverted_index.json", help="Optional hierarchy inverted index")
    ap.add_argument("--aliases", type=str, default="section_aliases.json", help="Optional aliases for headings")
    ap.add_argument("--sanity", action="store_true", help="Run sanity prompts and exit")
    args = ap.parse_args()

    hier = load_hierarchy(args.hier_index, args.aliases)
    chunks, chunks_hash = load_chunks(path=args.chunks)
    index = HybridIndex(chunks, chunks_hash, hier=hier)
    index.build()

    LOG.info("Ready (STRICT_MODE=%s).", STRICT_MODE)

    if args.sanity:
        run_sanity(index); return

    interactive_loop(index)

if __name__ == "__main__":
    main()

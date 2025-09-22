# -*- coding: utf-8 -*-
"""
retrival_model.py — lightweight hybrid retriever (pure Python)

Exports:
- load_hierarchy(path, aliases_path) -> dict|None
- load_chunks(path: str) -> (List[Chunk], chunks_hash: str)
- HybridIndex(chunks, chunks_hash, hier=None, model_name=None)
    - build(), search(query, top_k=12)
    - save(dir_path), load(dir_path)  # optional, best-effort
- classify_intent(question: str) -> str
- answer(question: str, index: HybridIndex, intent: str, use_rerank_flag: bool=False) -> str
- SANITY_PROMPTS: List[str]
- clean_display_text(text: str) -> str

Notes:
- Simple TF-IDF cosine with Arabic normalization/tokenization
- Intent anchors used for shallow on-topic rerank + snippet extraction
- Robust chunk loader for JSONL/JSON with common key variants
"""

from __future__ import annotations
import os, re, json, math, hashlib, pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

# =========================
# Arabic normalization utils
# =========================

_AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
_AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"٠١٢٣٤٥٦٧٨٩")}
_IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"۰۱۲۳۴۵۶۷۸۹")}
_JUNK_UNI = re.compile(r'/uni[0-9A-Fa-f]{4}')
_CTRL_WS  = re.compile(r'[\u200b\u200c\u200d\u200e\u200f]')
_TATWEEL  = '\u0640'

def _ar_norm_core(s: str) -> str:
    if not s: return ""
    s = s.replace(_TATWEEL, '')
    s = _AR_DIAC.sub('', s)
    s = (s.replace('أ','ا').replace('إ','ا').replace('آ','ا')
           .replace('ى','ي'))
    s = s.translate(_AR_NUMS).translate(_IR_NUMS)
    s = s.replace('،', ',').replace('٫','.')
    s = _JUNK_UNI.sub('', s)
    s = _CTRL_WS.sub('', s)
    # collapse whitespace
    return ' '.join(s.split())

def clean_display_text(s: str) -> str:
    """Stronger sanitizer for PDF artifacts (used by NewRag before printing)."""
    return _ar_norm_core(s)

_TOKEN_SPLIT = re.compile(r'[^0-9A-Za-z\u0621-\u064A]+')

def _tokenize_ar(s: str) -> List[str]:
    s = _ar_norm_core(s)
    toks = [t for t in _TOKEN_SPLIT.split(s) if t]
    return toks

# =========================
# Intents & anchors
# =========================

INTENT_ANCHORS: Dict[str, List[str]] = {
    "work_hours": ["ساعات","دوام","الدوام","العمل"],
    "ramadan_hours": ["ساعات","دوام","الدوام","رمضان"],
    "break": ["استراحة","راحة","بريك"],
    "workdays": ["ايام","أيام","العمل","الدوام","السبت","الاحد","الأحد","الخميس","الجمعة"],
    "overtime": ["ساعات اضافية","ساعات إضافية","العمل الاضافي","العمل الإضافي","اضافي","إضافي","وقت إضافي","موافقة"],
    "per_diem": ["مياومة","مياومات","بدل سفر","بدل المياومة","مصاريف سفر"],
    "procurement": ["شراء","مشتريات","توريد","عروض","عرض","مناقصة","توريدات"],
    "leave": ["اجازة","إجازة","عطلة","مرضي","سنوية","حداد","حج","امومة","أمومة","طارئة"],
    "gifts": ["هدايا","ضيافة","قبول","إبلاغ","ابلاغ"],
}

TIME_RANGE = re.compile(
    r'(?:من\s*)?'
    r'(\d{1,2}(?::|\.)?\d{0,2})\s*(?:[-–—]|الى|إلى|حتى|حتي)\s*'
    r'(\d{1,2}(?::|\.)?\d{0,2})'
)
TIME_TOKEN = re.compile(r'\b\d{1,2}[:\.]\d{0,2}\b')
DUR_TOKEN  = re.compile(r'\b(\d{1,3})\s*(?:دقيقة|دقائق|ساعة|ساعات)\b', re.I)
ANY_DIGIT  = re.compile(r'\d')

# =========================
# Data model for chunks
# =========================

@dataclass
class Chunk:
    text: str
    page: int
    source: str = "Data_pdf.pdf"
    meta: Optional[Dict[str, Any]] = None

# =========================
# Loaders
# =========================

def load_hierarchy(path: Optional[str], aliases_path: Optional[str]=None) -> Optional[Dict[str, Any]]:
    """Best-effort loader; index can work without this."""
    if not path and not aliases_path:
        return None
    out: Dict[str, Any] = {}
    try:
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                out["inverted"] = json.load(f)
    except Exception:
        out["inverted"] = {}
    try:
        if aliases_path and os.path.exists(aliases_path):
            with open(aliases_path, 'r', encoding='utf-8') as f:
                out["aliases"] = json.load(f)
    except Exception:
        out["aliases"] = {}
    return out or None

def _guess_text_key(d: Dict[str, Any]) -> Optional[str]:
    for k in ("text","chunk","content","body"):
        if k in d: return k
    return None

def _guess_page_key(d: Dict[str, Any]) -> Optional[str]:
    for k in ("page","page_num","pageNumber","pageno"):
        if k in d: return k
    return None

def _guess_source_key(d: Dict[str, Any]) -> Optional[str]:
    for k in ("source","file","doc","path","document"):
        if k in d: return k
    return None

def load_chunks(path: str) -> Tuple[List[Chunk], str]:
    """Load JSONL/JSON chunks; returns (chunks, content_hash)."""
    chunks: List[Chunk] = []
    # read file
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                tkey = _guess_text_key(d); pkey = _guess_page_key(d); skey = _guess_source_key(d)
                text  = d.get(tkey or "text","")
                page  = int(d.get(pkey or "page", -1))
                src   = d.get(skey or "source","Data_pdf.pdf")
                chunks.append(Chunk(text=text, page=page, source=os.path.basename(src), meta=d))
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and "chunks" in data:
            data = data["chunks"]
        if not isinstance(data, list):
            raise ValueError("Unsupported JSON shape for chunks")
        for d in data:
            tkey = _guess_text_key(d); pkey = _guess_page_key(d); skey = _guess_source_key(d)
            text  = d.get(tkey or "text","")
            page  = int(d.get(pkey or "page", -1))
            src   = d.get(skey or "source","Data_pdf.pdf")
            chunks.append(Chunk(text=text, page=page, source=os.path.basename(src), meta=d))

    # compute a short hash to help cache/save
    h = hashlib.sha1()
    for c in chunks:
        h.update(str(c.page).encode('utf-8', 'ignore'))
        h.update(str(len(c.text)).encode('utf-8', 'ignore'))
    return chunks, h.hexdigest()[:12]

# =========================
# HybridIndex (TF-IDF cosine)
# =========================

class HybridIndex:
    def __init__(self, chunks: List[Chunk], chunks_hash: str, hier: Optional[Dict[str, Any]]=None, model_name: Optional[str]=None):
        self.chunks = chunks
        self.chunks_hash = chunks_hash
        self.hier = hier or {}
        # index stores: vocab -> idf, doc_vectors: List[Dict[token,weight]], norms, doc_meta
        self._idf: Dict[str, float] = {}
        self._doc_vecs: List[Dict[str,float]] = []
        self._norms: List[float] = []
        self._doc_meta: List[Tuple[int, str]] = []  # (page, source)

    def build(self) -> None:
        # DF
        df: Dict[str, int] = {}
        docs_tokens: List[List[str]] = []
        for ch in self.chunks:
            toks = _tokenize_ar(ch.text)
            docs_tokens.append(toks)
            seen = set()
            for t in toks:
                if t not in seen:
                    df[t] = df.get(t, 0) + 1
                    seen.add(t)
        N = max(1, len(self.chunks))
        # IDF
        idf: Dict[str,float] = {}
        for t, c in df.items():
            # BM25-ish idf; add-1 smoothing
            idf[t] = math.log(1 + (N - c + 0.5)/(c + 0.5))
        self._idf = idf

        # TF-IDF (log-tf)
        self._doc_vecs = []
        self._norms = []
        self._doc_meta = []
        for i, toks in enumerate(docs_tokens):
            tf: Dict[str,int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            vec: Dict[str,float] = {}
            for t, f in tf.items():
                w = (1 + math.log(f)) * idf.get(t, 0.0)
                if w != 0.0:
                    vec[t] = w
            norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
            self._doc_vecs.append(vec)
            self._norms.append(norm)
            ch = self.chunks[i]
            self._doc_meta.append((ch.page, ch.source))

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "tfidf.pkl"), "wb") as f:
            pickle.dump({
                "idf": self._idf,
                "norms": self._norms,
                "meta": self._doc_meta,
                "chunks_hash": self.chunks_hash,
            }, f)

    def load(self, dir_path: str) -> bool:
        path = os.path.join(dir_path, "tfidf.pkl")
        if not os.path.exists(path): return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        # basic freshness check
        if data.get("chunks_hash") != self.chunks_hash:
            return False
        self._idf = data["idf"]
        self._norms = data["norms"]
        self._doc_meta = data["meta"]
        # re-build doc vectors (we need raw tokens)
        # safer to rebuild everything to prevent mismatch
        self.build()
        return True

    def _query_vec(self, query: str) -> Dict[str,float]:
        toks = _tokenize_ar(query)
        tf: Dict[str,int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        vec: Dict[str,float] = {}
        for t, f in tf.items():
            w = (1 + math.log(f)) * self._idf.get(t, 0.0)
            if w != 0.0:
                vec[t] = w
        return vec

    def search(self, query: str, top_k: int = 12) -> List[Tuple[float, int]]:
        """Return [(score, doc_idx)] best matches."""
        qvec = self._query_vec(query)
        if not qvec:  # empty after normalization
            return []
        # dot
        scores: List[Tuple[float,int]] = []
        qnorm = math.sqrt(sum(v*v for v in qvec.values())) or 1.0
        for i, dvec in enumerate(self._doc_vecs):
            dot = 0.0
            # iterate smaller dict
            if len(qvec) < len(dvec):
                for t, w in qvec.items():
                    dv = dvec.get(t)
                    if dv: dot += w * dv
            else:
                for t, dv in dvec.items():
                    qw = qvec.get(t)
                    if qw: dot += qw * dv
            score = dot / (qnorm * self._norms[i])
            scores.append((score, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

# =========================
# Intent classification
# =========================

def classify_intent(q: str) -> str:
    qn = _ar_norm_core(q)
    # explicit
    if "رمضان" in qn:
        return "ramadan_hours"
    if any(w in qn for w in ["دوام","ساعات العمل","اوقات العمل","ساعات الدوام"]):
        # if also "استراحة" present, it's break
        if any(w in qn for w in ["استراحة","راحة","بريك"]):
            return "break"
        return "work_hours"
    if any(w in qn for w in ["استراحة","راحة","بريك"]):
        return "break"
    if any(w in qn for w in ["ايام","أيام","السبت","الاحد","الأحد","الخميس","الجمعة"]):
        return "workdays"
    if any(w in qn for w in ["ساعات اضافية","ساعات إضافية","العمل الاضافي","العمل الإضافي","اضافي","إضافي"]):
        return "overtime"
    if any(w in qn for w in ["مياومة","مياومات","بدل سفر","بدل المياومة"]):
        return "per_diem"
    if any(w in qn for w in ["عروض","عرض","شراء","مشتريات","توريد","توريدات","مناقصة"]):
        return "procurement"
    if any(w in qn for w in ["اجازة","إجازة","عطلة","مرضي","سنوية","حداد","حج","امومة","أمومة","طارئة"]):
        return "leave"
    if any(w in qn for w in ["هدايا","ضيافة"]):
        return "gifts"
    # fallback
    return "general"

# =========================
# Snippet extraction helpers
# =========================

def _split_sentences_ar(text: str) -> List[str]:
    # rough splitter on Arabic/Latin punctuation and newlines
    text = clean_display_text(text)
    parts = re.split(r'[\.\!\؟\?\n]+', text)
    return [p.strip() for p in parts if p.strip()]

def _prefer_time_line(lines: List[str], want_ramadan: bool=False) -> Optional[str]:
    best = None
    best_score = -1
    for ln in lines:
        tn = _ar_norm_core(ln)
        score = 0
        if TIME_RANGE.search(tn): score += 3
        if TIME_TOKEN.search(tn): score += 1
        if "دوام" in tn or "العمل" in tn: score += 1
        if want_ramadan and "رمضان" in tn: score += 1
        if score > best_score:
            best_score = score; best = ln
    return best

def _prefer_anchored_numeric(lines: List[str], anchors: List[str]) -> Optional[str]:
    best = None; best_score = -1
    for ln in lines:
        tn = _ar_norm_core(ln)
        score = 0
        if ANY_DIGIT.search(tn): score += 1
        if any(a in tn for a in anchors): score += 2
        if score > best_score:
            best_score = score; best = ln
    return best

def _anchor_score(text: str, anchors: List[str]) -> int:
    tn = _ar_norm_core(text)
    return sum(1 for a in anchors if a in tn)

# =========================
# Answer generator
# =========================

def answer(question: str, index: HybridIndex, intent: str, use_rerank_flag: bool=False) -> str:
    # 1) retrieve
    hits = index.search(question, top_k=16)
    if not hits:
        return "لم أعثر على إجابة مناسبة."

    anchors = INTENT_ANCHORS.get(intent, [])

    # 2) light rerank: boost hits containing anchors / suitable signals
    reranked: List[Tuple[float,int]] = []
    for score, idx in hits:
        ch = index.chunks[idx]
        base = score
        textn = _ar_norm_core(ch.text)
        # anchor boost
        if anchors:
            base += 0.10 * _anchor_score(textn, anchors)
        # signal boosts
        if intent in ("work_hours","ramadan_hours") and (TIME_RANGE.search(textn) or TIME_TOKEN.search(textn)):
            base += 0.15
        if intent == "break" and (DUR_TOKEN.search(textn) or TIME_TOKEN.search(textn)):
            base += 0.15
        if intent in ("procurement","per_diem","overtime","leave") and ANY_DIGIT.search(textn):
            base += 0.05
        reranked.append((base, idx))

    reranked.sort(key=lambda x: x[0], reverse=True)

    # 3) pick best snippet from the top doc(s)
    best_text = ""
    best_meta: List[Tuple[int,str]] = []  # (page, source)

    # consider top 3 docs to pick a concise line
    for base, idx in reranked[:3]:
        ch = index.chunks[idx]
        lines = _split_sentences_ar(ch.text)
        picked: Optional[str] = None
        if intent in ("work_hours","ramadan_hours"):
            picked = _prefer_time_line(lines, want_ramadan=(intent=="ramadan_hours"))
        elif intent == "break":
            # prefer lines with duration tokens first
            cand = [ln for ln in lines if DUR_TOKEN.search(_ar_norm_core(ln))]
            picked = cand[0] if cand else _prefer_anchored_numeric(lines, anchors)
        elif intent == "workdays":
            # any line with weekday names or "ايام"+"العمل/الدوام"
            picked = None
            for ln in lines:
                tn = _ar_norm_core(ln)
                if any(d in ln for d in ["السبت","الاحد","الأحد","الاثنين","الإثنين","الثلاثاء","الاربعاء","الأربعاء","الخميس","الجمعة"]) \
                   or ("ايام" in tn and ("العمل" in tn or "الدوام" in tn)):
                    picked = ln; break
            if not picked:
                picked = _prefer_anchored_numeric(lines, anchors)
        else:
            picked = _prefer_anchored_numeric(lines, anchors) or (lines[0] if lines else ch.text.strip())

        if picked:
            best_text = picked
            best_meta.append((ch.page, ch.source))
            break

    # Fallback if no line was picked
    if not best_text:
        ch = index.chunks[reranked[0][1]]
        best_text = clean_display_text(ch.text.strip())
        best_meta.append((ch.page, ch.source))

    # 4) compose sources: include up to 8 sources (pages)
    src_lines: List[str] = []
    used = set()
    for _, idx in reranked[:8]:
        p, s = index.chunks[idx].page, index.chunks[idx].source
        key = (p, s)
        if key in used: continue
        used.add(key)
        src_lines.append(f"{len(src_lines)+1}. {s} - page {p}")

    body = clean_display_text(best_text).strip()
    if not body:
        body = "لم أعثر على إجابة مناسبة."
    out = body + "\nSources:\n" + "\n".join(src_lines)
    return out

# =========================
# Sanity prompts (30 Qs)
# =========================

SANITY_PROMPTS: List[str] = [
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

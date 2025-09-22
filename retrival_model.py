# -*- coding: utf-8 -*-
"""
retrival_model.py — Arabic-friendly hybrid retriever for AlNayzak-Chatbot.

Public API (used by NewRag.py):
- load_hierarchy(hier_path: str|None, aliases_path: str|None) -> dict|None
- load_chunks(path: str) -> (List[Chunk], chunks_hash: str)
- class HybridIndex: build(), save(dir), load(dir), search(query, k=8)
- classify_intent(question: str) -> str
- answer(question: str, index: HybridIndex, intent: str, use_rerank_flag=False) -> str
- SANITY_PROMPTS: List[str]

Design notes
------------
- Loader accepts JSONL or JSON and many common keys:
  text: "text" | "content" | "chunk" | "body" | "paragraph" | "chunk_text"
  page: "page" | "page_number" | "page_num" | "pageno" | "pageIndex" | meta.page_number
  section/heading: "heading" | "title" | meta.heading
  doc id/name: "doc" | "doc_id" | "document" | "source" | meta.source
- Cleans PDF artifacts like '/uni06BE' and collapses whitespace.
- Arabic normalization keeps letters, removes tatweel/diacritics, normalizes digits (٠١٢… -> 012…).
- Search = BM25-lite over wordpieces + short exact-phrase boost; no external models required.
- Answer chooses the *most relevant line* from top chunk based on intent (times, durations, weekdays, numbers).
- Outputs "Sources:" exactly like your runner expects: "Data_pdf.pdf - page X".
"""

from __future__ import annotations
import os, re, json, math, hashlib, pickle, io
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable

# ---------------- Arabic text utils ----------------

_AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
_TATWEEL = '\u0640'
# Eastern Arabic and Persian digits -> ASCII 0-9
_AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"٠١٢٣٤٥٦٧٨٩")}
_IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate(u"۰۱۲۳۴۵۶۷۸۹")}
# Weird PDF glyph names like /uni06BE
_PDF_UNI = re.compile(r'/uni[0-9A-Fa-f]{3,6}')
# Remove control chars but keep bidi wrappers (handled later by caller)
_CTRL = re.compile(r'[\u0000-\u0008\u000B-\u000C\u000E-\u001F]')

def ar_norm(s: str) -> str:
    if not s: return ""
    s = _CTRL.sub('', s)
    s = _PDF_UNI.sub('', s)
    s = s.replace(_TATWEEL, '')
    s = _AR_DIAC.sub('', s)
    s = (s.replace('أ','ا').replace('إ','ا').replace('آ','ا')
           .replace('ى','ي').replace('ة','ه'))
    s = s.translate(_AR_NUMS).translate(_IR_NUMS)
    # unify punctuation spacing
    s = s.replace('،', ',').replace('٫','.')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

_AR_LETTER = re.compile(r'[^\u0600-\u06FFa-zA-Z0-9]+')
def tokenize_ar(s: str) -> List[str]:
    s = ar_norm(s)
    # split on non-letters but keep mixed tokens like "8:30"
    s = re.sub(r'[:/\\\-\u2212]+', ' ', s)
    toks = [t for t in _AR_LETTER.split(s) if t]
    # prune tiny tokens unless digits
    out = []
    for t in toks:
        if t.isdigit():
            out.append(t)
        elif len(t) >= 2:
            out.append(t)
    return out

def ascii_digits(s: str) -> str:
    return s.translate(_AR_NUMS).translate(_IR_NUMS) if s else s

# time/duration detectors
TIME_TOKEN = re.compile(r'\b\d{1,2}[:\.]\d{1,2}\b')  # 8:30, 15.00
DUR_TOKEN  = re.compile(r'\b(\d{1,3})\s*(?:دقيقه|دقيقة|دقائق|ساعة|ساعه|ساعات)\b', re.I)
WEEKDAYS   = ("السبت","الأحد","الاحد","الإثنين","الاثنين","الثلاثاء","الأربعاء","الاربعاء","الخميس","الجمعة")

# ---------------- Data structures ----------------

@dataclass
class Chunk:
    text: str
    page: int
    doc: str = "Data_pdf.pdf"
    section: str = ""
    meta: dict = None

# ---------------- Loaders ----------------

def _coerce_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _first(*vals):
    for v in vals:
        if v is None: continue
        if isinstance(v, str) and v.strip(): return v
        if isinstance(v, (int, float)) and v != 0: return v
    return None

def _extract_text(rec: dict) -> Optional[str]:
    return _first(
        rec.get("text"), rec.get("content"), rec.get("chunk"),
        rec.get("body"), rec.get("paragraph"), rec.get("chunk_text"),
        rec.get("Text"), rec.get("Content")
    )

def _extract_page(rec: dict) -> Optional[int]:
    page = _first(
        rec.get("page"), rec.get("page_number"), rec.get("page_num"),
        rec.get("pageno"), rec.get("pageIndex"), rec.get("page_index")
    )
    if page is None and isinstance(rec.get("meta"), dict):
        m = rec["meta"]
        page = _first(m.get("page"), m.get("page_number"), m.get("page_num"), m.get("pageno"))
    return _coerce_int(page, 0) if page is not None else 0

def _extract_doc(rec: dict) -> str:
    doc = _first(rec.get("doc"), rec.get("doc_id"), rec.get("document"),
                 rec.get("source"), rec.get("file"), rec.get("filename"))
    if doc is None and isinstance(rec.get("meta"), dict):
        doc = _first(rec["meta"].get("source"), rec["meta"].get("file"))
    return str(doc) if doc else "Data_pdf.pdf"

def _extract_section(rec: dict) -> str:
    sec = _first(rec.get("heading"), rec.get("title"), rec.get("section"))
    if sec is None and isinstance(rec.get("meta"), dict):
        sec = _first(rec["meta"].get("heading"), rec["meta"].get("title"))
    return str(sec) if sec else ""

def _iter_records(data) -> Iterable[dict]:
    if isinstance(data, list):
        for r in data: 
            if isinstance(r, dict): yield r
    elif isinstance(data, dict):
        # Some exporters wrap under 'chunks'
        items = data.get("chunks") or data.get("data") or data.get("records") or []
        if isinstance(items, list):
            for r in items:
                if isinstance(r, dict): yield r

def load_chunks(path: str) -> Tuple[List[Chunk], str]:
    """
    Accepts .jsonl (one JSON per line) or .json (array/object).
    Returns (chunks, sha1_hash).
    """
    chunks: List[Chunk] = []
    h = hashlib.sha1()

    if not os.path.isfile(path):
        raise FileNotFoundError(f"chunks file not found: {path}")

    _, ext = os.path.splitext(path.lower())
    with io.open(path, "r", encoding="utf-8") as f:
        if ext == ".jsonl":
            for line in f:
                line = line.strip()
                if not line: continue
                h.update(line.encode("utf-8", errors="ignore"))
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                txt = _extract_text(rec)
                if not txt: continue
                txt = ar_norm(txt)
                if not txt or len(txt) < 3: continue
                pg  = _extract_page(rec)
                doc = _extract_doc(rec)
                sec = _extract_section(rec)
                chunks.append(Chunk(text=txt, page=pg, doc=doc, section=sec, meta=rec))
        else:
            raw = f.read()
            h.update(raw.encode("utf-8", errors="ignore"))
            try:
                data = json.loads(raw)
            except Exception:
                raise ValueError("invalid JSON in chunks file")
            for rec in _iter_records(data):
                txt = _extract_text(rec)
                if not txt: continue
                txt = ar_norm(txt)
                if not txt or len(txt) < 3: continue
                pg  = _extract_page(rec)
                doc = _extract_doc(rec)
                sec = _extract_section(rec)
                chunks.append(Chunk(text=txt, page=pg, doc=doc, section=sec, meta=rec))

    if not chunks:
        # Help the caller see what's wrong instead of silently passing.
        raise ValueError(
            "No chunks loaded. Please confirm your JSON keys. "
            "Expected one of: text|content|chunk|body|paragraph|chunk_text and page|page_number|pageno etc."
        )

    return chunks, h.hexdigest()

# ---------------- Hierarchy / aliases (optional) ----------------

def load_hierarchy(hier_path: Optional[str], aliases_path: Optional[str]) -> Optional[dict]:
    hier = None
    if hier_path and os.path.isfile(hier_path):
        try:
            with io.open(hier_path, "r", encoding="utf-8") as f:
                hier = json.load(f)
        except Exception:
            hier = None
    if aliases_path and os.path.isfile(aliases_path):
        try:
            with io.open(aliases_path, "r", encoding="utf-8") as f:
                aliases = json.load(f)
            if hier is None: hier = {}
            hier["_aliases"] = aliases
        except Exception:
            pass
    return hier

# ---------------- HybridIndex (BM25-lite + exact phrase boost) ----------------

class HybridIndex:
    def __init__(self, chunks: List[Chunk], chunks_hash: str, hier: Optional[dict]=None, model_name: Optional[str]=None):
        self.chunks: List[Chunk] = chunks
        self.chunks_hash = chunks_hash
        self.hier = hier or {}
        # inverted
        self.df: Dict[str, int] = {}
        self.doc_tfs: List[Dict[str, int]] = []
        self.doc_len: List[int] = []
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        # simple cache
        self._built = False

    def build(self):
        self.df.clear(); self.doc_tfs.clear(); self.doc_len.clear(); self.idf.clear()
        for ch in self.chunks:
            toks = tokenize_ar(ch.text)
            tf: Dict[str,int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.doc_tfs.append(tf)
            self.doc_len.append(sum(tf.values()))
            for t in tf.keys():
                self.df[t] = self.df.get(t, 0) + 1

        N = max(1, len(self.chunks))
        self.avgdl = max(1.0, sum(self.doc_len) / N if self.doc_len else 1.0)
        for t, df in self.df.items():
            # BM25 idf (plus 1 to avoid neg idf when df > N/2)
            self.idf[t] = math.log(1 + (N - df + 0.5) / (df + 0.5))
        self._built = True

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "bm25.pkl"), "wb") as f:
            pickle.dump({
                "chunks_hash": self.chunks_hash,
                "df": self.df,
                "doc_tfs": self.doc_tfs,
                "doc_len": self.doc_len,
                "idf": self.idf,
                "avgdl": self.avgdl,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, in_dir: str) -> bool:
        try:
            with open(os.path.join(in_dir, "bm25.pkl"), "rb") as f:
                obj = pickle.load(f)
            # trust loaded stats only if hashes match length
            if obj.get("df") and obj.get("doc_tfs") and obj.get("chunks_hash") == self.chunks_hash and len(obj["doc_tfs"]) == len(self.chunks):
                self.df = obj["df"]; self.doc_tfs = obj["doc_tfs"]
                self.doc_len = obj["doc_len"]; self.idf = obj["idf"]
                self.avgdl = obj["avgdl"]; self._built = True
                return True
        except Exception:
            return False
        return False

    def _bm25(self, q_toks: List[str], k1=1.5, b=0.75) -> List[Tuple[int,float]]:
        if not self._built:
            self.build()
        # term frequencies in query (ok to de-duplicate)
        uniq = {}
        for t in q_toks:
            uniq[t] = 1
        scores: List[Tuple[int,float]] = []
        for i, tf in enumerate(self.doc_tfs):
            dl = self.doc_len[i] or 1
            score = 0.0
            for t in uniq.keys():
                if t not in tf: continue
                idf = self.idf.get(t, 0.0)
                f = tf[t]
                denom = f + k1 * (1 - b + b * dl / self.avgdl)
                score += idf * (f * (k1 + 1)) / (denom if denom != 0 else 1.0)
            if score > 0:
                scores.append((i, score))
        # exact small-phrase boost (2-3 tokens adjacency)
        phrase = None
        if len(q_toks) >= 2:
            phrase = " ".join(q_toks[:3]).strip()
        if phrase:
            pat = re.compile(re.escape(phrase))
            for j, (i, sc) in enumerate(scores):
                if pat.search(self.chunks[i].text):
                    scores[j] = (i, sc * 1.15)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def search(self, query: str, k: int=8) -> List[Tuple[int, float]]:
        query = ar_norm(query)
        q_toks = tokenize_ar(query)
        if not q_toks:
            return []
        ranked = self._bm25(q_toks)
        return ranked[:k]

# ---------------- Intent & line picking ----------------

def classify_intent(question: str) -> str:
    q = ar_norm(question)
    q_no_digits = re.sub(r'\d+', '', q)

    if "رمضان" in q_no_digits:
        return "ramadan_hours"
    if ("ساعات" in q_no_digits or "الدوام" in q_no_digits or "العمل" in q_no_digits) and ("من" in q_no_digits or "الى" in q_no_digits or "إلى" in q_no_digits):
        return "work_hours"
    if "استراح" in q_no_digits or "بريك" in q_no_digits or "راحة" in q_no_digits:
        return "break"
    if "ايام" in q or "أيام" in question or "السبت" in q or "الجمعة" in q:
        return "workdays"
    if "ساعات اضافية" in q or "العمل الاضافي" in q or "العمل الإضافي" in q or "اوفر تايم" in q or "overtime" in q.lower():
        return "overtime"
    if "مياومات" in q or "بدل سفر" in q or "سفر" in q:
        return "per_diem"
    if "شراء" in q or "عروض اسعار" in q or "عروض أسعار" in q or "توريد" in q:
        return "procurement"
    if "اجازه" in q or "إجازة" in q or "اجازة" in q:
        if "اموم" in q: return "maternity"
        if "مرض" in q: return "sick_leave"
        if "طارئ" in q or "طارئة" in q: return "emergency_leave"
        return "leave"
    if "هدايا" in q or "الضيافه" in q or "الضيافة" in q:
        return "gifts"
    if "عهدة" in q or "عهده" in q:
        return "custody"
    if "رواتب" in q or "راتب" in q:
        return "payroll"
    if "تضارب المصالح" in q:
        return "conflict_of_interest"
    if "السلوك" in q or "التحرش" in q:
        return "conduct_harassment"
    if "السرية" in q or "حماية المعلومات" in q:
        return "confidentiality"
    return "general"

def _line_score_by_intent(line: str, intent: str) -> float:
    ln = ar_norm(line)
    score = 0.0
    # generic substance
    if len(ln) >= 12: score += 0.5
    if intent in ("work_hours", "ramadan_hours"):
        if TIME_TOKEN.search(ascii_digits(ln)): score += 3.0
        if "من" in ln and ("الى" in ln or "إلى" in ln): score += 1.0
        if "رمضان" in ln and intent == "ramadan_hours": score += 1.0
    elif intent == "break":
        if DUR_TOKEN.search(ln): score += 2.0
        if "استراح" in ln or "راحة" in ln or "بريك" in ln: score += 1.0
    elif intent == "workdays":
        if any(d in ln for d in WEEKDAYS): score += 2.0
        if "ايام" in ln and ("العمل" in ln or "الدوام" in ln): score += 1.0
    elif intent in ("overtime", "per_diem", "procurement", "leave", "maternity", "sick_leave", "emergency_leave", "payroll"):
        if re.search(r'\d', ln): score += 1.8
        if "نموذج" in ln or "طلب" in ln: score += 0.5
    elif intent in ("gifts","custody","conflict_of_interest","conduct_harassment","confidentiality"):
        if "يمنع" in ln or "ممنوع" in ln or "سياسة" in ln: score += 1.0
    return score

def _best_line_for_intent(text: str, intent: str) -> str:
    # split on hard breaks; also try bullet-ish splits
    parts = []
    for seg in re.split(r'[\r\n]+', text):
        seg = seg.strip()
        if not seg: continue
        # further split very long segments on "•" or " - "
        sub = re.split(r'[•\-\u2022]\s+', seg)
        for s in sub:
            s = s.strip()
            if s: parts.append(s)
    if not parts:
        return text.strip()
    scored = [(p, _line_score_by_intent(p, intent)) for p in parts]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0].strip() if scored[0][1] > 0 else parts[0].strip()

# ---------------- Answer synthesis ----------------

def _format_sources(hit_indices: List[int], chunks: List[Chunk], limit: int = 8) -> str:
    seen = []
    out_lines = []
    for i in hit_indices[:limit]:
        ch = chunks[i]
        key = (ch.doc, ch.page)
        if key in seen: 
            continue
        seen.append(key)
        # the runner prints exactly "Data_pdf.pdf - page N"
        out_lines.append(f"{len(out_lines)+1}. {ch.doc} - page {ch.page}")
    return "\n".join(out_lines) if out_lines else "—"

def answer(question: str, index: HybridIndex, intent: str, use_rerank_flag: bool=False) -> str:
    """
    Returns body + "\nSources:\n<lines>"
    Never fabricates numeric/time facts: selects the best matching line from top chunks.
    """
    hits = index.search(question, k=8)
    if not hits:
        return "لم أعثر على إجابة مناسبة."
    hit_ids = [i for (i, _) in hits]
    # choose the single best line among top-3 chunks for precision
    best_text = ""
    best_score = -1.0
    for i, _sc in hits[:3]:
        ch = index.chunks[i]
        candidate = _best_line_for_intent(ch.text, intent)
        sc = _line_score_by_intent(candidate, intent)
        # light question term presence boost
        q_norm = set(tokenize_ar(question))
        cand_norm = set(tokenize_ar(candidate))
        overlap = len(q_norm & cand_norm)
        sc += 0.1 * overlap
        if sc > best_score:
            best_score = sc
            best_text = candidate

    best_text = best_text.strip()
    if not best_text:
        return "لم أعثر على إجابة مناسبة."

    # Final polish: normalize duplicate spaces, remove stray '/uniXXXX'
    best_text = ar_norm(best_text)

    sources = _format_sources(hit_ids, index.chunks, limit=8)
    return f"{best_text}\nSources:\n{sources}"

# ---------------- Sanity prompts (stable) ----------------

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

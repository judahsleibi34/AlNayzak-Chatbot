# -*- coding: utf-8 -*-
"""
RAG Orchestrator (Arabic) — no-bias priors & strictness booster (single-file, no external helper imports)
- Data-driven section "priors" derived ONLY from *current* headings (no hardcoded section numbers)
- Priors are soft boosts (ordering) — never filter content out
- Robust to arbitrary hierarchy objects (e.g., HierData) via deep introspection
- Negation-aware scoring for approval-like intents
- Numeric/time/day strictness recovery for hours/limits/% questions
- Backwards-compatible with retrival_model (RET) interface

USAGE (strict sanity, extractive):
python NewRag.py --chunks Data_pdf_clean_chunks.jsonl --hier-index heading_inverted_index.json --aliases section_aliases.json --sanity --no-llm --regex-hunt --hourlines-only --out-dir runs

USAGE (interactive chatbot):
python NewRag.py --chunks Data_pdf_clean_chunks.jsonl --hier-index heading_inverted_index.json --aliases section_aliases.json --no-llm --regex-hunt --hourlines-only --out-dir runs
"""

import os, sys, re, json, time, argparse, logging
from datetime import datetime
from collections import defaultdict
from types import SimpleNamespace

# --------------------------------------------------------------------------------------
# Inline minimal helpers (replacing prior_deriver.py + intent_patterns.py)
# --------------------------------------------------------------------------------------

# Arabic normalizer
_AR_DIAC = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
def norm(s: str) -> str:
    if s is None: return ""
    t = s.strip().lower()
    t = _AR_DIAC.sub("", t)
    t = t.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    t = t.replace("ى","ي").replace("ئ","ي").replace("ؤ","و")
    t = t.replace("ة","ه")
    t = re.sub(r"[^\w\s%٪:–\-]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Intent → loose stems that we try to find inside headings (normalized)
_INTENT_TO_HEADING_STEMS = {
    "work_hours": ["ساعات العمل", "الدوام", "ايام الدوام", "الحضور", "الانصراف", "اوقات العمل"],
    "ramadan_hours": ["ساعات العمل","الدوام","رمضان","ايام الصوم","اوقات العمل"],
    "work_days": ["ايام العمل","ايام الدوام","نهاية الاسبوع","السبت"],
    "breaks": ["استراحه","راحه","فترات الاستراحه"],
    "overtime": ["العمل الاضافي","الساعات الاضافيه","اضافي","بدل الساعات الاضافيه"],
    "public_holiday_comp": ["العطل الرسميه","الاعياد","تعويض العطل"],
    "annual_leave": ["اجازه سنويه","سنويه","رصيد الاجازات"],
    "sick_leave": ["اجازه مرضيه","مرضيه","تقرير طبي"],
    "maternity_leave": ["اجازه امومه","الامومه"],
    "bereavement_leave": ["اجازه حداد","الحداد"],
    "payroll": ["الرواتب","الراتب","صرف الرواتب","شؤون ماليه"],
    "transport_allowance": ["بدل المواصلات","مواصلات","السفر الداخلي"],
    "salary_advance": ["سلفه","سلف الرواتب","السلف"],
    "petty_cash": ["النثريات","المصروفات النثريه","مصاريف"],
    "procurement_thresholds": ["المشتريات","عروض الاسعار","سقف الشراء"],
    "conflict_of_interest": ["تضارب مصالح","الهدايا","الضيافه","مدونه السلوك"],
    "asset_custody": ["العهد","العهدة","التسليم","الاستلام"],
    "remote_work": ["العمل عن بعد","العمل من المنزل","سياسه العمل عن بعد"],
    "hourly_exit": ["مغادره ساعيه","اذن مغادره","اذن ساعه"],
    "performance_review": ["تقييم الاداء","الاداء السنوي"],
    "discipline": ["الانذار","العقوبات","التدرج التاديبي","لائحه العقوبات"],
    "confidentiality": ["السرية","حماية المعلومات","البيانات"],
    "conduct_harassment": ["السلوك المهني","التحرش","مدونه السلوك"],
    "per_diem": ["مياومات","بدل سفر","السفر","بدل المياومات"],
}

def _coerce_pages(obj):
    """Return a set of page ints for many shapes: list/set/tuple/int/str (comma or space separated)."""
    if obj is None: return set()
    if isinstance(obj, (list, set, tuple)):
        out = set()
        for v in obj:
            try:
                out.add(int(str(v).strip()))
            except Exception:
                pass
        return out
    # ints
    try:
        return {int(str(obj).strip())}
    except Exception:
        pass
    # strings like "12, 13 14"
    try:
        toks = re.split(r"[,\s]+", str(obj))
        out = set()
        for t in toks:
            if not t: continue
            out.add(int(t))
        return out
    except Exception:
        return set()

def _extract_heading_and_pages(obj):
    """
    Given any small record/obj that *might* carry heading/pages fields, return (heading, pages_set) or (None, set()).
    Looks at common keys/attrs.
    """
    keys_heading = ("heading", "title", "name", "h", "label", "text")
    keys_pages   = ("pages", "page_set", "p", "page_numbers", "page_idxs", "page_index", "page")

    # dict-like
    if isinstance(obj, dict):
        hd = None
        for kh in keys_heading:
            if kh in obj and obj[kh]:
                hd = str(obj[kh]); break
        pgv = None
        for kp in keys_pages:
            if kp in obj and obj[kp] is not None:
                pgv = obj[kp]; break
        if hd:
            return hd, _coerce_pages(pgv)
        # dict of {heading: pages}
        if len(obj) == 1:
            try:
                k = next(iter(obj.keys()))
                v = obj[k]
                return str(k), _coerce_pages(v)
            except Exception:
                pass
        return None, set()

    # object with attributes
    hd = None; pgv = None
    for kh in keys_heading:
        if hasattr(obj, kh):
            val = getattr(obj, kh)
            if val:
                hd = str(val); break
    for kp in keys_pages:
        if hasattr(obj, kp):
            pgv = getattr(obj, kp)
            if pgv is not None:
                break
    if hd:
        return hd, _coerce_pages(pgv)
    return None, set()

def _iter_headings(hier):
    """
    Yield (heading_text, pages_set) from MANY possible hierarchy shapes.
    Supports:
    - dict: {heading: {"pages":[...]}} or {heading:[...]} or {heading:"12,13"}
    - list/tuple of dicts/tuples/objects
    - object: with .to_dict() / .as_dict() / .to_json(), or attributes like .headings/.nodes/.sections/.items
    - object: with iterable interface (__iter__) yielding items that themselves carry heading/pages
    This NEVER raises on unknown types; at worst yields nothing.
    """
    if not hier:
        return

    # If already a mapping
    if isinstance(hier, dict):
        # try mapping of heading->(mapping or pages)
        for k, v in hier.items():
            if isinstance(v, dict):
                _, pset = _extract_heading_and_pages(v)
                if pset:
                    yield str(k), pset
                    continue
            # try direct pages list/int/str
            pset = _coerce_pages(v)
            if pset:
                yield str(k), pset
            else:
                # maybe value is nested small dict with heading inside
                hd, pset2 = _extract_heading_and_pages(v)
                if hd and pset2:
                    yield hd, pset2
        return

    # Try well-known converters on objects
    for conv_name in ("to_dict", "as_dict"):
        if hasattr(hier, conv_name) and callable(getattr(hier, conv_name)):
            try:
                mp = getattr(hier, conv_name)()
                if isinstance(mp, dict):
                    for k, v in mp.items():
                        # same handling as dict branch
                        if isinstance(v, dict):
                            _, pset = _extract_heading_and_pages(v)
                            if pset:
                                yield str(k), pset
                                continue
                        pset = _coerce_pages(v)
                        if pset:
                            yield str(k), pset
                        else:
                            hd, pset2 = _extract_heading_and_pages(v)
                            if hd and pset2:
                                yield hd, pset2
                    return
            except Exception:
                pass

    # Common containers on custom objects
    for attr in ("headings", "nodes", "sections", "items", "entries", "children"):
        if hasattr(hier, attr):
            try:
                seq = getattr(hier, attr)
                # some are callables
                if callable(seq):
                    seq = seq()
                for it in seq or []:
                    # tuple like (heading, pages) or (heading, payload)
                    if isinstance(it, (tuple, list)) and it:
                        hd = str(it[0])
                        pages = set()
                        if len(it) > 1:
                            pages = _coerce_pages(it[1])
                            if not pages and isinstance(it[1], dict):
                                _, pages = _extract_heading_and_pages(it[1])
                        yield hd, pages
                        continue
                    # try dict/object shapes
                    hd, pages = _extract_heading_and_pages(it)
                    if hd:
                        yield hd, pages
            except Exception:
                pass

    # If the object itself is iterable
    try:
        for it in hier:
            # support tuples/lists
            if isinstance(it, (tuple, list)) and it:
                hd = str(it[0])
                pages = set()
                if len(it) > 1:
                    pages = _coerce_pages(it[1])
                    if not pages and isinstance(it[1], dict):
                        _, pages = _extract_heading_and_pages(it[1])
                yield hd, pages
                continue
            # support dict/object records
            hd, pages = _extract_heading_and_pages(it)
            if hd:
                yield hd, pages
    except TypeError:
        # not iterable; last-chance: introspect attributes looking like single record
        hd, pages = _extract_heading_and_pages(hier)
        if hd:
            yield hd, pages

def derive_section_priors(intent: str, hier) -> dict:
    """
    Return {"pages": set[int], "headings": [matched_headings]} based ONLY on current hierarchy headings.
    No hardcoded section numbers; matching is via normalized stem inclusion and/or token overlap.
    Robust to any hier type (dict/list/tuples/custom object).
    """
    intent = intent or ""
    norm_int = norm(intent)
    stems = _INTENT_TO_HEADING_STEMS.get(intent, [])

    # If classifier label is unknown, guess from tokens in intent string
    if not stems and norm_int:
        if "دوام" in norm_int or "ساعات" in norm_int:
            stems = _INTENT_TO_HEADING_STEMS["work_hours"]
        elif "رمضان" in norm_int:
            stems = _INTENT_TO_HEADING_STEMS["ramadan_hours"]
        elif "عروض" in norm_int or "سقف" in norm_int:
            stems = _INTENT_TO_HEADING_STEMS["procurement_thresholds"]

    pages, matched = set(), []
    toks = set(norm_int.split()) if norm_int else set()

    for heading, pset in _iter_headings(hier):
        nh = norm(heading or "")
        ok = False
        for stem in stems:
            if stem and stem in nh:
                ok = True; break
        if not ok and toks:
            if any(t and t in nh for t in toks):
                ok = True
        if ok:
            matched.append(heading)
            pages |= set(pset or set())

    return {"pages": pages, "headings": matched}

# --------------------------------------------------------------------------------------
# End inlined helpers
# --------------------------------------------------------------------------------------

# Quiet noisy libs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

try:
    import torch  # optional
except Exception:
    torch = None

# Your retriever module (existing in your repo)
import retrival_model as RET

# ---------------- Utilities for dict-or-object chunks ----------------
def _get_attr_or_key(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    val = getattr(obj, key, None)
    if val is not None: return val
    for nest in ("meta", "metadata", "__dict__"):
        container = getattr(obj, nest, None)
        if isinstance(container, dict) and key in container:
            return container.get(key)
    return default

def _first_non_empty(*vals):
    for v in vals:
        if v is None: continue
        s = str(v)
        if s.strip(): return s
    return ""

def _first_page_like(obj):
    candidates = (
        _get_attr_or_key(obj, "page"),
        _get_attr_or_key(obj, "page_number"),
        _get_attr_or_key(obj, "page_num"),
        _get_attr_or_key(obj, "page_idx"),
        _get_attr_or_key(obj, "pageno"),
        _get_attr_or_key(obj, "page_start"),
    )
    for c in candidates:
        if c is None or str(c).strip() == "": continue
        try: return int(str(c).strip())
        except Exception: pass
    return None

# ---------------- Global page index ----------------
CHUNKS_BY_PAGE = {}  # {int page: full concatenated text}

def _build_page_text_index(chunks):
    pages = defaultdict(list)
    for ch in chunks:
        txt = _first_non_empty(
            _get_attr_or_key(ch, "text"),
            _get_attr_or_key(ch, "content"),
            _get_attr_or_key(ch, "chunk"),
            _get_attr_or_key(ch, "body"),
            "",
        )
        if isinstance(txt, bytes):
            try: txt = txt.decode("utf-8", "ignore")
            except Exception: txt = ""
        pg = _first_page_like(ch)
        if pg is None:
            src = _get_attr_or_key(ch, "source") or _get_attr_or_key(ch, "doc_source") or ""
            m = re.search(r"(?:[Pp]age|(?:ال)?صفحة)\s+(\d+)", str(src))
            if m:
                try: pg = int(m.group(1))
                except Exception: pg = None
        if pg is not None and txt:
            pages[int(pg)].append(str(txt))
    return {p: "\n".join(v) for p, v in pages.items()}

# ---------------- Logging ----------------
def setup_logger(log_path: str):
    logger = logging.getLogger("rag_orchestrator")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
    logger.handlers = []; logger.addHandler(ch); logger.addHandler(fh)
    return logger

LOG = logging.getLogger("rag_orchestrator")  # reset in main()

# ---------------- Sanity Prompts (default) ----------------
DEFAULT_SANITY_PROMPTS = [
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

# ---------------- Arabic helpers / checks ----------------
_HEADING_PATTERNS = [
    r"^\s*الإجابة\s*:?$",
    r"^\s*الخلاصة\s*:?\s*$",
    r"^\s*الملخص\s*:?\s*$",
    r"^\s*Summary\s*:?\s*$",
    r"^\s*Answer\s*:?\s*$",
]
_AR_DAYS = ["الأحد", "الإثنين", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت"]

_TIME_PATTERNS = [
    r"\b\d{1,2}:\d{2}\b",                     # 8:30
    r"\b\d{1,2}[:٫]\d{2}\b",                  # 8٫30
    r"\b\d{1,2}\s*[-–]\s*\d{1,2}\b",          # 8-5
    r"\b\d{1,2}\s*(?:إلى|حتى)\s*\d{1,2}\b",   # 8 إلى 5
    r"\b\d{1,2}\s*(?:ص|م)\b",                 # 8 ص/5 م
]
_PERCENT_RX = re.compile(r"\b\d{1,3}\s*[%٪]\b")
_DURATION_RX = re.compile(r"\b\d{1,2}\s*(?:دقيقة|دقائق|ساعة|ساعات|يوم|أيام)\b")
_SECTION_HEAVY = re.compile(r"(?:\d+\.){2,}\d+")  # like 3.1.5.7 (TOC-ish)

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_AR_LETTER_RX = re.compile(r"[ء-ي]")
def _to_western_digits(s): return (s or "").translate(_ARABIC_DIGITS)
def _strip_mojibake(s): return "" if not s else s.replace("\ufeff","").replace("�","").replace("\uFFFD","")
def _arabic_ratio(s):
    if not s: return 1.0
    letters = re.findall(r"\w", s, flags=re.UNICODE)
    if not letters: return 1.0
    arabic = _AR_LETTER_RX.findall(s)
    return (len(arabic) / max(1, len(letters)))

def _purge_non_arabic_lines(s, min_ratio=0.66):
    if not s: return s
    keep = []
    for line in s.splitlines():
        ln = line.strip()
        if not ln: continue
        # keep numeric/time lines even if not Arabic-heavy
        t = _to_western_digits(ln)
        if any(re.search(p, t) for p in _TIME_PATTERNS) or _PERCENT_RX.search(t) or _DURATION_RX.search(t):
            keep.append(ln); continue
        ratio = _arabic_ratio(ln)
        if ratio >= min_ratio: keep.append(ln)
    return "\n".join(keep)

def _has_times_or_days(txt):
    if not txt: return False
    t = _to_western_digits(txt)
    if any(day in t for day in _AR_DAYS): return True
    if any(re.search(p, t) for p in _TIME_PATTERNS): return True
    if _DURATION_RX.search(t): return True
    if _PERCENT_RX.search(t): return True
    return False

def _sentences(txt):
    if not txt: return []
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    keep = []
    for l in lines:
        if any(re.match(p, l) for p in _HEADING_PATTERNS): continue
        keep.append(l)
    txt2 = " ".join(keep)
    parts = re.split(r"(?<=[\.\!\؟])\s+|[\n\r]+|[•\-–]\s+", txt2)
    parts = [p.strip(" -–•\t") for p in parts if p and len(p.strip()) > 0]
    merged = []
    for p in parts:
        if merged and len(p) < 10:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged

def _clean_text(txt):
    if not txt: return ""
    txt = _strip_mojibake(txt)
    txt = re.sub(r"^```.*?$", "", txt, flags=re.M | re.S)
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    keep = []
    for l in lines:
        if any(re.match(p, l) for p in _HEADING_PATTERNS): continue
        if "فهرس المحتويات" in l or "جميع الحقوق محفوظة" in l: continue
        if _SECTION_HEAVY.search(l): continue
        keep.append(l)
    txt = " ".join(keep).strip()
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def _paginate_text(text, max_chars=600):
    text = text.strip()
    if len(text) <= max_chars: return [text]
    parts, cur, count = [], [], 0
    for line in text.splitlines():
        if count + len(line) + 1 > max_chars:
            parts.append("\n".join(cur).strip())
            cur, count = [line], len(line)
        else:
            cur.append(line); count += len(line) + 1
    if cur: parts.append("\n".join(cur).strip())
    return parts

def _split_answer(answer_text):
    if not answer_text: return "", ""
    parts = re.split(r"\n(?=Sources:|المصادر:)", answer_text, maxsplit=1)
    body = parts[0].strip(); sources = parts[1].strip() if len(parts) > 1 else ""
    return body, sources

# ---------------- Q intent + keywords ----------------
POLICY_VERBS = ("يجب","يلزم","يُمنع","يُحظر","يتعيّن","لا يجوز","يحق","تكون","يتم","وفقاً","حسب")
NEG = ["لا","غير","عدم","دون","إلا","لا يتم","لا يجوز"]
REQ = ["موافقه","اذن","كتابي","خطي","مسبق","اعتماد"]

def _norm_tokens(s):
    s = _to_western_digits(s or "")
    s = re.sub(r"[^\w\s%٪:–\-]+"," ", s, flags=re.UNICODE)
    return [t for t in s.split() if t.strip()]

def _question_keywords(q):
    qn = norm(q or "")
    toks = _norm_tokens(qn)
    extras = []
    if "ساعات" in qn or "دوام" in qn: extras += ["ساعات","دوام","من","إلى","حتى"]
    if "رمضان" in qn: extras += ["رمضان","الصوم"]
    if "اضاف" in qn: extras += ["اضافيه","العمل الاضافي","اجر"]
    if "عطل" in qn or "عطله" in qn: extras += ["العطل","الرسمية","عطله","نهاية","أسبوع"]
    if "استراح" in qn or "راحه" in qn: extras += ["استراحه","راحه","مده","مدتها"]
    if "مغادر" in qn: extras += ["مغادره","ساعيه","الحد","الاقصى","شهري"]
    if "اجازه" in qn: extras += ["اجازه","ايام","مده","سنو"]
    if "سقف" in qn or "عروض" in qn: extras += ["سقف","عروض","اسعار","ثلاثه"]
    if "تضارب" in qn: extras += ["تضارب","مصالح","الهدايا"]
    return list(dict.fromkeys(toks + extras))

def _expects_numerics(q):
    q = norm(q or "")
    cues = ("كم","مده","ما الحد","الحد","من وإلى","من والى","من الى","متى","النسبة","%","٪","ساعات","دقائق","يوم","أيام","اجر","تعويض","سقف","بدل","مياومات","الحد الاقصى","شهري")
    return any(c in q for c in cues)

# ---------------- Sources → pages helpers ----------------
def _parse_pages_from_sources(sources_text):
    if not sources_text: return []
    s = _to_western_digits(sources_text)
    pages = set()
    for m in re.findall(r"(?:\b[Pp]age\b|(?:ال)?صفحة)\s+(\d+)", s):
        try: pages.add(int(m))
        except Exception: pass
    return sorted(pages)

def _page_ctx_from_pages(pages, max_chars=3500):
    if not pages: return ""
    buf, total = [], 0
    for p in pages:
        txt = CHUNKS_BY_PAGE.get(p, "")
        if not txt: continue
        t = _clean_text(txt)
        if total + len(t) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                buf.append(t[:remaining]); total = max_chars; break
        else:
            buf.append(t); total += len(t)
    return " ".join(buf).strip()

# ---------------- Generic regex hunter (question-driven) ----------------
GENERIC_RXS = [
    re.compile(r"\b\d{1,2}:\d{2}\b"),
    re.compile(r"\b\d{1,2}[:٫]\d{2}\b"),
    re.compile(r"\b\d{1,2}\s*(?:ص|م)\b"),
    re.compile(r"\b\d{1,2}\s*(?:إلى|حتى|-\s*|–\s*)\s*\d{1,2}\b"),
    re.compile(r"\b\d{1,3}\s*[%٪]\b"),
    re.compile(r"\b\d{1,2}\s*(?:دقيقة|دقائق|ساعة|ساعات|يوم|أيام)\b"),
    re.compile(r"\b\d+\b"),
]
ACCOUNTING_BAN = {"اهلاك","الاستهلاك","أصل","الاصول","الميزانيه","الاهلاك","القيمه الدفتريه"}

def _line_has_generic_numeric(t):
    T = _to_western_digits(t)
    for rx in GENERIC_RXS:
        if rx.search(T): return True
    if any(d in T for d in _AR_DAYS): return True
    return False

def _regex_hunt_generic(text, q_kws, intent=None):
    if not text: return []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    hits = []
    for l in lines:
        L = _clean_text(l)
        if not L: continue
        if _SECTION_HEAVY.search(L): continue
        if "فهرس المحتويات" in L or "جميع الحقوق محفوظة" in L: continue
        if _arabic_ratio(L) < 0.4 and not _line_has_generic_numeric(L):
            continue
        if intent in ("work_hours","ramadan_hours","overtime","annual_leave","sick_leave","hourly_exit"):
            if any(b in L for b in ACCOUNTING_BAN): 
                continue
        score = 0
        if _line_has_generic_numeric(L): score += 3
        for kw in q_kws:
            if kw and kw in norm(L): score += 1
        if any(v in L for v in POLICY_VERBS): score += 1
        if intent in ("overtime","salary_advance","remote_work","hourly_exit"):
            has_req = any(k in norm(L) for k in REQ)
            has_neg = any(k in norm(L) for k in NEG)
            if has_req and not has_neg: score += 3
            if has_req and has_neg: score -= 3
        if score > 0:
            hits.append((score, L))
    hits.sort(key=lambda x: (-x[0], len(x[1])))
    return [h[1] for h in hits]

# ---------------- Bullet formatting ----------------
def _clip(s: str, n: int) -> str:
    s = s.strip()
    if n and len(s) > n:
        return (s[:max(1, n-1)]).rstrip() + "…"
    return s

def _as_bullets_clipped(sents, limit=5, max_chars=120):
    sents = sents[:max(1, limit)]
    sents = [_clip(s, max_chars) for s in sents]
    out, seen = [], set()
    for s in sents:
        if s and s not in seen:
            seen.add(s); out.append(f"• {s}")
    return "\n".join(out)

def _filter_hourlines(sents):
    out = []
    for s in sents:
        t = _to_western_digits(s)
        if _has_times_or_days(t):
            out.append(s)
    return out

def _bullets_for_display(text: str, question: str, intent: str, cfg):
    sents = _sentences(text)
    if cfg.hourlines_only and (_is_hours_like(question, intent) or _expects_numerics(question)):
        sents = _filter_hourlines(sents)
    if not sents:
        sents = _sentences(text)
    return _as_bullets_clipped(sents, limit=cfg.max_bullets, max_chars=cfg.bullet_max_chars)

def _is_hours_like(question: str, intent: str = "") -> bool:
    q = norm((question or "").strip())
    hours_kws = ["ساعات","الدوام","رمضان","ايام الدوام","الساعات الاضافيه","العطل","استراحه","مغادره ساعيه","وقت","من الى","من وإلى"]
    return any(kw in q for kw in hours_kws) or intent in ("work_hours","ramadan_hours","overtime","work_days","breaks")

# ---------------- Core Q&A ----------------
def ask_once(index: RET.HybridIndex, tokenizer, model, question: str,
             use_llm: bool = True, use_rerank_flag: bool = True, cfg: SimpleNamespace = None, hier=None) -> str:
    t0 = time.time()
    cfg = cfg or SimpleNamespace(max_bullets=5, bullet_max_chars=120, paginate_chars=600,
                                 hourlines_only=False, regex_hunt=True)
    intent = RET.classify_intent(question)

    # --------- derive priors from current document (NO section numbers) ---------
    priors = derive_section_priors(intent, hier) if hier is not None else {"pages": set(), "headings": []}
    prior_pages = list(priors.get("pages", []) or [])

    # Use existing extractive path first (keeps compatibility)
    extractive_answer = RET.answer(question, index, intent, use_rerank_flag=use_rerank_flag)

    # split body/sources
    lines = str(extractive_answer or "").split('\n')
    body_lines, source_lines, sources_started = [], [], False
    for line in lines:
        ls = line.strip()
        if ls.startswith("Sources:") or ls.startswith("المصادر:"):
            sources_started = True; source_lines.append(line)
        elif sources_started:
            source_lines.append(line)
        else:
            body_lines.append(line)
    body_raw = '\n'.join(body_lines).strip()
    sources = '\n'.join(source_lines).strip()

    # pages from citations + priors (priors first)
    cited_pages = _parse_pages_from_sources(sources)
    ordered_pages = (prior_pages or []) + [p for p in cited_pages if p not in prior_pages]

    # Build page context preferring prior pages
    page_ctx = _page_ctx_from_pages(ordered_pages, max_chars=3500)

    # 1) Cleanup + minimal rescue if empty or weak
    hours_like = _is_hours_like(question, intent)
    tmp_body = _clean_text(body_raw)
    if (not tmp_body) or ("لا يقدّم النص" in tmp_body) or ("لم أعثر" in tmp_body):
        if page_ctx:
            body_raw = page_ctx
    elif (hours_like or _expects_numerics(question)) and not _has_times_or_days(tmp_body):
        if _has_times_or_days(page_ctx):
            body_raw = page_ctx

    # 2) Generic regex hunt (question-driven) — prefer prior pages
    if cfg.regex_hunt:
        q_kws = _question_keywords(question)
        hunted = []
        # search prior pages first
        if prior_pages:
            for p in prior_pages:
                t = CHUNKS_BY_PAGE.get(p, "")
                if not t: continue
                hunted.extend(_regex_hunt_generic(t, q_kws, intent=intent))
                if len(hunted) >= 10: break
        # then cited pages
        if not hunted and cited_pages:
            for p in cited_pages:
                t = CHUNKS_BY_PAGE.get(p, "")
                if not t: continue
                hunted.extend(_regex_hunt_generic(t, q_kws, intent=intent))
                if len(hunted) >= 10: break
        # global fallback ordered by priors first
        if not hunted:
            all_pages = (prior_pages or []) + [p for p in sorted(CHUNKS_BY_PAGE) if p not in prior_pages]
            for p in all_pages:
                t = CHUNKS_BY_PAGE.get(p, "")
                hunted.extend(_regex_hunt_generic(t, q_kws, intent=intent))
                if len(hunted) >= 10: break
        # prefer hunted lines when strict numerics expected
        if hunted and (_expects_numerics(question) or hours_like):
            body_raw = "\n".join(hunted[:8])

    # 3) Final formatting (LLM optional, but we keep it off-safe by default)
    def _final(dt, text, srcs):
        parts = _paginate_text(text, max_chars=cfg.paginate_chars)
        if len(parts) > 1:
            labeled = []
            for i, p in enumerate(parts, 1):
                labeled.append(f"الجزء {i}/{len(parts)}:\n{p}")
            text = "\n\n".join(labeled)
        return f"⏱ {dt:.2f}s | 🤖 {text}\n{srcs}" if srcs else f"⏱ {dt:.2f}s | 🤖 {text}"

    if not body_raw or len(body_raw.strip()) == 0:
        dt = time.time() - t0
        return _final(dt, "لا يقدّم النص المسترجَع تفاصيل كافية للإجابة بشكل قاطع من المصدر نفسه.", sources)

    body_clean = _clean_text(body_raw)
    body_clean = _purge_non_arabic_lines(body_clean)

    # Enforce strictness recovery: if numerics expected but missing, try again from prior pages
    if (_expects_numerics(question) or hours_like) and not _has_times_or_days(body_clean):
        q_kws = _question_keywords(question)
        hunted2 = []
        for p in (prior_pages or []):
            t = CHUNKS_BY_PAGE.get(p, "")
            hunted2.extend(_regex_hunt_generic(t, q_kws, intent=intent))
            if len(hunted2) >= 10: break
        if hunted2:
            body_clean = _clean_text("\n".join(hunted2[:8]))
            body_clean = _purge_non_arabic_lines(body_clean)

    # No LLM path (recommended for strict accuracy)
    if (not use_llm) or (tokenizer is None) or (model is None):
        dt = time.time() - t0
        bullets = _bullets_for_display(body_clean or body_raw, question, intent, cfg)
        if not bullets:
            bullets = _as_bullets_clipped(_sentences(body_clean or body_raw), limit=cfg.max_bullets, max_chars=cfg.bullet_max_chars)
        formatted = f"استنادًا إلى النصوص المسترجَعة من المصدر، إليك الخلاصة:\n{bullets}" if bullets else (body_clean or body_raw)
        return _final(dt, formatted, sources)

    # (Optional) LLM refine — guarded
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM  # load already done in main
        system_prompt = (
            "لخّص بوضوح من النص التالي دون إضافة أي معلومات جديدة أو أرقام غير موجودة. "
            "أعد بالبنود (•) وبالعربية فقط."
        )
        user_prompt = f"السؤال: {question}\nالنص:\n{body_clean or body_raw}"
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"[system]\n{system_prompt}\n\n[user]\n{user_prompt}\n\n[assistant]\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k,v in inputs.items()}
        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = eos_id if eos_id is not None else getattr(tokenizer, "pad_token_id", None)
        out_ids = model.generate(**inputs, max_new_tokens=160, do_sample=False, repetition_penalty=1.05,
                                 eos_token_id=eos_id, pad_token_id=pad_id)
        start = inputs["input_ids"].shape[1]
        raw = tokenizer.decode(out_ids[0][start:], skip_special_tokens=True).strip()
        resp = _clean_text(raw); resp = _purge_non_arabic_lines(resp)
        if not resp:
            dt = time.time() - t0
            bullets = _bullets_for_display(body_clean or body_raw, question, intent, cfg)
            formatted = f"استنادًا إلى النصوص المسترجَعة من المصدر، إليك الخلاصة:\n{bullets}" if bullets else (body_clean or body_raw)
            return _final(dt, formatted, sources)
        dt = time.time() - t0
        return _final(dt, resp, sources)
    except Exception as e:
        LOG.warning(f"LLM generation failed: {e}")
        dt = time.time() - t0
        bullets = _bullets_for_display(body_clean or body_raw, question, intent, cfg)
        formatted = f"استنادًا إلى النصوص المسترجَعة من المصدر، إليك الخلاصة:\n{bullets}" if bullets else (body_clean or body_raw)
        return _final(dt, formatted, sources)

# ---------------- Sanity runner ----------------
def _gather_sanity_prompts() -> list:
    ret_prompts = []
    try: ret_prompts = list(getattr(RET, "SANITY_PROMPTS", []) or [])
    except Exception: ret_prompts = []
    seen, merged = set(), []
    for q in (ret_prompts + DEFAULT_SANITY_PROMPTS):
        if q not in seen:
            seen.add(q); merged.append(q)
    return merged

def _pass_loose(answer_text: str) -> bool:
    has_sources = ("Sources:" in answer_text) or ("المصادر:" in answer_text)
    bad = ("لا يقدّم النص المسترجَع تفاصيل كافية" in answer_text)
    return bool(has_sources and not bad)

def _is_meaningful(txt: str) -> bool:
    return bool(txt and len(re.sub(r"\s+","", txt)) >= 12)

def _pass_strict(question: str, body_only: str) -> bool:
    if not _is_meaningful(body_only): return False
    if _is_hours_like(question, "") or _expects_numerics(question):
        return _has_times_or_days(body_only)
    return True

def run_test_prompts(index: RET.HybridIndex, tokenizer, model,
                     use_llm: bool, use_rerank_flag: bool, artifacts_dir: str, cfg: SimpleNamespace, hier=None):
    os.makedirs(artifacts_dir, exist_ok=True)
    results_path = os.path.join(artifacts_dir, "results.jsonl")
    summary_md   = os.path.join(artifacts_dir, "summary.md")
    report_txt   = os.path.join(artifacts_dir, "report.txt")

    results_f = open(results_path, "w", encoding="utf-8")
    report_f  = open(report_txt,  "w", encoding="utf-8")

    def _tee(line=""):
        print(line); report_f.write(line + "\n"); report_f.flush()

    tests = _gather_sanity_prompts()
    if not tests:
        _tee("❌ No sanity prompts available.")
        results_f.close(); report_f.close(); return

    _tee("🧪 Running sanity prompts ...")
    _tee("=" * 80)

    total = len(tests)
    pass_loose_count, pass_strict_count = 0, 0

    for i, q in enumerate(tests, 1):
        _tee(f"\n📝 Test {i}/{total}: {q}")
        _tee("-" * 60)
        try:
            result = ask_once(index, tokenizer, model, q, use_llm=use_llm, use_rerank_flag=use_rerank_flag, cfg=cfg, hier=hier)
            _tee(result)

            body_only, _src_blk = _split_answer(result)
            loose = _pass_loose(result)
            strict = _pass_strict(q, body_only)

            pass_loose_count += int(loose)
            pass_strict_count += int(strict)

            _tee("✅ PASS_LOOSE" if loose else "❌ FAIL_LOOSE")
            _tee("✅ PASS_STRICT" if strict else "❌ FAIL_STRICT")
            _tee("=" * 80)

            rec = {
                "index": i, "question": q, "answer": result, "body_only": body_only,
                "pass_loose": loose, "pass_strict": strict,
            }
            results_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); results_f.flush()

        except Exception as e:
            _tee(f"❌ Error: {e}")
            _tee("=" * 80)

    summary = (
        f"# Sanity Summary\n\n"
        f"- Total: {total}\n"
        f"- PASS_LOOSE: {pass_loose_count}/{total}\n"
        f"- PASS_STRICT: {pass_strict_count}/{total}\n"
        f"\nArtifacts:\n- results.jsonl\n- report.txt\n"
    )
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(summary)

    _tee(f"\nSummary: PASS_LOOSE {pass_loose_count}/{total} | PASS_STRICT {pass_strict_count}/{total}")
    _tee(f"Artifacts saved in: {artifacts_dir}")

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=str, default="Data_pdf_clean_chunks.jsonl", help="Path to chunks (JSONL/JSON)")
    parser.add_argument("--hier-index", type=str, default="heading_inverted_index.json")
    parser.add_argument("--aliases", type=str, default="section_aliases.json")
    parser.add_argument("--save-index", type=str, default=None)
    parser.add_argument("--load-index", type=str, default=None)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--ask", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--out-dir", type=str, default="runs")

    # General controls
    parser.add_argument("--regex-hunt", action="store_true", help="Generic numeric/time/day hunter (question-driven).")
    parser.add_argument("--hourlines-only", action="store_true", help="Keep only lines with times/days/numbers/% when relevant.")
    parser.add_argument("--max-bullets", type=int, default=5, help="Max bullets.")
    parser.add_argument("--bullet-max-chars", type=int, default=120, help="Max characters per bullet.")
    parser.add_argument("--paginate-chars", type=int, default=600, help="Pagination threshold of body.")

    args = parser.parse_args()
    cfg = SimpleNamespace(
        regex_hunt=args.regex_hunt,
        hourlines_only=args.hourlines_only,
        max_bullets=args.max_bullets,
        bullet_max_chars=args.bullet_max_chars,
        paginate_chars=args.paginate_chars,
    )

    # Artifacts dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Logger
    global LOG
    LOG = setup_logger(os.path.join(run_dir, "run.log"))
    LOG.info("Artifacts will be saved under: %s", run_dir)

    # Build/load index
    hier = RET.load_hierarchy(args.hier_index, args.aliases)
    if not os.path.exists(args.chunks):
        LOG.error("Chunks file not found: %s", args.chunks); return
    chunks, chunks_hash = RET.load_chunks(path=args.chunks)

    global CHUNKS_BY_PAGE
    CHUNKS_BY_PAGE = _build_page_text_index(chunks)

    index = RET.HybridIndex(chunks, chunks_hash, hier=hier)

    loaded = False
    if args.load_index and os.path.exists(args.load_index):
        try:
            rlog = logging.getLogger("retrival_model"); lvl = rlog.level; rlog.setLevel(logging.ERROR)
            loaded = index.load(args.load_index); rlog.setLevel(lvl)
            if loaded: LOG.info("Index loaded successfully from %s", args.load_index)
        except Exception as e:
            LOG.info("Will rebuild index: %s", e)

    if not loaded:
        LOG.info("Building index ..."); index.build()
        if args.save_index:
            try: index.save(args.save_index); LOG.info("Index saved to %s", args.save_index)
            except Exception as e: LOG.warning("Failed to save index: %s", e)

    # Optional LLM
    tok = mdl = None
    use_llm = not args.no_llm
    if use_llm:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            use_cuda = (args.device != "cpu") and (torch is not None) and hasattr(torch, "cuda") and torch.cuda.is_available()
            if args.device == "cuda" and not use_cuda:
                LOG.warning("CUDA requested but not available; falling back to CPU.")
            bf16_supported = use_cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            dtype_fp = torch.bfloat16 if (bf16_supported and torch is not None) else (torch.float16 if (use_cuda and torch is not None) else None)
            model_kwargs = {"trust_remote_code": True}
            if args.device == "cpu" or not use_cuda:
                model_kwargs["device_map"] = "cpu"
                if torch is not None: model_kwargs["torch_dtype"] = torch.float32
            else:
                model_kwargs["device_map"] = "auto"
                if dtype_fp is not None: model_kwargs["torch_dtype"] = dtype_fp
            if args.use_4bit or args.use_8bit:
                try:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=bool(args.use_4bit),
                        load_in_8bit=bool(args.use_8bit),
                        bnb_4bit_compute_dtype=(torch.bfloat16 if bf16_supported else (torch.float16 if use_cuda else None)),
                    )
                except Exception as e:
                    LOG.warning("Quantization setup failed: %s", e)
            tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        except Exception as e:
            LOG.warning(f"Failed to load LLM ({args.model}); continuing retrieval-only. Error: {e}")
            tok = mdl = None; use_llm = False

    use_rerank_flag = not args.no_rerank

    if args.test or args.sanity:
        run_test_prompts(index, tok, mdl, use_llm=use_llm, use_rerank_flag=use_rerank_flag, artifacts_dir=run_dir, cfg=cfg, hier=hier)
        print(f"\n✅ Saved artifacts under: {run_dir}")
        return

    if args.ask:
        ans = ask_once(index, tok, mdl, args.ask, use_llm=use_llm, use_rerank_flag=use_rerank_flag, cfg=cfg, hier=hier)
        single_path = os.path.join(run_dir, "single_answer.txt")
        with open(single_path, "w", encoding="utf-8") as f: f.write(ans)
        print(ans); print(f"\n✅ Saved single answer to: {single_path}")
        return

    # Interactive
    print("Ready. اطرح سؤالك (اكتب 'exit' للخروج)\n")
    interactive_path = os.path.join(run_dir, "interactive_transcript.txt")
    with open(interactive_path, "w", encoding="utf-8") as trans:
        while True:
            try: q = input("سؤالك: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting."); break
            if not q: continue
            if q.lower() in ("exit","quit","q"): print("Exiting."); break
            ans = ask_once(index, tok, mdl, q, use_llm=use_llm, use_rerank_flag=use_rerank_flag, cfg=cfg, hier=hier)
            print(ans); trans.write(f"\nQ: {q}\n{ans}\n"); trans.flush()
    print(f"\n✅ Interactive transcript saved to: {interactive_path}")

if __name__ == "__main__":
    main()

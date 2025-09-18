# -*- coding: utf-8 -*-
"""
RAG Orchestrator (Arabic-only, generalized) — TEXT-ANCHOR ONLY
- No access to hierarchy object at all (avoids 'HierData has no attribute get')
- Arabic normalization, digit unification, and diacritics stripping
- Page-aware fallback (from cited pages → text-anchored pages → global)
- Arabic-centric regex hunter for numbers/times/days/%/durations
- Policy/finance signal scoring (procurement & currency cues)
- Clean bullets + pagination; strict numeric/time guard

Typical use:
python NewRag_text_anchor_only.py --chunks Data_pdf_clean_chunks.jsonl --sanity \
  --device cpu --no-llm --regex-hunt --hourlines-only --max-bullets 5 \
  --bullet-max-chars 120 --paginate-chars 600 --out-dir runs
"""

import os, sys, re, json, time, argparse, logging
from datetime import datetime
from collections import defaultdict
from types import SimpleNamespace

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

try:
    import torch  # optional
except Exception:
    torch = None

# External retriever
import retrival_model as RET

# ---------------- Helpers ----------------

def _get_attr_or_key(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    val = getattr(obj, key, None)
    if val is not None:
        return val
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
        if c is None or str(c).strip() == "":
            continue
        try:
            return int(str(c).strip())
        except Exception:
            pass
    return None


CHUNKS_BY_PAGE = {}


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
            try:
                txt = txt.decode("utf-8", "ignore")
            except Exception:
                txt = ""
        pg = _first_page_like(ch)
        if pg is None:
            src = _get_attr_or_key(ch, "source") or _get_attr_or_key(ch, "doc_source") or ""
            m = re.search(r"(?:[Pp]age|(?:ال)?صفحة)\s+(\d+)", str(src))
            if m:
                try:
                    pg = int(m.group(1))
                except Exception:
                    pg = None
        if pg is not None and txt:
            pages[int(pg)].append(str(txt))
    return {p: "\n".join(v) for p, v in pages.items()}


# ---------------- Logging ----------------

def setup_logger(log_path: str):
    logger = logging.getLogger("rag_orchestrator")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
    logger.handlers = []
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


LOG = logging.getLogger("rag_orchestrator")


# ---------------- Sanity prompts ----------------
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


# ---------------- Arabic-only normalization ----------------
_ARABIC_DIACRITICS = dict.fromkeys(map(ord, "ًٌٍَُِّْـ"), None)
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _to_western_digits(s: str) -> str:
    return (s or "").translate(_ARABIC_DIGITS)


def _norm_arabic_letters(s: str) -> str:
    if not s:
        return ""
    s = s.translate(_ARABIC_DIACRITICS)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ى", "ي").replace("ة", "ه")
    s = s.replace("ؤ", "و").replace("ئ", "ي")
    return s


def _normalize_text_ar(s: str) -> str:
    s = _to_western_digits(s or "").strip()
    s = _norm_arabic_letters(s)
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------- Arabic checks ----------------
_HEADING_PATTERNS = [
    r"^\s*الإجابة\s*:?$",
    r"^\s*الخلاصة\s*:?\s*$",
    r"^\s*الملخص\s*:?\s*$",
    r"^\s*Summary\s*:?\s*$",
    r"^\s*Answer\s*:?\s*$",
]
_AR_DAYS = [
    "الاحد", "الأحد", "الاثنين", "الإثنين", "الثلاثاء", "الاربعاء", "الأربعاء",
    "الخميس", "الجمعة", "السبت",
]

_TIME_PATTERNS = [
    r"\b\d{1,2}:\d{2}\b",
    r"\b\d{1,2}[:٫]\d{2}\b",
    r"\b\d{1,2}\s*[-–]\s*\d{1,2}\b",
    r"\b\d{1,2}\s*(?:إلى|حتى)\s*\d{1,2}\b",
    r"\b\d{1,2}\s*(?:ص|م)\b",
]
_PERCENT_RX = re.compile(r"\b\d{1,3}\s*[%٪]\b")
_DURATION_RX = re.compile(r"\b\d{1,2}\s*(?:دقيقة|دقائق|ساعة|ساعات|يوم|أيام)\b")
_SECTION_HEAVY = re.compile(r"(?:\d+\.){2,}\d+")


def _strip_mojibake(s: str) -> str:
    if not s:
        return ""
    return (
        s.replace("\ufeff", "")
        .replace("\uFFFD", "")
        .replace("�", "")
        .replace("\xa0", " ")
    )


def _arabic_ratio(s: str) -> float:
    if not s:
        return 1.0
    letters = re.findall(r"\w", s, flags=re.UNICODE)
    if not letters:
        return 1.0
    arabic = re.findall(r"[ء-ي]", s)
    return (len(arabic) / max(1, len(letters)))


def _purge_non_arabic_lines(s: str, min_ratio: float = 0.66) -> str:
    if not s:
        return s
    keep = []
    for line in s.splitlines():
        ln = line.strip()
        if not ln:
            continue
        t = _to_western_digits(ln)
        if any(re.search(p, t) for p in _TIME_PATTERNS) or _PERCENT_RX.search(t) or _DURATION_RX.search(t):
            keep.append(ln)
            continue
        ratio = _arabic_ratio(ln)
        if ratio >= min_ratio:
            keep.append(ln)
    return "\n".join(keep)


def _has_times_or_days(txt: str) -> bool:
    if not txt:
        return False
    t = _to_western_digits(txt)
    if any(day in t for day in _AR_DAYS):
        return True
    if any(re.search(p, t) for p in _TIME_PATTERNS):
        return True
    if _DURATION_RX.search(t):
        return True
    if _PERCENT_RX.search(t):
        return True
    return False


def _sentences(txt: str):
    if not txt:
        return []
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    keep = []
    for l in lines:
        if any(re.match(p, l) for p in _HEADING_PATTERNS):
            continue
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


def _clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = _strip_mojibake(txt)
    txt = re.sub(r"^```.*?$", "", txt, flags=re.M | re.S)
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    keep = []
    for l in lines:
        if any(re.match(p, l) for p in _HEADING_PATTERNS):
            continue
        if "فهرس المحتويات" in l or "جميع الحقوق محفوظة" in l:
            continue
        if _SECTION_HEAVY.search(l):
            continue
        keep.append(l)
    txt = " ".join(keep).strip()
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _paginate_text(text: str, max_chars: int = 600):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    parts, cur, count = [], [], 0
    for line in text.splitlines():
        if count + len(line) + 1 > max_chars:
            parts.append("\n".join(cur).strip())
            cur, count = [line], len(line)
        else:
            cur.append(line)
            count += len(line) + 1
    if cur:
        parts.append("\n".join(cur).strip())
    return parts


def _split_answer(answer_text: str):
    if not answer_text:
        return "", ""
    parts = re.split(r"\n(?=Sources:|المصادر:)", answer_text, maxsplit=1)
    body = parts[0].strip()
    sources = parts[1].strip() if len(parts) > 1 else ""
    return body, sources


# ---------------- Intent & aliasing ----------------
INTENT_ALIASES_AR = {
    "work_hours": ["ساعات العمل", "اوقات الدوام", "الدوام الرسمي", "نظام الدوام"],
    "ramadan_hours": ["رمضان", "ساعات رمضان", "دوام رمضان", "الصوم"],
    "overtime": ["العمل الاضافي", "الساعات الاضافيه", "اضافي"],
    "holidays_work": ["العطل", "العطل الرسميه", "العمل في العطل"],
    "timesheets": ["الجداول الزمنيه", "التقارير", "الحضور", "الانصراف"],
    "payroll": ["الرواتب", "صرف الرواتب", "الاجر"],
    "transport": ["بدل المواصلات", "المواصلات", "نقل"],
    "hourly_leave": ["مغادره ساعيه", "اذن مغادره"],
    "annual_leave": ["الاجازه السنويه", "سنويه"],
    "sick_leave": ["الاجازه المرضيه", "مرضيه"],
    "carryover": ["ترحيل الاجازات", "غير المستخدمه", "رصيد الاجازات"],
    "maternity": ["اجازه الامومه", "امومه"],
    "procurement_conflict": ["تضارب المصالح", "اخلاقيات الشراء"],
    "per_diem": ["مياومات", "بدل سفر"],
}


# -------- Text-only anchoring (NO hierarchy access) --------

def _find_anchor_pages_by_text_alias(alias_keywords):
    """Score pages by presence of alias keywords in page text (Arabic-normalized)."""
    if not alias_keywords:
        return []
    norm_aliases = [_normalize_text_ar(a) for a in alias_keywords]
    scored = []
    for p, txt in CHUNKS_BY_PAGE.items():
        T = _normalize_text_ar(_clean_text(txt))
        score = sum(1 for a in norm_aliases if a and a in T)
        if score:
            scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [p for _, p in scored[:8]]


# ---------------- Regex hunter ----------------
GENERIC_RXS = [
    re.compile(r"\b\d{1,2}:\d{2}\b"),
    re.compile(r"\b\d{1,2}[:٫]\d{2}\b"),
    re.compile(r"\b\d{1,2}\s*(?:ص|م)\b"),
    re.compile(r"\b\d{1,2}\s*(?:إلى|حتى|-\s*|–\s*)\s*\d{1,2}\b"),
    re.compile(r"\b\d{1,3}\s*[%٪]\b"),
    re.compile(r"\b\d{1,2}\s*(?:دقيقة|دقائق|ساعة|ساعات|يوم|أيام)\b"),
    re.compile(r"\b\d+\b"),
]


def _line_has_generic_numeric(t: str) -> bool:
    T = _to_western_digits(t)
    for rx in GENERIC_RXS:
        if rx.search(T):
            return True
    if any(d in T for d in _AR_DAYS):
        return True
    return False


def _regex_hunt_generic(text: str, q_kws):
    if not text:
        return []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    hits = []
    DOMAIN_KWS_AR = [
        "موافقه", "موافقه مسبقه", "موافقه خطيه", "اذن", "اذن خطي",
        "تعويض", "يحتسب", "يعتمد", "يتطلب", "يلزم", "يجب", "لا يجوز", "يحق", "وفق", "حسب",
        "دوام", "ساعات", "اجازه", "اجازات", "عطله", "العطل", "مغادره", "استراحه",
        "بدل", "بدل مواصلات", "مياومات", "بدل سفر", "رواتب", "صرف", "موعد", "الحضور", "الجداول الزمنيه",
        "عروض اسعار", "عرض سعر", "عروض", "مناقصة", "توريد", "تضارب المصالح",
        "شيكل", "شواقل", "دينار", "دولار", "₪", "NIS", "ILS", "USD",
    ]
    for l in lines:
        L = _clean_text(l)
        if not L:
            continue
        if _SECTION_HEAVY.search(L):
            continue
        if "فهرس المحتويات" in L or "جميع الحقوق محفوظة" in L:
            continue
        score = 0
        T = _normalize_text_ar(L)
        if _line_has_generic_numeric(L):
            score += 3
        domain_kws = set(DOMAIN_KWS_AR + (q_kws or []))
        if _line_has_generic_numeric(L) and any(kw in T for kw in domain_kws):
            score += 2
        if re.search(r"\b\d+\b", T) and not any(k in T for k in domain_kws):
            score -= 2
        if any(stem in T for stem in ("يجب", "يلزم", "لا يجوز", "يحق", "وفق", "حسب", "على", "تكون")):
            score += 1
        if 20 <= len(T) <= 160:
            score += 1
        if score > 0:
            hits.append((score, L))
    hits.sort(key=lambda x: (-x[0], len(x[1])))
    return [h[1] for h in hits]


# ---------------- Bullets & display ----------------

def _clip(s: str, n: int) -> str:
    s = s.strip()
    return (s[: max(1, n - 1)].rstrip() + "…") if n and len(s) > n else s


def _as_bullets_clipped(sents, limit=5, max_chars=120):
    sents = sents[: max(1, limit)]
    sents = [_clip(s, max_chars) for s in sents]
    out, seen = [], set()
    for s in sents:
        if s and s not in seen:
            seen.add(s)
            out.append("• " + s)
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


# ---------------- Intent helpers ----------------

def _norm_tokens(s: str):
    s = _to_western_digits(s or "")
    s = re.sub(r"[^\w\s%٪:–\-]+", " ", s, flags=re.UNICODE)
    return [t for t in s.split() if t.strip()]


def _question_keywords(q: str):
    toks = _norm_tokens(q)
    extras = []
    if "ساعات" in q or "دوام" in q:
        extras += ["ساعات", "دوام", "من", "إلى", "حتى", "اوقات"]
    if "رمضان" in q or "الصوم" in q:
        extras += ["رمضان", "الصوم"]
    if "إضاف" in q or "اضافي" in q:
        extras += ["إضافية", "العمل الاضافي", "الساعات الاضافيه", "اجر"]
    if "عطل" in q or "عطلة" in q:
        extras += ["العطل", "الرسمية", "عطلة", "نهاية", "أسبوع"]
    if "استراحة" in q or "راحة" in q:
        extras += ["استراحة", "راحة", "مدة", "مدتها"]
    if "مغادرة" in q or "اذن" in q:
        extras += ["مغادرة", "ساعية", "الحد", "الأقصى", "شهري", "اذن"]
    if "إجازة" in q or "اجازه" in q:
        extras += ["إجازة", "أيام", "مدة", "سنو"]
    if "سقف" in q or "عروض" in q:
        extras += ["سقف", "عروض", "أسعار", "ثلاثة"]
    if "تضارب" in q:
        extras += ["تضارب", "مصالح", "الهدايا"]
    return list(dict.fromkeys(toks + extras))


def _expects_numerics(q: str) -> bool:
    q = q or ""
    cues = (
        "كم", "مدة", "ما الحد", "الحد", "من وإلى", "من والى", "من إلى", "متى", "النسبة", "%", "٪",
        "ساعات", "دقائق", "يوم", "أيام", "أجر", "تعويض", "سقف", "بدل", "مياومات", "الحد الأقصى", "شهري",
    )
    return any(c in q for c in cues)


def _is_hours_like(question: str, intent: str = "") -> bool:
    q = _normalize_text_ar(question or "")
    base = [
        "ساعات", "دوام", "اوقات", "رمضان", "الصوم", "ايام", "اضافي", "الساعات الاضافيه",
        "استراحه", "مغادره", "اذن", "وقت", "بدل", "مياومات", "سفر", "رواتب", "صرف", "موعد",
        "العطل", "العمل في العطل", "تعويض", "اجر", "جداول زمنيه", "الحضور",
    ]
    has_num = bool(re.search(r"\d", _to_western_digits(q))) or ("%" in q or "٪" in q)
    if has_num:
        return True
    return any(tok in q for tok in base) or intent in ("work_hours", "ramadan_hours", "overtime", "work_days", "breaks")


# ---------------- Core Q&A ----------------

def ask_once(index: RET.HybridIndex, tokenizer, model, question: str,
             use_llm: bool = True, use_rerank_flag: bool = True, cfg: SimpleNamespace = None) -> str:
    t0 = time.time()
    cfg = cfg or SimpleNamespace(max_bullets=5, bullet_max_chars=120, paginate_chars=600,
                                 hourlines_only=False, regex_hunt=True)

    intent = RET.classify_intent(question)
    extractive_answer = RET.answer(question, index, intent, use_rerank_flag=use_rerank_flag)

    # split body/sources
    lines = str(extractive_answer or "").split("\n")
    body_lines, source_lines, sources_started = [], [], False
    for line in lines:
        ls = line.strip()
        if ls.startswith("Sources:") or ls.startswith("المصادر:"):
            sources_started = True
            source_lines.append(line)
        elif sources_started:
            source_lines.append(line)
        else:
            body_lines.append(line)
    body_raw = "\n".join(body_lines).strip()
    sources = "\n".join(source_lines).strip()

    # page context from cited pages
    def _parse_pages_from_sources(sources_text):
        if not sources_text:
            return []
        s = _to_western_digits(sources_text)
        pages = set()
        for m in re.findall(r"(?:\b[Pp]age\b|(?:ال)?صفحة)\s+(\d+)", s):
            try:
                pages.add(int(m))
            except Exception:
                pass
        return sorted(pages)

    def _page_ctx_from_pages(pages, max_chars=3500):
        if not pages:
            return ""
        buf, total = [], 0
        for p in pages:
            txt = CHUNKS_BY_PAGE.get(p, "")
            if not txt:
                continue
            t = _clean_text(txt)
            if total + len(t) > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    buf.append(t[:remaining])
                    total = max_chars
                    break
            else:
                buf.append(t)
                total += len(t)
        return " ".join(buf).strip()

    cited_pages = _parse_pages_from_sources(sources)
    page_ctx = _page_ctx_from_pages(cited_pages, max_chars=3500)

    # Arabic-only dynamic aliasing → TEXT-ONLY anchoring (no hierarchy access)
    alias_list = INTENT_ALIASES_AR.get(intent, [])
    if not alias_list:
        q_norm = _normalize_text_ar(question)
        if any(k in q_norm for k in ["رمضان", "الصوم"]):
            alias_list = INTENT_ALIASES_AR["ramadan_hours"]
        elif any(k in q_norm for k in ["اضافي", "الاضافي", "الساعات الاضافيه"]):
            alias_list = INTENT_ALIASES_AR["overtime"]
        elif any(k in q_norm for k in ["اجازه سنويه", "الاجازه السنويه", "سنويه"]):
            alias_list = INTENT_ALIASES_AR["annual_leave"]
        elif any(k in q_norm for k in ["مرضيه", "الاجازه المرضيه"]):
            alias_list = INTENT_ALIASES_AR["sick_leave"]
        elif any(k in q_norm for k in ["مغادره", "اذن مغادره"]):
            alias_list = INTENT_ALIASES_AR["hourly_leave"]
        elif any(k in q_norm for k in ["رواتب", "صرف الرواتب", "اجر"]):
            alias_list = INTENT_ALIASES_AR["payroll"]
        elif any(k in q_norm for k in ["مواصلات", "بدل المواصلات"]):
            alias_list = INTENT_ALIASES_AR["transport"]
        elif any(k in q_norm for k in ["مياومات", "بدل سفر"]):
            alias_list = INTENT_ALIASES_AR["per_diem"]
        elif any(k in q_norm for k in ["تضارب", "تضارب المصالح"]):
            alias_list = INTENT_ALIASES_AR["procurement_conflict"]
        elif any(k in q_norm for k in ["عطل", "العطل"]):
            alias_list = INTENT_ALIASES_AR["holidays_work"]
        else:
            if _is_hours_like(question, intent):
                alias_list = INTENT_ALIASES_AR["work_hours"]

    anchor_pages = _find_anchor_pages_by_text_alias(alias_list) if alias_list else []
    if anchor_pages:
        anchored_ctx = _page_ctx_from_pages(anchor_pages, max_chars=3500)
        if anchored_ctx:
            page_ctx = anchored_ctx

    # 1) Cleanup & fallback
    hours_like = _is_hours_like(question, intent)
    tmp_body = _clean_text(body_raw)
    if (not tmp_body) or ("لا يقدّم النص" in tmp_body) or ("لم أعثر" in tmp_body):
        body_raw = page_ctx if page_ctx else body_raw
    elif (hours_like or _expects_numerics(question)) and not _has_times_or_days(tmp_body):
        if _has_times_or_days(page_ctx):
            body_raw = page_ctx
        else:
            q_kws = _question_keywords(question)
            candidate_pages = []
            for p, txt in CHUNKS_BY_PAGE.items():
                t = _clean_text(txt)
                if any(kw in t for kw in q_kws):
                    candidate_pages.append(p)
            if candidate_pages and not page_ctx:
                body_raw = _page_ctx_from_pages(candidate_pages[:6], max_chars=3500)

    # 2) Regex hunter
    if cfg.regex_hunt:
        q_kws = _question_keywords(question)
        hunted = []
        for p in cited_pages or []:
            t = CHUNKS_BY_PAGE.get(p, "")
            if not t:
                continue
            hunted.extend(_regex_hunt_generic(t, q_kws))
            if len(hunted) >= 10:
                break
        if not hunted:
            anchor_pages2 = []
            for p, txt in CHUNKS_BY_PAGE.items():
                t = _clean_text(txt)
                if any(kw in t for kw in q_kws):
                    anchor_pages2.append(p)
            for p in anchor_pages2:
                t = CHUNKS_BY_PAGE.get(p, "")
                hunted.extend(_regex_hunt_generic(t, q_kws))
                if len(hunted) >= 12:
                    break
        if not hunted:
            all_text = "\n".join(CHUNKS_BY_PAGE.get(p, "") for p in sorted(CHUNKS_BY_PAGE))
            hunted = _regex_hunt_generic(all_text, q_kws)
        if hunted and (hours_like or _expects_numerics(question)):
            body_raw = "\n".join(hunted[:12])

    # 3) Final formatting
    def _final(dt, text):
        nonlocal sources
        if sources:
            lines = [l.strip() for l in sources.splitlines() if l.strip()]
            seen = set(); clean = []
            for l in lines:
                k = re.sub(r"(page|صفحة)\s+\d+", "page", l, flags=re.I)
                if k in seen:
                    continue
                seen.add(k)
                clean.append(l)
            sources = "\n".join(clean[:5])
        parts = _paginate_text(text, max_chars=cfg.paginate_chars)
        if len(parts) > 1:
            labeled = []
            for i, p in enumerate(parts, 1):
                labeled.append(f"الجزء {i}/{len(parts)}:\n{p}")
            text = "\n\n".join(labeled)
        return (f"⏱ {dt:.2f}s | 🤖 {text}\n{sources}" if sources else f"⏱ {dt:.2f}s | 🤖 {text}")

    body_clean = _clean_text(body_raw)
    body_clean = _purge_non_arabic_lines(body_clean)

    if (not body_clean) or ((hours_like or _expects_numerics(question)) and not _has_times_or_days(body_clean)):
        if not _has_times_or_days(body_clean):
            dt = time.time() - t0
            msg = "لا يذكر النص رقماً/وقتاً محدداً لهذا السؤال ضمن الصفحات المستند إليها."
            return _final(dt, msg)

    # No LLM refine by default
    if (not use_llm) or (tokenizer is None) or (model is None):
        dt = time.time() - t0
        bullets = _bullets_for_display(body_clean or body_raw, question, intent, cfg)
        if not bullets:
            bullets = _as_bullets_clipped(_sentences(body_clean or body_raw), limit=cfg.max_bullets, max_chars=cfg.bullet_max_chars)
        formatted = f"استنادًا إلى النصوص المسترجَعة من المصدر، إليك الخلاصة:\n{bullets}" if bullets else (body_clean or body_raw)
        return _final(dt, formatted)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        system_prompt = (
            "لخّص بوضوح من النص التالي دون إضافة أي معلومات جديدة أو أرقام غير موجودة. "
            "أعد بالبنود (•) وبالعربية فقط."
        )
        user_prompt = f"السؤال: {question}\nالنص:\n{body_clean or body_raw}"
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"[system]\n{system_prompt}\n\n[user]\n{user_prompt}\n\n[assistant]\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = eos_id if eos_id is not None else getattr(tokenizer, "pad_token_id", None)
        out_ids = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        start = inputs["input_ids"].shape[1]
        raw = tokenizer.decode(out_ids[0][start:], skip_special_tokens=True).strip()
        resp = _clean_text(raw)
        resp = _purge_non_arabic_lines(resp)
        if not resp:
            dt = time.time() - t0
            bullets = _bullets_for_display(body_clean or body_raw, question, intent, cfg)
            formatted = f"استنادًا إلى النصوص المسترجَعة من المصدر، إليك الخلاصة:\n{bullets}" if bullets else (body_clean or body_raw)
            return _final(dt, formatted)
        dt = time.time() - t0
        return _final(dt, resp)
    except Exception as e:
        LOG.warning("LLM generation failed: %s", e)
        dt = time.time() - t0
        bullets = _bullets_for_display(body_clean or body_raw, question, intent, cfg)
        formatted = f"استنادًا إلى النصوص المسترجَعة من المصدر، إليك الخلاصة:\n{bullets}" if bullets else (body_clean or body_raw)
        return _final(dt, formatted)


# ---------------- Sanity runner ----------------

def _gather_sanity_prompts() -> list:
    ret_prompts = []
    try:
        ret_prompts = list(getattr(RET, "SANITY_PROMPTS", []) or [])
    except Exception:
        ret_prompts = []
    seen, merged = set(), []
    for q in (ret_prompts + DEFAULT_SANITY_PROMPTS):
        if q not in seen:
            seen.add(q)
            merged.append(q)
    return merged


def _pass_loose(answer_text: str) -> bool:
    has_sources = ("Sources:" in answer_text) or ("المصادر:" in answer_text)
    bad = ("لا يقدّم النص المسترجَع تفاصيل كافية" in answer_text)
    return bool(has_sources and not bad)


def _is_meaningful(txt: str) -> bool:
    return bool(txt and len(re.sub(r"\s+", "", txt)) >= 12)


def _pass_strict(question: str, body_only: str) -> bool:
    if not _is_meaningful(body_only):
        return False
    if _is_hours_like(question, "") or _expects_numerics(question):
        return _has_times_or_days(body_only)
    return True


def run_test_prompts(index: RET.HybridIndex, tokenizer, model,
                     use_llm: bool, use_rerank_flag: bool, artifacts_dir: str, cfg: SimpleNamespace):
    os.makedirs(artifacts_dir, exist_ok=True)
    results_path = os.path.join(artifacts_dir, "results.jsonl")
    summary_md = os.path.join(artifacts_dir, "summary.md")
    report_txt = os.path.join(artifacts_dir, "report.txt")

    results_f = open(results_path, "w", encoding="utf-8")
    report_f = open(report_txt, "w", encoding="utf-8")

    def _tee(line=""):
        print(line)
        report_f.write(line + "\n")
        report_f.flush()

    tests = _gather_sanity_prompts()
    if not tests:
        _tee("❌ No sanity prompts available.")
        results_f.close()
        report_f.close()
        return

    _tee("🧪 Running sanity prompts ...")
    _tee("=" * 80)

    total = len(tests)
    pass_loose_count, pass_strict_count = 0, 0

    for i, q in enumerate(tests, 1):
        _tee(f"\n📝 Test {i}/{total}: {q}")
        _tee("-" * 60)
        try:
            result = ask_once(index, tokenizer, model, q, use_llm=use_llm, use_rerank_flag=use_rerank_flag, cfg=cfg)
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
                "index": i,
                "question": q,
                "answer": result,
                "body_only": body_only,
                "pass_loose": loose,
                "pass_strict": strict,
            }
            results_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results_f.flush()

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

    results_f.close()
    report_f.close()


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
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", type=str, default="runs")

    # Controls
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    global LOG
    LOG = setup_logger(os.path.join(run_dir, "run.log"))
    LOG.info("Artifacts will be saved under: %s", run_dir)

    # Build/load index (we still load hierarchy for RET, but we never touch it here)
    hier = RET.load_hierarchy(args.hier_index, args.aliases)
    if not os.path.exists(args.chunks):
        LOG.error("Chunks file not found: %s", args.chunks)
        return
    chunks, chunks_hash = RET.load_chunks(path=args.chunks)

    global CHUNKS_BY_PAGE
    CHUNKS_BY_PAGE = _build_page_text_index(chunks)

    index = RET.HybridIndex(chunks, chunks_hash, hier=hier)

    loaded = False
    if args.load_index and os.path.exists(args.load_index):
        try:
            rlog = logging.getLogger("retrival_model")
            lvl = rlog.level
            rlog.setLevel(logging.ERROR)
            loaded = index.load(args.load_index)
            rlog.setLevel(lvl)
            if loaded:
                LOG.info("Index loaded successfully from %s", args.load_index)
        except Exception as e:
            LOG.info("Will rebuild index: %s", e)

    if not loaded:
        LOG.info("Building index ...")
        index.build()
        if args.save_index:
            try:
                index.save(args.save_index)
                LOG.info("Index saved to %s", args.save_index)
            except Exception as e:
                LOG.warning("Failed to save index: %s", e)

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
            dtype_fp = torch.bfloat16 if (bf16_supported and torch is not None) else (torch.float16 if use_cuda and torch is not None else None)
            model_kwargs = {"trust_remote_code": True}
            if args.device == "cpu" or not use_cuda:
                model_kwargs["device_map"] = "cpu"
                if torch is not None:
                    model_kwargs["torch_dtype"] = torch.float32
            else:
                model_kwargs["device_map"] = "auto"
                if dtype_fp is not None:
                    model_kwargs["torch_dtype"] = dtype_fp
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
            LOG.warning("Failed to load LLM (%s); continuing retrieval-only. Error: %s", args.model, e)
            tok = mdl = None
            use_llm = False

    use_rerank_flag = not args.no_rerank

    if args.test or args.sanity:
        run_test_prompts(index, tok, mdl, use_llm=use_llm, use_rerank_flag=use_rerank_flag, artifacts_dir=run_dir, cfg=cfg)
        print(f"\n✅ Saved artifacts under: {run_dir}")
        return

    if args.ask:
        ans = ask_once(index, tok, mdl, args.ask, use_llm=use_llm, use_rerank_flag=use_rerank_flag, cfg=cfg)
        single_path = os.path.join(run_dir, "single_answer.txt")
        with open(single_path, "w", encoding="utf-8") as f:
            f.write(ans)
        print(ans)
        print(f"\n✅ Saved single answer to: {single_path}")
        return

    print("Ready. اطرح سؤالك (اكتب 'exit' للخروج)\n")
    interactive_path = os.path.join(run_dir, "interactive_transcript.txt")
    with open(interactive_path, "w", encoding="utf-8") as trans:
        while True:
            try:
                q = input("سؤالك: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                print("Exiting.")
                break
            ans = ask_once(index, tok, mdl, q, use_llm=use_llm, use_rerank_flag=use_rerank_flag, cfg=cfg)
            print(ans)
            trans.write(f"\nQ: {q}\n{ans}\n")
            trans.flush()
    print(f"\n✅ Interactive transcript saved to: {interactive_path}")


if __name__ == "__main__":
    main()

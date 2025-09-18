# -*- coding: utf-8 -*-
"""
RAG Orchestrator (Arabic) — grounded-only answers, page-aware fallback, compact bullets,
pagination, and persistent artifacts for sanity tests. (HARDENED + COMPACT)

Adds on top of your version:
- Compact bullets: salient-line selection, noise filtering, deduplication, smart truncation
- CLI knobs: --max-bullets, --bullet-max-chars, --paginate-chars, --hourlines-only
- Hours/days hardening still in place (page-aware rescue + time/day focus)
- Arabic-only purge + mojibake cleanup

Usage examples:
# Strict, compact, no LLM (recommended for your strict criteria)
python NewRag.py \
  --chunks Data_pdf_clean_chunks.jsonl \
  --sanity \
  --device cuda \
  --use-4bit \
  --no-llm \
  --max-bullets 5 \
  --bullet-max-chars 140 \
  --paginate-chars 700 \
  --hourlines-only \
  --out-dir runs

# Single question
python NewRag.py \
  --chunks Data_pdf_clean_chunks.jsonl \
  --ask "ما هي ساعات الدوام الرسمية من وإلى؟" \
  --device cuda \
  --use-4bit \
  --no-llm \
  --max-bullets 5 \
  --bullet-max-chars 140 \
  --hourlines-only
"""

import os
import sys
import re
import json
import time
import argparse
import logging
from datetime import datetime
from collections import defaultdict

# Quieter logs / stable console
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import torch
except Exception:
    torch = None

# Your retriever module
import retrival_model as RET

# ---------------- Utilities for dict-or-object chunks ----------------
def _get_attr_or_key(obj, key, default=None):
    """Return obj[key] if dict-like, else getattr(obj, key, default).
       Also tries common nesting spots like .meta / .metadata if present."""
    # dict-like
    if isinstance(obj, dict):
        return obj.get(key, default)
    # object attribute
    val = getattr(obj, key, None)
    if val is not None:
        return val
    # nested common containers
    for nest in ("meta", "metadata", "__dict__"):
        container = getattr(obj, nest, None)
        if isinstance(container, dict) and key in container:
            return container.get(key)
    return default

def _first_non_empty(*vals):
    for v in vals:
        if v is None:
            continue
        s = str(v)
        if s.strip():
            return s
    return ""

def _first_page_like(obj):
    """Try multiple possible page fields and coerce to int if possible."""
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

# ---------------- Global page index (built in main) ----------------
CHUNKS_BY_PAGE = {}  # {int page: "full concatenated text for that page"}

def _build_page_text_index(chunks):
    pages = defaultdict(list)
    for ch in chunks:
        # robust text extraction (support dicts and objects)
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
            # sometimes chunks carry page in a nested "source" string (e.g., "Data_pdf.pdf - page 16")
            src = _get_attr_or_key(ch, "source") or _get_attr_or_key(ch, "doc_source") or ""
            try:
                pg_guess = re.search(r"(?:[Pp]age|صفحة|الصفحة)\s+(\d+)", str(src))
                if pg_guess:
                    pg = int(pg_guess.group(1))
            except Exception:
                pg = None
        if pg is not None and txt:
            pages[int(pg)].append(str(txt))
    # join per page
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

LOG = logging.getLogger("rag_orchestrator")  # reconfigured in main()

# ---------------- Sanity Prompts ----------------
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
    r"^\s*الإجابة\s+المختصرة\s*:?\s*$",
    r"^\s*الخلاصة\s*:?\s*$",
    r"^\s*الملخص\s*:?\s*$",
    r"^\s*Summary\s*:?\s*$",
    r"^\s*Answer\s*:?\s*$",
]

_AR_DAYS = ["الأحد", "الإثنين", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت"]

# Accepts: 8:30, 8٫30, 8-5, 8 – 5, 8 إلى 5, 8 حتى 5, 8 ص/5 م
_TIME_PATTERNS = [
    r"\b\d{1,2}:\d{2}\b",                 # 8:30
    r"\b\d{1,2}[:٫]\d{2}\b",              # 8٫30
    r"\b\d{1,2}\s*[-–]\s*\d{1,2}\b",      # 8-5 / 8–5
    r"\b\d{1,2}\s*(?:إلى|حتى)\s*\d{1,2}\b",  # 8 إلى 5 / 8 حتى 5
    r"\b\d{1,2}\s*(?:ص|م)\b",             # 8 ص / 5 م
]

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_AR_LETTER_RX = re.compile(r"[ء-ي]")  # coarse Arabic block

def _to_western_digits(s: str) -> str:
    return (s or "").translate(_ARABIC_DIGITS)

def _strip_mojibake(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\ufeff", "").replace(" ", "").replace("\uFFFD", "")
    s = re.sub(r"uni[0-9A-Fa-f]{4}", "", s)
    return s

def _arabic_ratio(s: str) -> float:
    if not s:
        return 1.0
    letters = re.findall(r"\w", s, flags=re.UNICODE)
    if not letters:
        return 1.0
    arabic = _AR_LETTER_RX.findall(s)
    return (len(arabic) / max(1, len(letters)))

def _purge_non_arabic_lines(s: str, min_ratio: float = 0.80) -> str:
    """Drop lines with too many Latin/CJK letters; keep numbers/punct."""
    if not s:
        return s
    keep = []
    for line in s.splitlines():
        ln = line.strip()
        if not ln:
            continue
        ratio = _arabic_ratio(ln)
        # allow short tokens like "8:30"
        has_word = bool(re.search(r"[^\W\d_]", ln, flags=re.UNICODE))
        if (not has_word) or ratio >= min_ratio:
            keep.append(ln)
    return "\n".join(keep)

def _extract_numbers_set(s: str):
    if not s:
        return set()
    s2 = _to_western_digits(s)
    pats = [
        r"\b\d{1,2}:\d{2}\b",
        r"\b\d{1,2}\s*[-–]\s*\d{1,2}\b",
        r"\b\d+(?:\.\d+)?\b",
    ]
    vals = set()
    for p in pats:
        for m in re.findall(p, s2):
            vals.add(m)
    return vals

def _has_times_or_days(txt: str) -> bool:
    if not txt:
        return False
    t = _to_western_digits(txt)
    if any(day in t for day in _AR_DAYS):
        return True
    return any(re.search(p, t) for p in _TIME_PATTERNS)

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
    parts = re.split(r"(?<=[\.!\؟])\s+|[\n\r]+|[•\-–]\s+", txt2)
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
        keep.append(l)
    txt = " ".join(keep).strip()
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def _is_meaningful(txt: str) -> bool:
    return bool(txt and len(re.sub(r"\s+", "", txt)) >= 12)

def _as_bullets(sents, max_items=8):
    out = []
    for s in sents[:max_items]:
        s = s.strip()
        if s:
            out.append(f"• {s}")
    return "\n".join(out)

def _closest_bullets(txt: str, max_sents: int = 6) -> str:
    sents = _sentences(txt)[:max_sents]
    return _as_bullets(sents, max_items=max_sents)

def _paginate_text(text: str, max_chars: int = 900):
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

def _format_with_intro_and_bullets(body_text: str, intro: str = "استنادًا إلى النصوص المسترجَعة من المصدر، إليك الخلاصة:"):
    sents = _sentences(body_text)
    if len(sents) <= 1:
        content = f"{intro}\n{(sents[0] if sents else body_text.strip())}"
    else:
        content = f"{intro}\n{_as_bullets(sents)}"
    return content

def _parse_pages_from_sources(sources_text: str):
    """Extract page numbers from Sources lines like 'Data_pdf.pdf - page 16'
       and Arabic 'الصفحة/صفحة 16'. Handles Arabic digits too."""
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

def _page_ctx_from_sources(sources_text: str, max_chars: int = 3500):
    pages = _parse_pages_from_sources(sources_text)
    if not pages:
        return ""
    buf = []
    total = 0
    for p in pages:
        txt = CHUNKS_BY_PAGE.get(p, "")
        if not txt:
            continue
        if total + len(txt) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                buf.append(txt[:remaining])
                total = max_chars
                break
        else:
            buf.append(txt)
            total += len(txt)
    return _clean_text("\n".join(buf))

def _is_hours_like(question: str, intent: str = "") -> bool:
    q = (question or "").strip()
    hours_kws = ["ساعات", "الدوام", "رمضان", "أيام الدوام", "الساعات الإضافية", "العطل الرسمية", "استراحة", "مغادرة ساعية"]
    if any(kw in q for kw in hours_kws):
        return True
    return intent in ("work_hours", "ramadan_hours", "overtime", "work_days", "breaks")

# ---------------- Compact bullets helpers ----------------
_NOISE_PATTERNS = [
    r"جميع الحقوق محفوظة",
    r"فهرس المحتويات",
    r"الدليل\s+الإداري",
    r"للتعليم المساند والإبداع العلمي",
    r"صفحة\s*\d+",
]

def _is_noise_line(ln: str) -> bool:
    if not ln or len(ln) < 6:
        return True
    for rx in _NOISE_PATTERNS:
        if re.search(rx, ln):
            return True
    # super short numbers-only lines are noise
    if re.fullmatch(r"[0-9:؛–\-\s\(\)]+", _to_western_digits(ln) or ""):
        return True
    return False

def _split_to_lines(txt: str):
    if not txt: return []
    # break on newlines and common list separators
    raw = re.split(r"[\n\r]+|[•·\-\u2022]\s+|(?<=\.)\s+|(?<=\؟)\s+|(?<=\!)\s+", txt)
    # normalize spaces
    lines = [re.sub(r"\s+", " ", l).strip(" .-–•\t") for l in raw]
    lines = [l for l in lines if l]
    return lines

def _smart_truncate(s: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= limit:
        return s
    # try to cut at a natural boundary before limit
    cut = max(
        s.rfind("،", 0, limit),
        s.rfind("؛", 0, limit),
        s.rfind(".", 0, limit),
        s.rfind(" - ", 0, limit),
        s.rfind(" — ", 0, limit),
        s.rfind(":", 0, limit),
    )
    if cut < max(20, limit // 2):
        cut = limit
    return s[:cut].rstrip() + "…"

def _select_salient_lines(text: str, question: str, hours_like: bool, max_lines: int, force_time_lines: bool = False):
    lines = _split_to_lines(text)
    # purge noise & non-Arabic-heavy lines
    cleaned = []
    for ln in lines:
        if _is_noise_line(ln):
            continue
        # keep Arabic-heavy or purely numeric/time lines
        if _arabic_ratio(ln) < 0.6 and not _has_times_or_days(ln):
            continue
        cleaned.append(ln)

    if hours_like:
        if force_time_lines:
            cleaned = [l for l in cleaned if _has_times_or_days(l)]
        else:
            with_time = [l for l in cleaned if _has_times_or_days(l)]
            if with_time:
                cleaned = with_time

    # dedupe exact duplicates
    seen, deduped = set(), []
    for l in cleaned:
        if l in seen:
            continue
        seen.add(l); deduped.append(l)

    return deduped[:max_lines]

def _format_compact(question: str, text: str, hours_like: bool, max_bullets: int, bullet_max_chars: int, hourlines_only: bool):
    lines = _select_salient_lines(text, question, hours_like, max_bullets, force_time_lines=(hours_like and hourlines_only))
    if not lines:
        # fallback to old behavior
        return _format_with_intro_and_bullets(text)
    clipped = [_smart_truncate(l, bullet_max_chars) for l in lines]
    bullets = "\n".join(f"• {l}" for l in clipped)
    return "استنادًا إلى النصوص المسترجَعة من المصدر، إليك الخلاصة:\n" + bullets

# ---------------- Q&A ----------------
def ask_once(index: RET.HybridIndex,
             tokenizer,
             model,
             question: str,
             use_llm: bool = True,
             use_rerank_flag: bool = True,
             paginate_chars: int = 900,
             max_bullets: int = 6,
             bullet_max_chars: int = 160,
             hourlines_only: bool = False) -> str:
    """
    1) classify intent
    2) retrieve via RET.answer (returns text+sources)
    3) if extractive is weak → FALL BACK to full page text from Sources pages
    4) optional LLM refine — strictly grounded (no new facts)
    5) format: compact bullets; paginate long outputs
    """
    t0 = time.time()
    intent = RET.classify_intent(question)

    extractive_answer = RET.answer(question, index, intent, use_rerank_flag=use_rerank_flag)

    # Split body/sources
    lines = str(extractive_answer or "").split('\n')
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

    body_raw = '\n'.join(body_lines).strip()
    sources = '\n'.join(source_lines).strip()

    # Build page-aware context if needed
    page_ctx = _page_ctx_from_sources(sources, max_chars=3500)

    # -------------- HARDENING START --------------
    hours_like = _is_hours_like(question, intent)
    if page_ctx:
        # Normalize both before checks
        tmp_body = _clean_text(body_raw)
        if (not tmp_body) or ("لا توجد معلومات" in tmp_body) or ("لم أعثر" in tmp_body):
            body_raw = page_ctx
        elif hours_like and not _has_times_or_days(tmp_body):
            if _has_times_or_days(page_ctx):
                body_raw = page_ctx
            else:
                # collect only segments from cited pages that contain times/days
                pages = _parse_pages_from_sources(sources)
                focused = []
                for p in pages:
                    t = CHUNKS_BY_PAGE.get(p, "")
                    if t and _has_times_or_days(t):
                        focused.append(t)
                if focused:
                    body_raw = _clean_text("\n".join(focused))

    # If still nothing → grounded insufficiency
    if not body_raw or len(body_raw.strip()) == 0:
        dt = time.time() - t0
        return _final(dt, "لا يقدّم النص المسترجَع تفاصيل كافية للإجابة بشكل قاطع من المصدر نفسه.", paginate_chars, sources)

    body_clean = _clean_text(body_raw)

    # If LLM disabled/unavailable → format extractive/page_ctx directly (compact)
    if (not use_llm) or (tokenizer is None) or (model is None):
        # Arabic-only purge of obvious noise before formatting
        body_clean2 = _purge_non_arabic_lines(body_clean)
        if hours_like and not _has_times_or_days(body_clean2) and _has_times_or_days(page_ctx):
            body_clean2 = _clean_text(page_ctx)
        dt = time.time() - t0
        formatted = _format_compact(
            question,
            (body_clean2 or body_clean or body_raw),
            hours_like=hours_like,
            max_bullets=max_bullets,
            bullet_max_chars=bullet_max_chars,
            hourlines_only=hourlines_only,
        )
        return _final(dt, formatted, paginate_chars, sources)

    # Short-circuit for hour-like answers already containing times/days
    if hours_like and _has_times_or_days(body_clean):
        dt = time.time() - t0
        formatted = _format_compact(
            question,
            body_clean,
            hours_like=hours_like,
            max_bullets=max_bullets,
            bullet_max_chars=bullet_max_chars,
            hourlines_only=hourlines_only,
        )
        return _final(dt, formatted, paginate_chars, sources)

    # LLM refinement (strictly grounded; Arabic-only)
    try:
        system_prompt = (
            "أعد صياغة المقتطف العربي التالي بوضوح واختصار دون إضافة أي معلومة جديدة أو استنتاج. "
            "اعتمد حصراً على هذا النص. لا تولّد أرقاماً/أوقات/أياماً غير موجودة. "
            "حافظ على جميع الأرقام/الأوقات/الأيام كما وردت حرفياً. "
            "اكتب الإجابة بالعربية الفصحى فقط دون استخدام أي لغة أخرى أو حروف لاتينية."
        )
        user_prompt = f"السؤال: {question}\nالنص لإعادة الصياغة:\n{body_clean or body_raw}"

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
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

        # Guardrails: fidelity checks vs the true context (body_raw/body_clean)
        src_text_for_check = body_clean or body_raw
        src_nums = _extract_numbers_set(src_text_for_check)
        out_nums = _extract_numbers_set(resp)

        # Hour-like strictness: must retain times/days if asked
        if hours_like and not _has_times_or_days(resp):
            resp = ""  # reject; will fallback

        # No new numbers
        if not out_nums.issubset(src_nums):
            resp = ""  # introduced new numbers → reject

        dt = time.time() - t0

        if not _is_meaningful(resp):
            # Safe fallback: bullets from grounded context (prefer page_ctx if it has times/days)
            fallback_txt = src_text_for_check
            if hours_like and _has_times_or_days(page_ctx):
                fallback_txt = _clean_text(page_ctx)
            if _is_meaningful(fallback_txt):
                formatted = _format_compact(
                    question,
                    fallback_txt,
                    hours_like=hours_like,
                    max_bullets=max_bullets,
                    bullet_max_chars=bullet_max_chars,
                    hourlines_only=hourlines_only,
                )
                return _final(dt, formatted, paginate_chars, sources)
            return _final(dt, "لا يقدّم النص المسترجَع تفاصيل كافية للإجابة بشكل قاطع من المصدر نفسه.", paginate_chars, sources)

        # Final formatting (compact)
        formatted = _format_compact(
            question,
            resp,
            hours_like=hours_like,
            max_bullets=max_bullets,
            bullet_max_chars=bullet_max_chars,
            hourlines_only=hourlines_only,
        )
        return _final(dt, formatted, paginate_chars, sources)

    except Exception as e:
        LOG.warning(f"LLM generation failed: {e}")
        dt = time.time() - t0
        # Prefer page_ctx if it's more informative for hour-like
        fallback_txt = body_clean if _is_meaningful(body_clean) else body_raw
        if hours_like and _has_times_or_days(page_ctx):
            fallback_txt = _clean_text(page_ctx)
        formatted = _format_compact(
            question,
            fallback_txt,
            hours_like=hours_like,
            max_bullets=max_bullets,
            bullet_max_chars=bullet_max_chars,
            hourlines_only=hourlines_only,
        )
        return _final(dt, formatted, paginate_chars, sources)

# helper for final formatting with pagination
ndef _final(dt, text, paginate_chars, sources):
    parts = _paginate_text(text, max_chars=paginate_chars)
    if len(parts) > 1:
        labeled = []
        for i, p in enumerate(parts, 1):
            labeled.append(f"الجزء {i}/{len(parts)}:\n{p}")
        text = "\n\n".join(labeled)
    return f"⏱ {dt:.2f}s | 🤖 {text}\n{sources}" if sources else f"⏱ {dt:.2f}s | 🤖 {text}"

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
            seen.add(q); merged.append(q)
    return merged

def _pass_loose(answer_text: str) -> bool:
    has_sources = ("Sources:" in answer_text) or ("المصادر:" in answer_text)
    bad = ("لا يقدّم النص المسترجَع تفاصيل كافية" in answer_text)
    return bool(has_sources and not bad)

def _pass_strict(question: str, body_only: str) -> bool:
    if not _is_meaningful(body_only):
        return False
    q = question or ""
    hours_like = any(kw in q for kw in [
        "ساعات", "الدوام", "رمضان", "أيام الدوام", "الساعات الإضافية", "العطل الرسمية", "استراحة", "مغادرة ساعية"
    ])
    if hours_like:
        return _has_times_or_days(body_only)
    return True

def run_test_prompts(index: RET.HybridIndex, tokenizer, model,
                     use_llm: bool, use_rerank_flag: bool, artifacts_dir: str,
                     max_bullets: int, bullet_max_chars: int, paginate_chars: int, hourlines_only: bool):
    os.makedirs(artifacts_dir, exist_ok=True)
    results_path = os.path.join(artifacts_dir, "results.jsonl")
    summary_md   = os.path.join(artifacts_dir, "summary.md")
    report_txt   = os.path.join(artifacts_dir, "report.txt")

    results_f = open(results_path, "w", encoding="utf-8")
    report_f  = open(report_txt,  "w", encoding="utf-8")

    def _tee(line=""):
        print(line)
        report_f.write(line + "\n")
        report_f.flush()

    tests = _gather_sanity_prompts()
    if not tests:
        _tee("❌ No sanity prompts available.")
        results_f.close(); report_f.close()
        return

    _tee("🧪 Running sanity prompts ...")
    _tee("=" * 80)

    total = len(tests)
    pass_loose_count, pass_strict_count = 0, 0

    for i, q in enumerate(tests, 1):
        _tee(f"\n📝 Test {i}/{total}: {q}")
        _tee("-" * 60)
        try:
            result = ask_once(index, tokenizer, model, q,
                              use_llm=use_llm, use_rerank_flag=use_rerank_flag,
                              paginate_chars=paginate_chars,
                              max_bullets=max_bullets,
                              bullet_max_chars=bullet_max_chars,
                              hourlines_only=hourlines_only)
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

    results_f.close(); report_f.close()

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
    parser.add_argument("--test", action="store_true", help="Run sanity prompts")
    parser.add_argument("--sanity", action="store_true", help="Alias for --test")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM refinement")
    parser.add_argument("--use-4bit", action="store_true", help="Quantize 4-bit (bitsandbytes)")
    parser.add_argument("--use-8bit", action="store_true", help="Quantize 8-bit (bitsandbytes)")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranker (save VRAM)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="LLM device")
    parser.add_argument("--out-dir", type=str, default="runs", help="Directory for run artifacts")

    # NEW knobs for compacting output
    parser.add_argument("--max-bullets", type=int, default=6, help="Max bullets per answer")
    parser.add_argument("--bullet-max-chars", type=int, default=160, help="Max chars per bullet")
    parser.add_argument("--paginate-chars", type=int, default=900, help="Chars per page chunk before pagination")
    parser.add_argument("--hourlines-only", action="store_true", help="For hours-like questions, keep only lines that contain times/days")

    args = parser.parse_args()

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
        LOG.error("Chunks file not found: %s", args.chunks)
        return
    chunks, chunks_hash = RET.load_chunks(path=args.chunks)

    # Build global page index for page-aware fallback (works for dicts or Chunk objects)
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
            dtype_fp = torch.bfloat16 if (bf16_supported and torch is not None) else (torch.float16 if (use_cuda and torch is not None) else None)

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

    # Execute
    use_rerank_flag = not args.no_rerank

    if args.test or args.sanity:
        run_test_prompts(index, tok, mdl,
                         use_llm=use_llm, use_rerank_flag=use_rerank_flag, artifacts_dir=run_dir,
                         max_bullets=args.max_bullets, bullet_max_chars=args.bullet_max_chars,
                         paginate_chars=args.paginate_chars, hourlines_only=args.hourlines_only)
        print(f"\n✅ Saved artifacts under: {run_dir}")
        return

    if args.ask:
        ans = ask_once(index, tok, mdl, args.ask,
                       use_llm=use_llm, use_rerank_flag=use_rerank_flag,
                       paginate_chars=args.paginate_chars,
                       max_bullets=args.max_bullets,
                       bullet_max_chars=args.bullet_max_chars,
                       hourlines_only=args.hourlines_only)
        single_path = os.path.join(run_dir, "single_answer.txt")
        with open(single_path, "w", encoding="utf-8") as f:
            f.write(ans)
        print(ans)
        print(f"\n✅ Saved single answer to: {single_path}")
        return

    # Interactive
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
            ans = ask_once(index, tok, mdl, q,
                           use_llm=use_llm, use_rerank_flag=use_rerank_flag,
                           paginate_chars=args.paginate_chars,
                           max_bullets=args.max_bullets,
                           bullet_max_chars=args.bullet_max_chars,
                           hourlines_only=args.hourlines_only)
            print(ans)
            trans.write(f"\nQ: {q}\n{ans}\n"); trans.flush()
    print(f"\n✅ Interactive transcript saved to: {interactive_path}")

if __name__ == "__main__":
    main()

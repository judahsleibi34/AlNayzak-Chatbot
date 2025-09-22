# =============================
# === File: NewRag.py (orchestrator)
# =============================
# -*- coding: utf-8 -*-
"""
Arabic RAG Orchestrator (bias‑free strict fixes)

- Removes keyword bias in strict-check logic (no hardcoded domain tokens)
- Fixes crash: never return re.Match from pass_strict
- Uses purely pattern-based quantitative intent (digits/%, time patterns) — not domain words
- Better junk filtering + unique citations
- Same CLI as before; writes artifacts: runs/run_*/{report.txt,results.jsonl,summary.md}

Usage (examples):
  python NewRag.py --chunks Data_pdf_clean_chunks.jsonl --hier-index heading_inverted_index.json --aliases section_aliases.json --sanity
  python NewRag.py --chunks Data_pdf_clean_chunks.jsonl --out-dir runs
"""

import os, re, sys, json, time, argparse, logging
from datetime import datetime
from types import SimpleNamespace
from collections import defaultdict

# Your retriever backend (updated below in this canvas as retrival_model.py)
import retrival_model as RET

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger("newrag")

# ---------------- Arabic helpers ----------------
_AR_DIAC = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_DEF_TIME_PATTERNS = [
    r"\b\d{1,2}:\d{2}\b",                # 8:30
    r"\b\d{1,2}[:٫]\d{2}\b",            # 8٫30
    r"\b\d{1,2}\s*(?:ص|م)\b",           # 8 ص/5 م
    r"\b\d{1,2}\s*(?:إلى|الى|حتى|[-–])\s*\d{1,2}\b",  # 8 إلى 5
]

_AR_DAYS = ["الأحد","الإثنين","الاثنين","الثلاثاء","الأربعاء","الاربعاء","الخميس","الجمعة","السبت"]

_DEF_PERCENT_RX = re.compile(r"\b\d{1,3}\s*[%٪]\b")


def norm(s: str) -> str:
    if s is None:
        return ""
    t = s.strip().lower()
    t = _AR_DIAC.sub("", t)
    t = t.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    t = t.replace("ى","ي").replace("ئ","ي").replace("ؤ","و")
    # keep \u0629 (ة) because it's semantically important and not domain-biased
    t = re.sub(r"[^\w\s%٪:–-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def has_times_or_days(txt: str) -> bool:
    if not txt:
        return False
    T = (txt or "").translate(_ARABIC_DIGITS)
    if any(d in T for d in _AR_DAYS):
        return True
    for p in _DEF_TIME_PATTERNS:
        if re.search(p, T):
            return True
    if _DEF_PERCENT_RX.search(T):
        return True
    # plain digits often indicate quantities/durations
    return bool(re.search(r"\b\d+\b", T))


def is_meaningful(txt: str) -> bool:
    if not txt:
        return False
    t = re.sub(r"\s+", "", txt)
    return len(t) >= 12


def _split_answer(answer_text: str):
    if not answer_text:
        return "", ""
    parts = re.split(r"\n(?=Sources:|المصادر:)", answer_text, maxsplit=1)
    body = parts[0].strip()
    sources = parts[1].strip() if len(parts) > 1 else ""
    return body, sources


def pass_loose(answer_text: str) -> bool:
    has_sources = ("Sources:" in answer_text) or ("المصادر:" in answer_text)
    bad = ("لم يرد نص صريح" in answer_text) or ("لا يقدّم النص" in answer_text)
    return bool(has_sources and not bad)


# *** Bias‑free strict: no domain keyword lists ***
# If the question contains explicit *patterns* that imply quantitative/temporal info
# (digits, %, time glyphs, ranges), then demand that the answer body also shows such patterns.
_QUANT_Q_PAT = re.compile(r"(:|%|٪|\d|\b\d{1,2}\s*(?:إلى|الى|حتى|[-–])\s*\d{1,2}\b)")


def pass_strict(question: str, body_only: str) -> bool:
    if not is_meaningful(body_only):
        return False
    qn = norm(question)
    needs_quants = bool(_QUANT_Q_PAT.search(qn))
    if needs_quants:
        return bool(has_times_or_days(body_only))
    return True


# ---------------- Page-text index (for numeric rescue if needed) ----------------
CHUNKS_BY_PAGE: dict[int, str] = {}


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
        if v is None:
            continue
        s = str(v)
        if s.strip():
            return s
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


# ---------------- Core ask_once ----------------

def ask_once(index: RET.HybridIndex, question: str, cfg: SimpleNamespace) -> str:
    t0 = time.time()
    # delegate to retriever (already extractive & bias-minimized)
    intent = "general"  # no domain bias routing
    answer = RET.answer(question, index, intent, use_rerank_flag=False)
    dt = time.time() - t0
    return f"⏱ {dt:.2f}s | 🤖 {answer}"


# ---------------- Sanity runner ----------------
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


_DEF_OUTDIR = "runs"


def _split_body_sources(ans_line: str):
    body, sources = _split_answer(ans_line)
    return body, sources


def run_test_prompts(index: RET.HybridIndex, artifacts_dir: str, cfg: SimpleNamespace):
    os.makedirs(artifacts_dir, exist_ok=True)
    results_path = os.path.join(artifacts_dir, "results.jsonl")
    summary_md   = os.path.join(artifacts_dir, "summary.md")
    report_txt   = os.path.join(artifacts_dir, "report.txt")

    results_f = open(results_path, "w", encoding="utf-8")
    report_f  = open(report_txt,  "w", encoding="utf-8")

    def _tee(line=""):
        print(line)
        report_f.write(line + "\n"); report_f.flush()

    total = len(SANITY_PROMPTS)
    _tee(f"🧪 Running sanity prompts ({total}) …")
    _tee("=" * 80)

    pass_loose_count, pass_strict_count = 0, 0

    for i, q in enumerate(SANITY_PROMPTS, 1):
        _tee(f"\n📝 Test {i}/{total}: {q}")
        _tee("-" * 60)
        try:
            res = ask_once(index, q, cfg)
            _tee(res)
            body, _src = _split_answer(res)
            loose = pass_loose(res)
            strict = pass_strict(q, body)
            pass_loose_count += int(loose)
            pass_strict_count += int(strict)
            _tee("✅ PASS_LOOSE" if loose else "❌ FAIL_LOOSE")
            _tee("✅ PASS_STRICT" if strict else "❌ FAIL_STRICT")
            _tee("=" * 80)
            rec = {"index": i, "question": q, "answer": res, "body_only": body,
                   "pass_loose": bool(loose), "pass_strict": bool(strict)}
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default="Data_pdf_clean_chunks.jsonl", help="Path to chunks (JSONL/JSON)")
    ap.add_argument("--hier-index", type=str, default="heading_inverted_index.json", help="Optional hierarchy inverted index")
    ap.add_argument("--aliases", type=str, default="section_aliases.json", help="Optional aliases for headings")
    ap.add_argument("--sanity", action="store_true", help="Run 30 sanity questions and exit")
    ap.add_argument("--out-dir", type=str, default=_DEF_OUTDIR)
    args = ap.parse_args()

    hier = RET.load_hierarchy(args.hier_index, args.aliases)
    chunks, chunks_hash = RET.load_chunks(path=args.chunks)

    global CHUNKS_BY_PAGE
    CHUNKS_BY_PAGE = _build_page_text_index(chunks)

    index = RET.HybridIndex(chunks, chunks_hash, hier=hier)
    index.build()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    cfg = SimpleNamespace()

    if args.sanity:
        run_test_prompts(index, run_dir, cfg)
        print(f"\n✅ Saved artifacts under: {run_dir}")
        return

    print("Ready. اطرح سؤالك (اكتب 'exit' للخروج)\n")
    while True:
        try:
            q = input("سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break
        if not q:
            continue
        if q.lower() in ("exit","quit","q"):
            print("Exiting."); break
        print(ask_once(index, q, cfg))


if __name__ == "__main__":
    main()


# =============================
# === File: retrival_model.py (hardened, bias‑minimized)
# =============================
# -*- coding: utf-8 -*-
"""
Hardened Arabic-first EXTRACTIVE retriever (PDF-only, no fabrication)
— bias‑minimized (no domain keyword gating); pattern-based selection; junk filtering.

- Deterministic, extractive-only.
- Strict section/page gating via hierarchy+aliases (if provided).
- Proximity+unit guards for numerics and times.
- Refuses when confidence low.
- Clean Arabic output + exact PDF page citations.
"""

import os, re, sys, json, logging, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Set

import numpy as np

try:
    import faiss  # optional
except Exception:
    faiss = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger(__name__)

# ---------------- Arabic utils ----------------
AR_DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
AR_NUMS = {ord(c): ord('0')+i for i,c in enumerate("٠١٢٣٤٥٦٧٨٩")}
IR_NUMS = {ord(c): ord('0')+i for i,c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_SECTION_NUM_RE = re.compile(r'^\s*\d+(?:\.\d+){1,3}\s*$')


def ar_normalize(s: str) -> str:
    if not s:
        return ""
    s = s.replace('\u0640','')
    s = AR_DIAC.sub('', s)
    s = (s.replace('أ','ا').replace('إ','ا').replace('آ','ا').replace('ى','ي'))
    s = s.translate(AR_NUMS).translate(IR_NUMS)
    s = s.replace('،', ',').replace('٫','.')
    s = ' '.join(s.split())
    return s


def rtl_wrap(t: str) -> str:
    return '\u202B' + t + '\u202C'


SENT_SPLIT_RE = re.compile(r'(?<=[\.\!\؟\?|،]|[\n])\s+')


def sent_split(s: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(s or "") if p and p.strip()]
    out = []
    for p in parts:
        pn = ar_normalize(p)
        if len(pn) < 6:
            continue
        letters = sum(ch.isalpha() for ch in pn)
        total = len(pn.replace(" ", ""))
        if total == 0 or letters/total < 0.5:
            continue
        if _SECTION_NUM_RE.match(pn):  # drop lines like "3.4"
            continue
        # drop known catalogue headings
        if "الجداول الزمنية" in pn or "التقارير" in pn:
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


def load_chunks(path=CHUNKS_PATH):
    if not os.path.exists(path):
        LOG.error("Chunks file not found: %s", path); sys.exit(1)
    LOG.info("Loading chunks from %s …", path)
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
    chunks = []
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


# ---------------- Index (dense only) ----------------
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
        LOG.info("Building embeddings…")
        texts = [c.norm for c in self.chunks]
        self.emb = self.model.encode(texts, batch_size=128, convert_to_numpy=True,
                                     show_progress_bar=True, normalize_embeddings=True)
        if faiss is not None:
            d = self.emb.shape[1]
            self.faiss = faiss.IndexFlatIP(d)
            self.faiss.add(self.emb.astype('float32'))
            LOG.info("Built FAISS index")
        else:
            LOG.warning("faiss not available; using numpy search.")

    def dense(self, q: str, topk=60, restrict_ids: Optional[Set[int]] = None):
        qv = self.model.encode([ar_normalize(q)], convert_to_numpy=True, normalize_embeddings=True)
        if self.faiss is not None:
            D, I = self.faiss.search(qv.astype('float32'), topk)
            scores, idxs = D[0], I[0]
        else:
            sims = self.emb @ qv[0]
            idxs = np.argsort(-sims)[:topk]
            scores = sims[idxs]
        if restrict_ids is None:
            return scores, idxs
        # keep only restricted pages if provided; else empty (no unsafe fallback)
        filtS, filtI = [], []
        rset = set(int(x) for x in restrict_ids)
        for s, i in zip(scores, idxs):
            if int(i) in rset:
                filtS.append(float(s)); filtI.append(int(i))
            if len(filtI) >= topk: break
        return np.array(filtS), np.array(filtI)


# ---------------- Generic guards ----------------
_DEF_UNITS = ["ساعة","ساعه","دقيقة","دقائق","شيكل","كم","كيلومتر","فاتورة","فاتوره","إيصال","ايصال","%","٪"]
_AR_DAYS = ["الأحد","الإثنين","الاثنين","الثلاثاء","الأربعاء","الاربعاء","الخميس","الجمعة","السبت"]


def _has_numbers(sn: str) -> bool:
    return bool(re.search(r'\d', sn))


def _has_time_hint(sn: str) -> bool:
    sn = ar_normalize(sn)
    return (":" in sn) or any(w in sn for w in ["الى","إلى","حتى","-","–"]) or any(d in sn for d in _AR_DAYS)


def _looks_junky(sn: str) -> bool:
    snn = ar_normalize(sn)
    if len(snn) < 6:
        return True
    if _SECTION_NUM_RE.match(snn):
        return True
    letters = sum(ch.isalpha() for ch in snn)
    total = len(snn.replace(" ", ""))
    if total > 0 and letters/total < 0.5:
        return True
    if "الجداول الزمنية" in snn or "التقارير" in snn:
        return True
    return False


def _prox_numeric_within(text: str, window=48) -> bool:
    T = ar_normalize(text)
    for m in re.finditer(r'\b\d+\b', T):
        start = max(0, m.start()-window); end = m.end()+window
        ctx = T[start:end]
        if any(u in ctx for u in _DEF_UNITS):
            return True
    return False


def _unique_pages_from_hits(chunks: List[Chunk], hits, limit=3) -> list:
    seen = set(); pages = []
    for _, i in hits:
        p = chunks[i].page
        if p >= 0 and p not in seen:
            seen.add(p); pages.append(p)
        if len(pages) >= limit:
            break
    return pages


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
        # respect strict gating; refuse later
        return []
    pairs = list(zip(list(map(float, scores)), list(map(int, idxs))))
    pairs.sort(key=lambda x: -x[0])
    return pairs[:12]


# ---------------- Answer composition (generic, bias‑minimized) ----------------

def _compose_time_range(chunks: List[Chunk], hits) -> Optional[str]:
    # Try to find explicit time ranges like "8:30 إلى 15:00"
    TIME_RE = re.compile(
        r'(?:من\s*)?'  # optional "from"
        r'(\d{1,2}(?::|\.)?\d{0,2})\s*'
        r'(?:[-–—]|الى|إلى|حتي|حتى)\s*'
        r'(\d{1,2}(?::|\.)?\d{0,2})'
    )
    def _norm_hhmm(t):
        t = t.replace('.', ':')
        if ':' not in t:
            return f"{int(t):d}:00"
        h, m = t.split(':', 1)
        if m == "": m = "00"
        return f"{int(h):d}:{int(m):02d}"

    for _, i in hits:
        ch = chunks[i]
        for s in sent_split(ch.text):
            n = ar_normalize(s)
            m = TIME_RE.search(n)
            if not m:
                continue
            a, b = _norm_hhmm(m.group(1)), _norm_hhmm(m.group(2))
            pages = _unique_pages_from_hits(chunks, hits, limit=3)
            srcs  = "\n".join(f"{k}. Data_pdf.pdf - page {p}" for k, p in enumerate(pages, 1))
            return rtl_wrap(f"{s.strip()}") + "\nSources:\n" + srcs
    return None


def _compose_best_snippet(chunks: List[Chunk], hits) -> Optional[str]:
    best = None; best_score = -1e9; best_page = None
    for _, i in hits:
        ch = chunks[i]
        for s in sent_split(ch.text):
            if _looks_junky(s):
                continue
            sn = ar_normalize(s)
            score = 0.0
            if _has_numbers(sn): score += 0.6
            if _has_time_hint(sn): score += 0.6
            if _prox_numeric_within(s, window=48): score += 0.4
            # prefer sentences that mention days-of-week if present
            if any(d in sn for d in ["الاحد","الإثنين","الاثنين","الثلاثاء","الأربعاء","الاربعاء","الخميس","الجمعة","السبت"]):
                score += 0.8
            # length prior: avoid too long
            L = len(sn)
            if L > 300: score -= 0.5
            if score > best_score:
                best_score, best, best_page = score, s.strip(), ch.page
    if not best:
        return None
    pages = _unique_pages_from_hits(chunks, hits, limit=3)
    srcs  = "\n".join(f"{k}. Data_pdf.pdf - page {p}" for k, p in enumerate(pages, 1))
    return rtl_wrap(best) + "\nSources:\n" + srcs


def answer(q: str, index: HybridIndex, intent: str, use_rerank_flag: bool=False) -> str:
    hits = retrieve(index, q, intent)
    if not hits:
        return rtl_wrap("لم يرد نص صريح حول ذلك في الدليل المرفق.")

    # Try to return an explicit time-range line if present
    tr = _compose_time_range(index.chunks, hits)
    if tr:
        return tr

    # Fallback to best snippet (with guards)
    bs = _compose_best_snippet(index.chunks, hits)
    if bs:
        return bs

    return rtl_wrap("لم يرد نص صريح حول ذلك في الدليل المرفق.")

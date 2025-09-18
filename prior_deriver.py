
# -*- coding: utf-8 -*-
"""
Derive section priors from the *current* PDF hierarchy.
No fixed section numbers → no cross-document bias.
"""

import re
from typing import Dict, Any, List, Set
from intent_patterns import INTENT_PATTERNS, norm

def _token_overlap(a: str, b: str) -> float:
    A = set(norm(a).split())
    B = set(norm(b).split())
    if not A or not B:
        return 0.0
    # Jaccard
    return len(A & B) / max(1, len(A | B))

def compile_heading_matchers(intent: str):
    patt = INTENT_PATTERNS.get(intent, {})
    rx = [re.compile(r, re.IGNORECASE) for r in patt.get("regex", [])]
    kws = [norm(k) for k in patt.get("kw", [])]
    return rx, kws

def headings_from_hierarchy(hier) -> List[Dict[str, Any]]:
    """
    Expect hier iterable like: {title, page_start, page_end, path}
    """
    out = []
    if not hier:
        return out
    for h in hier:
        t = h.get("title") or h.get("heading") or ""
        out.append({
            "title": t,
            "title_norm": norm(t),
            "page_start": int(h.get("page_start", -1)),
            "page_end": int(h.get("page_end", -1)),
            "path": h.get("path") or ""
        })
    return out

def derive_section_priors(intent: str, hier) -> Dict[str, Any]:
    rx_list, kw_norm = compile_heading_matchers(intent)
    heads = headings_from_hierarchy(hier)
    candidate_heads: List[Dict[str, Any]] = []

    for h in heads:
        t = h["title"]; tn = h["title_norm"]
        rx_hit = any(r.search(t) or r.search(tn) for r in rx_list)
        ov = _token_overlap(tn, " ".join(kw_norm)) if kw_norm else 0.0
        score = (2.0 if rx_hit else 0.0) + (1.0 * ov)
        if score > 0:
            cand = dict(h); cand["score"] = score
            candidate_heads.append(cand)

    # Fallback: allow fuzzy heading matches if nothing hit strictly
    if not candidate_heads and heads:
        for h in heads:
            ov = _token_overlap(h["title_norm"], " ".join(kw_norm))
            if ov >= 0.25:
                cand = dict(h); cand["score"] = 0.5 + ov
                candidate_heads.append(cand)

    candidate_heads.sort(key=lambda d: d["score"], reverse=True)
    top = candidate_heads[:5]

    pages: Set[int] = set()
    for h in top:
        ps = h["page_start"]; pe = h["page_end"]
        if ps >= 0 and pe >= 0 and pe >= ps:
            for p in range(ps, pe + 1):
                pages.add(int(p))
        elif ps >= 0:
            pages.add(int(ps))

    return {"pages": pages, "headings": top}

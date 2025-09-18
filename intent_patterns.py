
# -*- coding: utf-8 -*-
"""
Source-agnostic intent heading patterns (Arabic-first).
No section numbers; relies on language regex + keywords.
"""

import re

# Simple normalizer for Arabic strings (diacritics + common variants)
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")
def norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = _ARABIC_DIACRITICS.sub("", s)
    # normalize Hamza / taa marbuta / alef maqsura
    s = (s.replace("إ","ا").replace("أ","ا").replace("آ","ا")
           .replace("ة","ه").replace("ى","ي"))
    return re.sub(r"\s+", " ", s).strip()

# Each intent lists regex anchors and optional loose keywords (normalized).
INTENT_PATTERNS = {
    "work_hours": {
        "regex": [
            r"\bساعات?\s+(?:ال)?دوام\b",
            r"\bايام?\s+(?:ال)?دوام\b",
            r"\bالدوام\s+(?:الرسمي|الرئيسي)\b",
        ],
        "kw": ["دوام","ساعات","من","الى","حتى","الأحد","الخميس"]
    },
    "ramadan_hours": {
        "regex": [
            r"\bساعات?\s+العمل\s+(?:في|خلال)\s+رمضان\b",
            r"\bدوام\s+رمضان\b",
            r"\bايام\s+الصوم\b",
        ],
        "kw": ["رمضان","الصوم","دوام","ساعات","من","الى","حتى"]
    },
    "overtime": {
        "regex": [
            r"\bالعمل\s+الاضافي\b",
            r"\bساعات?\s+اضافيه\b",
        ],
        "kw": ["موافقه","اذن","كتابي","خطي","مسبق","اجر","تعويض"]
    },
    "work_days": {
        "regex": [r"\bايام\s+(?:ال)?عمل\b", r"\bنهايه\s+الاسبوع\b"],
        "kw": ["الأحد","الخميس","السبت","عطل"]
    },
    "breaks": {
        "regex": [r"\bاستراح[هة]\b", r"\bفتر[هة]\s+الراحه\b"],
        "kw": ["مده","دقائق","ساعات"]
    },
    "annual_leave": {
        "regex": [r"\bاجازه\s+سنوي[هة]\b"],
        "kw": ["ايام","سنه","ترحيل","حد","اقصى","زياده"]
    },
    "sick_leave": {
        "regex": [r"\bاجازه\s+مرضيه\b", r"\bتقرير\s+طبي\b"],
        "kw": ["ايام","اجر","نصف","كامل","تقرير"]
    },
    "maternity_leave": {
        "regex": [r"\bاجازه\s+اموم[هة]\b"],
        "kw": ["قبل","بعد","الولاده","مده","ايام"]
    },
    "bereavement_leave": {
        "regex": [r"\bاجازه\s+حداد\b", r"\bوفاه\b"],
        "kw": ["مدفوعه","الاجر","مده","ايام"]
    },
    "payroll_date": {
        "regex": [r"\bصرف\s+الرواتب\b", r"\bتاريخ\s+الراتب\b"],
        "kw": ["الشهر","اليوم","الخامس"]
    },
    "transport_allowance": {
        "regex": [r"\bبدل\s+المواصلات\b", r"\bنقل\b"],
        "kw": ["منزل","العمل","يصرف","نموذج"]
    },
    "salary_advance": {
        "regex": [r"\bسلف[هة]?\s+على\s+الراتب\b"],
        "kw": ["شروط","سقف","موافقه"]
    },
    "petty_cash": {
        "regex": [r"\bنثريات\b", r"\bمياومات\b", r"\bبدل\s+سفر\b"],
        "kw": ["حد","اقصى","تسويه","مستندات"]
    },
    "three_quotes": {
        "regex": [r"\bثلاث[هة]?\s+عروض\s+اسعار\b"],
        "kw": ["سقف","شراء","حد","اقصى"]
    },
    "conflict_of_interest": {
        "regex": [r"\bتضارب\s+المصالح\b"],
        "kw": ["مشتريات","ابلاغ","افصاح"]
    },
    "gifts_hospitality": {
        "regex": [r"\bالهدايا\b", r"\bالضيافه\b"],
        "kw": ["قبول","ابلاغ","حد","سقف"]
    },
    "asset_checkin": {
        "regex": [r"\bاستلام\s+عهد[هة]\b", r"\bعهد[هة]\b"],
        "kw": ["نموذج","استلام"]
    },
    "asset_checkout": {
        "regex": [r"\bتسليم\s+العهد[هة]\b", r"\bاعاده\s+العهد[هة]\b"],
        "kw": ["استقاله","انتقال","براءه","نموذج"]
    },
    "remote_work": {
        "regex": [r"\bالعمل\s+عن\s+بعد\b", r"\bمن\s+المنزل\b"],
        "kw": ["سياسه","اعتماد","موافقه","شروط"]
    },
    "hourly_exit": {
        "regex": [r"\bاذن\s+مغادر[هة]\s+ساعيه\b"],
        "kw": ["حد","اقصى","شهري","ساعات","اشعار"]
    },
    "performance_review": {
        "regex": [r"\bتقييم\s+الاداء\b"],
        "kw": ["سنويا","معايير","مؤشرات"]
    },
    "discipline": {
        "regex": [r"\bانذار\b", r"\bتدرج\s+تاديبي\b", r"\bجزاءات\b"],
        "kw": ["مخالفات","عقوبات","اجراءات"]
    },
    "confidentiality": {
        "regex": [r"\bسياس[هة]\s+(?:ال)?سري[هة]\b", r"\bحمايه\s+المعلومات\b"],
        "kw": ["سري","افشاء","وصول","تداول"]
    },
    "conduct_harassment": {
        "regex": [r"\bالسلوك\s+المهني\b", r"\bمكافحه\s+التحرش\b"],
        "kw": ["سلوك","تحرش","بيئه","كرامه","شكاوى"]
    },
}

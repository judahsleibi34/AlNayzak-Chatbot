import pandas as pd
import re
import unicodedata

class Normalization:
    def __init__(self, path, prefer_digits="latin"):
        """
        prefer_digits: 'latin' -> 0-9, 'arabic' -> ٠-٩, or None to leave as-is
        """
        self.dataset_path = path
        # Important for Arabic CSVs saved with BOM
        self.dataset = pd.read_csv(self.dataset_path, encoding="utf-8-sig")

        # Core cleanup regexes
        self.RE_TATWEEL    = re.compile("\u0640")
        self.RE_DIACRITICS = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
        self.RE_ZW_RTL     = re.compile("[\u200B-\u200F\u202A-\u202E\u2066-\u2069]")

        # Protect spans (URLs/emails/handles/hashtags)
        self.RE_URL     = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
        self.RE_HANDLE  = re.compile(r"(?:^|\s)@[A-Za-z0-9_]{1,50}\b")
        self.RE_HASHTAG = re.compile(r"(?:^|\s)#[\w\u0600-\u06FF_]{1,80}\b")
        self.RE_EMAIL   = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")

        # Options
        self.prefer_digits = prefer_digits

        # Digit maps
        self._arabic2latin = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        self._latin2arabic = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

    def _unicode_nfkc(self, s: str) -> str:
        return unicodedata.normalize("NFKC", s)

    def letters_normalizer(self, s: str) -> str:
        # Unify common look-alikes for Arabic
        s = re.sub("[\u0622\u0623\u0625\u0671]", "ا", s)  # آ/أ/إ/ٱ -> ا
        s = s.replace("ؤ", "ء").replace("ئ", "ء")
        s = s.replace("ى", "ي").replace("ی", "ي")
        s = s.replace("ک", "ك")
        return s

    def digits_normalizer(self, s: str) -> str:
        if self.prefer_digits == "latin":
            return s.translate(self._arabic2latin)
        if self.prefer_digits == "arabic":
            return s.translate(self._latin2arabic)
        return s

    def punctuation_normalizer(self, s: str) -> str:
        # Unify quotes/dashes/spaces; keep things simple & model-friendly
        s = s.replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"').replace("‟", '"')
        s = s.replace("’", "'").replace("‘", "'")
        # Normalize dashes to ASCII hyphen
        s = re.sub(r"[–—-]", "-", s)
        # Collapse multiple spaces
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s

    def _protect_spans(self, text: str):
        spans, i = {}, 0
        def sub(m):
            nonlocal i
            k = f"§§§PROT{i}§§§"
            spans[k] = m.group(0)
            i += 1
            return k
        for p in (self.RE_URL, self.RE_EMAIL, self.RE_HANDLE, self.RE_HASHTAG):
            text = p.sub(sub, text)
        return text, spans

    def _restore_spans(self, text: str, spans: dict) -> str:
        for k, v in spans.items():
            text = text.replace(k, v)
        return text

    def _normalize_ar_surface(self, s: str) -> str:
        s = self._unicode_nfkc(s)
        s = self.RE_ZW_RTL.sub("", s)      # remove RTL/LRM etc.
        s = self.RE_TATWEEL.sub("", s)     # remove tatweel
        s = self.RE_DIACRITICS.sub("", s)  # remove diacritics
        s = self.letters_normalizer(s)     # unify letters
        s = self.digits_normalizer(s)      # unify digits
        s = self.punctuation_normalizer(s) # unify punctuation/spacing
        return s

    def preprocess(self, text: str) -> str:
        txt, spans = self._protect_spans(str(text))
        txt = self._normalize_ar_surface(txt)
        return self._restore_spans(txt, spans)

    def normalize_content(self, text_col="content", out_col="content_normalized"):
        if text_col not in self.dataset.columns:
            raise ValueError(f"Column '{text_col}' not found. Available: {list(self.dataset.columns)}")
        self.dataset[out_col] = self.dataset[text_col].astype(str).apply(self.preprocess)
        return self.dataset

    def save(self, path="full_dataset_expanded_normalized.csv"):
        self.dataset.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved to {path}")
        return path


if __name__ == "__main__":
    # Point this to the file you confirmed
    Data = Normalization("full_dataset_expanded.csv", prefer_digits="latin")  # or 'arabic'
    Data.normalize_content(text_col="content", out_col="content_normalized")
    Data.save("full_dataset_expanded_normalized.csv")

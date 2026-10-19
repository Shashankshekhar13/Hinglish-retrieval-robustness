import json, re
from pathlib import Path
from collections import Counter, defaultdict
from indic_transliteration import sanscript
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import wordfreq

from .phonetic import phonetic_key, coarse_key

def casual_romanize(devanagari: str) -> str:
    itrans = sanscript.transliterate(devanagari, sanscript.DEVANAGARI, sanscript.ITRANS)
    s = itrans
    s = s.replace('R^i', 'ri')
    s = s.replace('.N', 'n')
    s = s.replace('~N', 'n')
    s = s.replace('JN', 'gy')
    s = s.replace('A', 'aa')
    s = s.replace('I', 'ee')
    s = s.replace('U', 'oo')
    s = s.replace('M', 'n')
    s = s.replace('x', 'ksh')
    s = re.sub(r'[\.\^~]', '', s)
    return s.lower()

class RomanLexicon:
    def __init__(self, exact_map=None, key_map=None, coarse_map=None):
        # exact_map: roman_hi -> list[str] (ranked glosses)
        # key_map: key -> list[str] (ranked glosses)
        # coarse_map: coarse_key -> list[str] (ranked glosses)
        self.exact_map = exact_map or {}
        self.key_map = key_map or {}
        self.coarse_map = coarse_map or {}

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, str]]):
        # pairs is (romanized_hi, english_gloss)
        exact_counts = defaultdict(Counter)
        key_counts = defaultdict(Counter)
        coarse_counts = defaultdict(Counter)

        for hi_roman, en_gloss in pairs:
            hi_roman = hi_roman.strip().lower()
            en_gloss = en_gloss.strip().lower()
            if not hi_roman or not en_gloss:
                continue
            # Drop entries where English side equals romanized Hindi side
            if hi_roman == en_gloss:
                continue
            # Drop glosses that are English stopwords
            if en_gloss in ENGLISH_STOP_WORDS:
                continue

            exact_counts[hi_roman][en_gloss] += 1
            pk = phonetic_key(hi_roman)
            if pk:
                key_counts[pk][en_gloss] += 1
            ck = coarse_key(hi_roman)
            if ck:
                coarse_counts[ck][en_gloss] += 1

        def rank_glosses(counter):
            # Rank by count desc, then wordfreq zipf_frequency desc, then alphabetical
            items = list(counter.keys())
            return sorted(items, key=lambda w: (-counter[w], -wordfreq.zipf_frequency(w, "en"), w))

        exact_map = {k: rank_glosses(cnt) for k, cnt in exact_counts.items()}
        key_map = {k: rank_glosses(cnt) for k, cnt in key_counts.items()}
        coarse_map = {k: rank_glosses(cnt) for k, cnt in coarse_counts.items()}

        return cls(exact_map, key_map, coarse_map)

    def lookup(self, word: str) -> tuple[list[str], str]:
        w = word.strip().lower()
        if w in self.exact_map:
            return self.exact_map[w], "exact"
        pk = phonetic_key(w)
        if pk and pk in self.key_map:
            return self.key_map[pk], "key"
        ck = coarse_key(w)
        if ck and ck in self.coarse_map:
            return self.coarse_map[ck], "coarse"
        return [], "none"

    def lookup_exact(self, word: str) -> list[str]:
        return self.exact_map.get(word.strip().lower(), [])

    def has(self, word: str) -> bool:
        _, level = self.lookup(word)
        return level != "none"

    def canonical_forms(self) -> list[str]:
        return sorted(self.exact_map.keys())

    def save(self, path: Path | str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "exact_map": self.exact_map,
            "key_map": self.key_map,
            "coarse_map": self.coarse_map,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path | str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["exact_map"], data["key_map"], data["coarse_map"])

def build_lexicon(muse_path: Path | str) -> RomanLexicon:
    p = Path(muse_path)
    pairs = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                hi_deva, en = parts[0], parts[1]
                # Casual romanize Hindi word
                hi_roman = casual_romanize(hi_deva)
                pairs.append((hi_roman, en))
    return RomanLexicon.from_pairs(pairs)

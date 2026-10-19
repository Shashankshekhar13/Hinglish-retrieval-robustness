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

def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            curr[j + 1] = prev[j] if c1 == c2 else 1 + min(prev[j], prev[j + 1], curr[j])
        prev = curr
    return prev[-1]

class RomanLexicon:
    def __init__(self, exact_map=None, key_candidates=None, coarse_candidates=None):
        # exact_map: roman_hi -> list[str] (ranked glosses)
        # key_candidates: key -> list[dict(headword=str, gloss=str, count=int, zipf=float)]
        # coarse_candidates: coarse_key -> list[dict(headword=str, gloss=str, count=int, zipf=float)]
        self.exact_map = exact_map or {}
        self.key_candidates = key_candidates or {}
        self.coarse_candidates = coarse_candidates or {}

    @property
    def key_map(self):
        # For backward compatibility and inspection
        return {k: [c["gloss"] for c in cands] for k, cands in self.key_candidates.items()}

    @property
    def coarse_map(self):
        return {k: [c["gloss"] for c in cands] for k, cands in self.coarse_candidates.items()}

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, str]]):
        # pairs is (romanized_hi, english_gloss)
        exact_counts = defaultdict(Counter)
        key_pair_counts = defaultdict(Counter)     # key -> (headword, gloss) -> count
        coarse_pair_counts = defaultdict(Counter)  # coarse_key -> (headword, gloss) -> count

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
                key_pair_counts[pk][(hi_roman, en_gloss)] += 1
            ck = coarse_key(hi_roman)
            if ck:
                coarse_pair_counts[ck][(hi_roman, en_gloss)] += 1

        def rank_exact(counter):
            items = list(counter.keys())
            return sorted(items, key=lambda w: (-counter[w], -wordfreq.zipf_frequency(w, "en"), w))

        def pack_candidates(pair_counter):
            cands = []
            for (hw, gl), cnt in pair_counter.items():
                z = round(wordfreq.zipf_frequency(gl, "en"), 4)
                cands.append({"headword": hw, "gloss": gl, "count": cnt, "zipf": z})
            # Base deterministic sorting
            cands.sort(key=lambda c: (-c["count"], -c["zipf"], c["headword"], c["gloss"]))
            return cands

        exact_map = {k: rank_exact(cnt) for k, cnt in exact_counts.items()}
        key_candidates = {k: pack_candidates(cnt) for k, cnt in key_pair_counts.items()}
        coarse_candidates = {k: pack_candidates(cnt) for k, cnt in coarse_pair_counts.items()}

        return cls(exact_map, key_candidates, coarse_candidates)

    def _rank_candidates_for_word(self, candidates: list[dict], word: str, max_dist: int | None = 2) -> list[str]:
        # Rank candidates by:
        # 1. Edit distance between typed word and headword (ascending)
        # 2. Count (descending)
        # 3. Zipf frequency (descending)
        # 4. Gloss string (alphabetical)
        scored = []
        for c in candidates:
            dist = levenshtein(word, c["headword"])
            if max_dist is not None and dist > max_dist:
                continue
            scored.append((dist, -c["count"], -c["zipf"], c["gloss"]))
        scored.sort()
        # Deduplicate glosses in ranked order
        seen = set()
        ranked = []
        for _, _, _, gl in scored:
            if gl not in seen:
                seen.add(gl)
                ranked.append(gl)
        return ranked

    def lookup(self, word: str, max_dist: int | None = 2) -> tuple[list[str], str]:
        w = word.strip().lower()
        if w in self.exact_map:
            return self.exact_map[w], "exact"
        pk = phonetic_key(w)
        if pk and pk in self.key_candidates:
            ranked = self._rank_candidates_for_word(self.key_candidates[pk], w, max_dist=max_dist)
            if ranked:
                return ranked, "key"
        ck = coarse_key(w)
        if ck and ck in self.coarse_candidates:
            ranked = self._rank_candidates_for_word(self.coarse_candidates[ck], w, max_dist=max_dist)
            if ranked:
                return ranked, "coarse"
        return [], "none"

    def lookup_exact(self, word: str) -> list[str]:
        return self.exact_map.get(word.strip().lower(), [])

    def has(self, word: str, max_dist: int | None = 2) -> bool:
        _, level = self.lookup(word, max_dist=max_dist)
        return level != "none"

    def canonical_forms(self) -> list[str]:
        return sorted(self.exact_map.keys())

    def save(self, path: Path | str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "exact_map": self.exact_map,
            "key_candidates": self.key_candidates,
            "coarse_candidates": self.coarse_candidates,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path | str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        key_cands = data.get("key_candidates")
        if key_cands is None and "key_map" in data:
            key_cands = {k: [{"headword": "", "gloss": g, "count": 1, "zipf": 0.0} for g in v] for k, v in data["key_map"].items()}
        coarse_cands = data.get("coarse_candidates")
        if coarse_cands is None and "coarse_map" in data:
            coarse_cands = {k: [{"headword": "", "gloss": g, "count": 1, "zipf": 0.0} for g in v] for k, v in data["coarse_map"].items()}
        return cls(data["exact_map"], key_cands, coarse_cands)

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

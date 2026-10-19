import re
from pathlib import Path
from wordfreq import zipf_frequency

TOK_RE = re.compile(r"[a-z0-9]+")

_DEFAULT_RESOURCES_DIR = Path(__file__).resolve().parent.parent.parent / "resources"

def tokenize(text: str) -> list[str]:
    return TOK_RE.findall(text.lower())

def load_stopword_groups(path=None) -> dict[str, str]:
    p = Path(path) if path else _DEFAULT_RESOURCES_DIR / "hi_stopwords.txt"
    if not p.exists():
        p = Path("resources/hi_stopwords.txt")
    mapping = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [w.strip().lower() for w in line.split(",") if w.strip()]
            if parts:
                canonical = parts[0]
                for variant in parts:
                    mapping[variant] = canonical
    return mapping

def load_ambiguous(path=None) -> set[str]:
    p = Path(path) if path else _DEFAULT_RESOURCES_DIR / "ambiguous.txt"
    if not p.exists():
        p = Path("resources/ambiguous.txt")
    ambiguous = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            for w in line.split(","):
                w = w.strip().lower()
                if w:
                    ambiguous.add(w)
    return ambiguous

class Tagger:
    def __init__(self, lexicon=None, stopwords_path=None, ambiguous_path=None):
        self.lexicon = lexicon
        self.stopword_map = load_stopword_groups(stopwords_path)
        self.stopwords = set(self.stopword_map.keys())
        self.ambiguous = load_ambiguous(ambiguous_path)

    def tag(self, tokens: list[str]) -> list[str]:
        tags = [None] * len(tokens)
        # Pass 1: Tag definite tokens
        is_rule2 = [False] * len(tokens)
        for i, tok in enumerate(tokens):
            t = tok.lower()
            if not t or t.isdigit():
                tags[i] = "OTHER"
            elif t in self.stopwords and t not in self.ambiguous:
                tags[i] = "HI"
                is_rule2[i] = True
            elif t in self.ambiguous:
                tags[i] = None
            elif zipf_frequency(t, "en") >= 3.0:
                tags[i] = "EN"
            elif self.lexicon is not None and self.lexicon.has(t):
                tags[i] = "HI"
            else:
                tags[i] = "EN"

        # Pass 2: Resolve ambiguous tokens based on rule-2 context only
        for i, tok in enumerate(tokens):
            if tags[i] is None:
                left_hi = i > 0 and is_rule2[i - 1]
                right_hi = i + 1 < len(tokens) and is_rule2[i + 1]
                if left_hi or right_hi:
                    tags[i] = "HI"
                else:
                    tags[i] = "EN"

        return tags

import re

_DIGRAPHS = [("chh", "ch"), ("ph", "f"), ("sh", "s"), ("kh", "k"), ("gh", "g"),
             ("th", "t"), ("dh", "d"), ("bh", "b"), ("jh", "j"),
             ("w", "v"), ("z", "j"), ("q", "k")]
_V = set("aeiou")

def _collapse(s):
    return re.sub(r"(.)\1+", r"\1", s)

def phonetic_key(word):
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    for a, b in _DIGRAPHS:
        w = w.replace(a, b)
    if len(w) >= 4 and w[-1] in "nm" and w[-2] in _V:
        w = w[:-1]
    w = _collapse(w)
    w = w[0] + "".join(c for c in w[1:] if c not in _V)
    return _collapse(w)

def coarse_key(word):
    k = phonetic_key(word)
    return k[:1] + k[1:].replace("h", "").replace("y", "") if k else ""

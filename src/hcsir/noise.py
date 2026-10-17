import random, re

VOW = "aeiou"
WORD = re.compile(r"[A-Za-z]+")

FINAL = {"hai": ["h", "hy", "he"], "hain": ["hn", "hen"], "nahi": ["nahin"],
         "nahin": ["nahi"], "kyun": ["kyu"], "mein": ["me", "men"]}

def _hi_ops(w):
    n = len(w)
    c = lambda i: w[i] not in VOW
    vd = [w[:i] + w[i+1:] for i in range(1, n-1)
          if w[i] in VOW and c(i-1) and c(i+1)] if n >= 4 else []
    vl = [w[:m.start()] + {"aa": "a", "ee": "i", "oo": "u"}[m.group()] + w[m.end():]
          for m in re.finditer(r"aa|ee|oo", w)]
    fv = FINAL.get(w, [])
    ca = []
    for a, b in [("v","w"),("w","v"),("j","z"),("z","j"),("f","ph"),("ph","f"),("s","sh"),("sh","s")]:
        for m in re.finditer(a, w):
            if a == "s" and w[m.end():m.end()+1] == "h":
                continue
            ca.append(w[:m.start()] + b + w[m.end():])
    ge = [w[:i] + w[i+1:] for i in range(n-1) if w[i] == w[i+1] and w[i] not in VOW]
    return [("vowel_delete", vd), ("vowel_length", vl), ("final_variant", fv),
            ("consonant_alt", ca), ("gemination", ge)]

def _en_ops(w):
    n = len(w)
    return [("swap", [w[:i] + w[i+1] + w[i] + w[i+2:] for i in range(1, n-2)]),
            ("delete", [w[:i] + w[i+1:] for i in range(1, n-1)]),
            ("dup", [w[:i] + w[i] + w[i:] for i in range(1, n-1)])]

def _valid(opmap, w):
    return [(n, [c for c in cs if c and c != w and c.isalpha()]) for n, cs in opmap(w)]

def _has_op(opmap, w):
    return any(cs for _, cs in _valid(opmap, w))

def _pick(opmap, w, rng):
    ops = [(n, cs) for n, cs in _valid(opmap, w) if cs]
    if not ops:
        return w, "none"
    n, cs = rng.choice(ops)
    return rng.choice(cs), n

def perturb_hi_word(word, rng):
    return _pick(_hi_ops, word.lower(), rng)

def _noise(text, eligible, perturb, levels, seed):
    toks = list(WORD.finditer(text))
    pos = [i for i, m in enumerate(toks) if eligible(m.group())]
    rng = random.Random(seed)
    rng.shuffle(pos)
    edits = []
    for i in pos:
        old = toks[i].group()
        new, op = perturb(old, rng)
        if old[0].isupper() and new[0].islower():
            new = new[0].upper() + new[1:]
        edits.append((i, old, new, op))
    out = {}
    for lv in levels:
        ed = edits[:round(lv * len(pos))]
        s = text
        for i, old, new, op in sorted(ed, key=lambda e: -toks[e[0]].start()):
            s = s[:toks[i].start()] + new + s[toks[i].end():]
        out[lv] = (s, [{"pos": i, "old": o, "new": n, "op": p} for i, o, n, p in ed])
    return out

def add_hi_noise(text, hindi_words, levels, seed):
    hw = {w.lower() for w in hindi_words}
    return _noise(text, lambda w: w.lower() in hw and _has_op(_hi_ops, w.lower()),
                  perturb_hi_word, levels, seed)

def add_en_noise(text, levels, seed):
    return _noise(text, lambda w: len(w) >= 4 and _has_op(_en_ops, w),
                  lambda w, rng: _pick(_en_ops, w, rng), levels, seed)

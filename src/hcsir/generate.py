import json, re, time
from dataclasses import dataclass, field
from pathlib import Path

DEVA = re.compile(r"[\u0900-\u097F]")
TOK = re.compile(r"[a-z0-9]+")

def _toks(s):
    return TOK.findall(s.lower())

@dataclass
class GenResult:
    hinglish: str
    hindi_words: list
    status: str
    attempts: int
    reasons: list = field(default_factory=list)

class RateLimiter:
    def __init__(self, rpm):
        self.gap, self.t = 60.0 / rpm, 0.0
    def wait(self):
        d = self.t + self.gap - time.monotonic()
        if d > 0:
            time.sleep(d)
        self.t = time.monotonic()

class Cache:
    def __init__(self, path):
        self.path, self.d = Path(path), {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.d[r["key"]] = r
    def get(self, k):
        return self.d.get(k)
    def put(self, k, rec):
        rec = {"key": k, **rec}
        self.d[k] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def validate(english, hinglish, hindi_words, min_mix=0.2):
    r = []
    if DEVA.search(hinglish):
        r.append("devanagari")
    if not hindi_words:
        r.append("no_hindi")
    toks = _toks(hinglish)
    hw = {t for w in hindi_words for t in _toks(w)}
    if any(t not in set(toks) for t in hw):
        r.append("hindi_word_missing")
    if toks and sum(t in hw for t in toks) / len(toks) < min_mix:
        r.append("too_little_mixing")
    if hinglish.strip().lower() == english.strip().lower():
        r.append("unchanged")
    return r

def _parse(raw):
    try:
        s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        d = json.loads(s)
        h, w = d["hinglish"], d["hindi_words"]
        assert isinstance(h, str) and isinstance(w, list)
        return h.strip(), [str(x) for x in w], []
    except Exception:
        return None, None, ["bad_json"]

def _call(client, model, prompt, temp, limiter):
    for i in range(6):
        limiter.wait()
        try:
            r = client.chat.completions.create(
                model=model, temperature=temp,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"})
            return r.choices[0].message.content
        except Exception:
            time.sleep(min(2 ** i, 60))
    raise RuntimeError("Groq call failed after retries (nothing cached)")

def generate_cs(dataset, qid, english, client, g, cache, limiter, template):
    key = f"{dataset}:{qid}:{g['prompt_version']}"
    c = cache.get(key)
    if c:
        return GenResult(c["hinglish"], c["hindi_words"], c["status"], c["attempts"], c["reasons"])
    prompt = template.replace("{english}", english)
    res, hing, hw, reasons = None, None, None, []
    for attempt in range(1, g["max_retries"] + 1):
        raw = _call(client, g["gen_model"], prompt, 0.2 if attempt == 1 else 0.7, limiter)
        hing, hw, reasons = _parse(raw)
        if hing is not None:
            reasons = validate(english, hing, hw, g.get("min_mix", 0.2))
        if not reasons:
            res = GenResult(hing, hw, "ok", attempt, [])
            break
    if res is None:
        res = GenResult(hing or "", hw or [], "failed", g["max_retries"], reasons)
    cache.put(key, {"hinglish": res.hinglish, "hindi_words": res.hindi_words,
                    "status": res.status, "attempts": res.attempts, "reasons": res.reasons})
    return res

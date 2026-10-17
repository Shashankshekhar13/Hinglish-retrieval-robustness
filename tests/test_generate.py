import json
from types import SimpleNamespace
from hcsir.generate import generate_cs, Cache, RateLimiter, validate

GOOD = json.dumps({"hinglish": "covid ka vaccine kitna safe hai", "hindi_words": ["ka", "kitna", "hai"]})
DEVA = json.dumps({"hinglish": "covid का vaccine safe hai", "hindi_words": ["hai"]})
EN = "how safe is the covid vaccine"
G = {"gen_model": "m", "max_retries": 3, "prompt_version": "v1"}

class Fake:
    def __init__(self, outs):
        self.outs, self.calls = list(outs), 0
        self.chat = self.completions = self
    def create(self, **kw):
        self.calls += 1
        out = self.outs.pop(0) if len(self.outs) > 1 else self.outs[0]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out))])

def run(outs, tmp_path, cache=None, client=None):
    client = client or Fake(outs)
    cache = cache or Cache(tmp_path / "c.jsonl")
    return generate_cs("d", "q1", EN, client, G, cache, RateLimiter(10**6), "{english}"), client

def test_valid_first_try(tmp_path):
    r, _ = run([GOOD], tmp_path)
    assert r.status == "ok" and r.attempts == 1

def test_devanagari_retry(tmp_path):
    r, _ = run([DEVA, GOOD], tmp_path)
    assert r.status == "ok" and r.attempts == 2

def test_fail_after_retries(tmp_path):
    r, _ = run(["not json"], tmp_path)
    assert r.status == "failed" and r.attempts == 3

def test_cache_skips_call(tmp_path):
    r, c = run([GOOD], tmp_path)
    run([GOOD], tmp_path, cache=Cache(tmp_path / "c.jsonl"), client=c)
    assert c.calls == 1

def test_validate_reasons():
    r = validate("q", "q", [])
    assert "unchanged" in r and "no_hindi" in r

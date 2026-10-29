import argparse, sys, yaml
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import os, winreg
for _k in ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]:
    if _k not in os.environ:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                val, _ = winreg.QueryValueEx(key, _k)
                if val:
                    os.environ[_k] = val
        except Exception:
            pass

from groq import Groq
from hcsir.data import load_beir
from hcsir.generate import Cache, RateLimiter, generate_cs
from hcsir.queries import build_query_file
from hcsir.annotation import pick_qids, N_PER_DS, SPARE

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--limit", type=int)
ap.add_argument("--annotation-first", action="store_true")
a = ap.parse_args()

cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
g = dict(cfg["groq"])
if "max_tokens" in cfg and a.dataset in cfg["max_tokens"]:
    g["max_tokens"] = cfg["max_tokens"][a.dataset]
ds = load_beir(a.dataset, cfg["datasets"][a.dataset], cfg["seed"])
qids = sorted(ds.queries)
if a.annotation_first:
    pool = [q for q in qids if len(ds.queries[q].split()) >= 3]
    main, spare = pick_qids(pool, N_PER_DS, SPARE, cfg["seed"], a.dataset)
    qids = main + spare
if a.limit:
    qids = qids[:a.limit]

template = open(f"prompts/cs_rewrite_{g['prompt_version']}.txt", encoding="utf-8").read()

import itertools
keys = []
if "GROQ_API_KEYS" in os.environ:
    keys = [k.strip() for k in os.environ["GROQ_API_KEYS"].split(",") if k.strip()]
else:
    # Auto-discover GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, etc.
    keys = []
    if os.environ.get("GROQ_API_KEY"):
        keys.append(os.environ["GROQ_API_KEY"])
    i = 2
    while f"GROQ_API_KEY_{i}" in os.environ:
        val = os.environ[f"GROQ_API_KEY_{i}"].strip()
        if val:
            keys.append(val)
        i += 1

if not keys:
    keys = [os.environ.get("GROQ_API_KEY", "")]

clients = [Groq(api_key=k, max_retries=1) for k in keys if k]
client_cycle = itertools.cycle(clients if clients else [Groq(max_retries=1)])
n_keys = max(1, len(clients))
rpm = (2 * n_keys) if a.dataset == "arguana" else min(45, 15 * n_keys)
cache, lim = Cache("data/cache/groq_gen.jsonl"), RateLimiter(rpm)
print(f"Using {n_keys} Groq key(s) with effective RPM={rpm}")

from hcsir.generate import GenResult
def gen(qid, en):
    if len(en.split()) < 3:
        return GenResult("", [], "skipped_short", 0)
    return generate_cs(a.dataset, qid, en, client_cycle, g, cache, lim, template)
print(build_query_file(a.dataset, ds.queries, gen, cfg["seed"], cfg["noise_levels"], qids=qids))

import argparse, sys, yaml
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import os
if "GROQ_API_KEY" not in os.environ:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            val, _ = winreg.QueryValueEx(key, "GROQ_API_KEY")
            if val:
                os.environ["GROQ_API_KEY"] = val
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
g = cfg["groq"]
ds = load_beir(a.dataset, cfg["datasets"][a.dataset], cfg["seed"])
qids = sorted(ds.queries)
if a.annotation_first:
    main, spare = pick_qids(qids, N_PER_DS, SPARE, cfg["seed"], a.dataset)
    qids = main + spare
if a.limit:
    qids = qids[:a.limit]

template = open(f"prompts/cs_rewrite_{g['prompt_version']}.txt", encoding="utf-8").read()
client, cache, lim = Groq(), Cache("data/cache/groq_gen.jsonl"), RateLimiter(g["rpm"])
from hcsir.generate import GenResult
def gen(qid, en):
    if len(en.split()) < 3:
        return GenResult("", [], "skipped_short", 0)
    return generate_cs(a.dataset, qid, en, client, g, cache, lim, template)
print(build_query_file(a.dataset, ds.queries, gen, cfg["seed"], cfg["noise_levels"], qids=qids))

import json, zlib
from pathlib import Path
from .noise import add_hi_noise, add_en_noise

def _k(lv):
    return str(int(round(lv * 100)))

def build_query_file(name, queries, gen_fn, seed, levels, out_dir="data/queries", qids=None):
    out = Path(out_dir) / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    recs = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["qid"]] = r
    target_qids = qids or sorted(queries)
    for idx, qid in enumerate(target_qids):
        if qid in recs:
            print(f"[{idx+1}/{len(target_qids)}] {qid}: {recs[qid]['gen_status']} (cached)", flush=True)
            continue
        print(f"[{idx+1}/{len(target_qids)}] {qid}: generating...", end="", flush=True)
        en = queries[qid]
        s = zlib.crc32(f"{seed}:{qid}".encode())
        g = gen_fn(qid, en)
        print(f" -> {g.status}", flush=True)
        if any("DailyTokenLimitError" in str(r) for r in getattr(g, "reasons", [])):
            print(f" [STOPPING: Daily token limit reached. Will resume after reset.]", flush=True)
            break
        if any(str(r).startswith("api_error") for r in getattr(g, "reasons", [])):
            print(f" [WARNING: {qid} failed with api_error ({g.reasons}), skipping writing to query file]", flush=True)
            continue
        en_n = add_en_noise(en, levels, s)
        rec = {"qid": qid, "dataset": name, "EN": en, "gen_status": g.status,
               "hindi_words": g.hindi_words,
               "noise_log": {"en": {_k(l): en_n[l][1] for l in levels}},
               "CS-clean": None}
        for l in levels:
            rec[f"EN-noise{_k(l)}"] = en_n[l][0]
            rec[f"CS-noise{_k(l)}"] = None
        if g.status == "ok":
            hi = add_hi_noise(g.hinglish, g.hindi_words, levels, s)
            rec["CS-clean"] = g.hinglish
            rec["noise_log"]["hi"] = {_k(l): hi[l][1] for l in levels}
            for l in levels:
                rec[f"CS-noise{_k(l)}"] = hi[l][0]
        recs[qid] = rec
        with open(out, "w", encoding="utf-8") as f:
            for q in sorted(recs):
                f.write(json.dumps(recs[q], ensure_ascii=False) + "\n")
    return out

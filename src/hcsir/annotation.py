import csv, json, random
from pathlib import Path

COLS = ["item_id", "dataset", "qid", "english", "hinglish",
        "natural", "meaning_preserved", "rewrite", "notes"]
N_PER_DS, SPARE = 40, 30

def pick_qids(all_qids, n, spare, seed, dataset):
    ids = sorted(all_qids)
    random.Random(f"{seed}:{dataset}").shuffle(ids)
    return ids[:n], ids[n:n + spare]

def _load(qdir, name):
    p = Path(qdir) / f"{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def export_sheet(qdir, out, qids_by_ds, n=N_PER_DS, seed=42):
    rows = []
    for name, qids in qids_by_ds.items():
        main, spare = pick_qids(qids, n, SPARE, seed, name)
        recs = {r["qid"]: r for r in _load(qdir, name)}
        good = [recs[q] for q in main + spare if q in recs and recs[q]["gen_status"] == "ok"][:n]
        if len(good) < n:
            print(f"WARNING {name}: only {len(good)} usable items")
        rows += [{"dataset": name, "qid": r["qid"], "english": r["EN"], "hinglish": r["CS-clean"]}
                 for r in good]
    random.Random(seed).shuffle(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"item_id": f"{i:03d}", **r})
    return Path(out)

def _yn(v, row):
    s = (v or "").strip().lower()
    if s in ("y", "yes"):
        return True
    if s in ("n", "no"):
        return False
    raise ValueError(f"row {row}: expected Y/N, got {v!r}")

def import_sheet(path, qdir):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))
    human, nat, mp, both, rew = {}, 0, 0, 0, 0
    for r in rows:
        a = _yn(r["natural"], r["item_id"])
        b = _yn(r["meaning_preserved"], r["item_id"])
        nat, mp, both = nat + a, mp + b, both + (a and b)
        rw = (r.get("rewrite") or "").strip()
        rew += bool(rw)
        text = rw or (r["hinglish"] if a and b else "")
        if text:
            human[(r["dataset"], r["qid"])] = text
    for name in {d for d, _ in human}:
        recs = _load(qdir, name)
        for rec in recs:
            if (name, rec["qid"]) in human:
                rec["CS-human"] = human[(name, rec["qid"])]
        with open(Path(qdir) / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n = len(rows)
    return {"n": n, "natural_pct": 100 * nat / n, "meaning_pct": 100 * mp / n,
            "both_pct": 100 * both / n, "rewritten_pct": 100 * rew / n,
            "cs_human_usable": len(human)}

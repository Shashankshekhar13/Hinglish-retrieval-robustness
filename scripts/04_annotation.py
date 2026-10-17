import sys, json, yaml
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hcsir.data import load_beir
from hcsir.annotation import export_sheet, import_sheet

cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
if sys.argv[1] == "export":
    q = {n: sorted(load_beir(n, k, cfg["seed"]).queries) for n, k in cfg["datasets"].items()}
    print(export_sheet("data/queries", "data/annotation/check_sheet.csv", q, seed=cfg["seed"]))
else:
    s = import_sheet("data/annotation/check_sheet_done.csv", "data/queries")
    json.dump(s, open("outputs/tables/annotation_stats.json", "w"), indent=2)
    print(s)

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hcsir.config import load_config
from hcsir.data import load_beir


def main():
    cfg = load_config("configs/experiment.yaml")
    seed = cfg.seed

    for name, n_queries in cfg.datasets.items():
        ds = load_beir(name, n_queries=n_queries, seed=seed)
        print(f"{name}: {len(ds.corpus)} docs, {len(ds.queries)} queries")


if __name__ == "__main__":
    main()

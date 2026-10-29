import argparse
import logging
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hcsir.run_experiments import run_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_grid")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Hinglish CS-IR experiment grid.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Filters in the form key=val1,val2 (e.g. --only retriever=bm25,minilm,e5small dataset=scifact)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional shorthand to filter by dataset (e.g. --dataset fiqa)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of existing run files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save runs, results.csv and norm_stats.csv",
    )
    parser.add_argument(
        "--queries_dir",
        type=str,
        default="data/queries",
        help="Directory containing dataset.jsonl query files",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Top-k documents to retrieve per query",
    )
    return parser.parse_args()


def parse_only_args(only_list: list[str]) -> dict[str, str]:
    only_dict = {}
    for item in only_list:
        if "=" in item:
            k, v = item.split("=", 1)
            only_dict[k.strip()] = v.strip()
    return only_dict


def main():
    args = parse_args()

    cfg_path = Path(args.config)
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {cfg_path} not found. Using defaults.")

    only_dict = parse_only_args(args.only)
    if args.dataset:
        only_dict["dataset"] = args.dataset
    logger.info(f"Running grid with filters: {only_dict if only_dict else 'None (Full grid)'}")

    df = run_grid(
        cfg=cfg,
        only=only_dict if only_dict else None,
        force=args.force,
        output_dir=args.output_dir,
        queries_dir=args.queries_dir,
        top_k=args.top_k,
    )

    logger.info(f"Grid execution complete! Total rows in results: {len(df)}")
    print(df.to_string(max_rows=50))


if __name__ == "__main__":
    main()

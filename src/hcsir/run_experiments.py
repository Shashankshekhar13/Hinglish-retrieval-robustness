import json
import logging
from pathlib import Path
import time
from typing import Any
import pandas as pd
import ranx
from ranx import Qrels, Run, evaluate

from .data import Dataset, load_beir
from .lexicon import RomanLexicon
from .tagger import Tagger
from .normalize import normalize, NormResult
from .retrievers import BaseRetriever, BM25Retriever, DenseRetriever, get_retriever
from .baselines.translit import translit_query
from .baselines.llm_translate import llm_translate

logger = logging.getLogger(__name__)


def load_clean_queries(dataset: str, queries_dir: str | Path = "data/queries") -> dict[str, dict[str, Any]]:
    """Loads queries for a dataset, keeping only those with gen_status == 'ok'."""
    path = Path(queries_dir) / f"{dataset}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    clean: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("gen_status") == "ok":
                clean[str(item["qid"])] = item

    # Check if human annotation sheet exists
    human_sheet = Path("data/annotation/check_sheet_done.csv")
    if human_sheet.exists():
        df_human = pd.read_csv(human_sheet)
        if "qid" in df_human.columns and "dataset" in df_human.columns:
            ds_human = df_human[df_human["dataset"] == dataset]
            for _, row in ds_human.iterrows():
                qid = str(row["qid"])
                if qid in clean:
                    rw = str(row.get("rewrite", "") or "").strip()
                    if rw and rw.lower() != "nan":
                        clean[qid]["CS-human"] = rw
                    elif (
                        str(row.get("natural", "")).strip().lower() in ("y", "yes")
                        and str(row.get("meaning_preserved", "")).strip().lower() in ("y", "yes")
                    ):
                        clean[qid]["CS-human"] = str(row.get("hinglish", ""))

    return clean


def get_grid_cells(include_human: bool = False) -> list[tuple[str, str, str]]:
    """Returns the full grid of (retriever, condition, normalizer) tuples."""
    cells = []

    # Row 1: bm25, minilm, e5small, bgem3 x EN, EN-noise25, EN-noise50 x none, key
    for ret in ["bm25", "minilm", "e5small", "bgem3"]:
        for cond in ["EN", "EN-noise25", "EN-noise50"]:
            for norm in ["none", "key"]:
                cells.append((ret, cond, norm))

    # Row 2: bm25, minilm, e5small, bgem3 x CS-clean, CS-noise25, CS-noise50 x none, stop, exact, key, key_expand
    for ret in ["bm25", "minilm", "e5small", "bgem3"]:
        for cond in ["CS-clean", "CS-noise25", "CS-noise50"]:
            for norm in ["none", "stop", "exact", "key", "key_expand"]:
                cells.append((ret, cond, norm))

    # Row 3: minilm_vocabexp x CS-clean, CS-noise25, CS-noise50 x none
    for cond in ["CS-clean", "CS-noise25", "CS-noise50"]:
        cells.append(("minilm_vocabexp", cond, "none"))

    # Row 4: e5small, bgem3 x CS-clean, CS-noise25, CS-noise50 x translit
    for ret in ["e5small", "bgem3"]:
        for cond in ["CS-clean", "CS-noise25", "CS-noise50"]:
            cells.append((ret, cond, "translit"))

    # Row 5: bm25, minilm, e5small, bgem3 x CS-clean, CS-noise50 x llm
    for ret in ["bm25", "minilm", "e5small", "bgem3"]:
        for cond in ["CS-clean", "CS-noise50"]:
            cells.append((ret, cond, "llm"))

    # Row 6: CS-human + EN on the same subset x none, key
    if include_human:
        for ret in ["bm25", "minilm", "e5small", "bgem3", "minilm_vocabexp"]:
            for cond in ["CS-human", "EN"]:
                for norm in ["none", "key"]:
                    cells.append((ret, cond, norm))

    return cells


def normalize_query_batch(
    dataset: str,
    condition: str,
    normalizer: str,
    clean_queries: dict[str, dict[str, Any]],
    lexicon: RomanLexicon,
    tagger: Tagger,
    client: Any = None,
    cfg: Any = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Normalizes a batch of queries for a given condition and normalizer mode."""
    norm_queries: dict[str, str] = {}
    aggregated_stats = {
        "tokens_hi": 0,
        "stopwords_dropped": 0,
        "resolved_exact": 0,
        "resolved_key": 0,
        "resolved_coarse": 0,
        "unresolved": 0,
        "gated_out_queries": 0,
    }

    for qid, record in clean_queries.items():
        raw_text = record.get(condition, "")
        if not raw_text:
            norm_queries[qid] = ""
            continue

        if normalizer == "none":
            norm_queries[qid] = raw_text

        elif normalizer in ("stop", "exact", "key", "key_expand"):
            res = normalize(raw_text, mode=normalizer, lexicon=lexicon, tagger=tagger)
            norm_queries[qid] = res.text
            s = res.stats
            aggregated_stats["tokens_hi"] += s.get("tokens_hi", 0)
            aggregated_stats["stopwords_dropped"] += s.get("stopwords_dropped", 0)
            aggregated_stats["resolved_exact"] += s.get("resolved_exact", 0)
            aggregated_stats["resolved_key"] += s.get("resolved_key", 0)
            aggregated_stats["resolved_coarse"] += s.get("resolved_coarse", 0)
            aggregated_stats["unresolved"] += s.get("unresolved", 0)
            if s.get("gated_out"):
                aggregated_stats["gated_out_queries"] += 1

        elif normalizer == "translit":
            trans_text = translit_query(raw_text, tagger=tagger)
            norm_queries[qid] = trans_text

        elif normalizer == "llm":
            from .baselines.llm_translate import llm_translate
            try:
                trans_text, _ = llm_translate(dataset, qid, raw_text, condition, client=client, cfg=cfg)
                if not trans_text or not trans_text.strip():
                    raise ValueError(f"Empty translation returned for {dataset}:{qid}:{condition}")
                norm_queries[qid] = trans_text
            except Exception as e:
                raise RuntimeError(
                    f"LLM translation failed for {dataset}:{qid}:{condition} (client={'provided' if client is not None else 'None'}): {e}. "
                    "Silent fallback to raw text is disabled."
                ) from e
        else:
            raise ValueError(f"Unknown normalizer: {normalizer!r}")

    return norm_queries, aggregated_stats


def _parse_only_filter(only: dict | None) -> dict[str, set[str]]:
    """Normalizes the `only` dict into sets of lowercase strings."""
    res = {}
    if not only:
        return res
    for k, v in only.items():
        if isinstance(v, str):
            vals = [x.strip() for x in v.split(",") if x.strip()]
        elif isinstance(v, (list, tuple, set)):
            vals = [str(x).strip() for x in v if str(x).strip()]
        else:
            vals = [str(v).strip()]
        res[k] = set(vals)
    return res


def run_grid(
    cfg: Any = None,
    datasets: dict[str, Dataset] | None = None,
    lexicon: RomanLexicon | None = None,
    tagger: Tagger | None = None,
    only: dict | None = None,
    force: bool = False,
    output_dir: str | Path = "outputs",
    queries_dir: str | Path = "data/queries",
    top_k: int = 100,
    llm_client: Any = None,
) -> pd.DataFrame:
    """Executes the factorial experiment grid and produces results.csv and norm_stats.csv."""
    out_path = Path(output_dir)
    runs_base_dir = out_path / "runs"
    runs_base_dir.mkdir(parents=True, exist_ok=True)

    filter_dict = _parse_only_filter(only)

    # 1. Determine datasets to evaluate
    if datasets is None:
        datasets = {}
        ds_cfg = getattr(cfg, "datasets", None)
        if isinstance(cfg, dict):
            ds_cfg = cfg.get("datasets", {})
        ds_names = list(ds_cfg.keys()) if ds_cfg else ["scifact", "nfcorpus", "arguana"]
        for name in ds_names:
            if "dataset" in filter_dict and name not in filter_dict["dataset"]:
                continue
            n_q = ds_cfg.get(name) if isinstance(ds_cfg, dict) else None
            datasets[name] = load_beir(name, n_queries=n_q, seed=getattr(cfg, "seed", 42) if not isinstance(cfg, dict) else cfg.get("seed", 42))

    # 2. Lazy load resources if not passed
    if lexicon is None:
        lexicon_file = Path("data/lexicon/roman_hi_en.json")
        if lexicon_file.exists():
            lexicon = RomanLexicon.load(lexicon_file)
        else:
            lexicon = RomanLexicon.from_pairs([])

    if tagger is None:
        tagger = Tagger(lexicon=lexicon)

    has_human = Path("data/annotation/check_sheet_done.csv").exists()
    all_grid_cells = get_grid_cells(include_human=has_human)

    # Normalization query cache: (dataset, condition, normalizer) -> (norm_queries, stats)
    norm_cache: dict[tuple[str, str, str], tuple[dict[str, str], dict[str, Any]]] = {}

    results_rows: list[dict[str, Any]] = []
    norm_stats_rows: list[dict[str, Any]] = []
    seen_norm_stats: set[tuple[str, str, str]] = set()

    for ds_name, ds in datasets.items():
        if "dataset" in filter_dict and ds_name not in filter_dict["dataset"]:
            continue

        clean_queries = load_clean_queries(ds_name, queries_dir=queries_dir)
        clean_qids = sorted(clean_queries.keys())
        n_queries = len(clean_qids)

        if n_queries == 0:
            logger.warning(f"No clean queries found for dataset {ds_name}, skipping.")
            continue

        # ranx Qrels matching evaluated query set exactly
        evaluated_qrels = Qrels({qid: ds.qrels[qid] for qid in clean_qids if qid in ds.qrels})

        # Filter cells applicable to this run
        ds_cells = []
        for ret, cond, norm in all_grid_cells:
            if "retriever" in filter_dict and ret not in filter_dict["retriever"]:
                continue
            if "condition" in filter_dict and cond not in filter_dict["condition"]:
                continue
            if "normalizer" in filter_dict and norm not in filter_dict["normalizer"]:
                continue
            ds_cells.append((ret, cond, norm))

        # Group cells by retriever to index only once
        retrievers_in_cells = sorted(set(r for r, _, _ in ds_cells))

        for ret_name in retrievers_in_cells:
            ret_cells = [(c, n) for r, c, n in ds_cells if r == ret_name]
            ret_run_dir = runs_base_dir / ds_name / ret_name
            ret_run_dir.mkdir(parents=True, exist_ok=True)

            # Check if all cells exist and do not need re-running
            cells_to_run = []
            for cond, norm in ret_cells:
                run_file = ret_run_dir / f"{cond}__{norm}.json"
                if force or not run_file.exists():
                    cells_to_run.append((cond, norm))

            retriever_inst = None
            if cells_to_run:
                logger.info(f"Indexing retriever {ret_name} on dataset {ds_name}...")
                retriever_inst = get_retriever(ret_name, cfg=cfg, lexicon=lexicon)
                retriever_inst.index(ds)

            for cond, norm in ret_cells:
                run_file = ret_run_dir / f"{cond}__{norm}.json"

                # Check if we should compute normalization stats
                norm_key = (ds_name, cond, norm)
                if norm_key not in norm_cache:
                    try:
                        nq, ns = normalize_query_batch(
                            ds_name, cond, norm, clean_queries, lexicon, tagger, client=llm_client, cfg=cfg
                        )
                        norm_cache[norm_key] = (nq, ns)
                    except Exception as e:
                        logger.warning(
                            f"Skipping cell ({ds_name}, {ret_name}, {cond}, {norm}): normalizer failed ({e}). "
                            "Never substituting original text."
                        )
                        norm_cache[norm_key] = None
                        continue
                elif norm_cache[norm_key] is None:
                    logger.warning(
                        f"Skipping cell ({ds_name}, {ret_name}, {cond}, {norm}): normalizer previously failed."
                    )
                    continue

                if norm_key not in seen_norm_stats:
                    seen_norm_stats.add(norm_key)
                    _, ns = norm_cache[norm_key]
                    t_hi = ns["tokens_hi"]
                    norm_stats_rows.append({
                        "dataset": ds_name,
                        "condition": cond,
                        "normalizer": norm,
                        "n_queries": n_queries,
                        "tokens_hi": t_hi,
                        "stopwords_dropped": ns["stopwords_dropped"],
                        "resolved_exact": ns["resolved_exact"],
                        "resolved_key": ns["resolved_key"],
                        "resolved_coarse": ns["resolved_coarse"],
                        "unresolved": ns["unresolved"],
                        "gated_out_queries": ns["gated_out_queries"],
                        "pct_exact": round(ns["resolved_exact"] / t_hi * 100, 2) if t_hi > 0 else 0.0,
                        "pct_key": round(ns["resolved_key"] / t_hi * 100, 2) if t_hi > 0 else 0.0,
                        "pct_coarse": round(ns["resolved_coarse"] / t_hi * 100, 2) if t_hi > 0 else 0.0,
                        "pct_unresolved": round(ns["unresolved"] / t_hi * 100, 2) if t_hi > 0 else 0.0,
                    })

                if run_file.exists() and not force:
                    # Skip search and load existing run
                    run = Run.from_file(str(run_file))
                else:
                    norm_queries, _ = norm_cache[norm_key]
                    t0 = time.monotonic()
                    raw_run = retriever_inst.search(norm_queries, k=top_k)
                    elapsed = time.monotonic() - t0
                    run = Run(raw_run)
                    run.save(str(run_file))
                    logger.info(f"[{ds_name}] {ret_name} | {cond} | {norm} finished in {elapsed:.2f}s")

                metrics = evaluate(evaluated_qrels, run, ["ndcg@10", "recall@100"])
                results_rows.append({
                    "dataset": ds_name,
                    "retriever": ret_name,
                    "condition": cond,
                    "normalizer": norm,
                    "ndcg@10": float(metrics["ndcg@10"]),
                    "recall@100": float(metrics["recall@100"]),
                    "n_queries": n_queries,
                })

    # Exact columns required by specification
    cols = ["dataset", "retriever", "condition", "normalizer", "ndcg@10", "recall@100", "n_queries"]
    results_df = pd.DataFrame(results_rows, columns=cols)
    results_csv_path = out_path / "results.csv"
    if results_csv_path.exists():
        try:
            prev_results = pd.read_csv(results_csv_path)
            results_df = pd.concat([prev_results, results_df], ignore_index=True)
            results_df = results_df.drop_duplicates(
                subset=["dataset", "retriever", "condition", "normalizer"], keep="last"
            )
        except Exception as e:
            raise IOError(f"Could not merge with existing results.csv at {results_csv_path}: {e}") from e
    results_df.to_csv(results_csv_path, index=False)

    if norm_stats_rows:
        norm_df = pd.DataFrame(norm_stats_rows)
        norm_csv_path = out_path / "norm_stats.csv"
        if norm_csv_path.exists():
            try:
                prev_norm = pd.read_csv(norm_csv_path)
                norm_df = pd.concat([prev_norm, norm_df], ignore_index=True)
                norm_df = norm_df.drop_duplicates(
                    subset=["dataset", "condition", "normalizer"], keep="last"
                )
            except Exception as e:
                raise IOError(f"Could not merge with existing norm_stats.csv at {norm_csv_path}: {e}") from e
        norm_df.to_csv(norm_csv_path, index=False)

    return results_df

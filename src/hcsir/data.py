import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Shared types (defined in Task 1, used everywhere)
Run = dict[str, dict[str, float]]  # qid -> doc_id -> score (top-100)


@dataclass
class Dataset:
    name: str
    corpus: dict[str, dict[str, str]]  # doc_id -> {"title": str, "text": str}
    queries: dict[str, str]  # qid -> English text
    qrels: dict[str, dict[str, int]]  # qid -> doc_id -> relevance score


def _parse_beir_from_hf(name: str) -> Dataset:
    from datasets import load_dataset

    # Load corpus
    corpus_ds = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    corpus: dict[str, dict[str, str]] = {}
    for row in corpus_ds:
        doc_id = str(row.get("_id", row.get("id", "")))
        title = str(row.get("title", ""))
        text = str(row.get("text", ""))
        corpus[doc_id] = {"title": title, "text": text}

    # Load queries
    queries_ds = load_dataset(f"BeIR/{name}", "queries", split="queries")
    raw_queries: dict[str, str] = {}
    for row in queries_ds:
        qid = str(row.get("_id", row.get("id", "")))
        text = str(row.get("text", ""))
        raw_queries[qid] = text

    # Load qrels (test split)
    qrels_ds = load_dataset(f"BeIR/{name}-qrels", split="test")
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row.get("query-id", row.get("query_id", row.get("qid", ""))))
        doc_id = str(row.get("corpus-id", row.get("corpus_id", row.get("doc_id", ""))))
        score = int(row.get("score", 1))
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = score

    # Keep only queries that appear in the test qrels
    queries = {qid: text for qid, text in raw_queries.items() if qid in qrels and len(qrels[qid]) > 0}

    return Dataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


def load_beir(
    name: str,
    n_queries: int | None = None,
    seed: int = 42,
    cache_dir: str | Path = "data/raw",
) -> Dataset:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    full_cache_file = cache_path / f"{name}_full.pkl"
    legacy_cache_file = cache_path / f"{name}.pkl"

    if full_cache_file.is_file():
        with open(full_cache_file, "rb") as f:
            ds = pickle.load(f)
    elif legacy_cache_file.is_file():
        with open(legacy_cache_file, "rb") as f:
            ds = pickle.load(f)
    else:
        ds = _parse_beir_from_hf(name)
        with open(full_cache_file, "wb") as f:
            pickle.dump(ds, f)

    # Subsample if requested and available queries exceed n_queries
    if n_queries is not None and len(ds.queries) > n_queries:
        rng = random.Random(seed)
        sampled_qids = rng.sample(sorted(ds.queries.keys()), n_queries)
        sampled_queries = {qid: ds.queries[qid] for qid in sorted(sampled_qids)}
        sampled_qrels = {qid: ds.qrels[qid] for qid in sorted(sampled_qids) if qid in ds.qrels}
        return Dataset(
            name=name,
            corpus=ds.corpus,
            queries=sampled_queries,
            qrels=sampled_qrels,
        )

    return ds

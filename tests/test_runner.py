import json
from pathlib import Path
import pandas as pd
import pytest

from hcsir.data import Dataset
from hcsir.lexicon import RomanLexicon
from hcsir.tagger import Tagger
from hcsir.run_experiments import run_grid, load_clean_queries


@pytest.fixture
def fake_experiment_env(tmp_path):
    # Setup temporary directory structure
    queries_dir = tmp_path / "data" / "queries"
    queries_dir.mkdir(parents=True)
    lexicon_dir = tmp_path / "data" / "lexicon"
    lexicon_dir.mkdir(parents=True)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)

    # 3 fake documents
    corpus = {
        "doc1": {"title": "Biomaterials properties", "text": "Inductive properties of biomaterials are important."},
        "doc2": {"title": "Stem cells", "text": "Hematopoietic stem cells divide and differentiate."},
        "doc3": {"title": "Goitre treatment", "text": "Radioiodine reduces goitre and thyroid volume."},
    }

    # 3 fake queries + 1 failed query to ensure filtering works
    records = [
        {
            "qid": "q1",
            "dataset": "fake_ds",
            "gen_status": "ok",
            "EN": "inductive properties of biomaterials",
            "EN-noise25": "inductve properties of biomaterials",
            "EN-noise50": "inductve propertees of biomaterals",
            "CS-clean": "biomaterials ke inductive properties",
            "CS-noise25": "biomaterials ke inductve properties",
            "CS-noise50": "biomaterials kee inductve propertees",
        },
        {
            "qid": "q2",
            "dataset": "fake_ds",
            "gen_status": "ok",
            "EN": "stem cells divide",
            "EN-noise25": "sem cells divide",
            "EN-noise50": "sem cells divde",
            "CS-clean": "stem cells divide hote hain",
            "CS-noise25": "stem cells divde hote hain",
            "CS-noise50": "stem cells divde hte hn",
        },
        {
            "qid": "q3",
            "dataset": "fake_ds",
            "gen_status": "ok",
            "EN": "radioiodine reduces thyroid volume",
            "EN-noise25": "radioiodine redces thyroid volume",
            "EN-noise50": "radioiodine redces thyrod volume",
            "CS-clean": "radioiodine thyroid volume kam karta hai",
            "CS-noise25": "radioiodine thyroid volume km karta hai",
            "CS-noise50": "radioiodine thyrod volume km krta h",
        },
        {
            "qid": "q4_failed",
            "dataset": "fake_ds",
            "gen_status": "fail",
            "EN": "should be ignored",
            "CS-clean": "ignored query",
        },
    ]

    with open(queries_dir / "fake_ds.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    qrels = {
        "q1": {"doc1": 1},
        "q2": {"doc2": 1},
        "q3": {"doc3": 1},
        "q4_failed": {"doc1": 1},
    }

    fake_ds = Dataset(
        name="fake_ds",
        corpus=corpus,
        queries={r["qid"]: r["EN"] for r in records},
        qrels=qrels,
    )

    lexicon = RomanLexicon.from_pairs([
        ("kam", "less"),
        ("karta", "does"),
        ("hote", "are"),
    ])
    tagger = Tagger(lexicon=lexicon)

    cfg = {
        "datasets": {"fake_ds": None},
        "retrievers": ["bm25"],
        "top_k": 10,
    }

    return {
        "tmp_path": tmp_path,
        "dataset": fake_ds,
        "lexicon": lexicon,
        "tagger": tagger,
        "cfg": cfg,
        "output_dir": output_dir,
        "queries_dir": queries_dir,
    }


def test_load_clean_queries_filters_status(fake_experiment_env):
    queries_dir = fake_experiment_env["queries_dir"]
    clean = load_clean_queries("fake_ds", queries_dir=queries_dir)
    assert len(clean) == 3
    assert "q1" in clean
    assert "q2" in clean
    assert "q3" in clean
    assert "q4_failed" not in clean


def test_run_grid_writes_files_and_exact_columns(fake_experiment_env):
    env = fake_experiment_env
    output_dir = env["output_dir"]

    # Run only bm25 on fake_ds
    df = run_grid(
        cfg=env["cfg"],
        datasets={"fake_ds": env["dataset"]},
        lexicon=env["lexicon"],
        tagger=env["tagger"],
        only={"dataset": ["fake_ds"], "retriever": ["bm25"]},
        output_dir=output_dir,
        queries_dir=env["queries_dir"],
    )

    # Check results.csv exists and exact columns
    results_csv_path = output_dir / "results.csv"
    assert results_csv_path.exists()

    expected_columns = [
        "dataset",
        "retriever",
        "condition",
        "normalizer",
        "ndcg@10",
        "recall@100",
        "n_queries",
    ]
    assert list(df.columns) == expected_columns

    df_loaded = pd.read_csv(results_csv_path)
    assert list(df_loaded.columns) == expected_columns

    # Check norm_stats.csv exists
    norm_stats_path = output_dir / "norm_stats.csv"
    assert norm_stats_path.exists()
    norm_df = pd.read_csv(norm_stats_path)
    assert "condition" in norm_df.columns
    assert "normalizer" in norm_df.columns
    assert "tokens_hi" in norm_df.columns

    # Check ranx run files exist
    runs_dir = output_dir / "runs" / "fake_ds" / "bm25"
    assert runs_dir.exists()
    # Check that at least EN__none and CS-clean__key run files exist
    assert (runs_dir / "EN__none.json").exists()
    assert (runs_dir / "CS-clean__key.json").exists()

    # Check n_queries equal between EN and CS rows
    en_rows = df[df["condition"] == "EN"]
    cs_rows = df[df["condition"] == "CS-clean"]
    assert len(en_rows) > 0
    assert len(cs_rows) > 0
    assert set(en_rows["n_queries"].unique()) == {3}
    assert set(cs_rows["n_queries"].unique()) == {3}
    assert en_rows["n_queries"].iloc[0] == cs_rows["n_queries"].iloc[0]


def test_run_grid_skips_existing(fake_experiment_env, monkeypatch):
    env = fake_experiment_env
    output_dir = env["output_dir"]

    # First run
    run_grid(
        cfg=env["cfg"],
        datasets={"fake_ds": env["dataset"]},
        lexicon=env["lexicon"],
        tagger=env["tagger"],
        only={"dataset": ["fake_ds"], "retriever": ["bm25"]},
        output_dir=output_dir,
        queries_dir=env["queries_dir"],
    )

    run_file = output_dir / "runs" / "fake_ds" / "bm25" / "EN__none.json"
    mtime_before = run_file.stat().st_mtime_ns

    # Mock search so if it's called again, it raises an error
    from hcsir.retrievers import BM25Retriever
    def fail_search(*args, **kwargs):
        raise AssertionError("Search was called when cell should be skipped!")

    monkeypatch.setattr(BM25Retriever, "search", fail_search)

    # Second run without force should skip search
    df_second = run_grid(
        cfg=env["cfg"],
        datasets={"fake_ds": env["dataset"]},
        lexicon=env["lexicon"],
        tagger=env["tagger"],
        only={"dataset": ["fake_ds"], "retriever": ["bm25"]},
        output_dir=output_dir,
        queries_dir=env["queries_dir"],
        force=False,
    )

    assert run_file.stat().st_mtime_ns == mtime_before
    assert len(df_second) > 0

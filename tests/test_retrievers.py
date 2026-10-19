import pytest
import numpy as np
from pathlib import Path
from hcsir.data import Dataset
from hcsir.retrievers import get_retriever, BM25Retriever, DenseRetriever

@pytest.fixture
def toy_dataset():
    corpus = {
        "doc1": {"title": "Vitamin D and Bone Health", "text": "Vitamin D deficiency causes rickets and weak bones."},
        "doc2": {"title": "Cardiovascular Disease", "text": "Heart disease is linked to high blood pressure and cholesterol."},
        "doc3": {"title": "Common Cold Prevention", "text": "Wash hands regularly and drink warm fluids to prevent cold."},
        "doc4": {"title": "ArguAna Test", "text": "This document shares an ID with a query for identical ID removal testing."}
    }
    queries = {
        "q1": "vitamin d deficiency",
        "doc4": "argument about identical id removal"
    }
    qrels = {
        "q1": {"doc1": 1},
        "doc4": {"doc4": 1, "doc2": 1}
    }
    return Dataset(name="toy_ds", corpus=corpus, queries=queries, qrels=qrels)

@pytest.fixture
def toy_cfg(tmp_path):
    class Cfg:
        seed = 42
        device = "cpu"
        batch_size = {"minilm": 2, "e5small": 2, "bgem3": 2}
        models = {
            "minilm": "sentence-transformers/all-MiniLM-L12-v2",
            "e5small": "intfloat/multilingual-e5-small",
            "bgem3": "BAAI/bge-m3"
        }
        cache_dir = tmp_path / "emb"
    return Cfg()

def test_bm25_ranks_exact_match_first(toy_dataset, toy_cfg):
    retriever = get_retriever("bm25", toy_cfg)
    retriever.index(toy_dataset)
    run = retriever.search({"q1": "vitamin d deficiency"}, k=3)
    assert "q1" in run
    top_doc = list(run["q1"].keys())[0]
    assert top_doc == "doc1"

def test_run_shape(toy_dataset, toy_cfg):
    retriever = get_retriever("bm25", toy_cfg)
    retriever.index(toy_dataset)
    run = retriever.search(toy_dataset.queries, k=2)
    for qid, doc_scores in run.items():
        assert len(doc_scores) <= 2
        scores = list(doc_scores.values())
        assert all(isinstance(s, float) for s in scores)
        # Check descending order
        assert scores == sorted(scores, reverse=True)

def test_identical_id_removed(toy_dataset, toy_cfg):
    # ArguAna dataset rule: remove doc from query's results if doc_id == qid
    toy_dataset.name = "arguana"
    retriever = get_retriever("bm25", toy_cfg)
    retriever.index(toy_dataset)
    run = retriever.search({"doc4": "argument about identical id removal"}, k=3)
    assert "doc4" not in run["doc4"]

@pytest.mark.slow
def test_dense_cache_reused(toy_dataset, toy_cfg, monkeypatch):
    toy_dataset.name = "toy_ds"
    retriever = get_retriever("minilm", toy_cfg)
    encode_calls = 0
    orig_encode = retriever._encode_passages

    def spy_encode(*args, **kwargs):
        nonlocal encode_calls
        encode_calls += 1
        return orig_encode(*args, **kwargs)

    monkeypatch.setattr(retriever, "_encode_passages", spy_encode)

    # First index: should encode
    retriever.index(toy_dataset)
    assert encode_calls == 1

    # Second index: should load from cache, no re-encode
    retriever2 = get_retriever("minilm", toy_cfg)
    monkeypatch.setattr(retriever2, "_encode_passages", spy_encode)
    retriever2.index(toy_dataset)
    assert encode_calls == 1  # Still 1!

@pytest.mark.slow
def test_e5_prefixes(toy_dataset, toy_cfg, monkeypatch):
    retriever = get_retriever("e5small", toy_cfg)
    captured_texts = []

    orig_encode = retriever.model.encode
    def spy_model_encode(sentences, *args, **kwargs):
        captured_texts.extend(list(sentences))
        return orig_encode(sentences, *args, **kwargs)

    monkeypatch.setattr(retriever.model, "encode", spy_model_encode)
    retriever.index(toy_dataset)
    # Check that passages had "passage: " prefix
    assert any(t.startswith("passage: ") for t in captured_texts)

    captured_texts.clear()
    retriever.search({"q1": "test query"}, k=2)
    # Check that query had "query: " prefix
    assert any(t.startswith("query: ") for t in captured_texts)

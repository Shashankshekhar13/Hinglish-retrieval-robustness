import pytest
from hcsir.config import load_config
from hcsir.data import load_beir, Dataset


def test_config_loads():
    cfg = load_config("configs/experiment.yaml")
    assert cfg.seed == 42
    assert cfg.datasets["arguana"] == 300
    assert cfg.datasets["scifact"] is None
    assert cfg.datasets["nfcorpus"] is None
    assert cfg.device == "auto"
    assert cfg.batch_size["minilm"] == 32
    assert cfg.batch_size["e5small"] == 32
    assert cfg.batch_size["bgem3"] == 8
    assert cfg.top_k == 100


@pytest.mark.slow
def test_load_scifact_shapes():
    ds = load_beir("scifact", None, 42)
    assert isinstance(ds, Dataset)
    assert len(ds.queries) == 300
    assert 5000 <= len(ds.corpus) <= 5300
    for qid in ds.queries:
        assert qid in ds.qrels
        assert len(ds.qrels[qid]) >= 1


@pytest.mark.slow
def test_subsample_deterministic():
    ds1 = load_beir("arguana", 300, 42)
    ds2 = load_beir("arguana", 300, 42)
    assert list(ds1.queries.keys()) == list(ds2.queries.keys())
    assert len(ds1.queries) == 300


@pytest.mark.slow
def test_subsample_respects_qrels():
    ds = load_beir("arguana", 300, 42)
    for qid in ds.queries:
        assert qid in ds.qrels
        assert len(ds.qrels[qid]) >= 1

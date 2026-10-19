import re
import pytest
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace

from hcsir.lexicon import RomanLexicon
from hcsir.tagger import Tagger
from hcsir.baselines.translit import translit_query
from hcsir.baselines.vocab_expansion import build_vocab_expanded_minilm
from hcsir.baselines.llm_translate import llm_translate

@pytest.fixture
def toy_lexicon():
    pairs = [
        ("paani", "water"),
        ("pani", "water"),
        ("bimari", "disease")
    ]
    return RomanLexicon.from_pairs(pairs)

@pytest.fixture
def toy_tagger(toy_lexicon):
    return Tagger(lexicon=toy_lexicon)

def test_translit_only_hi_tokens(toy_tagger):
    # In 'covid ka vaccine kitna safe hai', 'ka', 'kitna', 'hai' are HI stopwords.
    # EN tokens: 'covid', 'vaccine', 'safe' must remain pure ASCII English.
    # HI tokens must contain Devanagari characters.
    DEVA = re.compile(r"[\u0900-\u097F]")
    res = translit_query("covid ka vaccine kitna safe hai", toy_tagger)
    toks = res.split()
    assert toks[0] == "covid"
    assert DEVA.search(toks[1])  # 'ka' -> Devanagari
    assert toks[2] == "vaccine"
    assert DEVA.search(toks[3])  # 'kitna' -> Devanagari
    assert toks[4] == "safe"
    assert DEVA.search(toks[5])  # 'hai' -> Devanagari

def test_llm_translate_cached(tmp_path):
    class FakeMessage:
        content = "how safe is the vaccine"

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletions:
        def __init__(self):
            self.calls = 0
        def create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(choices=[FakeChoice()])

    class FakeClient:
        def __init__(self):
            self.completions = FakeCompletions()
            self.chat = SimpleNamespace(completions=self.completions)

    client = FakeClient()
    cache_path = tmp_path / "trans_cache.jsonl"
    cfg = {
        "groq": {
            "translate_model": "fake-model",
            "prompt_version": "v1"
        }
    }

    # First call: calls client and caches
    trans1, lat1 = llm_translate("scifact", "q1", "kya vaccine safe hai", "CS-clean", client, cfg, cache_path=cache_path)
    assert trans1 == "how safe is the vaccine"
    assert lat1 >= 0.0
    assert client.completions.calls == 1

    # Second call: loaded from cache, 0 client calls
    trans2, lat2 = llm_translate("scifact", "q1", "kya vaccine safe hai", "CS-clean", client, cfg, cache_path=cache_path)
    assert trans2 == "how safe is the vaccine"
    assert lat2 == lat1
    assert client.completions.calls == 1

@pytest.mark.slow
def test_added_token_single_word(toy_lexicon, tmp_path):
    cfg = SimpleNamespace(
        device="cpu",
        cache_dir=tmp_path / "emb",
        models={"minilm": "sentence-transformers/all-MiniLM-L12-v2"}
    )
    retriever = build_vocab_expanded_minilm(toy_lexicon, cfg)
    tokenizer = retriever.model.tokenizer

    # 'company' should remain untouched (not split into com-pani)
    toks_company = tokenizer.tokenize("company")
    assert toks_company == ["company"]

    # 'pani' should tokenize into exactly 1 single token, not ['pan', '##i']
    toks_pani = tokenizer.tokenize("pani")
    assert toks_pani == ["pani"]

@pytest.mark.slow
def test_new_embedding_is_gloss_mean(toy_lexicon, tmp_path):
    cfg = SimpleNamespace(
        device="cpu",
        cache_dir=tmp_path / "emb",
        models={"minilm": "sentence-transformers/all-MiniLM-L12-v2"}
    )
    retriever = build_vocab_expanded_minilm(toy_lexicon, cfg)
    tokenizer = retriever.model.tokenizer
    embeddings = retriever.model[0].auto_model.get_input_embeddings().weight.data

    # For 'pani', gloss is 'water'
    water_ids = tokenizer("water", add_special_tokens=False)["input_ids"]
    expected_emb = embeddings[water_ids].mean(dim=0).cpu().numpy()

    pani_id = tokenizer.convert_tokens_to_ids("pani")
    actual_emb = embeddings[pani_id].cpu().numpy()

    assert np.allclose(actual_emb, expected_emb, atol=1e-5)

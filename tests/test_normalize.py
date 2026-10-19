import pytest
from hcsir.lexicon import RomanLexicon
from hcsir.tagger import Tagger
from hcsir.normalize import normalize, NormResult

@pytest.fixture
def toy_setup():
    pairs = [
        ("paani", "water"),
        ("khana", "food"),
        ("bimari", "disease"),
    ]
    lexicon = RomanLexicon.from_pairs(pairs)
    tagger = Tagger(lexicon=lexicon)
    return lexicon, tagger

def test_normalize_mode_none(toy_setup):
    lexicon, tagger = toy_setup
    res = normalize("kya paani se bimari hai", "none", lexicon, tagger)
    assert res.text == "kya paani se bimari hai"

def test_normalize_mode_stop(toy_setup):
    lexicon, tagger = toy_setup
    # 'kya', 'se', 'hai' are in hi_stopwords.txt and dropped
    res = normalize("kya paani se bimari hai", "stop", lexicon, tagger)
    assert res.text == "paani bimari"
    assert res.stats["stopwords_dropped"] == 3

def test_normalize_mode_exact(toy_setup):
    lexicon, tagger = toy_setup
    # 'pani' is not an exact form in toy lexicon (only 'paani' is)
    # 'bimari' is an exact form -> 'disease'
    # 'se' is a stopword and dropped
    res = normalize("pani se bimari", "exact", lexicon, tagger)
    assert res.text == "pani disease"
    assert res.stats["resolved_exact"] == 1
    assert res.stats["stopwords_dropped"] == 1

def test_normalize_mode_key(toy_setup):
    lexicon, tagger = toy_setup
    # 'pani' resolves to 'water' via phonetic key
    # 'bimari' resolves to 'disease' via exact
    # 'se' is a stopword and dropped
    res = normalize("pani se bimari", "key", lexicon, tagger)
    assert res.text == "water disease"
    assert res.stats["resolved_key"] == 1
    assert res.stats["resolved_exact"] == 1
    assert res.stats["stopwords_dropped"] == 1

def test_normalize_mode_key_expand(toy_setup):
    lexicon, tagger = toy_setup
    res = normalize("pani se bimari", "key_expand", lexicon, tagger)
    assert res.text == "pani water bimari disease"
    assert res.stats["stopwords_dropped"] == 1

def test_normalize_english_untouched_all_en(toy_setup):
    lexicon, tagger = toy_setup
    res = normalize("what causes diabetes", "key", lexicon, tagger)
    assert res.text == "what causes diabetes"

def test_normalize_gated_out(toy_setup):
    lexicon, tagger = toy_setup
    # Pure English query without rule-2 stopwords: even if a token would match key lexicon, it is gated out
    res = normalize("what causes diabetes", "key", lexicon, tagger)
    assert res.text == "what causes diabetes"
    assert res.stats["gated_out"] is True

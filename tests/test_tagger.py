import pytest
from hcsir.tagger import Tagger, tokenize, load_stopword_groups

def test_basic_mix():
    tagger = Tagger()
    tokens = tokenize("covid ka vaccine kitna safe hai")
    assert tagger.tag(tokens) == ["EN", "HI", "EN", "HI", "EN", "HI"]

def test_ambiguous_hi_context():
    tagger = Tagger(lexicon=None)
    # 'main' has direct neighbor 'ja'/'raha' or 'ghar'
    # 'main ja raha hoon': 'ja' is a stopword (rule 2), so 'main' gets HI
    tokens = tokenize("main ja raha hoon")
    tags = tagger.tag(tokens)
    assert tags[0] == "HI"
    assert tags[1] == "HI"
    assert tags[2] == "HI"
    assert tags[3] == "HI"

def test_rule5_does_not_trigger_ambiguous_hi():
    class DummyLexicon:
        def has(self, w):
            return w == "drc"
    tagger = Tagger(lexicon=DummyLexicon())
    # 'in' is ambiguous, 'drc' is tagged HI by rule 5 (lexicon)
    # rule 3: 'in' must NOT become HI because 'drc' is rule 5, not rule 2
    tokens = tokenize("in drc")
    tags = tagger.tag(tokens)
    assert tags[0] == "EN"
    assert tags[1] == "HI"

def test_ambiguous_en_context():
    tagger = Tagger()
    tokens = tokenize("what is the main cause of diabetes")
    assert tagger.tag(tokens) == ["EN"] * len(tokens)

def test_variants_map():
    mapping = load_stopword_groups()
    assert mapping["nhi"] == "nahi"
    assert mapping["nahi"] == "nahi"
    assert mapping["kya"] == "kya"
    assert mapping["se"] == "se"
    assert mapping["hai"] == "hai"

def test_numbers():
    tagger = Tagger()
    tokens = tokenize("covid 19")
    assert tagger.tag(tokens) == ["EN", "OTHER"]

def test_tokenize():
    assert tokenize("Kya COVID-19 safe hai?") == ["kya", "covid", "19", "safe", "hai"]

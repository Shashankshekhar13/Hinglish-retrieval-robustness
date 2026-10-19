from pathlib import Path
import pytest
from hcsir.phonetic import phonetic_key, coarse_key
from hcsir.lexicon import casual_romanize, build_lexicon, RomanLexicon

def test_casual_romanize_key_match():
    # casual_romanize converts Devanagari to casual romanized Hindi
    # which should yield phonetic keys matching the expected casual variants
    pairs = [
        ("नहीं", "nahi"),
        ("पानी", "pani"),
        ("खाना", "khana"),
        ("कमरा", "kamra"),
        ("घर", "ghar")
    ]
    for deva, roman in pairs:
        assert phonetic_key(casual_romanize(deva)) == phonetic_key(roman), f"Mismatch for {deva} vs {roman}"

def test_lookup_levels():
    toy_path = Path("tests/fixtures/muse_toy.txt")
    lex = build_lexicon(toy_path)
    # पानी -> paanee (exact)
    glosses, level = lex.lookup("paanee")
    assert glosses == ["water"]
    assert level == "exact"

    # "paani" has phonetic_key == phonetic_key("paanee") -> "key"
    glosses, level = lex.lookup("paani")
    assert glosses == ["water"]
    assert level == "key"

    # "xyzq" matches nothing -> level "none"
    glosses, level = lex.lookup("xyzq")
    assert glosses == []
    assert level == "none"

def test_from_pairs():
    lex = RomanLexicon.from_pairs([("paani", "water")])
    assert lex.has("paani")
    glosses, level = lex.lookup("pani")
    assert glosses == ["water"]
    assert level == "key"

def test_gloss_ranking_deterministic(tmp_path):
    toy_path = Path("tests/fixtures/muse_toy.txt")
    lex1 = build_lexicon(toy_path)
    lex2 = build_lexicon(toy_path)
    p1 = tmp_path / "lex1.json"
    p2 = tmp_path / "lex2.json"
    lex1.save(p1)
    lex2.save(p2)
    assert p1.read_bytes() == p2.read_bytes()

def test_roundtrip(tmp_path):
    toy_path = Path("tests/fixtures/muse_toy.txt")
    lex = build_lexicon(toy_path)
    save_path = tmp_path / "saved_lex.json"
    lex.save(save_path)
    loaded = RomanLexicon.load(save_path)
    probes = ["paani", "paanee", "ghar", "house", "xyz"]
    for word in probes:
        assert lex.lookup(word) == loaded.lookup(word)

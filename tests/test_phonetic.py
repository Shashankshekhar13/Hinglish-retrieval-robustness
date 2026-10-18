import pytest
from hcsir.phonetic import phonetic_key, coarse_key

@pytest.mark.parametrize("words,key", [
    (["nahi", "nahin", "nhi"], "nh"),
    (["accha", "acha", "achha"], "ach"),
    (["khana", "khaana"], "kn"),
    (["zaroor", "jaroor", "zarur"], "jr"),
    (["theek", "thik"], "tk"),
    (["phir", "fir"], "fr"),
    (["hum"], "hm"),
    ([""], ""),
    (["Nahi!"], "nh"),
])
def test_phonetic_key(words, key):
    for w in words:
        assert phonetic_key(w) == key

def test_coarse_key():
    assert coarse_key("nai") == coarse_key("nahi") == "n"
    assert coarse_key("kya") == coarse_key("kia") == "k"

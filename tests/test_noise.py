import random
from hcsir.noise import _has_op, _hi_ops, perturb_hi_word, add_hi_noise, add_en_noise

HI = ("nahi hai kya kaise bimari paani accha karna theek kaam mujhe aur bhi zaroor "
      "phir kitna kyun dawai ghar khana raha hoon sakta chahiye zyada log pehle baad abhi").split()
T = "covid ka vaccine kitna safe hai aur dawai bhi nahi kaam karta"
HW = ["ka", "kitna", "hai", "aur", "dawai", "bhi", "nahi", "kaam", "karta"]

def test_operator_changes_word():
    rng = random.Random(0)
    for w in HI * 5:
        if not _has_op(_hi_ops, w):
            continue
        new, _ = perturb_hi_word(w, rng)
        assert new != w and new.isalpha() and new == new.lower()

def test_nested_rate_and_en_untouched():
    r = add_hi_noise(T, HW, [0.25, 0.5], 42)
    e25 = {(e["pos"], e["new"]) for e in r[0.25][1]}
    e50 = {(e["pos"], e["new"]) for e in r[0.5][1]}
    assert e25 <= e50
    eligible_count = sum(_has_op(_hi_ops, w) for w in HW)
    assert len(r[0.5][1]) == round(0.5 * eligible_count)
    for w in ["covid", "vaccine", "safe"]:
        assert w in r[0.5][0].split()

def test_deterministic():
    assert add_hi_noise(T, HW, [0.5], 1) == add_hi_noise(T, HW, [0.5], 1)
    assert add_hi_noise(T, HW, [0.5], 1) != add_hi_noise(T, HW, [0.5], 2)

def test_en_short_tokens_untouched():
    r = add_en_noise("is the covid vaccine safe for all", [0.5], 3)
    for e in r[0.5][1]:
        assert len(e["old"]) >= 4

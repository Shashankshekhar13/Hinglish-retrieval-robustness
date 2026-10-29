from dataclasses import dataclass
from .tagger import tokenize, Tagger
from .lexicon import RomanLexicon

@dataclass
class NormResult:
    text: str
    stats: dict

def normalize(text: str, mode: str, lexicon: RomanLexicon, tagger: Tagger) -> NormResult:
    valid_modes = {"none", "stop", "exact", "key", "key_expand"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown normalizer mode: {mode!r}. Valid modes are: {valid_modes}")

    stats = {
        "tokens_hi": 0,
        "resolved_exact": 0,
        "resolved_key": 0,
        "resolved_coarse": 0,
        "unresolved": 0,
        "stopwords_dropped": 0,
        "gated_out": False
    }

    if mode == "none" or not text.strip():
        return NormResult(text=text, stats=stats)

    tokens = tokenize(text)
    if not tokens:
        return NormResult(text="", stats=stats)

    # Check rule-2 Hindi stopword gating:
    # A token is rule 2 if it is in tagger.stopwords and not in tagger.ambiguous
    has_rule2 = any(t in tagger.stopwords and t not in tagger.ambiguous for t in tokens)

    # In key and key_expand modes, if there is no rule-2 Hindi token, gate out and return unchanged
    if mode in ("key", "key_expand") and not has_rule2:
        stats["gated_out"] = True
        return NormResult(text=" ".join(tokens), stats=stats)

    tags = tagger.tag(tokens)
    out_tokens = []

    for tok, tag in zip(tokens, tags):
        if tag != "HI":
            out_tokens.append(tok)
            continue

        stats["tokens_hi"] += 1

        # Check if it's a stopword variant
        if tok in tagger.stopwords:
            stats["stopwords_dropped"] += 1
            # Dropped in all active modes (stop, exact, key, key_expand)
            continue

        if mode == "stop":
            out_tokens.append(tok)
            continue

        if mode == "exact":
            glosses = lexicon.lookup_exact(tok)
            if glosses:
                stats["resolved_exact"] += 1
                out_tokens.append(glosses[0])
            else:
                stats["unresolved"] += 1
                out_tokens.append(tok)
            continue

        if mode in ("key", "key_expand"):
            glosses, level = lexicon.lookup(tok)
            if level == "exact":
                stats["resolved_exact"] += 1
            elif level == "key":
                stats["resolved_key"] += 1
            elif level == "coarse":
                stats["resolved_coarse"] += 1
            else:
                stats["unresolved"] += 1

            if glosses:
                top_gloss = glosses[0]
                if mode == "key":
                    out_tokens.append(top_gloss)
                else:  # key_expand
                    out_tokens.append(tok)
                    out_tokens.append(top_gloss)
            else:
                out_tokens.append(tok)
            continue

    return NormResult(text=" ".join(out_tokens), stats=stats)

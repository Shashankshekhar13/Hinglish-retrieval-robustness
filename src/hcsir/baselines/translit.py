from indic_transliteration import sanscript
from ..tagger import Tagger, tokenize

def translit_word_to_devanagari(word: str) -> str:
    # Use indic_transliteration to map Romanized/ITRANS Hindi token to Devanagari
    return sanscript.transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)

def translit_query(text: str, tagger: Tagger) -> str:
    tokens = tokenize(text)
    if not tokens:
        return ""
    tags = tagger.tag(tokens)
    out_tokens = []
    for tok, tag in zip(tokens, tags):
        if tag == "HI":
            out_tokens.append(translit_word_to_devanagari(tok))
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens)

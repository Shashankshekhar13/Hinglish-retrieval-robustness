from pathlib import Path
import torch
import numpy as np
from tokenizers import AddedToken
from sentence_transformers import SentenceTransformer

from ..retrievers import DenseRetriever
from ..lexicon import RomanLexicon

def build_vocab_expanded_minilm(lexicon: RomanLexicon, cfg) -> DenseRetriever:
    dev = getattr(cfg, "device", "auto")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    base_model_name = getattr(cfg, "models", {}).get("minilm", "sentence-transformers/all-MiniLM-L12-v2")
    st_model = SentenceTransformer(base_model_name, device="cpu")
    tokenizer = st_model.tokenizer
    hf_model = st_model[0].auto_model

    # Get input embeddings layer
    input_embeddings = hf_model.get_input_embeddings()
    orig_weights = input_embeddings.weight.data.clone()
    emb_dim = orig_weights.shape[1]

    # Find canonical romanized Hindi words of length >= 3 not already a single token
    words_to_add = []
    word_glosses = {}
    for w in lexicon.canonical_forms():
        if len(w) < 3:
            continue
        # Check if tokenizer already treats it as a single token
        subwords = tokenizer.tokenize(w)
        if len(subwords) == 1 and subwords[0] == w:
            continue
        glosses = lexicon.lookup_exact(w)
        if glosses:
            words_to_add.append(w)
            word_glosses[w] = glosses

    if words_to_add:
        # Precompute gloss subword vectors before expanding tokenizer
        word_vectors = {}
        for w in words_to_add:
            gloss_vectors = []
            for g in word_glosses[w]:
                sub_ids = tokenizer(g, add_special_tokens=False)["input_ids"]
                if sub_ids:
                    g_vec = orig_weights[sub_ids].mean(dim=0)
                    gloss_vectors.append(g_vec)
            if gloss_vectors:
                word_vectors[w] = torch.stack(gloss_vectors).mean(dim=0)

        # Add tokens with single_word=True
        tokens_to_add = [AddedToken(w, single_word=True) for w in words_to_add]
        num_added = tokenizer.add_tokens(tokens_to_add)

        # Resize embeddings
        hf_model.resize_token_embeddings(len(tokenizer))
        new_weights = hf_model.get_input_embeddings().weight.data

        # Initialize each new embedding
        for w, vec in word_vectors.items():
            token_id = tokenizer.convert_tokens_to_ids(w)
            new_weights[token_id] = vec

    if dev != "cpu":
        st_model = st_model.to(dev)

    # Wrap in DenseRetriever with separate cache dir outputs/emb/minilm_vocabexp/
    if hasattr(cfg, "cache_dir") and cfg.cache_dir:
        cache_dir = Path(cfg.cache_dir)
    else:
        cache_dir = Path("outputs/emb")

    vocabexp_cfg = type("VocabExpCfg", (), {
        "device": dev,
        "batch_size": getattr(cfg, "batch_size", {}),
        "cache_dir": cache_dir
    })()

    retriever = DenseRetriever("minilm_vocabexp", base_model_name, cfg=vocabexp_cfg)
    retriever._model = st_model
    return retriever

import json
from pathlib import Path
from typing import Any
import numpy as np
import Stemmer
import bm25s
from .data import Dataset, Run

class BaseRetriever:
    def index(self, ds: Dataset) -> None:
        raise NotImplementedError

    def search(self, queries: dict[str, str], k: int = 100) -> Run:
        raise NotImplementedError

class BM25Retriever(BaseRetriever):
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = None
        self.doc_ids = []
        self.ds_name = ""

    def index(self, ds: Dataset) -> None:
        self.ds_name = ds.name
        self.doc_ids = sorted(ds.corpus.keys())
        corpus_texts = []
        for did in self.doc_ids:
            doc = ds.corpus[did]
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            full_text = f"{title} {text}".strip()
            corpus_texts.append(full_text)

        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=self.stemmer)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def search(self, queries: dict[str, str], k: int = 100) -> Run:
        if self.retriever is None:
            raise RuntimeError("Index has not been built. Call index(ds) first.")

        qids = list(queries.keys())
        query_texts = [queries[qid] for qid in qids]
        query_tokens = bm25s.tokenize(query_texts, stopwords="en", stemmer=self.stemmer)

        # Retrieve top k + 1 in case we need to filter identical id
        fetch_k = min(len(self.doc_ids), k + 1)
        results, scores = self.retriever.retrieve(query_tokens, k=fetch_k)

        run: Run = {}
        for idx, qid in enumerate(qids):
            run[qid] = {}
            for doc_idx, score in zip(results[idx], scores[idx]):
                doc_id = self.doc_ids[int(doc_idx)]
                if self.ds_name == "arguana" and doc_id == qid:
                    continue
                run[qid][doc_id] = float(score)
                if len(run[qid]) >= k:
                    break
        return run

class DenseRetriever(BaseRetriever):
    def __init__(self, name: str, model_name: str, cfg=None):
        self.name = name
        self.model_name = model_name
        self.cfg = cfg
        self.ds_name = ""
        self.doc_ids = []
        self.corpus_embeddings = None
        self._model = None

        # Cache dir
        if hasattr(cfg, "cache_dir") and cfg.cache_dir:
            self.emb_dir = Path(cfg.cache_dir) / name
        else:
            self.emb_dir = Path("outputs/emb") / name

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = getattr(self.cfg, "device", "auto")
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=dev)
            self._model.max_seq_length = 512
        return self._model

    def _get_batch_size(self) -> int:
        bs_cfg = getattr(self.cfg, "batch_size", {})
        if isinstance(bs_cfg, dict):
            return bs_cfg.get(self.name, 32)
        return 32

    def _encode_passages(self, texts: list[str]) -> np.ndarray:
        import torch
        # Add passage prefix for e5
        if "e5" in self.name or "e5" in self.model_name:
            texts = [f"passage: {t}" for t in texts]

        # Sort texts by length before encoding to save memory and time
        indexed_texts = sorted(enumerate(texts), key=lambda x: len(x[1]))
        orig_indices = [idx for idx, _ in indexed_texts]
        sorted_texts = [txt for _, txt in indexed_texts]

        bs = self._get_batch_size()
        embeddings = None

        # OOM handling: halve batch size down to 1
        while bs >= 1:
            try:
                embeddings = self.model.encode(
                    sorted_texts,
                    batch_size=bs,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    bs //= 2
                    if bs < 1:
                        raise e
                else:
                    raise e

        # Restore original order
        restored = np.zeros_like(embeddings)
        for sorted_pos, orig_pos in enumerate(orig_indices):
            restored[orig_pos] = embeddings[sorted_pos]
        return restored

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if "e5" in self.name or "e5" in self.model_name:
            texts = [f"query: {t}" for t in texts]
        return self.model.encode(
            texts,
            batch_size=self._get_batch_size(),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    def index(self, ds: Dataset) -> None:
        import torch
        self.ds_name = ds.name
        self.doc_ids = sorted(ds.corpus.keys())
        self.emb_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.emb_dir / f"{ds.name}.npy"
        ids_path = self.emb_dir / f"{ds.name}_doc_ids.json"
        meta_path = self.emb_dir / "meta.json"

        # Check if cache exists and matches doc_ids
        if cache_path.exists() and ids_path.exists():
            with open(ids_path, "r", encoding="utf-8") as f:
                cached_ids = json.load(f)
            if cached_ids == self.doc_ids:
                self.corpus_embeddings = np.load(cache_path)
                return

        # Encode passages
        corpus_texts = []
        for did in self.doc_ids:
            doc = ds.corpus[did]
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            corpus_texts.append(f"{title} {text}".strip())

        self.corpus_embeddings = self._encode_passages(corpus_texts)

        # Save cache
        np.save(cache_path, self.corpus_embeddings)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f)

        # Save metadata
        device_name = str(self.model.device)
        gpu_model = torch.cuda.get_device_name(0) if torch.cuda.is_available() and "cuda" in device_name else "N/A"
        meta = {
            "model_name": self.model_name,
            "device": device_name,
            "gpu_model": gpu_model,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def search(self, queries: dict[str, str], k: int = 100) -> Run:
        if self.corpus_embeddings is None:
            raise RuntimeError("Index has not been built. Call index(ds) first.")

        qids = list(queries.keys())
        query_texts = [queries[qid] for qid in qids]
        q_embs = self.encode_queries(query_texts)

        # Exact dot product scores: (n_queries, n_docs)
        scores_matrix = np.dot(q_embs, self.corpus_embeddings.T)

        run: Run = {}
        fetch_k = min(len(self.doc_ids), k + 1)
        for idx, qid in enumerate(qids):
            row = scores_matrix[idx]
            # Get top indices
            top_indices = np.argpartition(-row, fetch_k - 1)[:fetch_k]
            top_indices = top_indices[np.argsort(-row[top_indices])]

            run[qid] = {}
            for doc_idx in top_indices:
                doc_id = self.doc_ids[doc_idx]
                if self.ds_name == "arguana" and doc_id == qid:
                    continue
                run[qid][doc_id] = float(row[doc_idx])
                if len(run[qid]) >= k:
                    break
        return run

def get_retriever(name: str, cfg, lexicon=None) -> BaseRetriever:
    if name == "bm25":
        return BM25Retriever(cfg=cfg)
    if name == "minilm_vocabexp":
        from .baselines.vocab_expansion import build_vocab_expanded_minilm
        if lexicon is None:
            from .lexicon import RomanLexicon
            lexicon = RomanLexicon.load("data/lexicon/roman_hi_en.json")
        return build_vocab_expanded_minilm(lexicon, cfg)
    models = getattr(cfg, "models", {})
    if isinstance(models, dict) and name in models:
        model_name = models[name]
    elif name == "minilm":
        model_name = "sentence-transformers/all-MiniLM-L12-v2"
    elif name == "e5small":
        model_name = "intfloat/multilingual-e5-small"
    elif name == "bgem3":
        model_name = "BAAI/bge-m3"
    else:
        raise ValueError(f"Unknown retriever: {name}")
    return DenseRetriever(name, model_name, cfg=cfg)

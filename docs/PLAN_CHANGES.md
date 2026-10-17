# Plan Changes (these override the implementation plan)

## 1. GPU for encoder inference (new)

Hardware: NVIDIA GeForce RTX 3050 laptop GPU, 6 GB VRAM, driver 581.86 (CUDA 13.0).

`configs/experiment.yaml` gets two new keys:
- `device: auto` (use cuda if `torch.cuda.is_available()`, otherwise cpu)
- `batch_size: {minilm: 32, e5small: 32, bgem3: 8}`

Task 9: the dense retriever uses `cfg.device` instead of the fixed value `cpu`. Keep fp32 (no half precision), so results match a CPU run apart from tiny floating-point differences.
Out-of-memory handling: if CUDA runs out of memory while encoding, halve the batch size and retry, down to batch size 1. Call `torch.cuda.empty_cache()` between corpora.
Sort texts by length before encoding to save memory and time; restore the original order afterward.
Write the device name (`cuda` or `cpu`) and the GPU model into `outputs/emb/{retriever}/meta.json` next to the cache.
bge-m3 now runs on all 3 datasets (the SciFact + NFCorpus fallback is no longer needed unless the GPU run fails).
Tests: the default test run must still pass on CPU. Mark GPU tests `@pytest.mark.slow`.
Unchanged: PhonoNorm, noise, BM25 and normalization stay CPU-only. Do not add bigger corpora (TREC-COVID, Touché).

## 2. Task order (new)

Build the human-check sheet first, because the user's annotation is the critical path: Task 1 → Task 6 → Task 7 → Task 8 → Tasks 2–5 → Tasks 9–12.

## 3. Task 6 test fix

In `test_operator_changes_word`, use a fixed list of about 30 common Hindi words written in the test file, instead of reading the stopword and lexicon files. This removes the dependency on Tasks 2 and 4.

## 4. Task 7 generation order

Before the full generation run, pick 120 qids (40 per dataset, seed 42) plus about 20 spare qids, and generate those first. Export the annotation sheet from them (Task 8) before generating the rest. Check the Groq daily request and token limits first. ArguAna queries are whole arguments and use the most tokens, so generate SciFact and NFCorpus first and spread ArguAna over several days.

## 5. Task 10 transliteration

Skip IndicXlit. Use `indic_transliteration` (ITRANS → Devanagari) from the start, and record this choice in the README.

## 6. Task 1 loader fallback

If `BeIR/*` loading fails with the installed datasets version, stop and ask the user before switching to the official BEIR zip files or the `mteb/<name>` parquet versions.

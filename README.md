# Spelling, Not Switching: Phonetic Normalization for Romanized Hinglish Code-Switched Retrieval

Empirical investigation into romanized Hindi-English (Hinglish) code-switched query retrieval degradation across BEIR datasets, isolating phonetic spelling variations from cross-lingual switching, and evaluating PhonoNorm against transliteration, vocabulary expansion, and LLM baselines.

## Setup

1. **Python Environment**:
   Requires Python 3.11:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

2. **PyTorch Installation**:
   - Dense encoders run on GPU inference (NVIDIA RTX 3050, 6 GB VRAM, CUDA 13.0 driver):
     ```powershell
     python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
     ```
   - All text manipulation, PhonoNorm, BM25, and noise operators run on CPU.

3. **Install Dependencies**:
   ```powershell
   python -m pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Set your Groq API key:
   ```powershell
   $env:GROQ_API_KEY = "your_groq_key"
   ```

## Design Choices & Transliteration

- **Transliteration**: We use `indic_transliteration` (ITRANS -> Devanagari) rather than IndicXlit for zero-shot script mapping without additional neural model download or latency overhead.
- **PhonoNorm**: Training-free phonetic normalization that projects Romanized Hindi variants to canonical keys.

## Implementation Details & Resource Refinements

- **`hindi_words` Pruning in Validation**:
  The LLM generation prompt outputs a JSON object with `"hinglish"` text and a list of `"hindi_words"`. In longer queries (such as full debate arguments in ArguAna), the model occasionally produces Hindi word lists containing minor spelling variants or hallucinated tokens not literally present in the rewritten text. To avoid false-positive validation failures while maintaining strict data quality, the validator prunes `hindi_words` to only those tokens that actually appear in the generated `hinglish` string before computing the mixing ratio and validating code-switching criteria.
- **Homographs Removed from `resources/hi_stopwords.txt`**:
  Common English words that could conflict with Romanized Hindi stopword variants (`she`, `mere`, `key`, `men`, `de`, `le`, `h`) were excluded from `hi_stopwords.txt`, and `"or"` was explicitly omitted from the variants of `"aur"`. This ensures pure English queries in the benchmark are never erroneously tagged as Hindi by the language tagger (verified 0 false positives across all 14,209 tokens in the BEIR query sets).
- **Additional Pronoun and Conjunction Forms in `resources/hi_stopwords.txt`**:
  Added inflected Hindi pronouns and conjunctions (`jabki`, `usme`, `isme`, `usse`, `isse`, `usko`, `isko`, `unka`, `inka`) to `hi_stopwords.txt` based on empirical analysis of frequent colloquial Hindi grammatical connectives in the generated queries.
- **Query Coverage & API Rate Limit Exclusions**:
  A small subset of queries (67 in ArguAna, 2 in SciFact, 1 in NFCorpus) were excluded from final evaluation strictly due to Groq daily token allowance limits (TPD) on the generation API, rather than quality or validation failures. Across all datasets, all evaluation conditions (English, noise variants, and code-switched queries) evaluate on the identical set of completed queries (`outputs/tables/generation_stats.json`).

## Reproduction Steps

1. **Data Download & Preparation**:
   ```powershell
   python scripts/01_download.py
   ```
2. **Run Tests**:
   ```powershell
   python -m pytest tests/ -v
   ```

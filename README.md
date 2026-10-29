# Switching vs. Spelling in Romanized Hinglish Information Retrieval

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![BEIR Benchmark](https://img.shields.io/badge/benchmark-BEIR-orange.svg)](https://github.com/beir-cellar/beir)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for the paper:
> **"Switching vs. Spelling in Romanized Hinglish Information Retrieval"**  
> **Authors:** Shahank Shekhar\*, Harshit Kumar\*, Sarthak Kumar\*  
> *\*Department of IoT and Intelligent Systems, Manipal University Jaipur, Jaipur, Rajasthan, India*

---

## Overview

Search queries that mix Hindi and English in Latin script (**romanized Hinglish**) face two potential sources of retrieval degradation:
1. **The language switch itself** (cross-lingual vocabulary mismatch and syntactic divergence), and
2. **The lack of standard orthography** for romanized Hindi words (e.g., *"nahi"*, *"nhi"*, *"nahin"*, *"nai"*).

Prior work on code-switched retrieval evaluated native scripts where orthography is standardized, leaving these two failure modes conflated. In this work, we introduce **Hinglish-CSR**, a controlled benchmark holding query content fixed while systematically varying language switching and phonetic spelling noise across **five BEIR datasets (1,166 usable queries across 100,785 documents)**.

We decompose retrieval loss into switching ($\Delta_{\text{sw}}$) and spelling ($\Delta_{\text{sp}}$) components across four representative retrievers (**BM25**, **MiniLM-L12-v2**, **multilingual-E5-small**, and **BGE-M3**), paired with 100,000-sample permutation tests and 10,000-sample bootstrap confidence intervals. We further evaluate **PhonoNorm**, a training-free query normalizer featuring an explicit ablation ladder (Stopword Removal $\rightarrow$ Exact Lexicon Lookup $\rightarrow$ Phonetic-Key Cascade).

```
                            ┌───────────────────────────────────┐
                            │    English BEIR Test Queries      │
                            │   (ArguAna, NFCorpus, SciFact,   │
                            │         FiQA, SCIDOCS)            │
                            └─────────────────┬─────────────────┘
                                              │ LLM Romanized Rewrite (Groq Qwen 2.5 / 3.8-27b)
                                              ▼
                            ┌───────────────────────────────────┐
                            │   Hinglish-CSR Benchmark (1,166)  │
                            ├───────────────────────────────────┤
                            │ • CS-clean  (0% spelling noise)   │
                            │ • CS-n25    (25% Hindi noise)     │
                            │ • CS-n50    (50% Hindi noise)     │
                            │ • EN-n50    (English typo control)│
                            │ • CS-human  (120 human-verified)  │
                            └─────────────────┬─────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     │                                                 │
                     ▼                                                 ▼
     ┌───────────────────────────────┐                 ┌───────────────────────────────┐
     │      Raw / Unnormalized       │                 │     PhonoNorm Normalizer      │
     │      (Direct Evaluation)      │                 │    (Ablation Ladder Stages)   │
     └───────────────┬───────────────┘                 └───────────────┬───────────────┘
                     │                                                 │
                     │   ┌─────────────────────────────────────────────┘
                     │   │ • Stage 1: +Stop (Drop Hindi function words)
                     │   │ • Stage 2: +Exact (Stopwords + exact MUSE gloss)
                     │   │ • Stage 3: +Key (Full consonant skeleton cascade)
                     │   │ • Stage 4: +Gloss Expansion (Token + English gloss)
                     │   │ • Baseline: Devanagari Transliteration
                     │   │ • Baseline: MiniLM-VExp (MUSE Vocab Expansion)
                     ▼   ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │             Information Retrieval Systems (Top-100)             │
     ├────────────────────────────────┬────────────────────────────────┤
     │       Lexical & English        │          Multilingual          │
     │  • BM25 (bm25s English stem)   │  • multilingual-E5-small       │
     │  • all-MiniLM-L12-v2           │  • BAAI/bge-m3                 │
     └────────────────────────────────┴────────────────────────────────┘
                                      │
                                      ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │       Statistical Analysis (ranx: nDCG@10 & Recall@100)         │
     │   • 100k-sample Fisher Randomization Test (Holm-corrected α=0.05)│
     │   • 10k-sample Paired Bootstrap CIs for (Δ_sp - Δ_sw)           │
     └─────────────────────────────────────────────────────────────────┘
```

---

## Key Contributions & Empirical Findings

### 1. Language Switching Dominates Spelling Noise
- Across lexical, English dense, and multilingual retrievers, **the language switch causes the vast majority of retrieval degradation** ($\Delta_{\text{sw}} = +0.023$ to $+0.092$ nDCG@10, all $p_{\text{holm}} < 10^{-5}$).
- In contrast, 50% phonetic spelling noise costs at most **$0.022$ nDCG@10** on average, and is **statistically insignificant** for BM25 ($\Delta_{\text{sp}} = -0.001$, $p = 1.00$) and MiniLM ($\Delta_{\text{sp}} = +0.002$, $p = 1.00$).
- For BM25, MiniLM, and E5-small, spelling cost represents at most **13%** of the switching cost.
- In paired bootstrap testing (10,000 resamples), the 95% confidence interval for $(\Delta_{\text{sp}} - \Delta_{\text{sw}})$ is **strictly negative for all four retrievers**, proving that switching dominates spelling across the board.

### 2. Multilingual Encoders Show a Distinct Trade-off
- **BGE-M3** is the sole model for which spelling noise clearly degrades performance ($\Delta_{\text{sp}} = +0.022$, $p < 10^{-5}$), representing **61%** of its switching cost ($\Delta_{\text{sw}} = +0.036$).
- On ArguAna, BGE-M3 suffers zero switching loss ($\Delta_{\text{sw}} = -0.011$, n.s.), but degrades consistently under spelling noise across all five datasets ($+0.003$ to $+0.047$).

### 3. Normalization Gains Are Driven by Stopword Removal
- An explicit ablation ladder reveals that query normalization improvements on English-only models (e.g., MiniLM $+0.033$ nDCG@10, $p < 10^{-5}$) are **almost entirely explained by Hindi stopword removal** ($+0.032$, $p < 10^{-5}$).
- The phonetic-key lookup cascade yields **no statistically significant improvement over simple stopword dropping** for any retriever ($\text{Key} - \text{Stop}$ differences range from $-0.004$ to $+0.007$, all Holm-adjusted $p \ge 0.87$).
- Exact lexicon lookup performs on par with the full phonetic key cascade (mean differences $\le 0.005$).

### 4. Normalization Degrades Multilingual Representations
- Every PhonoNorm mode degrades **BGE-M3** relative to unnormalized noisy queries ($\text{Stop Gain} = -0.017$, $p = 0.007$).
- Because multilingual models exploit Hindi function words as syntactic and contextual signals, dropping or altering them damages cross-lingual contextual representations.
- Devanagari transliteration is the only strategy that improves BGE-M3 ($0.386$ vs. $0.361$ at CS-n50), but it severely impairs E5-small.

---

## Hinglish-CSR Benchmark

Hinglish-CSR spans five diverse BEIR benchmarks covering scientific claims, medical literature, arguments, financial QA, and academic citations. Queries were converted into casual romanized Hinglish via LLM rewrites (`qwen/qwen3.8-27b` on Groq, prompt v3, $T=0.2$ / $0.7$), verified against strict linguistic constraints, and systematically corrupted using five reproducible phonetic operators.

### Benchmark Statistics (Table II)

| Dataset | Corpus Docs | Test Sampled | Usable Queries | Short Skipped | Failed Validation | API Limit | Domain |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **ArguAna** | 8,674 | 300 | **233** | 0 | 0 | 67 | Counter-arguments |
| **NFCorpus** | 3,633 | 323 | **133** | 172 | 17 | 1 | Medical IR |
| **SciFact** | 5,183 | 300 | **296** | 0 | 2 | 2 | Scientific fact checking |
| **FiQA** | 57,638 | 300 | **297** | 1 | 2 | 0 | Financial QA |
| **SCIDOCS** | 25,657 | 300 | **207** | 0 | 0 | 93 | Paper citation matching |
| **Total** | **100,785** | **1,523** | **1,166** | **173** | **21** | **163** | Heterogeneous BEIR |

### Controlled Noise Injection
1. **Hindi Spelling Noise (CS-n25, CS-n50):** Applied exclusively to LLM-marked Hindi tokens using five realistic informal variation operators:
   - *Vowel deletion* (interior vowel deletion: *karna* $\rightarrow$ *krna*)
   - *Vowel shortening* (*aa/ee/oo* $\rightarrow$ *a/i/u*)
   - *Final function-word variants* (*hai* $\rightarrow$ *h/hy/he*, *nahi* $\rightarrow$ *nahin*)
   - *Consonant alternation* (*v* $\leftrightarrow$ *w*, *j* $\leftrightarrow$ *z*, *f* $\leftrightarrow$ *ph*, *s* $\leftrightarrow$ *sh*)
   - *Consonant undoubling* (*achha* $\rightarrow$ *acha*)
2. **English Typo Control (EN-n50):** Character swaps, deletions, and duplications applied to English words ($\ge 4$ characters) to contrast native lexical fragility against phonetic Hindi noise.
3. **Human-Checked Split (CS-human):** 120 queries (40 per dataset across ArguAna, NFCorpus, SciFact) verified by a native speaker: 97.5% judged natural, 97.5% meaning-preserving, 95.8% both.

---

## Experimental Results

### 1. Main Retrieval Performance (Table III)
Mean nDCG@10 across all five BEIR datasets on unnormalized queries:

| Retriever | Architecture | English (`EN`) | Clean CS (`CS-clean`) | 25% Noise (`CS-n25`) | 50% Noise (`CS-n50`) | English Typo (`EN-n50`) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BM25** | Lexical (`bm25s`) | 0.382 | 0.359 | 0.361 | 0.360 | 0.274 |
| **MiniLM** | Dense (`all-MiniLM-L12-v2`) | **0.422** | 0.330 | 0.330 | 0.328 | 0.290 |
| **E5-small** | Dense (`multilingual-e5-small`) | 0.375 | 0.305 | 0.302 | 0.296 | 0.297 |
| **BGE-M3** | Dense (`BAAI/bge-m3`) | 0.419 | **0.383** | **0.373** | **0.361** | **0.368** |
| *MiniLM-VExp* | Vocab Expansion (MUSE) | — | 0.337 | 0.336 | 0.334 | — |

---

### 2. Switching vs. Spelling Cost Decomposition (Table IV)
Decomposition of mean nDCG@10 degradation, with Fisher permutation $p$-values (Holm-corrected) and 10,000 paired bootstrap 95% confidence intervals:

| Retriever | Switching Cost $\Delta_{\text{sw}}$ | $p_{\text{sw}}$ | Spelling Cost $\Delta_{\text{sp}}$ (50%) | $p_{\text{sp}}$ | Typo Drop $\Delta_{\text{typo}}$ | Bootstrap 95% CI $(\Delta_{\text{sp}} - \Delta_{\text{sw}})$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BM25** | +0.023 | $< 10^{-5}$ | **-0.001** | 1.000 | +0.107 | `[-0.035, -0.015]` |
| **MiniLM** | +0.092 | $< 10^{-5}$ | **+0.002** | 1.000 | +0.132 | `[-0.116, -0.084]` |
| **E5-small** | +0.070 | $< 10^{-5}$ | +0.009 | 0.032 | +0.078 | `[-0.082, -0.051]` |
| **BGE-M3** | +0.036 | $< 10^{-5}$ | **+0.022** | $< 10^{-5}$ | +0.052 | `[-0.034, -0.004]` |

> **Key takeaway:** The bootstrap interval for $(\Delta_{\text{sp}} - \Delta_{\text{sw}})$ excludes zero and is entirely negative for every model, confirming that language switching dominates phonetic spelling noise in romanized retrieval.

---

### 3. Per-Dataset Breakdown (Table V)
Switching cost $\Delta_{\text{sw}} = \text{EN} - \text{CS-clean}$ and spelling cost $\Delta_{\text{sp}} = \text{CS-clean} - \text{CS-n50}$ across each dataset:

| Dataset | BM25 $\Delta_{\text{sw}}$ | BM25 $\Delta_{\text{sp}}$ | MiniLM $\Delta_{\text{sw}}$ | MiniLM $\Delta_{\text{sp}}$ | E5-small $\Delta_{\text{sw}}$ | E5-small $\Delta_{\text{sp}}$ | BGE-M3 $\Delta_{\text{sw}}$ | BGE-M3 $\Delta_{\text{sp}}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ArguAna** | +0.045 | -0.007 | +0.102 | -0.008 | +0.067 | -0.010 | **-0.011** | +0.011 |
| **NFCorpus** | +0.017 | -0.001 | +0.043 | -0.001 | +0.059 | +0.014 | +0.017 | +0.021 |
| **SciFact** | +0.025 | -0.005 | +0.064 | +0.009 | +0.059 | +0.008 | +0.021 | +0.027 |
| **FiQA** | +0.024 | +0.006 | **+0.220** | +0.004 | **+0.150** | +0.033 | **+0.134** | +0.047 |
| **SCIDOCS** | +0.003 | +0.000 | +0.031 | +0.007 | +0.014 | +0.001 | +0.020 | +0.003 |
| **Mean** | **+0.023** | **-0.001** | **+0.092** | **+0.002** | **+0.070** | **+0.009** | **+0.036** | **+0.022** |

---

### 4. Normalization Ablation Ladder (MiniLM at CS-noise50)
Ablating each stage of PhonoNorm reveals where the recovery comes from:

| Stage | Query Transformation | MiniLM nDCG@10 | $\Delta$ vs. Unnormalized | $\Delta$ vs. Previous Stage | Significance |
|:---|:---|:---:|:---:|:---:|:---:|
| **None** | Unnormalized noisy query | 0.328 | — | — | Baseline |
| **+Stop** | Drop Hindi stopwords / function words | 0.354 | **+0.026** | **+0.026** | $p < 10^{-5}$ |
| **+Exact** | Stopwords + exact MUSE lexicon lookup | 0.355 | +0.027 | +0.001 | $p = 1.00$ |
| **+Key** | Full phonetic consonant-skeleton cascade | **0.357** | **+0.029** | +0.002 | $p = 1.00$ vs. Stop |
| *Native English* | Original English query | 0.422 | +0.094 | — | Upper Bound |

---

### 5. Human-Checked Evaluation Split (`CS-human`, 120 Queries)
Evaluated across 40 verified queries per dataset (ArguAna, NFCorpus, SciFact):

| Retriever | English (`EN`) | CS-human (Raw) | CS-human (`+Key`) | Relative Gap Recovered |
|:---|:---:|:---:|:---:|:---:|
| **BM25** | 0.489 | 0.489 | **0.499** | +0.010 gain |
| **MiniLM** | 0.526 | 0.454 | **0.506** | **~73% gap recovered** |
| **E5-small** | 0.461 | 0.454 | 0.444 | Degradation (-0.010) |
| **BGE-M3** | 0.527 | **0.550** | 0.516 | Degradation (-0.034) |
| *MiniLM-VExp* | — | 0.454 | — | 0.000 gain |

---

## PhonoNorm Normalizer & Diagnostics

### Algorithm Details
PhonoNorm operates purely at inference time with zero model fine-tuning:
1. **Transliterated Lexicon:** 29,704 bilingual entries from the MUSE `hi-en` dictionary mapped to informal Roman script via `indic-transliteration` (ITRANS) with casual orthography post-processing (12,522 unique phonetic keys, 10,761 coarse keys).
2. **Consonant Skeleton Reduction:** Merges common digraphs (`ph` $\rightarrow$ `f`, `sh` $\rightarrow$ `s`, `kh` $\rightarrow$ `k`, `th` $\rightarrow$ `t`, `dh` $\rightarrow$ `d`, `bh` $\rightarrow$ `b`), standardizes glides/sibilants (`w` $\rightarrow$ `v`, `z` $\rightarrow$ `j`, `q` $\rightarrow$ `k`), strips final nasals after vowels, collapses consecutive duplicate characters, and removes non-initial vowels (`nahi`, `nahin`, `nhi` $\rightarrow$ `nh`).
3. **Language & Ambiguity Tagger:** Identifies Hindi tokens using Zipf frequencies ($\ge 3.0$ reserved for English) and context gating for cross-lingual homographs (e.g., *to*, *me*, *hi* are tagged Hindi only adjacent to unambiguous Hindi function words).

### Why Does the Phonetic Key Add So Little?
1. **Syntactic Glue Phenomenon:** In casual Hinglish queries, **80–92% of Hindi tokens are grammatical function words** (*mein*, *ke*, *hai*, *bhi*). Dropping them eliminates out-of-vocabulary noise that English encoders cannot parse, which explains why stopword filtering captures nearly the entire gain.
2. **Lexicon Coverage & Collisions:** Exact lookup glosses only 2–12% of remaining Hindi tokens, whereas the phonetic key matches 97–100%. However, phonetic reduction causes severe semantic collisions (e.g., *kal* [yesterday/tomorrow] and *kaal* [era/time] collapse to `kl`; *sab* [all] and *saab* [sir] collapse to `sb`), neutralizing potential precision benefits.
3. **Escaped Function Words under Noise:** Stopword filtering relies on surface variant tables. When heavily noised, function words escape stopword detection (recognition drops from 88.4% to 74.0% on SciFact), causing the normalized pipeline to degrade under extreme noise.

---

## Repository Structure

```
.
├── configs/
│   └── experiment.yaml             # Master benchmark & model configuration
├── data/
│   ├── annotation/
│   │   └── check_sheet_done.csv    # 120-query human verification sheets & ratings
│   ├── cache/
│   │   └── groq_gen.jsonl          # Cached LLM query rewrites
│   ├── lexicon/
│   │   └── roman_hi_en.json        # 3-tier phonetic hash dictionary (exact, key, coarse)
│   ├── queries/                    # Benchmark queries across all 5 BEIR datasets
│   │   ├── arguana.jsonl           # English, CS-clean, CS-n25, CS-n50, EN-n50, CS-human
│   │   ├── fiqa.jsonl
│   │   ├── nfcorpus.jsonl
│   │   ├── scidocs.jsonl
│   │   └── scifact.jsonl
│   └── raw/                        # Pre-cached BEIR corpora & qrels (.pkl)
├── docs/                           # Research specifications, writing plans, and prompts
├── outputs/
│   ├── emb/                        # Precomputed dense corpus embeddings
│   ├── figures/                    # Publication figures (PDF & high-res PNG)
│   │   ├── fig_decomposition.{pdf,png}   # Switching vs. spelling degradation bar chart
│   │   ├── fig_noise_curve.{pdf,png}      # nDCG@10 degradation curves over noise levels
│   │   ├── fig_methods_ladder.{pdf,png}   # PhonoNorm ablation ladder gains
│   │   └── fig_methodology_framework.{pdf,png}
│   ├── norm_stats.csv              # Token resolution & tagging statistics
│   ├── results.csv                 # Master grid evaluation output
│   ├── runs/                       # Individual ranx run files
│   └── tables/                     # LaTeX tables and statistical outputs
│       ├── bootstrap_spelling_vs_switching.csv # 10k paired bootstrap tests
│       ├── decomposition.csv       # Per-dataset & mean empirical decomposition
│       ├── human_results.csv       # 120-query human split evaluation results
│       ├── significance_v2.csv     # 100k permutation Fisher tests (Holm-corrected)
│       └── *.tex                   # Generated publication LaTeX tables
├── resources/
│   ├── ambiguous.txt               # Cross-lingual homograph collision tokens
│   └── hi_stopwords.txt            # Curated Romanized Hindi stopwords & variants
├── scripts/
│   ├── 01_download.py              # Download BEIR benchmark corpora
│   ├── 02_build_lexicon.py         # Build romanized MUSE phonetic hash dictionary
│   ├── 03_generate_queries.py      # LLM rewrite generation & validation pipeline
│   ├── 04_annotation.py            # Annotation sheet exporter & quality validator
│   ├── 05_run_grid.py              # Master evaluation grid runner (5 datasets)
│   ├── 06_analysis.py              # Analysis, table generation, and decompositions
│   ├── 07_figures.py               # Generates camera-ready publication figures
│   ├── 08_human_eval.py            # Evaluates retrievers on 120 human queries
│   └── run_significance_v2.py      # 100k Fisher permutation & bootstrap CI tests
└── src/
    └── hcsir/                      # Core research package
        ├── annotation.py           # Verification and sampling utilities
        ├── config.py               # Benchmark configuration parser
        ├── data.py                 # BEIR data loaders and cache managers
        ├── generate.py             # Groq LLM client & validation logic
        ├── lexicon.py              # 3-tier phonetic lookup & transliteration
        ├── noise.py                # 5 reproducible phonetic noise operators
        ├── normalize.py            # PhonoNorm pipeline & ablation modes
        ├── phonetic.py             # Consonant-skeleton transformations
        ├── retrievers.py           # BM25, SentenceTransformers, and BGE-M3 wrappers
        ├── run_experiments.py      # Grid execution logic
        └── tagger.py               # Rule-based language tagger & Zipf filtering
```

---

## Setup & Reproduction

### 1. Environment Setup
Requires Python 3.11+:
```powershell
# Create and activate virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install PyTorch with CUDA support
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install project dependencies
python -m pip install -r requirements.txt
```

### 2. Lexicon & Benchmark Preparation
Build the 3-tier romanized MUSE phonetic dictionary:
```powershell
python scripts/02_build_lexicon.py
```
*(Query generation is precomputed in `data/queries/` with all validation checks passed).*

### 3. Run Master Evaluation Grid (All 5 Datasets)
Evaluates BM25, MiniLM, E5-small, and BGE-M3 across all noise levels and normalizer modes:
```powershell
python scripts/05_run_grid.py
```
Outputs master results to `outputs/results.csv` and individual runs to `outputs/runs/`.

### 4. Run Human Evaluation (120 CS-human Queries)
Evaluates retrievers on the verified human test split:
```powershell
python scripts/08_human_eval.py
```
Outputs: `outputs/tables/human_results.csv` and `outputs/tables/table_human.tex`.

### 5. Run Statistical Significance & Bootstrap Tests
Executes 100,000 paired Fisher randomization tests with step-down Holm correction and 10,000 paired bootstrap resamples:
```powershell
python scripts/run_significance_v2.py
```
Outputs: `outputs/tables/significance_v2.csv` and `outputs/tables/bootstrap_spelling_vs_switching.csv`.

### 6. Generate Tables & Publication Figures
Generate LaTeX tables and camera-ready figures:
```powershell
# Generate LaTeX tables and decomposition summary
python scripts/06_analysis.py

# Generate publication figures (PDF & high-res PNG)
python scripts/07_figures.py
```
Outputs: `outputs/tables/` and `outputs/figures/`.

---

## Limitations

As detailed in Section VII of the paper:
1. **Synthetic Queries:** Queries are LLM-generated (`qwen3.8-27b`) from a single prompt; Hindi appears predominantly as grammatical function words around English content terms.
2. **Noise Operators:** The 5 synthetic spelling operators mirror common phonetic simplifications but were not fitted to empirical social-media error distributions.
3. **Coverage & Filtering:** Due to strict validation filters and length constraints, short queries (particularly in NFCorpus) and rate-limited queries were omitted, yielding non-uniform subsets.
4. **Scope:** Evaluated specifically on romanized Hindi–English code-switching over English BEIR document corpora.

---

## Citation

```bibtex
@article{shekhar2026switching,
  title={Switching vs. Spelling in Romanized Hinglish Information Retrieval},
  author={ Shahank Shekhar, Harshit Kumar and Sarthak Kumar},
  journal={Department of IoT and Intelligent Systems, Manipal University Jaipur},
  year={2026}
}
```

---

## License & Data Ethics
This benchmark is built on public BEIR evaluation collections under their respective research licenses. Human annotations conform to established ethical guidelines for language evaluation. Code is released under the [MIT License](LICENSE).

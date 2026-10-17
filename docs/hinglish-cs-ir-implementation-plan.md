# Hinglish Code-Switched IR — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Note on format:** the user asked for planning documents only, so this plan has **no implementation code**. Each step gives exact behavior, interfaces, test cases (input → expected), and commands. The implementer writes the code when execution starts.

**Spec:** `C:\Users\Harshit Kumar\Downloads\NLP CWS\.plans\hinglish-cs-ir-paper-spec.md`
**Companion docs:** `hinglish-cs-ir-paper-writing-plan.md`, `hinglish-cs-ir-annotation-guide.md` (same folder)

**Goal:** Build a CPU-only experiment pipeline that does three things for romanized Hinglish queries over 3 BEIR datasets: (1) measures how much retrieval drops on them, (2) splits that drop into a switching part and a spelling part, and (3) tests a training-free phonetic normalizer (PhonoNorm) against prior-work baselines.

**Architecture:** A small Python package `hcsir` with one module per responsibility: data, tagging, phonetic keys, lexicon, normalizer, noise, generation, retrievers, baselines, experiment runner, and analysis. Each step's outputs are cached to disk (JSONL/NPY/ranx runs), so any stage can be rerun alone. The experiment runner writes one tidy `results.csv`, and all tables and figures are generated from it.

**Tech Stack:** Python 3.11, `datasets` (HF BEIR loaders), `sentence-transformers` + CPU `torch`, `bm25s` + `PyStemmer`, `ranx` (metrics + significance tests), `groq` SDK, `indic-transliteration`, `wordfreq`, `pandas`, `numpy`, `matplotlib`, `pyyaml`, `pytest`.

## Global Constraints
- CPU only. No model training or fine-tuning. Dense encoders are used for inference only.
- LLM = Groq free tier only, through env var `GROQ_API_KEY`. The key never goes in files, logs, or git.
- Every Groq call is cached on disk (JSONL keyed by query id + prompt version). Reruns must never re-bill or re-hit rate limits for completed items.
- All randomness is seeded from `configs/experiment.yaml: seed: 42`.
- Hinglish = **romanized only**. Any Devanagari character in a generated query counts as a validation failure.
- Noise is injected using the **LLM-reported Hindi word list**, never with our own tagger. This avoids a circular advantage for PhonoNorm.
- Datasets: BEIR SciFact (test, 300 q), NFCorpus (test, 323 q), ArguAna (test, subsample 300 q, seed 42).
- ArguAna: drop any retrieved doc whose id equals the query id, following BEIR's `ignore_identical_ids`.
- Metrics: nDCG@10 (primary) and Recall@100, from `ranx`.
- Project root: `C:\Users\Harshit Kumar\Downloads\NLP CWS\hinglish-cs-ir\` (new git repo; the folder name has a space, so always quote paths). Large data/outputs are git-ignored.
- The paper is written in English with the IEEEtran conference template (see the paper-writing plan).

---

## File Structure

```
hinglish-cs-ir/
  README.md                      how to reproduce, in order
  requirements.txt               pinned deps (CPU torch note in README)
  .gitignore                     data/raw, outputs/emb, outputs/runs, .env
  configs/experiment.yaml        seed, datasets, models, groq models, rpm, noise levels
  resources/
    hi_stopwords.txt             curated Hinglish function words + spelling variants (1 group per line)
    ambiguous.txt                tokens valid in both EN and HI (main, to, me, hi, is, us, par, so, he, in)
  src/hcsir/
    config.py                    load yaml -> Config dataclass
    data.py                      BEIR loading, query subsampling          (Task 1)
    tagger.py                    token language tagging EN/HI/OTHER       (Task 2)
    phonetic.py                  phonetic_key, coarse_key                 (Task 3)
    lexicon.py                   romanized hi->en lexicon                 (Task 4)
    normalize.py                 PhonoNorm modes                          (Task 5)
    noise.py                     HI spelling noise + EN typo control      (Task 6)
    generate.py                  Groq Hinglish generation + validation    (Task 7)
    queries.py                   assemble all query conditions per dataset(Task 7)
    annotation.py                export/import human-check sheet          (Task 8)
    retrievers.py                BM25 + dense w/ embedding cache          (Task 9)
    baselines/vocab_expansion.py base-paper §6 reimplementation (MiniLM)  (Task 10)
    baselines/translit.py        Roman HI -> Devanagari                   (Task 10)
    baselines/llm_translate.py   Groq translate-to-English ceiling        (Task 10)
    run_experiments.py           full grid -> outputs/results.csv         (Task 11)
    analysis.py                  decomposition, recovery, stats, emb shift(Task 12)
    figures.py                   paper figures + LaTeX tables             (Task 12)
  scripts/                       thin CLIs: 01_download.py ... 07_figures.py
  tests/                         one test file per module, tiny fixtures
  data/raw/                      BEIR + MUSE downloads (ignored)
  data/queries/{dataset}.jsonl   all query conditions (committed: this is a contribution)
  data/annotation/               human-check CSVs (committed)
  outputs/emb/ outputs/runs/     caches (ignored)
  outputs/results.csv            tidy results (committed)
  outputs/tables/ outputs/figures/
```

### Shared types (defined in Task 1, used everywhere)
- `Dataset` (dataclass): `name: str`, `corpus: dict[str, dict]` (doc_id → `{"title": str, "text": str}`), `queries: dict[str, str]` (qid → English text), `qrels: dict[str, dict[str, int]]`.
- `Run` = `dict[str, dict[str, float]]` (qid → doc_id → score, top-100). It converts directly to `ranx.Run`.
- Query record (one JSONL line in `data/queries/{dataset}.jsonl`):
  `{"qid", "dataset", "EN", "EN-noise25", "EN-noise50", "CS-clean", "CS-noise25", "CS-noise50", "hindi_words": [...], "noise_log": {...}, "gen_status": "ok"|"failed"}`
- Condition names (exact strings): `EN`, `EN-noise25`, `EN-noise50`, `CS-clean`, `CS-noise25`, `CS-noise50`, `CS-human`.
- Normalizer names (exact strings): `none`, `stop`, `exact`, `key`, `key_expand`, `translit`, `vocab_exp`, `llm`.

---

### Task 1: Project scaffold, config, and BEIR data loading

**Files:**
- Create: `requirements.txt`, `.gitignore`, `README.md`, `configs/experiment.yaml`, `src/hcsir/__init__.py`, `src/hcsir/config.py`, `src/hcsir/data.py`, `scripts/01_download.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `load_config(path="configs/experiment.yaml") -> Config`; `load_beir(name: str, n_queries: int | None, seed: int) -> Dataset`; the `Dataset` and `Run` types above.

**Config keys (exact):** `seed: 42`; `datasets: {scifact: null, nfcorpus: null, arguana: 300}` (value = query subsample size, null = all test queries); `retrievers: [bm25, minilm, e5small, bgem3]`; `models: {minilm: sentence-transformers/all-MiniLM-L12-v2, e5small: intfloat/multilingual-e5-small, bgem3: BAAI/bge-m3}`; `groq: {gen_model: <set after checking console.groq.com/docs/models — a 70B-class Llama>, translate_model: <an 8B-class instant model>, rpm: 20, max_retries: 3, prompt_version: v1}`; `noise_levels: [0.25, 0.5]`; `top_k: 100`.

- [ ] **Step 1: Environment.** Create a venv with Python 3.11. Install CPU torch first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`), then `pip install -r requirements.txt`. Record exact versions with `pip freeze > requirements.lock.txt`.
- [ ] **Step 2: Write failing tests** in `tests/test_data.py`:
  | Test | Input | Expected |
  |---|---|---|
  | `test_load_scifact_shapes` | `load_beir("scifact", None, 42)` | 300 queries; corpus size within 5,000–5,300; every qid in `queries` has ≥1 entry in `qrels` |
  | `test_subsample_deterministic` | `load_beir("arguana", 300, 42)` twice | identical qid lists, length 300 |
  | `test_subsample_respects_qrels` | arguana subsample | every sampled qid has qrels |
  | `test_config_loads` | `load_config()` | `seed == 42`, `datasets["arguana"] == 300` |
- [ ] **Step 3: Run tests, expect FAIL** (`python -m pytest tests/test_data.py -v` → ImportError / not implemented).
- [ ] **Step 4: Implement.** Load from the HF hub: `BeIR/{name}` (configs `corpus`, `queries`) and `BeIR/{name}-qrels` (split `test`). Keep only queries that appear in the test qrels. Subsample with `random.Random(seed).sample(sorted(qids), n)`. Cache the parsed Dataset to `data/raw/{name}.pkl`. `scripts/01_download.py` loads all 3 datasets and prints their counts.
- [ ] **Step 5: Run tests, expect PASS.** Run `python scripts/01_download.py`. It should print 3 lines with doc/query counts that match the spec table.
- [ ] **Step 6: Commit** (`git init` first): `feat: scaffold project and BEIR loaders`.

---

### Task 2: Hinglish stopword resource and token language tagger

**Files:**
- Create: `resources/hi_stopwords.txt`, `resources/ambiguous.txt`, `src/hcsir/tagger.py`
- Test: `tests/test_tagger.py`

**Interfaces:**
- Consumes: the lexicon membership check is optional at construction (`lexicon=None`), so this task does not depend on Task 4.
- Produces: `tokenize(text: str) -> list[str]` (lowercase, split on whitespace/punctuation, keep alphanumerics); `Tagger(lexicon: RomanLexicon | None = None)` with `.tag(tokens: list[str]) -> list[str]` returning `"EN" | "HI" | "OTHER"` per token; `load_stopword_groups() -> dict[str, str]` (variant → canonical form, e.g. `"nhi" → "nahi"`).

**Resource content rules:**
- `hi_stopwords.txt`: about 120–150 Hindi function words, one canonical form per line followed by its common romanized variants, comma-separated, for example `nahi,nahin,nhi,nai,nahee`, `hai,h,he,hy,hain`, `kya,kia,kyaa`, `ka,ki,ke`, `mein,me,main,mai`, `kitna,kitni,kitne`, `aur,or,aor`, `bhi,bi`, `kaise,kese,kaisa`, `kyun,kyon,kyu,kyoun`. Cover pronouns, postpositions, auxiliaries, question words, negation, and conjunctions.
- `ambiguous.txt`: `main, to, me, hi, is, us, par, so, he, in, an, the` (exactly these; any later additions must be logged in README).

**Tagging rules (applied in order):**
1. Digits-only or empty token → `OTHER`.
2. In the stopword variants **and not** in `ambiguous.txt` → `HI`.
3. In `ambiguous.txt` → decided in a second pass: `HI` if a neighbor at ±1 was tagged `HI` by rule 2 or 5, else `EN`.
4. `wordfreq.zipf_frequency(tok, "en") >= 3.0` → `EN`.
5. Lexicon given and `lexicon.has(tok)` (exact, key, or coarse match) → `HI`.
6. Otherwise → `EN`. Unknown tokens are left untouched downstream.

- [ ] **Step 1: Write the resource files** following the content rules above.
- [ ] **Step 2: Write failing tests:**
  | Test | Input tokens | Expected tags |
  |---|---|---|
  | `test_basic_mix` | `covid ka vaccine kitna safe hai` | `EN HI EN HI EN HI` |
  | `test_ambiguous_hi_context` | `main ghar ja raha hoon` (lexicon=None) | `main`→`HI` because `raha`/`hoon` are stopwords; `ghar`→`EN` without a lexicon (rule 6), which is accepted |
  | `test_ambiguous_en_context` | `what is the main cause of diabetes` | all `EN` |
  | `test_variants_map` | `load_stopword_groups()["nhi"]` | `"nahi"` |
  | `test_numbers` | `covid 19` | `EN OTHER` |
  | `test_tokenize` | `"Kya COVID-19 safe hai?"` | `["kya","covid","19","safe","hai"]` |
- [ ] **Step 3: Run, expect FAIL.** `python -m pytest tests/test_tagger.py -v`
- [ ] **Step 4: Implement** `tagger.py` as specified.
- [ ] **Step 5: Run, expect PASS.**
- [ ] **Step 6: Commit** `feat: Hinglish stopwords and token tagger`.

---

### Task 3: Phonetic key functions (the heart of the tweak)

**Files:**
- Create: `src/hcsir/phonetic.py`
- Test: `tests/test_phonetic.py`

**Interfaces:**
- Produces: `phonetic_key(word: str) -> str`; `coarse_key(word: str) -> str`.

**Algorithm (exact order):**
1. Lowercase, then keep only `a–z`.
2. Digraph and aspiration maps, applied left to right in this order: `chh→ch`, `ph→f`, `sh→s`, `kh→k`, `gh→g`, `th→t`, `dh→d`, `bh→b`, `jh→j`, `w→v`, `z→j`, `q→k`.
3. Final-nasal strip: if the length is ≥4 and the word ends in `n` or `m` preceded by a vowel, drop that final letter.
4. Collapse runs of the same letter (`cc→c`, `aa→a`).
5. Keep the first character. From the rest, delete the vowels `a e i o u`.
6. Collapse repeated letters again.
- `coarse_key` = `phonetic_key`, then also delete every non-initial `h` and `y`.

- [ ] **Step 1: Write failing tests:**
  | Input(s) | `phonetic_key` expected | Notes |
  |---|---|---|
  | `nahi`, `nahin`, `nhi` | all `nh` | nasal strip + vowel deletion |
  | `accha`, `acha`, `achha` | all `ach` | chh map + collapse |
  | `khana`, `khaana` | both `kn` | aspiration + length |
  | `zaroor`, `jaroor`, `zarur` | all `jr` | z→j, repeat collapse |
  | `theek`, `thik` | both `tk` | |
  | `phir`, `fir` | both `fr` | |
  | `hum` | `hm` | len 3: no nasal strip |
  | `""` | `""` | |
  | `Nahi!` | `nh` | case + punctuation |
  | coarse: `nai`, `nahi` | `coarse_key` both `n` | h-drop fallback |
  | coarse: `kya`, `kia` | `coarse_key` both `k` | y-drop fallback |
- [ ] **Step 2: Run, expect FAIL.** `python -m pytest tests/test_phonetic.py -v`
- [ ] **Step 3: Implement** exactly as specified. No other rules (YAGNI). Ambiguities are handled by lexicon tie-breaking.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat: Hinglish phonetic and coarse keys`.

---

### Task 4: Romanized Hindi→English lexicon

**Files:**
- Create: `src/hcsir/lexicon.py`, `scripts/02_build_lexicon.py`
- Test: `tests/test_lexicon.py`
- Output: `data/lexicon/roman_hi_en.json`, `outputs/tables/lexicon_stats.json`

**Interfaces:**
- Consumes: `phonetic_key`, `coarse_key` (Task 3).
- Produces: `casual_romanize(devanagari: str) -> str`; `build_lexicon(muse_path: Path) -> RomanLexicon`; class `RomanLexicon` with:
  - `.lookup(word: str) -> tuple[list[str], str]` returns `(ranked English glosses, match_level)`, where `match_level` is one of `"exact"`, `"key"`, `"coarse"`, `"none"`, tried in that order.
  - `.lookup_exact(word: str) -> list[str]`
  - `.has(word: str) -> bool`
  - `.canonical_forms() -> list[str]` returns the casual-romanized Hindi headwords (needed by Task 10 vocab expansion).
  - `.save(path)` / `RomanLexicon.load(path)`.
  - `RomanLexicon.from_pairs(pairs: list[tuple[str, str]])` builds a lexicon directly from (romanized Hindi, English) pairs. It is used by test fixtures in Tasks 2, 5, and 10.

**Behavior:**
- Source: the MUSE `hi-en.txt` dictionary (Conneau et al. 2018), downloaded from the MUSE GitHub dictionary links into `data/raw/muse/`. It is the same resource family as the base paper's baseline, so the comparison is fair.
- `casual_romanize`: run `indic_transliteration` Devanagari→ITRANS, then apply these maps: `A→aa, I→ee, U→oo, M→n, .N→n, R^i→ri, x→ksh, ~N→n, JN→gy`. Remove the remaining `.`, `^`, and `~` characters and lowercase the result. Exact schwa handling doesn't matter, because phonetic keys delete medial vowels.
- Indexing: build an exact-form map plus `key→Counter(gloss)` and `coarse→Counter(gloss)`.
- Gloss ranking: higher count first, then higher `wordfreq.zipf_frequency(gloss,"en")`, then alphabetical. Fully deterministic.
- Drop entries where the English side equals the romanized Hindi side (transliterated loanwords). Drop glosses that are English stopwords.

- [ ] **Step 1: Write failing tests:**
  | Test | Input | Expected |
  |---|---|---|
  | `test_casual_romanize_key_match` | `phonetic_key(casual_romanize("नहीं"))` | `== phonetic_key("nahi")` |
  | same for | `पानी`/`pani`, `खाना`/`khana`, `कमरा`/`kamra`, `घर`/`ghar` | keys equal |
  | `test_lookup_levels` | toy lexicon built from 3-line fixture `tests/fixtures/muse_toy.txt` (`पानी water`, `घर house`, `घर home`) | `lookup("paanee")` → (`["water"]`, `"exact"`); `lookup("paani")` → (`["water"]`, `"key"`); `lookup("xyzq")` → (`[]`, `"none"`) |
  | `test_from_pairs` | `RomanLexicon.from_pairs([("paani","water")])` | `lookup("pani")` → (`["water"]`, `"key"`) |
  | `test_gloss_ranking_deterministic` | build twice | identical JSON bytes |
  | `test_roundtrip` | save → load | equal lookups on 5 probe words |
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `lexicon.py` and the script. The script writes `lexicon_stats.json` with: entries, unique exact forms, unique keys, unique coarse keys, mean glosses per key, and the collision rate (keys with ≥2 distinct top-glosses). **These numbers go in the paper's method section.**
- [ ] **Step 4: Run, expect PASS.** Then run `python scripts/02_build_lexicon.py` and check that `lexicon_stats.json` exists and that entries > 20,000.
- [ ] **Step 5: Commit** `feat: romanized MUSE lexicon with phonetic indexing`.

---

### Task 5: PhonoNorm normalizer (with its ablation ladder)

**Files:**
- Create: `src/hcsir/normalize.py`
- Test: `tests/test_normalize.py`

**Interfaces:**
- Consumes: `tokenize`, `Tagger`, `load_stopword_groups` (Task 2); `RomanLexicon` (Task 4).
- Produces: `normalize(text: str, mode: str, lexicon: RomanLexicon, tagger: Tagger) -> NormResult`, where `NormResult` has `text: str` and `stats: dict` (counts of HI tokens, resolved_exact, resolved_key, resolved_coarse, unresolved, stopwords_dropped).

**Modes (exact strings, cumulative ladder):**
| mode | Behavior |
|---|---|
| `none` | return text unchanged |
| `stop` | drop HI-tagged tokens that are stopword variants; keep everything else as is |
| `exact` | `stop` + replace each remaining HI token by its top gloss if `lookup_exact` finds it |
| `key` | `stop` + replace each remaining HI token by its top gloss using the full cascade exact→key→coarse |
| `key_expand` | like `key`, but **keep** the original token and append the gloss right after it |

EN and OTHER tokens are never changed. The output is the tokens joined by a single space.

- [ ] **Step 1: Write failing tests** (fixture: `RomanLexicon.from_pairs([("paani","water"),("khana","food"),("bimari","disease")])`):
  | Input | Mode | Expected `text` |
  |---|---|---|
  | `kya paani se bimari hai` | `none` | unchanged |
  | same | `stop` | `paani bimari` (`kya`, `se`, `hai` must be in `hi_stopwords.txt`) |
  | `pani se bimari` | `exact` | `pani disease` (`pani` is not an exact form) |
  | `pani se bimari` | `key` | `water disease` |
  | `pani se bimari` | `key_expand` | `pani water bimari disease` |
  | `what causes diabetes` | `key` | unchanged (all EN) |
  | stats on `pani se bimari`, `key` | | `resolved_key==1`, `resolved_exact==1`, `stopwords_dropped==1` |
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat: PhonoNorm normalizer with ablation modes`.

---

### Task 6: Spelling-noise generator and English typo control

**Files:**
- Create: `src/hcsir/noise.py`
- Test: `tests/test_noise.py`

**Interfaces:**
- Produces:
  - `perturb_hi_word(word: str, rng: random.Random) -> tuple[str, str]` returns (new word, operator name).
  - `add_hi_noise(text: str, hindi_words: list[str], levels: list[float], seed: int) -> dict[float, tuple[str, list[dict]]]` returns, per level, the noised text and an edit log.
  - `add_en_noise(text: str, levels: list[float], seed: int) -> dict[float, tuple[str, list[dict]]]`.

**HI operators** (pick uniformly among those applicable to the word; the output must differ from the input):
- `vowel_delete`: delete one non-initial, non-final vowel (`nahi→nhi`, `karna→krna`).
- `vowel_length`: shorten one of `aa/ee/oo` to `a/i/u`; if there is none, lengthen an interior `a→aa` or a final `i→ee`.
- `final_variant`: final `ai→e` (`hai→he`); else, if the word ends in a vowel, append `n` (`nahi→nahin`).
- `consonant_alt`: swap one occurrence of `v↔w`, `j↔z`, `f↔ph`, `s↔sh`.
- `gemination`: undouble a doubled consonant (`accha→acha`), or double one interior consonant.

**Selection is nested:** collect the eligible Hindi-token positions (tokens matching `hindi_words` that have ≥1 applicable operator). Shuffle them with the seed. Level 0.25 edits the first `round(0.25·n)` positions; level 0.5 edits the first `round(0.5·n)`, **reusing the identical edits** for the shared positions. Use one `random.Random(seed)` per call: shuffle once, then draw operators in shuffled order, so the 0.5 edits are a strict superset of the 0.25 edits. The caller (`build_query_file`, Task 7) passes a per-query seed `zlib.crc32(f"{cfg.seed}:{qid}".encode())`. This makes every query reproducible and independent of processing order.

**EN control:** eligible tokens = tokens of length ≥4. Operators: adjacent-character swap, character deletion, character duplication. The same nested scheme and the same levels apply.

- [ ] **Step 1: Write failing tests:**
  | Test | Check |
  |---|---|
  | `test_operator_changes_word` | for 200 random words from the stopword and lexicon files, `perturb_hi_word` output ≠ input, lowercase alpha only |
  | `test_only_hindi_tokens_touched` | `add_hi_noise("covid ka vaccine kitna safe hai", ["ka","kitna","hai"], [0.5], 42)` changes no EN token |
  | `test_nested` | every edited position at 0.25 is edited identically at 0.5 |
  | `test_deterministic` | same seed → identical output; different seed → different output on a 10-Hindi-word query |
  | `test_rate` | 10 Hindi words, level 0.5 → exactly 5 edits in the log |
  | `test_en_noise_only_long_tokens` | tokens of length <4 are never edited |
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat: controlled Hinglish spelling noise and EN typo control`.

---

### Task 7: Groq Hinglish query generation and assembly of all conditions

**Files:**
- Create: `src/hcsir/generate.py`, `src/hcsir/queries.py`, `scripts/03_generate_queries.py`, `prompts/cs_rewrite_v1.txt`
- Test: `tests/test_generate.py` (uses a fake client; no network)
- Output: `data/cache/groq_gen.jsonl`, `data/queries/{dataset}.jsonl`

**Interfaces:**
- Consumes: `Dataset` (Task 1), `add_hi_noise`, `add_en_noise` (Task 6).
- Produces: `generate_cs(qid: str, english: str, client, cfg) -> GenResult` (fields `hinglish: str`, `hindi_words: list[str]`, `status: "ok"|"failed"`, `attempts: int`); `validate(english: str, hinglish: str, hindi_words: list[str]) -> list[str]` (list of failure reasons, empty = valid); `build_query_file(dataset: Dataset, cfg) -> Path` (writes the query-record JSONL defined under Shared types).

**Prompt requirements** (`prompts/cs_rewrite_v1.txt`, adapted from base paper Appendix D):
- Rewrite the English search query the way an urban Indian user would type it in **romanized Hinglish** (Latin script only).
- Keep technical, medical, and scientific terms and named entities in English. Mix in Hindi grammar and function words, and Hindi content words where natural.
- Use the most common casual spelling for every Hindi word. Preserve the full meaning.
- Return JSON only: `{"hinglish": "...", "hindi_words": ["...", ...]}`, where `hindi_words` lists every Hindi word exactly as it appears in the output.

**Validation rules (any failure triggers a retry, up to `max_retries`, with temperature 0.7):**
- Contains a Devanagari character (U+0900–U+097F) → `devanagari`
- `hindi_words` is empty → `no_hindi`
- A word in `hindi_words` is not a token of `hinglish` → `hindi_word_missing`
- Hindi-token fraction < 0.2 → `too_little_mixing`
- Output identical to the English input (case-insensitive) → `unchanged`
- JSON parse failure → `bad_json`

After the retries run out, set `status="failed"`. Failed queries are excluded from the CS conditions and counted in the paper.

**Rate limiting and caching:** a token-bucket limiter at `cfg.groq.rpm`. Retry on HTTP 429/5xx with exponential backoff (1, 2, 4, 8 s, then up to 60 s). Cache key = `f"{dataset}:{qid}:{prompt_version}"`; skip any key already cached. Expected volume is about 923 generation calls in total. **Check the current Groq free-tier daily request limit for the chosen model before running; if it is below about 1,000/day, split the run across 2 days. The cache makes resuming safe.**

- [ ] **Step 1: Write failing tests** with a `FakeClient` that returns scripted responses:
  | Test | Fake response(s) | Expected |
  |---|---|---|
  | `test_valid_first_try` | valid JSON | status ok, attempts 1 |
  | `test_devanagari_retry` | first has Devanagari, second valid | status ok, attempts 2 |
  | `test_fail_after_retries` | always `bad_json` | status failed, attempts 3 |
  | `test_cache_skips_call` | run twice | fake client called once |
  | `test_validate_reasons` | `("q", "q", [])` | contains `unchanged` and `no_hindi` |
  | `test_build_query_file_fields` | tiny 2-query dataset + fake client | each JSONL line has all 6 condition keys + `hindi_words` + `noise_log` |
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `generate.py`, `queries.py`, and the prompt file.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Smoke run (needs the user's key):** `python scripts/03_generate_queries.py --dataset scifact --limit 5`. Check by eye that the 5 outputs are romanized, natural, and meaning-preserving. **Show them to the user before the full run.**
- [ ] **Step 6: Full run** for all 3 datasets. Record the failure count per dataset in `outputs/tables/generation_stats.json`.
- [ ] **Step 7: Commit** `feat: Groq Hinglish generation and query conditions` (commit `data/queries/*.jsonl`, not the cache).

---

### Task 8: Human-check export and import

**Files:**
- Create: `src/hcsir/annotation.py`, `scripts/04_annotation.py`
- Test: `tests/test_annotation.py`
- Output: `data/annotation/check_sheet.csv` (to the user), `data/annotation/check_sheet_done.csv` (from the user), `outputs/tables/annotation_stats.json`

**Interfaces:**
- Consumes: the query JSONL files (Task 7).
- Produces: `export_sheet(n_per_dataset: int = 40, seed: int = 42) -> Path`; `import_sheet(path: Path) -> dict` (stats); and it adds the `CS-human` field to the matching records in `data/queries/{dataset}.jsonl`.

**Sheet columns (exact):** `item_id, dataset, qid, english, hinglish, natural (Y/N), meaning_preserved (Y/N), rewrite (optional), notes`. Rows are shuffled across datasets with the seed, and only `gen_status=="ok"` queries are included.

**Import rules:**
- `CS-human` = `rewrite` if it is non-empty.
- Else `hinglish`, if `natural==Y` and `meaning_preserved==Y`.
- Else the item is excluded from the CS-human subset.
- Stats: % natural, % meaning-preserved, % both, % rewritten, and the count of the usable CS-human subset.

- [ ] **Step 1: Write failing tests:** export gives 120 rows, 40 per dataset, deterministic across runs; import on a fixture filled sheet produces the correct percentages and writes `CS-human` only for qualifying rows; Y/N parsing accepts `y/Y/yes/n/N/no` and rejects anything else with a clear error naming the row.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5:** Run `python scripts/04_annotation.py export` and hand the CSV to the user with `hinglish-cs-ir-annotation-guide.md`. **The import waits until the user returns the sheet; Tasks 9–11 continue in parallel.**
- [ ] **Step 6: Commit** `feat: human-check sheet export/import`.

---

### Task 9: Retrievers (BM25 + dense with embedding cache)

**Files:**
- Create: `src/hcsir/retrievers.py`
- Test: `tests/test_retrievers.py` (tiny 4-doc corpus; dense tests marked `@pytest.mark.slow`)

**Interfaces:**
- Consumes: `Dataset`, `Run`.
- Produces: `get_retriever(name: str, cfg) -> Retriever`, where `Retriever` has `.index(ds: Dataset) -> None` and `.search(queries: dict[str, str], k: int = 100) -> Run`. For dense retrievers only: `.encode_queries(texts: list[str]) -> np.ndarray` (needed by Task 12 embedding analysis).

**Behavior:**
- `bm25`: `bm25s` with the English `PyStemmer` stemmer and English stopwords. Document text = `title + " " + text`.
- Dense (`minilm`, `e5small`, `bgem3`): `sentence-transformers` on device `cpu`, `normalize_embeddings=True`, batch size 32, `max_seq_length=512`. e5 needs prefixes `"query: "` / `"passage: "`. MiniLM and bge-m3 use none. Scoring = exact dot product (corpora ≤9k docs, so no ANN index is needed).
- Corpus embedding cache: `outputs/emb/{retriever}/{dataset}.npy` plus `doc_ids.json`. If the cache exists, skip encoding. The model name is stored next to the cache and checked on load.
- ArguAna: remove a doc from a query's results if `doc_id == qid`.
- Expected CPU time: minutes for MiniLM and e5-small. bge-m3 may take 1–3 h for all 3 corpora, so run it overnight in the background. If it is too slow, restrict bge-m3 to SciFact + NFCorpus and note this as a limitation (spec §8).

- [ ] **Step 1: Write failing tests:**
  | Test | Check |
  |---|---|
  | `test_bm25_ranks_exact_match_first` | the toy query "vitamin d deficiency" ranks the doc containing it first |
  | `test_run_shape` | Run has ≤k docs per query, float scores, descending order |
  | `test_identical_id_removed` | arguana-style toy where a doc id equals the qid → that id is absent |
  | `test_dense_cache_reused` (slow) | the second `.index()` call does not re-encode (spy on the encode call count) |
  | `test_e5_prefixes` (slow) | the encoder receives prefixed strings |
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (`python -m pytest tests/test_retrievers.py -v -m "not slow"`, then once with slow tests).
- [ ] **Step 5: Sanity check against published numbers:** EN-condition nDCG@10 for BM25 on SciFact and NFCorpus should land within ±3 points of the published BEIR BM25 numbers (SciFact about 0.66–0.69, NFCorpus about 0.32–0.33; `bm25s` differs slightly from Pyserini). If it's off by more, debug before moving on.
- [ ] **Step 6: Commit** `feat: BM25 and dense retrievers with caching`.

---

### Task 10: Baselines — vocab expansion (base paper), transliteration, LLM ceiling

**Files:**
- Create: `src/hcsir/baselines/__init__.py`, `vocab_expansion.py`, `translit.py`, `llm_translate.py`, `prompts/translate_v1.txt`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `RomanLexicon.canonical_forms()` and `.lookup_exact` (Task 4); the `Retriever` interface (Task 9); `Tagger` (Task 2); the Groq client and limiter from `generate.py` (Task 7).
- Produces:
  - `build_vocab_expanded_minilm(lexicon, cfg) -> Retriever`, registered as retriever `minilm_vocabexp`.
  - `translit_query(text: str, tagger: Tagger) -> str`.
  - `llm_translate(qid: str, text: str, condition: str, client, cfg) -> tuple[str, float]` returns (English text, latency in seconds). It is cached like Task 7 with key `f"{dataset}:{qid}:{condition}:translate_v1"`.

**Vocab expansion (mirrors base paper §6.1, Eqs. 1–2):**
- For each canonical romanized Hindi form `w_t` (length ≥3) that the MiniLM tokenizer does not already treat as a single token, add it with `AddedToken(w_t, single_word=True)`. Without `single_word`, a token like `pani` would match inside `company`.
- Initialize each new embedding row as the mean, over the word's English glosses, of the mean of each gloss's subword embeddings.
- Resize the embeddings and re-encode the corpus into a separate cache dir `outputs/emb/minilm_vocabexp/`.
- No training.

**Transliteration baseline:** HI-tagged tokens become Devanagari. Try `ai4bharat-transliteration` (IndicXlit) first. If it doesn't install on Windows within 30 minutes of trying, fall back to `indic_transliteration` ITRANS→Devanagari and record which one was used in README and the paper. Evaluate it only with multilingual retrievers (`e5small`, `bgem3`); BM25 and MiniLM cannot match Devanagari against an English corpus.

**LLM ceiling:** the Groq `translate_model` with the prompt "Translate this search query into English. Output only the translation." Run it only on conditions `CS-clean` and `CS-noise50`, about 1,846 calls. Record latency per query, and report the mean latency and the call count in the paper as the cost.

- [ ] **Step 1: Write failing tests:**
  | Test | Check |
  |---|---|
  | `test_added_token_single_word` | after expansion, `tokenizer.tokenize("company")` is unchanged; `tokenize("pani")` is one token |
  | `test_new_embedding_is_gloss_mean` | for a fixture word with gloss `water`, the new row equals the mean subword embedding of `water` (allclose) |
  | `test_translit_only_hi_tokens` | EN tokens are unchanged; HI tokens contain Devanagari |
  | `test_llm_translate_cached` | fake client called once over two runs; latency recorded |
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat: vocab-expansion, transliteration and LLM-ceiling baselines`.

---

### Task 11: Experiment runner (full grid → results.csv)

**Files:**
- Create: `src/hcsir/run_experiments.py`, `scripts/05_run_grid.py`
- Test: `tests/test_runner.py` (tiny fake dataset + BM25 only)
- Output: `outputs/runs/{dataset}/{retriever}/{condition}__{normalizer}.json` (ranx runs), `outputs/results.csv`, `outputs/norm_stats.csv`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_grid(cfg, only: dict | None = None) -> pd.DataFrame`. `results.csv` has columns (exact): `dataset, retriever, condition, normalizer, ndcg@10, recall@100, n_queries`. `norm_stats.csv` holds the per-dataset/condition/mode aggregates of `NormResult.stats`.

**Grid (exact):**
| retrievers | conditions | normalizers |
|---|---|---|
| bm25, minilm, e5small, bgem3 | EN, EN-noise25, EN-noise50 | none, key (key on EN shows the normalizer does no harm) |
| bm25, minilm, e5small, bgem3 | CS-clean, CS-noise25, CS-noise50 | none, stop, exact, key, key_expand |
| minilm_vocabexp | CS-clean, CS-noise25, CS-noise50 | none |
| e5small, bgem3 | CS-clean, CS-noise25, CS-noise50 | translit |
| bm25, minilm, e5small, bgem3 | CS-clean, CS-noise50 | llm |
| all of the above | CS-human (on its 120-query subset) + EN on the **same subset** | none, key |

**Rules:**
- All conditions for a dataset use the **same query set**: the qids with `gen_status=="ok"`. EN is evaluated on that same set, so comparisons are paired.
- Skip a cell whose run file already exists, unless `--force` is given.
- Log the wall-clock time per cell.

- [ ] **Step 1: Write failing tests:** on a 3-query fake dataset with BM25, `run_grid` writes the expected run files and a results.csv with the exact columns; the second call skips existing cells; the EN and CS rows have equal `n_queries`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Full run:** `python scripts/05_run_grid.py`. Start with `--only retriever=bm25,minilm,e5small`, then run bgem3 in the background (`run_in_background`).
- [ ] **Step 6: Commit** `feat: experiment grid runner` (commit `results.csv` and `norm_stats.csv`).

---

### Task 12: Analysis, significance, embedding shift, figures and LaTeX tables

**Files:**
- Create: `src/hcsir/analysis.py`, `src/hcsir/figures.py`, `scripts/06_analysis.py`, `scripts/07_figures.py`
- Test: `tests/test_analysis.py`
- Output: `outputs/tables/*.tex`, `outputs/figures/*.pdf`, `outputs/tables/significance.csv`

**Interfaces:**
- Consumes: `results.csv`, run files, `norm_stats.csv`, and `Retriever.encode_queries`.
- Produces:
  - `decompose(df) -> pd.DataFrame` with columns `dataset, retriever, switch_drop, spell_drop_25, spell_drop_50, en_noise_drop_50`, where `switch_drop = EN − CS-clean`, `spell_drop_k = CS-clean − CS-noisek`, and `en_noise_drop_50 = EN − EN-noise50`.
  - `gap_recovery(df, condition) -> pd.DataFrame` = `(norm − none) / (EN − none)` in %.
  - `significance(baseline_run, system_runs, qrels) -> pd.DataFrame` via `ranx.compare` with the Fisher randomization test (`stat_test="fisher"`), α=0.05.
  - `embedding_shift(retriever, ds, queries) -> pd.DataFrame`: the mean cosine between each EN query embedding and its variant (CS-clean, CS-noise25, CS-noise50, and each of those after the `key` normalizer), plus the centroid distance, mirroring base paper Fig. 2.

**Paper artifacts (names fixed; the writing plan refers to them):**
- `table_main.tex`: nDCG@10. Rows = retrievers; column groups = datasets; within each group: EN | CS-clean | CS-n25 | CS-n50.
- `fig_decomposition.pdf`: one stacked bar per retriever (averaged over datasets) with segments switch_drop and spell_drop_50, plus a marker for en_noise_drop_50 as the generic-typo control.
- `table_methods.tex`: on CS-noise50 (and CS-clean), the ladder none/stop/exact/key/key_expand plus vocab_exp/translit/llm. Shows nDCG@10, gap recovery %, and a † for significance vs `none`.
- `fig_noise_curve.pdf`: nDCG@10 vs noise level (0/25/50) per retriever, solid line = none, dashed = key.
- `table_coverage.tex`: % of HI tokens resolved by exact / key / coarse / unresolved at each noise level (from `norm_stats.csv`). This is the most direct evidence that the phonetic keys are what help.
- `fig_embshift.pdf`: mean EN↔variant cosine per condition, before and after `key`, for minilm, e5small, and bgem3.
- `table_human.tex`: EN vs CS-human vs CS-human+key on the 120-query verified subset.
- Figure style: single-column IEEE width (3.5 in), 8-pt fonts, colorblind-safe palette, vector PDF. Follow the `scientific-visualization` / `dataviz` skill guidance when implementing.

- [ ] **Step 1: Write failing tests** on a synthetic results DataFrame with known numbers:
  - `decompose` with EN=0.60, CS-clean=0.50, CS-noise50=0.40 gives switch 0.10 and spell_50 0.10.
  - `gap_recovery` with none=0.40, key=0.50, EN=0.60 gives 50.0.
  - `significance` on two identical runs gives p ≈ 1 and not significant.
  - Every `.tex` table compiles inside a minimal IEEEtran doc (run `pdflatex` in the test only if it is on PATH; otherwise skip).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.** Then run `python scripts/06_analysis.py` and `python scripts/07_figures.py`.
- [ ] **Step 5: Results review gate (with the user):** read the numbers before writing anything. Check them against the contingency branches in the paper-writing plan §5.
- [ ] **Step 6: Commit** `feat: analysis, significance tests, paper figures and tables`.

---

## Schedule (maps to spec §14)
| Dates (2026) | Tasks | Blocking inputs from the user |
|---|---|---|
| Wed Oct 14 – Sun Oct 18 | 1, 2, 3, 4, 5, 6 | none |
| Fri Oct 16 – Mon Oct 19 | 7 (smoke run → user OK → full run) | `GROQ_API_KEY`; approve the 5 smoke samples |
| Mon Oct 19 – Wed Oct 21 | 8 export; user annotates ~120 items (2–3 h) | the filled sheet back by **Wed Oct 21** |
| Sun Oct 18 – Fri Oct 23 | 9, 10, 11 (bge-m3 runs overnight) | none |
| Sat Oct 24 – Mon Oct 26 | 12 + results review gate | 30-min review with the user |
| Tue Oct 27 – Tue Nov 3 | paper writing (see writing plan) | venue name, page limit, authors |
| Wed Nov 4 | full draft ready | |

## Risks to watch during execution
- **Groq model names and free-tier limits change.** Check them at the start of Task 7 and record the exact model IDs and the date in README and in the paper's setup section.
- **Hinglish too clean.** If the LLM writes mostly English (failure `too_little_mixing` is common), tighten the prompt to v2 and bump `prompt_version`, so the cache doesn't mix the two prompt versions.
- **Phonetic key collisions** (e.g. `kal` = yesterday/tomorrow, `kaal` = time). Expected; collect examples for the error analysis instead of adding rules.
- **bge-m3 too slow on CPU.** Restrict it to 2 datasets (decided in Task 9 Step 5).

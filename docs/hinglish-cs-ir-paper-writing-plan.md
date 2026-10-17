# Hinglish Code-Switched IR — Paper Writing Plan

**Spec:** `hinglish-cs-ir-paper-spec.md` · **Experiments:** `hinglish-cs-ir-implementation-plan.md` · same folder
**Writing window:** Tue Oct 27 – Tue Nov 3, 2026 · **Full draft:** Wed Nov 4, 2026
**Language:** English · **Template:** IEEEtran conference (`\documentclass[conference]{IEEEtran}`), two-column

## 1. Format and page budget (6 pages incl. references; adjust once the venue is known)
| Section | Pages | ≈ Words | Main artifacts |
|---|---|---|---|
| Abstract | — | 150–180 | — |
| I. Introduction | 0.75 | 550 | — |
| II. Related Work | 0.5 | 400 | — |
| III. Hinglish-CSR: Query Construction | 0.75 | 450 | Table: noise operators with examples; generation and human-check stats |
| IV. PhonoNorm | 0.6 | 400 | Algorithm box; `lexicon_stats.json` numbers |
| V. Experimental Setup | 0.4 | 300 | — |
| VI. Results | 1.6 | 700 | `table_main.tex`, `fig_decomposition.pdf`, `table_methods.tex`, `table_coverage.tex` (+ `fig_noise_curve.pdf` if space allows) |
| VII. Analysis & Error Cases | 0.4 | 300 | `fig_embshift.pdf`, 4–6 failure examples |
| VIII. Limitations, Ethics | 0.2 | 150 | — |
| IX. Conclusion | 0.15 | 100 | — |
| References | 0.65 | — | ~20–25 refs |

If space is tight, cut in this order: `fig_noise_curve` → `table_human` (move it into the text as one sentence) → Related Work length → the Analysis section. Never cut `fig_decomposition` or `table_methods`.

## 2. Title and claims
**Working title:** *Spelling, Not Switching: Disentangling Romanization Noise in Hinglish Code-Mixed Information Retrieval*
**Alternative (use under contingency C, §5):** *Switching, Not Spelling: ...*

**Claim map:** every claim in the paper must point to an artifact.
| Claim | Evidence artifact |
|---|---|
| C1: Retrievers drop on romanized Hinglish queries | `table_main.tex` (EN vs CS-clean) |
| C2: A large share of the drop is spelling, not switching | `fig_decomposition.pdf`; the EN-noise control shows the effect is specific to romanization |
| C3: The phonetic key resolves Hindi tokens that exact lookup misses, more so as noise grows | `table_coverage.tex` |
| C4: PhonoNorm recovers X% of the gap, beats the vocab-expansion baseline, and costs ~0 compared with the LLM ceiling | `table_methods.tex` + latency numbers |
| C5: Findings hold on human-verified queries | `table_human.tex` |
| C6: Normalization moves CS query embeddings back toward the EN ones | `fig_embshift.pdf` |

## 3. Section-by-section content
**Abstract (write last).** Use 5 sentences: (1) the problem: code-switched queries break retrievers; (2) the gap: prior work studies only native scripts; (3) what we do: build Hinglish-CSR with controlled spelling noise; (4) the findings, with 2 numbers: the share of the drop due to spelling, and the gap recovered by PhonoNorm; (5) the release statement.

**I. Introduction.**
- Para 1: Code-switching is common, and mixed-language queries show up in search logs (Gupta et al. 2014: Bing logs). Hinglish is typed in Roman script with no standard spelling.
- Para 2: Zeng et al. (2026) show retrievers fail on code-switched queries, and lexicon vocabulary expansion only partly fixes this. But their languages have a fixed orthography.
- Para 3: In romanized languages, two things change at once: the language and the spelling. No prior work separates them for neural retrieval.
- Para 4: Contributions, as a 3-bullet list copied from spec §5, with final numbers.
- Do not claim to be "the first" to do anything until the novelty check (§6, step 1) is done.

**II. Related Work.** Three short paragraphs, and every reference must come from §4:
- (a) Code-switching and cross-lingual IR: Zeng 2026, MiLQ, ContrastiveMix, Litschko 2023, the FIRE CMIR shared task.
- (b) Mixed-script and transliterated IR: Gupta 2014. **State the difference explicitly**: their work predates dense retrievers and has no controlled disentanglement.
- (c) Hinglish resources and normalization: HingBERT/L3Cube, PHINC, the 2026 Hinglish papers, and the Functionalizer (orthographic variation in tokenizers).

**III. Hinglish-CSR.** Cover: source datasets and why they were chosen (CPU-feasible corpora, overlap with the base paper's domains); the generation prompt, summarized, with the full prompt in the repo; the validation rules and the failure counts; the noise operators table (operator → example → frequency in our edits); the nested noise design; the EN typo control; the human check (acceptance rates, CS-human subset size). One sentence explains why noise uses the LLM-reported Hindi words, not our tagger: it avoids circularity.

**IV. PhonoNorm.** Cover: the tagger rules (brief); the phonetic and coarse key rules as a compact table; the lexicon construction from MUSE (same source as the base paper's baseline, so the comparison is fair); lexicon stats; the cascade lookup; the modes ladder (stop/exact/key/key_expand). Include an algorithm box of about 10 lines.

**V. Setup.** Cover: retrievers and exact model IDs; CPU hardware and runtime; metrics; the Fisher randomization test at α=0.05; the exact Groq model IDs and access dates; baselines (vocab expansion details, transliteration tool used, LLM ceiling only on CS-clean and CS-noise50).

**VI. Results.** In this order: C1 → C2 → C3 → C4 → C5. Each paragraph opens with the takeaway sentence, then gives the numbers. Report gap recovery as a percentage with the base number (e.g. "recovers 58% of the gap (0.41 → 0.52 nDCG@10)").

**VII. Analysis.** Cover: the embedding shift; 4–6 failure cases (phonetic collisions like kal/kaal; Hindi words missing from MUSE; English words wrongly tagged HI); a note on which retriever family benefits most and why.

**VIII. Limitations and ethics.** Cover: synthetic queries (mitigated by the human subset); one language pair; small corpora (CPU constraint) with TREC-COVID and Touché excluded; the MUSE lexicon's coverage of colloquial Hindi; the noise operators approximate real variation; licenses of the released data.

## 4. Reference list (must verify every entry before it goes into `references.bib`)
Use the `citation-management` skill to fetch BibTeX by DOI or arXiv ID. **Any entry that cannot be verified is dropped, never guessed.**
| Key (planned) | Work | Source to verify |
|---|---|---|
| zeng2026csir | Code-Switching IR: Benchmarks, Analysis, and the Limits of Current Retrievers | arXiv 2604.17632 |
| gupta2014mixed | Query expansion for mixed-script IR (SIGIR 2014) | ACM DL |
| conneau2018muse | Word translation without parallel data (MUSE) | ICLR 2018 / arXiv 1710.04087 |
| thakur2021beir | BEIR benchmark | NeurIPS 2021 Datasets & Benchmarks |
| muennighoff2023mteb | MTEB | EACL 2023 |
| robertson2009bm25 | BM25 and beyond | FnTIR 2009 |
| reimers2019sbert | Sentence-BERT | EMNLP 2019 |
| wang2024me5 | Multilingual E5 technical report | arXiv 2402.05672 |
| chen2024bgem3 | BGE M3-Embedding | arXiv 2402.03216 |
| kim2025milq | MiLQ: mixed-language query benchmark | as cited in Zeng 2026 |
| do2024contrastivemix | ContrastiveMix | as cited in Zeng 2026 |
| litschko2023cs | Boosting zero-shot CLIR with artificial code-switching | as cited in Zeng 2026 |
| chanda2025fire | Overview of the code-mixed IR shared task (FIRE) | as cited in Zeng 2026 |
| srivastava2020phinc | PHINC parallel Hinglish corpus | arXiv 2004.09447 |
| nayak2022hingbert | L3Cube-HingCorpus and HingBERT | arXiv 2204.08398 |
| makowski2026functionalizer | The Functionalizer | arXiv 2609.15991 |
| smallwinsbig2026 | Small Wins Big: sarcasm in code-mixed Hinglish | arXiv 2602.21933 |
| cssurvey2026 | Beyond Monolingual Assumptions: survey of code-switched NLP | ACL 2026 (2026.acl-long.386) |
| madhani2023aksharantar | Aksharantar / IndicXlit (only if IndicXlit is used) | arXiv 2205.03018 |
| bassani2022ranx | ranx | ECIR 2022 |
| smucker2007significance | Comparison of significance tests for IR evaluation | CIKM 2007 |
| bm25s2024 | bm25s library | arXiv 2407.03618 |

## 5. Contingency branches (decided at the results review gate, implementation plan Task 12 Step 5)
| Outcome | How to recognize it | Framing change |
|---|---|---|
| **A. Expected** | spell_drop_50 ≥ ~30% of the total drop for most retrievers; `key` significantly beats `none` and `exact` | Keep the title. Lead with C2 and support it with C4. |
| **B. Mixed** | `key` helps BM25/MiniLM but not e5/bge-m3 | Make retriever-family dependence the headline: "multilingual encoders absorb spelling noise; lexical and English-centric ones do not." Still publishable. |
| **C. Reversed** | Switching dominates; spelling drop is small | Flip the title to "Switching, Not Spelling." The decomposition is still the contribution. PhonoNorm becomes a negative or partial result, reported honestly. |
| **D. Ceiling gap** | The LLM ceiling is far above PhonoNorm | Frame it as a cost–quality trade-off: give latency and call counts, and position PhonoNorm as the zero-cost option. |
| **E. Generation broken** | >15% of generations fail, or the human check finds <70% natural | Pause and revise the prompt (v2) before any results are reported. |

## 6. Writing process (step by step)
1. **Final novelty check (Oct 27, ~1 h).** Search Semantic Scholar, the ACL Anthology, and arXiv for: "Hinglish retrieval", "romanized code-mixed dense retrieval", "transliteration variation retrieval", "code-mixed query normalization". Record the results in `paper/novelty_check.md`. Rephrase any claim a found paper overlaps.
2. **Set up `paper/`:** `main.tex`, `references.bib`, `figures/` (copy from `outputs/figures/`), `tables/` (`\input` from `outputs/tables/`, so the numbers always match the code).
3. **Draft order:** IV Method → V Setup → III Data → VI Results → VII Analysis → I Intro → II Related → VIII/IX → Abstract. Methods first, because they are fixed facts.
4. **Use the `scientific-writing` skill** for drafting, and follow `jmo-workflow:ste` style when explaining to the user.
5. **Self-review with the `peer-review` skill** as a mock reviewer. Fix every "major" item.
6. **Polish:** one pass for numbers (each one traced to `results.csv`), one for terminology consistency (Hinglish-CSR, PhonoNorm, CS-clean...), and one for length.

## 7. Pre-submission checklist
- [ ] Page count within the venue limit (incl./excl. references as the venue requires)
- [ ] Anonymized if the venue is double-blind (no names, no GitHub link; use "anonymized repository")
- [ ] All fonts embedded (`pdffonts main.pdf` shows "yes" in the emb column)
- [ ] Figures readable when printed in grayscale
- [ ] Every number in the text matches the generated tables
- [ ] Every reference verified (§4); no unverifiable entries
- [ ] Data release terms checked: the licenses of SciFact, NFCorpus, ArguAna, and MUSE, and the Groq/model terms for redistributing generated text
- [ ] Limitations section present
- [ ] Exact model IDs, Groq access dates, and the seed reported for reproducibility
- [ ] Venue name, authors, and affiliations filled in (currently TBD)

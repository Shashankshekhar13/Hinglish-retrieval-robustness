# Agent Rules — Hinglish CS-IR project

Read this file and PLAN_CHANGES.md at the start of every conversation. Where PLAN_CHANGES.md conflicts with the implementation plan, PLAN_CHANGES.md wins.

## Project

Hinglish code-switched retrieval paper ("Spelling, Not Switching"). The spec and plan are in this folder (docs/):

- hinglish-cs-ir-paper-spec.md
- hinglish-cs-ir-implementation-plan.md
- hinglish-cs-ir-paper-writing-plan.md (reference only)
- hinglish-cs-ir-annotation-guide.md (reference only)

## Rules
1. No training or fine-tuning of any model. Encoders are used for inference only. Inference may run on the laptop GPU (RTX 3050, 6 GB VRAM) when available. PhonoNorm, the noise operators and BM25 must run on CPU only.
2. One plan task per conversation. Do only the task named in the prompt. Do not skip ahead.
3. Follow the task's steps in order: write failing tests, run them (expect FAIL), implement, run them (expect PASS), commit.
4. Use the project virtual environment (.venv). The project path contains a space (D:\NLP Research), so quote all paths and use python -m pip, never bare pip.
5. Secrets: never print, log, write to a file, or commit GROQ_API_KEY. To check it, print only True/False.
6. Tests must never call Groq and must never download large models. Use fakes. Mark slow tests with @pytest.mark.slow.
7. Ask before any command that downloads models or datasets, or that calls Groq.
8. Stop and report to the user at: Task 7 Step 5, Task 8 Step 5, Task 12 Step 5.
9. Ignore references to superpowers:*, citation-management, scientific-writing, jmo-workflow and other Claude Code skills. Just follow the steps.
10. Commit after each task using the commit message given in the plan.
11. Do not invent results, citations or numbers. If something fails or is unclear, stop and say so.
12. At the end of every task, give a short summary: what was done, what passed, what failed.

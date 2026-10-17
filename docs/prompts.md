# Prompt 1 — Environment setup (no task implementation)

Read AGENT_RULES.md, PLAN_CHANGES.md and hinglish-cs-ir-implementation-plan.md (all in the docs/ folder). Do NOT implement any plan task yet. Only set up the environment, in this order.

The workspace path is D:\NLP Research (it contains a space). Quote every path and use python -m pip.

Step 1: Folder layout

The project root is the workspace root. Make sure the 7 .md files (4 original + AGENT_RULES.md, PLAN_CHANGES.md, and the prompt files) are inside docs/; move them there if they are in the root.

Create the folder structure from the plan's "File Structure" section. Put a .gitkeep in each empty folder.

Create .gitignore with:

data/raw/
data/cache/
outputs/emb/
outputs/runs/
.env
.venv/
__pycache__/
*.pyc

Step 2: requirements

Create requirements.txt with one package per line (no torch line, torch is installed separately):

datasets
sentence-transformers
bm25s
PyStemmer
ranx
groq
indic-transliteration
wordfreq
pandas
numpy
matplotlib
pyyaml
pytest

Step 3: Git and virtual environment

Ask me before running commands that need permission.

git init
py -3.11 -m venv .venv
Activate it (.\.venv\Scripts\Activate.ps1), then python -m pip install --upgrade pip
Install CUDA PyTorch (my GPU is an NVIDIA RTX 3050, driver 581.86, CUDA 13.0). Try: python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 If that URL fails or the build is not found, tell me and look up the current Windows pip command on pytorch.org/get-started/locally. Pick a CUDA version that is not newer than 13.0. Do not guess a URL.
python -m pip install -r requirements.txt
python -m pip freeze > requirements.lock.txt

If PowerShell blocks activation, tell me and suggest Set-ExecutionPolicy -Scope CurrentUser RemoteSigned. Do not change system settings yourself.

Step 4: Verify

Run these checks:

python -c "import torch, sentence_transformers, bm25s, ranx, datasets; print('imports ok')"
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
python -c "import os; print('GROQ key set:', bool(os.environ.get('GROQ_API_KEY')))"

Print only True/False for the key, never its value. If cuda available is False, stop and tell me. Do not continue with a CPU-only torch without asking.

Step 5: Commit

Commit everything with the message: chore: project scaffold and environment

Stop

Stop after the commit. Report: what worked, what failed, and the Python, torch and datasets versions, plus whether CUDA is available.

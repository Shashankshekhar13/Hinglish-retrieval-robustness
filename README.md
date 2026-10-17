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

## Reproduction Steps

1. **Data Download & Preparation**:
   ```powershell
   python scripts/01_download.py
   ```
2. **Run Tests**:
   ```powershell
   python -m pytest tests/ -v
   ```

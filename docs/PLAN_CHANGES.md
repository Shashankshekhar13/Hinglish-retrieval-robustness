# Plan Changes — Hinglish CS-IR Project

Where this file conflicts with `hinglish-cs-ir-implementation-plan.md`, this file wins.

1. **Hardware & Acceleration:**
   - The original plan specified CPU-only execution across all components.
   - **Change:** Dense encoders for inference may run on the laptop GPU (NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB VRAM) when available.
   - PhonoNorm, noise operators, BM25, and all other text manipulation components must run on CPU only.
   - PyTorch is installed with CUDA support (e.g. cu128 / cu124 / cu126 compatible with CUDA 13.0 driver) rather than CPU-only wheels.
   - **New Config Keys in `configs/experiment.yaml`:**
     - `device: auto` (resolves to `"cuda"` if `torch.cuda.is_available()` else `"cpu"`)
     - `batch_sizes:` mapping per dense retriever for GPU inference (e.g. `minilm: 64`, `e5small: 64`, `bgem3: 16`)

2. **Workspace Location:**
   - Workspace root is `D:\NLP Research` (paths contain spaces, so always quote paths in commands and scripts).

3. **No Model Training:**
   - Strictly no training or fine-tuning of any model. All models are used for zero-shot / frozen inference only.

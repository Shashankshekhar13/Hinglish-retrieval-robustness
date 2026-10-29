import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Set publication style for single-column IEEE figures (width 3.5 inches)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 7.5,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def draw_card(ax, x, y, w, h, bg_color, border_color, border_w=1.0, radius=1.0, zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=bg_color, edgecolor=border_color,
                         linewidth=border_w, zorder=zorder)
    ax.add_patch(box)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color="#4A5568", lw=1.2, mutation=8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=mutation, shrinkA=1, shrinkB=1),
                zorder=5)

# ==============================================================================
# PART 1: Benchmark Construction & Controlled Noise Pipeline (fig_pipeline_data)
# ==============================================================================
def make_fig_pipeline_data(out_dir: Path):
    fig, ax = plt.subplots(figsize=(3.5, 4.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Colors
    C_BORDER = "#0072B2"
    C_BG = "#F4F8FC"
    BOX_WHITE = "#FFFFFF"
    TEXT_MAIN = "#1A202C"
    TEXT_MUTED = "#4A5568"

    # Outer container
    draw_card(ax, 2, 2, 96, 96, C_BG, C_BORDER, border_w=1.3, radius=1.5)
    ax.text(50, 94.5, "Hinglish-CSR Benchmark Pipeline", 
            fontsize=8.2, fontweight="bold", ha="center", color=C_BORDER)

    # Step 1: Original English Query
    draw_card(ax, 6, 78.5, 88, 12.5, BOX_WHITE, "#CBD5E0")
    ax.text(50, 87.2, "1. Clean English Query (EN)", fontsize=7.2, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(50, 83.0, "5 BEIR Benchmark Collections (1,166 Queries)", fontsize=6.2, ha="center", color=TEXT_MUTED)
    ax.text(50, 79.5, '"Why diversify investments?"', fontsize=6.0, style="italic", ha="center", color="#2B6CB0")

    draw_arrow(ax, 28, 78.5, 28, 73.0, color=C_BORDER)
    ax.text(32, 75.8, "Llama-3.3-70B Matrix-Language Rewrite", fontsize=5.6, ha="left", va="center", color=TEXT_MUTED)

    # Step 2: Clean CS Query
    draw_card(ax, 6, 56.5, 88, 16.5, BOX_WHITE, "#CBD5E0")
    ax.text(50, 69.5, "2. Clean Hinglish Query (CS-clean)", fontsize=7.2, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(50, 65.5, '"investments ko diversify kyun karte hain?"', fontsize=6.0, style="italic", ha="center", color="#2C5282")
    
    # Verification Badge
    draw_card(ax, 10, 57.8, 80, 6.2, "#EBF8FF", "#3182CE", radius=0.6)
    ax.text(50, 62.2, "120 Expert-Verified Queries", fontsize=5.6, fontweight="bold", ha="center", va="center", color="#2B6CB0")
    ax.text(50, 59.2, "(97.5% Naturalness, 97.5% Meaning-Preserving)", fontsize=5.0, ha="center", va="center", color="#2B6CB0")

    draw_arrow(ax, 28, 56.5, 28, 51.0, color=C_BORDER)
    ax.text(32, 53.8, "Targeted Romanization Noise (Hindi Only)", fontsize=5.6, ha="left", va="center", color="#9B2C2C")

    # Step 3: Noise Injection Operators
    draw_card(ax, 6, 17.5, 88, 33.0, BOX_WHITE, "#CBD5E0")
    ax.text(50, 47.0, "3. Controlled Spelling Perturbations", fontsize=7.2, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(50, 43.5, "Simulates Unstandardized Romanized Typing (0%, 25%, 50%)", fontsize=5.6, ha="center", color=TEXT_MUTED)

    ops = [
        ("Vowel Deletion", "nahi -> nhi, mujhe -> mjhe"),
        ("Vowel Length Swap", "kaam -> kam, theek -> thik"),
        ("Consonant Alternation", "v <-> w, z <-> j, ph <-> f"),
        ("Gemination / Typo", "kyun -> kkyun, hai -> h / hy")
    ]
    for idx, (op_name, op_ex) in enumerate(ops):
        oy = 37.5 - idx * 5.6
        draw_card(ax, 9, oy, 82, 4.8, "#FFF5F5", "#FEB2B2", radius=0.6)
        ax.text(12, oy + 2.4, op_name, fontsize=5.6, fontweight="bold", va="center", color="#9B2C2C")
        ax.text(88, oy + 2.4, op_ex, fontsize=5.4, style="italic", ha="right", va="center", color=TEXT_MUTED)

    # Step 4: Control condition
    draw_card(ax, 6, 4.0, 88, 10.5, "#EDF2F7", "#CBD5E0")
    ax.text(50, 11.2, "Control: Generic English Typos (EN-noise50)", fontsize=6.4, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(50, 6.8, "Isolates Romanization Noise vs. General Typo Sensitivity", fontsize=5.6, ha="center", color=TEXT_MUTED)

    plt.tight_layout()
    out_pdf = out_dir / "fig_pipeline_data.pdf"
    out_png = out_dir / "fig_pipeline_data.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_pdf} and {out_png}")

# ==============================================================================
# PART 2: PhonoNorm & Multi-Retriever Disentanglement (fig_pipeline_method)
# ==============================================================================
def make_fig_pipeline_method(out_dir: Path):
    fig, ax = plt.subplots(figsize=(3.5, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    C_NORM_BORDER = "#009E73"
    C_NORM_BG = "#F0FAF7"
    C_EVAL_BORDER = "#7B1FA2"
    C_EVAL_BG = "#F9F4FB"
    BOX_WHITE = "#FFFFFF"
    TEXT_MAIN = "#1A202C"
    TEXT_MUTED = "#4A5568"

    # Top Section: PhonoNorm Normalizer (y: 50.5 to 98)
    draw_card(ax, 2, 50.5, 96, 47.5, C_NORM_BG, C_NORM_BORDER, border_w=1.3, radius=1.5)
    ax.text(50, 94.8, "PhonoNorm (Query Normalizer, CPU-Only)", 
            fontsize=8.0, fontweight="bold", ha="center", color=C_NORM_BORDER)

    norm_steps = [
        ("1. Token Language ID", "English wordlist vs. Hindi Lexicon -> Marks Romanized HI tokens", "#E6FFFA", "#285E61"),
        ("2. Stopword Filter", "Curated romanized particles (hai, ka, ke, ko, mein...) -> Dropped", "#FEFCBF", "#744210"),
        ("3. Phonetic Skeleton", "Digraph collapse (ph->f) & nasal strip -> nahi/nhi unified to 'nh'", "#E6FFFA", "#234E52"),
        ("4. Bilingual Glossing", "Cascaded lookup in MUSE hi-en: Exact -> Key -> Coarse -> EN gloss", "#C6F6D5", "#22543D")
    ]
    for s_i, (s_title, s_desc, s_bg, s_fg) in enumerate(norm_steps):
        sy = 83.2 - s_i * 9.6
        draw_card(ax, 5, sy, 90, 8.4, BOX_WHITE, "#A7F3D0")
        ax.text(8, sy + 5.6, s_title, fontsize=6.6, fontweight="bold", color=TEXT_MAIN)
        ax.text(8, sy + 2.0, s_desc, fontsize=5.3, color=s_fg)

    # Arrow from Normalizer to Retrieval Grid
    draw_arrow(ax, 30, 50.5, 30, 45.0, color="#4A5568", lw=1.3)
    ax.text(34, 47.8, "Normalized Queries ->", fontsize=5.8, ha="left", va="center", color=TEXT_MUTED)

    # Bottom Section: Multi-Retriever & Disentanglement (y: 2 to 44)
    draw_card(ax, 2, 2.0, 96, 42.0, C_EVAL_BG, C_EVAL_BORDER, border_w=1.3, radius=1.5)
    ax.text(50, 40.8, "Retrieval Evaluation & Disentanglement", 
            fontsize=8.0, fontweight="bold", ha="center", color=C_EVAL_BORDER)

    # Model grid box
    draw_card(ax, 5, 23.5, 90, 15.5, BOX_WHITE, "#E9D8FD")
    ax.text(50, 35.8, "Multi-Retriever Execution Grid", fontsize=6.8, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(50, 31.8, "BM25 (Lexical)  *  all-MiniLM-L12-v2 (EN Dense)", fontsize=5.6, ha="center", color=TEXT_MUTED)
    ax.text(50, 28.2, "multilingual-e5-small  *  bge-m3 (568M Multilingual Dense)", fontsize=5.6, ha="center", color=TEXT_MUTED)
    ax.text(50, 24.8, "Over 5 Unaltered Benchmark Corpora (100k+ Documents)", fontsize=5.4, style="italic", ha="center", color="#6B46C1")

    # Decomposition box
    draw_card(ax, 5, 4.0, 90, 17.5, BOX_WHITE, "#E9D8FD")
    ax.text(50, 18.0, r"Empirical Decomposition (nDCG@10)", fontsize=6.8, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(50, 13.8, r"$\Delta_{\mathrm{switch}} = \mathrm{EN} - \mathrm{CS}_{\mathrm{clean}} \quad \vert \quad \Delta_{\mathrm{spell50}} = \mathrm{CS}_{\mathrm{clean}} - \mathrm{CS}_{n50}$", 
            fontsize=6.2, ha="center", color="#2B6CB0")
    ax.text(50, 8.5, "100,000-Permutation Fisher Tests (Holm-corrected α=0.05)\n10,000-Resample Paired Bootstrap 95% Confidence Intervals", 
            fontsize=5.4, ha="center", color=TEXT_MUTED)

    plt.tight_layout()
    out_pdf = out_dir / "fig_pipeline_method.pdf"
    out_png = out_dir / "fig_pipeline_method.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_pdf} and {out_png}")

# ==============================================================================
# UNIFIED VERTICAL DIAGRAM: All in one concise single-column figure (fig_methodology_vertical)
# ==============================================================================
def make_fig_methodology_vertical(out_dir: Path):
    fig, ax = plt.subplots(figsize=(3.5, 7.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    BOX_WHITE = "#FFFFFF"
    TEXT_MAIN = "#1A202C"
    TEXT_MUTED = "#4A5568"

    # Container 1: Data
    draw_card(ax, 2, 68.0, 96, 30.0, "#F4F8FC", "#0072B2", border_w=1.2, radius=1.2)
    ax.text(50, 95.0, "I. Hinglish-CSR Benchmark", fontsize=7.6, fontweight="bold", ha="center", color="#0072B2")
    
    draw_card(ax, 5, 84.5, 90, 8.2, BOX_WHITE, "#CBD5E0")
    ax.text(50, 89.8, "Clean EN Query: 5 BEIR Corpora (1,166 Qs)", fontsize=6.2, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(50, 86.2, '"Why diversify investments?"', fontsize=5.6, style="italic", ha="center", color="#2B6CB0")
    
    draw_arrow(ax, 50, 84.5, 50, 80.5, color="#0072B2", mutation=6)
    
    draw_card(ax, 5, 70.0, 90, 10.5, BOX_WHITE, "#CBD5E0")
    ax.text(50, 77.2, "Llama-3.3-70B Matrix-Language CS (CS-clean)", fontsize=6.2, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(50, 73.8, "+ Romanization Noise Operators (0%, 25%, 50% on Hindi)", fontsize=5.6, ha="center", color="#9B2C2C")
    ax.text(50, 70.8, "120 Expert-Verified Subset (97.5% Natural)", fontsize=5.2, style="italic", ha="center", color="#2B6CB0")

    draw_arrow(ax, 50, 68.0, 50, 64.0, color="#4A5568", mutation=7)
    ax.text(50, 66.0, "Code-Switched Queries", fontsize=5.4, ha="center", va="center", color=TEXT_MUTED)

    # Container 2: PhonoNorm
    draw_card(ax, 2, 38.0, 96, 26.0, "#F0FAF7", "#009E73", border_w=1.2, radius=1.2)
    ax.text(50, 61.2, "II. PhonoNorm (CPU-Only Normalizer)", fontsize=7.6, fontweight="bold", ha="center", color="#009E73")
    
    p_steps = [
        "1. Token Tagger: Marks Hindi tokens (English wordlist check)",
        "2. Stopword Filter: Drops romanized particles (hai, ka, ke, ko...)",
        "3. Phonetic Keying: Digraph collapse & consonant skeleton (nahi -> 'nh')",
        "4. Lexicon Cascade: MUSE hi-en lookup (Exact -> Key -> Coarse -> Gloss)"
    ]
    for pi, pt in enumerate(p_steps):
        py = 55.5 - pi * 5.0
        draw_card(ax, 5, py, 90, 4.2, BOX_WHITE, "#A7F3D0", radius=0.6)
        ax.text(7, py + 2.1, pt, fontsize=5.3, va="center", color=TEXT_MAIN)

    draw_arrow(ax, 50, 38.0, 50, 34.0, color="#4A5568", mutation=7)
    ax.text(50, 36.0, "Normalized Queries", fontsize=5.4, ha="center", va="center", color=TEXT_MUTED)

    # Container 3: Evaluation Grid
    draw_card(ax, 2, 17.5, 96, 16.5, "#FEF9F2", "#D55E00", border_w=1.2, radius=1.2)
    ax.text(50, 31.2, "III. Multi-Retriever Grid", fontsize=7.6, fontweight="bold", ha="center", color="#D55E00")
    
    draw_card(ax, 5, 19.5, 90, 9.5, BOX_WHITE, "#FBD38D")
    ax.text(50, 25.8, "BM25 * all-MiniLM-L12-v2 * m-E5-small * BGE-M3 * VocabExp", fontsize=5.6, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(50, 22.0, "Evaluated over 5 Unaltered BEIR Corpora (100k+ Documents)", fontsize=5.4, style="italic", ha="center", color=TEXT_MUTED)

    draw_arrow(ax, 50, 17.5, 50, 14.5, color="#4A5568", mutation=7)
    ax.text(50, 16.0, "Ranked Runs", fontsize=5.4, ha="center", va="center", color=TEXT_MUTED)

    # Container 4: Disentanglement
    draw_card(ax, 2, 2.0, 96, 12.5, "#F9F4FB", "#7B1FA2", border_w=1.2, radius=1.2)
    ax.text(50, 11.8, "IV. Empirical Disentanglement & Rigor", fontsize=7.4, fontweight="bold", ha="center", color="#7B1FA2")
    ax.text(50, 7.8, r"$\Delta_{\mathrm{switch}} = \mathrm{EN} - \mathrm{CS}_{\mathrm{clean}} \quad \vert \quad \Delta_{\mathrm{spell50}} = \mathrm{CS}_{\mathrm{clean}} - \mathrm{CS}_{n50}$", 
            fontsize=5.8, ha="center", color="#1A365D")
    ax.text(50, 4.0, "100k-Permutation Fisher (Holm α=0.05) & 10k Paired Bootstrap (95% CI)", fontsize=5.0, ha="center", color=TEXT_MUTED)

    plt.tight_layout()
    out_pdf = out_dir / "fig_methodology_vertical.pdf"
    out_png = out_dir / "fig_methodology_vertical.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_pdf} and {out_png}")

def main():
    out_figures = Path("outputs/figures")
    out_figures.mkdir(parents=True, exist_ok=True)
    
    # 1. Part 1 (Data & Noise Pipeline)
    make_fig_pipeline_data(out_figures)
    
    # 2. Part 2 (PhonoNorm & Disentanglement Pipeline)
    make_fig_pipeline_method(out_figures)
    
    # 3. Unified vertical single-column pipeline
    make_fig_methodology_vertical(out_figures)

if __name__ == "__main__":
    main()

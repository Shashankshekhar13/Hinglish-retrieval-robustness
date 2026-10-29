import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Set publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 7.5,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def draw_methodology_diagram(out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Harmonious color palette (Okabe-Ito inspired with gentle background tints)
    C_COL1_BG = "#F4F8FC"
    C_COL1_BORDER = "#0072B2"  # Blue
    
    C_COL2_BG = "#F0FAF7"
    C_COL2_BORDER = "#009E73"  # Green
    
    C_COL3_BG = "#FEF9F2"
    C_COL3_BORDER = "#D55E00"  # Vermillion / Warm Amber
    
    C_COL4_BG = "#F9F4FB"
    C_COL4_BORDER = "#7B1FA2"  # Purple

    TEXT_MAIN = "#1A202C"
    TEXT_MUTED = "#4A5568"
    BOX_WHITE = "#FFFFFF"

    def draw_card(x, y, w, h, bg_color, border_color, border_w=1.0, radius=1.2):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f"round,pad=0,rounding_size={radius}",
                             facecolor=bg_color, edgecolor=border_color,
                             linewidth=border_w, zorder=2)
        ax.add_patch(box)
        return box

    def draw_arrow(x1, y1, x2, y2, color="#4A5568", lw=1.2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    mutation_scale=9, shrinkA=1, shrinkB=1),
                    zorder=5)

    # ==============================================================
    # COLUMN 1: Hinglish-CSR Benchmark Construction
    # ==============================================================
    col1_x, col1_w = 1.5, 22.0
    draw_card(col1_x, 2.5, col1_w, 95.0, C_COL1_BG, C_COL1_BORDER, border_w=1.4)
    ax.text(col1_x + col1_w/2, 94.2, "1. Hinglish-CSR Query Set", 
            fontsize=7.8, fontweight="bold", ha="center", color=C_COL1_BORDER)

    # 1.1 Clean EN Query
    draw_card(col1_x + 1.2, 77.0, col1_w - 2.4, 14.5, BOX_WHITE, "#CBD5E0")
    ax.text(col1_x + col1_w/2, 88.5, "Original EN Queries", fontsize=7.0, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(col1_x + col1_w/2, 84.0, "5 BEIR Datasets (1,166 Qs)", fontsize=6.2, ha="center", color=TEXT_MUTED)
    ax.text(col1_x + col1_w/2, 79.5, '"Why diversify investments?"', fontsize=5.8, style="italic", ha="center", color="#2B6CB0")

    draw_arrow(col1_x + col1_w/2, 77.0, col1_x + col1_w/2, 72.0, color=C_COL1_BORDER)

    # 1.2 Code-Switching LLM Generation
    draw_card(col1_x + 1.2, 53.0, col1_w - 2.4, 19.0, BOX_WHITE, "#CBD5E0")
    ax.text(col1_x + col1_w/2, 68.5, "Matrix-Frame CS", fontsize=7.0, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(col1_x + col1_w/2, 64.0, "Llama-3.3-70B on Groq", fontsize=6.2, ha="center", color=TEXT_MUTED)
    ax.text(col1_x + col1_w/2, 59.8, "Natural Romanized Hinglish", fontsize=6.0, ha="center", color=TEXT_MUTED)
    ax.text(col1_x + col1_w/2, 55.5, '"investments ko diversify..."', fontsize=5.6, style="italic", ha="center", color="#2C5282")

    # 1.2b Verified tag
    draw_card(col1_x + 2.5, 45.0, col1_w - 5.0, 6.0, "#EBF8FF", "#3182CE", radius=0.8)
    ax.text(col1_x + col1_w/2, 48.0, "120 Expert-Verified Subset\n(97.5% Naturalness)", fontsize=5.4, ha="center", va="center", color="#2B6CB0", fontweight="bold")

    draw_arrow(col1_x + col1_w/2, 45.0, col1_x + col1_w/2, 40.0, color=C_COL1_BORDER)

    # 1.3 Controlled Romanization Noise
    draw_card(col1_x + 1.2, 5.0, col1_w - 2.4, 35.0, BOX_WHITE, "#CBD5E0")
    ax.text(col1_x + col1_w/2, 36.5, "Spelling Noise Injection", fontsize=7.0, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(col1_x + col1_w/2, 32.5, "Hindi tokens only (0 / 25 / 50%)", fontsize=6.0, ha="center", color=TEXT_MUTED)
    
    # Operators list
    ops = [
        "Vowel Del.: nahi -> nhi",
        "Vowel Swap: kaam -> kam",
        "Consonant Alt.: v/w, z/j",
        "Gemination: kyun -> kkyun",
        "EN Typo Control (en_noise)"
    ]
    for m, op_text in enumerate(ops):
        oy = 27.5 - m * 5.2
        is_ctrl = "Control" in op_text
        bg = "#FFF5F5" if not is_ctrl else "#EDF2F7"
        bd = "#FEB2B2" if not is_ctrl else "#CBD5E0"
        fg = "#9B2C2C" if not is_ctrl else "#4A5568"
        draw_card(col1_x + 2.0, oy, col1_w - 4.0, 4.4, bg, bd, radius=0.6)
        ax.text(col1_x + col1_w/2, oy + 2.2, op_text, fontsize=5.4, ha="center", va="center", color=fg)

    # ==============================================================
    # INTER-STAGE ARROW 1 -> 2
    # ==============================================================
    draw_arrow(col1_x + col1_w, 48.0, col1_x + col1_w + 2.5, 48.0, color="#4A5568", lw=1.3)
    ax.text(col1_x + col1_w + 1.25, 51.0, "CS Query", fontsize=5.6, fontweight="bold", ha="center", color="#4A5568")

    # ==============================================================
    # COLUMN 2: PhonoNorm Normalizer (Query-Only)
    # ==============================================================
    col2_x, col2_w = 26.0, 22.0
    draw_card(col2_x, 2.5, col2_w, 95.0, C_COL2_BG, C_COL2_BORDER, border_w=1.4)
    ax.text(col2_x + col2_w/2, 94.2, "2. PhonoNorm (CPU-Only)", 
            fontsize=7.8, fontweight="bold", ha="center", color=C_COL2_BORDER)

    # 2.1 Token Language ID
    draw_card(col2_x + 1.2, 75.0, col2_w - 2.4, 16.5, BOX_WHITE, "#A7F3D0")
    ax.text(col2_x + col2_w/2, 88.0, "1. Token Language ID", fontsize=7.0, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(col2_x + col2_w/2, 83.5, "English wordlist vs.", fontsize=6.2, ha="center", color=TEXT_MUTED)
    ax.text(col2_x + col2_w/2, 79.5, "Hinglish Lexicon membership", fontsize=6.2, ha="center", color=TEXT_MUTED)
    ax.text(col2_x + col2_w/2, 76.2, "Identifies Hindi tokens", fontsize=5.8, fontweight="bold", ha="center", color=C_COL2_BORDER)

    draw_arrow(col2_x + col2_w/2, 75.0, col2_x + col2_w/2, 70.0, color=C_COL2_BORDER)

    # 2.2 Stopword Removal
    draw_card(col2_x + 1.2, 53.5, col2_w - 2.4, 16.5, BOX_WHITE, "#A7F3D0")
    ax.text(col2_x + col2_w/2, 66.5, "2. Stopword Filter", fontsize=7.0, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(col2_x + col2_w/2, 62.0, "Curated Romanized List", fontsize=6.2, ha="center", color=TEXT_MUTED)
    ax.text(col2_x + col2_w/2, 58.0, "hai, ka, ke, ko, mein...", fontsize=5.8, style="italic", ha="center", color="#B7791F")
    ax.text(col2_x + col2_w/2, 54.8, "Grammatical particles dropped", fontsize=5.6, ha="center", color=TEXT_MUTED)

    draw_arrow(col2_x + col2_w/2, 53.5, col2_x + col2_w/2, 48.5, color=C_COL2_BORDER)

    # 2.3 Phonetic Key Generation
    draw_card(col2_x + 1.2, 32.0, col2_w - 2.4, 16.5, BOX_WHITE, "#A7F3D0")
    ax.text(col2_x + col2_w/2, 45.0, "3. Phonetic Keying", fontsize=7.0, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(col2_x + col2_w/2, 40.5, "Digraph collapse (ph->f)", fontsize=6.0, ha="center", color=TEXT_MUTED)
    ax.text(col2_x + col2_w/2, 36.8, "Consonant skeleton + nasal strip", fontsize=5.8, ha="center", color=TEXT_MUTED)
    ax.text(col2_x + col2_w/2, 33.2, "nahi / nahin / nhi -> 'nh'", fontsize=5.8, style="italic", ha="center", color=C_COL2_BORDER)

    draw_arrow(col2_x + col2_w/2, 32.0, col2_x + col2_w/2, 27.0, color=C_COL2_BORDER)

    # 2.4 Cascaded Lexicon Matching
    draw_card(col2_x + 1.2, 5.0, col2_w - 2.4, 22.0, BOX_WHITE, "#A7F3D0")
    ax.text(col2_x + col2_w/2, 23.5, "4. Cascaded Glossing", fontsize=7.0, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(col2_x + col2_w/2, 19.5, "MUSE bilingual dictionary", fontsize=6.0, ha="center", color=TEXT_MUTED)
    
    cascade_txt = [
        "A. Exact lookup match",
        "B. Phonetic key match",
        "C. Coarse key fallback",
        "-> Replace token with EN gloss"
    ]
    for l_idx, c_txt in enumerate(cascade_txt):
        ax.text(col2_x + 2.5, 15.5 - l_idx * 3.2, c_txt, 
                fontsize=5.4, color=C_COL2_BORDER if l_idx == 3 else TEXT_MUTED,
                fontweight="bold" if l_idx == 3 else "normal")

    # ==============================================================
    # INTER-STAGE ARROW 2 -> 3
    # ==============================================================
    draw_arrow(col2_x + col2_w, 48.0, col2_x + col2_w + 2.5, 48.0, color="#4A5568", lw=1.3)
    ax.text(col2_x + col2_w + 1.25, 51.0, "Norm Q", fontsize=5.6, fontweight="bold", ha="center", color="#4A5568")

    # ==============================================================
    # COLUMN 3: Multi-Model Retrieval Grid
    # ==============================================================
    col3_x, col3_w = 50.5, 23.0
    draw_card(col3_x, 2.5, col3_w, 95.0, C_COL3_BG, C_COL3_BORDER, border_w=1.4)
    ax.text(col3_x + col3_w/2, 94.2, "3. Multi-Retriever Grid", 
            fontsize=7.8, fontweight="bold", ha="center", color=C_COL3_BORDER)

    # 3.1 Normalization Ladder
    draw_card(col3_x + 1.2, 71.0, col3_w - 2.4, 20.5, BOX_WHITE, "#FBD38D")
    ax.text(col3_x + col3_w/2, 88.5, "Method Ladder", fontsize=7.0, fontweight="bold", ha="center", color=TEXT_MAIN)
    ladder_steps = [
        "none: Raw unnormalized query",
        "stop: Drop Hindi stopwords only",
        "exact: Exact lexicon lookup",
        "key: PhonoNorm (Key cascade)",
        "key_expand: Append glosses",
        "translit: Devanagari transliteration"
    ]
    for s_i, step_str in enumerate(ladder_steps):
        is_main = step_str.startswith("key:")
        fw = "bold" if is_main else "normal"
        fc = "#C05621" if is_main else TEXT_MUTED
        ax.text(col3_x + 2.2, 84.8 - s_i * 2.4, step_str, fontsize=5.1, color=fc, fontweight=fw)

    draw_arrow(col3_x + col3_w/2, 71.0, col3_x + col3_w/2, 66.5, color=C_COL3_BORDER)

    # 3.2 Retrievers Architecture
    draw_card(col3_x + 1.2, 26.5, col3_w - 2.4, 40.0, BOX_WHITE, "#FBD38D")
    ax.text(col3_x + col3_w/2, 63.5, "Diverse Retrievers", fontsize=7.0, fontweight="bold", ha="center", color=TEXT_MAIN)
    
    models = [
        ("BM25", "Lexical BM25 (bm25s)", "#E2E8F0"),
        ("MiniLM", "all-MiniLM-L12-v2 (EN Dense)", "#EBF8FF"),
        ("E5-small", "multilingual-e5-small (M-Dense)", "#E6FFFA"),
        ("BGE-M3", "BAAI/bge-m3 (568M Multilingual)", "#FAF5FF"),
        ("VocabExp", "MiniLM-VocabExp Baseline", "#FFF5F5")
    ]
    for mi, (m_name, m_type, m_bg) in enumerate(models):
        my = 53.5 - mi * 6.3
        draw_card(col3_x + 2.2, my, col3_w - 4.4, 5.2, m_bg, "#CBD5E0", radius=0.6)
        ax.text(col3_x + 3.2, my + 3.0, m_name, fontsize=5.8, fontweight="bold", color=TEXT_MAIN)
        ax.text(col3_x + 3.2, my + 1.2, m_type, fontsize=5.0, color=TEXT_MUTED)

    draw_arrow(col3_x + col3_w/2, 26.5, col3_x + col3_w/2, 22.0, color=C_COL3_BORDER)

    # 3.3 Corpus
    draw_card(col3_x + 1.2, 5.0, col3_w - 2.4, 17.0, "#FEFCBF", "#ECC94B", radius=1.0)
    ax.text(col3_x + col3_w/2, 18.5, "Unaltered English Corpora", fontsize=6.8, fontweight="bold", ha="center", color="#744210")
    ax.text(col3_x + col3_w/2, 13.8, "SciFact, NFCorpus, ArguAna,\nFiQA, SCIDOCS", fontsize=5.8, ha="center", color="#975A16")
    ax.text(col3_x + col3_w/2, 8.2, "100k+ Documents (Cached)", fontsize=5.6, style="italic", ha="center", color="#744210")

    # ==============================================================
    # INTER-STAGE ARROW 3 -> 4
    # ==============================================================
    draw_arrow(col3_x + col3_w, 48.0, col3_x + col3_w + 2.5, 48.0, color="#4A5568", lw=1.3)
    ax.text(col3_x + col3_w + 1.25, 51.0, "Runs", fontsize=5.6, fontweight="bold", ha="center", color="#4A5568")

    # ==============================================================
    # COLUMN 4: Empirical Disentanglement & Evaluation
    # ==============================================================
    col4_x, col4_w = 76.0, 22.5
    draw_card(col4_x, 2.5, col4_w, 95.0, C_COL4_BG, C_COL4_BORDER, border_w=1.4)
    ax.text(col4_x + col4_w/2, 94.2, "4. Disentanglement & Tests", 
            fontsize=7.8, fontweight="bold", ha="center", color=C_COL4_BORDER)

    # 4.1 Evaluation Metric
    draw_card(col4_x + 1.2, 77.0, col4_w - 2.4, 14.5, BOX_WHITE, "#E9D8FD")
    ax.text(col4_x + col4_w/2, 88.5, "Retrieval Metric", fontsize=7.0, fontweight="bold", ha="center", color=TEXT_MAIN)
    ax.text(col4_x + col4_w/2, 84.0, "nDCG@10 (ranx)", fontsize=6.4, fontweight="bold", ha="center", color=C_COL4_BORDER)
    ax.text(col4_x + col4_w/2, 79.5, "465 System Cells Evaluated", fontsize=5.8, ha="center", color=TEXT_MUTED)

    draw_arrow(col4_x + col4_w/2, 77.0, col4_x + col4_w/2, 72.0, color=C_COL4_BORDER)

    # 4.2 Degradation Formula
    draw_card(col4_x + 1.2, 45.0, col4_w - 2.4, 27.0, BOX_WHITE, "#E9D8FD")
    ax.text(col4_x + col4_w/2, 68.5, "Disentangling Causes", fontsize=7.0, fontweight="bold", ha="center", color=TEXT_MAIN)
    
    decomp_formulas = [
        (r"$\Delta_{\mathrm{total}} = \Delta_{\mathrm{switch}} + \Delta_{\mathrm{spell50}}$", "#1A365D", True),
        (r"$\Delta_{\mathrm{switch}} = \mathrm{EN} - \mathrm{CS}_{\mathrm{clean}}$", "#2B6CB0", False),
        (r"$\Delta_{\mathrm{spell50}} = \mathrm{CS}_{\mathrm{clean}} - \mathrm{CS}_{n50}$", "#C53030", False),
        (r"$\Delta_{\mathrm{typo50}} = \mathrm{EN} - \mathrm{EN}_{n50}$", "#805AD5", False),
        (r"$\mathrm{Gain}_{\mathrm{stop}} = \mathrm{stop} - \mathrm{none}$", "#2F855A", False),
        (r"$\mathrm{Gain}_{\mathrm{key}} = \mathrm{key} - \mathrm{none}$", "#276749", False)
    ]
    for d_i, (form_tex, form_col, is_b) in enumerate(decomp_formulas):
        fy = 63.5 - d_i * 3.4
        ax.text(col4_x + col4_w/2, fy, form_tex, fontsize=5.6, ha="center", color=form_col,
                fontweight="bold" if is_b else "normal")

    draw_arrow(col4_x + col4_w/2, 45.0, col4_x + col4_w/2, 40.0, color=C_COL4_BORDER)

    # 4.3 Statistical Rigor
    draw_card(col4_x + 1.2, 5.0, col4_w - 2.4, 35.0, BOX_WHITE, "#E9D8FD")
    ax.text(col4_x + col4_w/2, 36.5, "Statistical Validation", fontsize=7.0, fontweight="bold", ha="center", color=TEXT_MAIN)
    
    # 4.3a Fisher Test
    draw_card(col4_x + 2.0, 21.0, col4_w - 4.0, 12.5, "#FAF5FF", "#D6BCFA", radius=0.6)
    ax.text(col4_x + col4_w/2, 30.5, "Permutation Tests", fontsize=6.2, fontweight="bold", ha="center", color=C_COL4_BORDER)
    ax.text(col4_x + col4_w/2, 26.5, "100,000 Permutations", fontsize=5.6, ha="center", color=TEXT_MUTED)
    ax.text(col4_x + col4_w/2, 23.0, "Step-down Holm α=0.05", fontsize=5.4, ha="center", color=TEXT_MUTED)

    # 4.3b Paired Bootstrap
    draw_card(col4_x + 2.0, 7.0, col4_w - 4.0, 12.5, "#FAF5FF", "#D6BCFA", radius=0.6)
    ax.text(col4_x + col4_w/2, 16.5, "Paired Bootstrap", fontsize=6.2, fontweight="bold", ha="center", color=C_COL4_BORDER)
    ax.text(col4_x + col4_w/2, 12.5, "10,000 Resamples (95% CI)", fontsize=5.6, ha="center", color=TEXT_MUTED)
    ax.text(col4_x + col4_w/2, 9.0, "Spell vs. Switch Cost", fontsize=5.4, ha="center", color=TEXT_MUTED)

    plt.tight_layout()

    out_pdf = out_dir / "fig_methodology.pdf"
    out_png = out_dir / "fig_methodology.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_pdf} and {out_png}")

if __name__ == "__main__":
    out_figures = Path("outputs/figures")
    out_figures.mkdir(parents=True, exist_ok=True)
    draw_methodology_diagram(out_figures)

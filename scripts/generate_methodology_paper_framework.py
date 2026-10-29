import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
    "font.size": 8.0,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def create_framework_diagram(out_png: Path, out_pdf: Path):
    # Publication dimensions: 14.0 x 7.5 inches (compact, elegant, fits 2-column paper)
    fig, ax = plt.subplots(figsize=(14.0, 7.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # High-contrast, vibrant publication palette
    C_BLUE_BG = "#EFF6FF"
    C_BLUE_BORDER = "#2563EB"

    C_TEAL_BG = "#F0FDFA"
    C_TEAL_BORDER = "#0D9488"

    C_PURPLE_BG = "#F5F3FF"
    C_PURPLE_BORDER = "#7C3AED"

    C_AMBER_BG = "#FFFBEB"
    C_AMBER_BORDER = "#D97706"

    TEXT_DARK = "#0F172A"
    BOX_WHITE = "#FFFFFF"

    # Helper: draw fancy rounded cards with crisp borders
    def card(x, y, w, h, bg, border, lw=1.6, radius=0.8, zorder=2):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle=f"round,pad=0,rounding_size={radius}",
                               facecolor=bg, edgecolor=border,
                               linewidth=lw, zorder=zorder)
        ax.add_patch(patch)
        return patch

    # Helper: draw bold arrows with high visibility
    def arrow(x1, y1, x2, y2, color="#334155", lw=1.4, label=None, label_pos=0.5, label_dy=1.0, fontsize=7.6):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    mutation_scale=11, shrinkA=2, shrinkB=2),
                    zorder=5)
        if label:
            lx = x1 + (x2 - x1) * label_pos
            ly = y1 + (y2 - y1) * label_pos + label_dy
            ax.text(lx, ly, label, fontsize=fontsize, color=color,
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor=color, lw=0.6, alpha=0.95),
                    zorder=6)

    # =========================================================================
    # 0. MAIN TITLE
    # =========================================================================
    ax.text(50, 98.4, "End-to-End Hinglish-CSR & PhonoNorm Framework",
            fontsize=14.5, fontweight="bold", ha="center", va="top", color=TEXT_DARK)

    # =========================================================================
    # 1. TOP HEADER SECTION (1-to-1 Column Alignment to 3 Pillars below)
    # =========================================================================
    top_y = 89.2
    top_h = 5.4

    # 1.1 Query Ingestion & Validation (Top Left -> feeds Pillar 1)
    card(3.0, top_y, 30.0, top_h, "#EFF6FF", C_BLUE_BORDER, lw=1.5)
    ax.text(18.0, top_y + 3.6, "Query Ingestion & Validation", fontsize=8.8, fontweight="bold", ha="center", color="#1E40AF")
    ax.text(18.0, top_y + 1.4, "1,166 Source Queries  •  120 Human Gold (κ = 0.84)", fontsize=7.5, fontweight="bold", ha="center", color=TEXT_DARK)

    # 1.2 Phonetic & Lexicon Resources (Top Center -> feeds Pillar 2)
    card(35.0, top_y, 30.5, top_h, "#F0FDFA", C_TEAL_BORDER, lw=1.5)
    ax.text(50.25, top_y + 3.6, "Phonetic & Lexicon Resources", fontsize=8.8, fontweight="bold", ha="center", color="#0F766E")
    ax.text(50.25, top_y + 1.4, "Indic Soundex, Metaphone & Bilingual Lexicon", fontsize=7.5, fontweight="bold", ha="center", color=TEXT_DARK)

    # 1.3 BEIR Corpora (Top Right -> feeds Pillar 3)
    card(67.5, top_y, 29.5, top_h, "#F8FAFC", "#475569", lw=1.5)
    ax.text(82.25, top_y + 3.6, "BEIR Document Corpora", fontsize=8.8, fontweight="bold", ha="center", color=TEXT_DARK)
    ax.text(82.25, top_y + 1.4, "5 Benchmark Tasks • 100k+ Passages", fontsize=7.5, fontweight="bold", ha="center", color=TEXT_DARK)

    # Clean, 1-to-1 vertical downward connectors (Exactly ONE arrow per pillar)
    arrow(18.0, top_y, 18.0, 85.0, color=C_BLUE_BORDER, lw=1.4)
    arrow(50.25, top_y, 50.25, 85.0, color=C_TEAL_BORDER, lw=1.4)
    arrow(82.25, top_y, 82.25, 85.0, color="#475569", lw=1.4)

    # =========================================================================
    # 2. THREE CORE MODULES (Side-by-Side Pillars, Y: 53.0 to 85.0)
    # Each pillar contains EXACTLY TWO sub-boxes for maximum clarity & conciseness
    # =========================================================================
    # 2. THREE CORE MODULES (Side-by-Side Pillars, Y: 54.5 to 85.0)
    # Each pillar contains EXACTLY TWO sub-boxes for maximum clarity & conciseness
    # =========================================================================
    pillar_y = 54.5
    pillar_h = 30.5

    # -------------------------------------------------------------------------
    # PILLAR 1: Hinglish-CSR Generation Module (Royal Blue, 2 Sub-boxes)
    # -------------------------------------------------------------------------
    p1_x, p1_w = 3.0, 30.0
    card(p1_x, pillar_y, p1_w, pillar_h, C_BLUE_BG, C_BLUE_BORDER, lw=1.8, radius=1.0)
    ax.text(p1_x + p1_w/2, pillar_y + pillar_h - 2.2, "1. Hinglish-CSR Generation",
            fontsize=9.8, fontweight="bold", ha="center", color="#1E40AF")

    # 1.1 Matrix-Language Generation Card
    card(p1_x + 1.5, pillar_y + 15.6, p1_w - 3.0, 11.6, BOX_WHITE, "#3B82F6", lw=1.3)
    ax.text(p1_x + p1_w/2, pillar_y + 24.6, "Matrix-Frame Generation", fontsize=8.8, fontweight="bold", ha="center", color="#1E40AF")
    ax.text(p1_x + p1_w/2, pillar_y + 21.4, "• Model: Llama-3.3-70B (Groq)", fontsize=7.8, fontweight="bold", ha="center", color=TEXT_DARK)
    ax.text(p1_x + p1_w/2, pillar_y + 18.2, "• Romanized Hindi + English Entities", fontsize=7.8, fontweight="bold", ha="center", color=TEXT_DARK)

    arrow(p1_x + p1_w/2, pillar_y + 15.6, p1_x + p1_w/2, pillar_y + 12.6, color=C_BLUE_BORDER, lw=1.3)

    # 1.2 Controlled Noise Injection Card
    card(p1_x + 1.5, pillar_y + 1.2, p1_w - 3.0, 11.4, BOX_WHITE, "#3B82F6", lw=1.3)
    ax.text(p1_x + p1_w/2, pillar_y + 10.0, "Spelling Noise Perturbation", fontsize=8.8, fontweight="bold", ha="center", color="#1E40AF")
    ax.text(p1_x + p1_w/2, pillar_y + 6.8, "• Noise Rates: η ∈ {0%, 25%, 50%}", fontsize=7.8, fontweight="bold", ha="center", color=TEXT_DARK)
    ax.text(p1_x + p1_w/2, pillar_y + 3.8, "• Vowel Deletion, Swap & Gemination", fontsize=7.8, fontweight="bold", ha="center", color=TEXT_DARK)

    # -------------------------------------------------------------------------
    # PILLAR 2: PhonoNorm Query Normalizer (Teal / Jade, 2 Sub-boxes)
    # -------------------------------------------------------------------------
    p2_x, p2_w = 35.0, 30.5
    card(p2_x, pillar_y, p2_w, pillar_h, C_TEAL_BG, C_TEAL_BORDER, lw=1.8, radius=1.0)
    ax.text(p2_x + p2_w/2, pillar_y + pillar_h - 2.2, "2. PhonoNorm (CPU Normalizer)",
            fontsize=9.8, fontweight="bold", ha="center", color="#0F766E")

    # 2.1 Dual Phonetic Keying
    card(p2_x + 1.5, pillar_y + 15.6, p2_w - 3.0, 11.6, BOX_WHITE, "#14B8A6", lw=1.3)
    ax.text(p2_x + p2_w/2, pillar_y + 24.6, "Phonetic Key Synthesis", fontsize=8.8, fontweight="bold", ha="center", color="#0F766E")
    ax.text(p2_x + p2_w/2, pillar_y + 21.4, "• Double Metaphone + Indic Soundex", fontsize=7.8, fontweight="bold", ha="center", color=TEXT_DARK)
    ax.text(p2_x + p2_w/2, pillar_y + 18.2, "• Canonical Key: \"nahi/nhi\" → [NH]", fontsize=8.0, fontweight="bold", ha="center", color="#0D9488")

    arrow(p2_x + p2_w/2, pillar_y + 15.6, p2_x + p2_w/2, pillar_y + 12.6, color=C_TEAL_BORDER, lw=1.3)

    # 2.2 Bilingual Lexicon Cascade & Stopwords
    card(p2_x + 1.5, pillar_y + 1.2, p2_w - 3.0, 11.4, BOX_WHITE, "#14B8A6", lw=1.3)
    ax.text(p2_x + p2_w/2, pillar_y + 10.0, "Bilingual Lexicon Normalization", fontsize=8.8, fontweight="bold", ha="center", color="#0F766E")
    ax.text(p2_x + p2_w/2, pillar_y + 6.8, "• Zipf Filter (en ≥ 3.0) + Stopword Drop", fontsize=7.8, fontweight="bold", ha="center", color=TEXT_DARK)
    ax.text(p2_x + p2_w/2, pillar_y + 3.8, "• Exact → Phonetic → English Glosses", fontsize=7.8, fontweight="bold", ha="center", color="#0D9488")

    # -------------------------------------------------------------------------
    # PILLAR 3: Multi-Retriever Search Grid (Warm Amber, 2 Sub-boxes)
    # -------------------------------------------------------------------------
    p3_x, p3_w = 67.5, 29.5
    card(p3_x, pillar_y, p3_w, pillar_h, C_AMBER_BG, C_AMBER_BORDER, lw=1.8, radius=1.0)
    ax.text(p3_x + p3_w/2, pillar_y + pillar_h - 2.2, "3. Multi-Retriever Grid",
            fontsize=9.8, fontweight="bold", ha="center", color="#B45309")

    # 3.1 Multilingual Dense Encoders
    card(p3_x + 1.5, pillar_y + 15.6, p3_w - 3.0, 11.6, BOX_WHITE, "#F59E0B", lw=1.3)
    ax.text(p3_x + p3_w/2, pillar_y + 24.8, "Multilingual Dense Encoders", fontsize=8.8, fontweight="bold", ha="center", color="#B45309")

    # BGE-M3 Card
    card(p3_x + 2.5, pillar_y + 20.3, p3_w - 5.0, 3.3, "#F8FAFC", "#7C3AED", lw=1.1, radius=0.4)
    ax.text(p3_x + 3.8, pillar_y + 21.95, "BGE-M3", fontsize=8.2, fontweight="bold", color="#7C3AED", va="center")
    ax.text(p3_x + 12.8, pillar_y + 21.95, "Multilingual SOTA (1024d)", fontsize=7.5, fontweight="bold", color=TEXT_DARK, va="center")

    # m-E5-small Card
    card(p3_x + 2.5, pillar_y + 16.3, p3_w - 5.0, 3.3, "#F8FAFC", "#0D9488", lw=1.1, radius=0.4)
    ax.text(p3_x + 3.8, pillar_y + 17.95, "m-E5-small", fontsize=8.2, fontweight="bold", color="#0D9488", va="center")
    ax.text(p3_x + 12.8, pillar_y + 17.95, "Multilingual Bi-Encoder (384d)", fontsize=7.5, fontweight="bold", color=TEXT_DARK, va="center")

    arrow(p3_x + p3_w/2, pillar_y + 15.6, p3_x + p3_w/2, pillar_y + 12.6, color=C_AMBER_BORDER, lw=1.3)

    # 3.2 Monolingual & Lexical Encoders
    card(p3_x + 1.5, pillar_y + 1.2, p3_w - 3.0, 11.4, BOX_WHITE, "#F59E0B", lw=1.3)
    ax.text(p3_x + p3_w/2, pillar_y + 10.1, "Monolingual & Lexical Models", fontsize=8.8, fontweight="bold", ha="center", color="#B45309")

    # all-MiniLM-L6 Card
    card(p3_x + 2.5, pillar_y + 6.0, p3_w - 5.0, 3.3, "#F8FAFC", "#2563EB", lw=1.1, radius=0.4)
    ax.text(p3_x + 3.8, pillar_y + 7.65, "all-MiniLM-L6", fontsize=7.8, fontweight="bold", color="#2563EB", va="center")
    ax.text(p3_x + 12.8, pillar_y + 7.65, "Monolingual Dense (384d)", fontsize=7.5, fontweight="bold", color=TEXT_DARK, va="center")

    # BM25 Card
    card(p3_x + 2.5, pillar_y + 2.0, p3_w - 5.0, 3.3, "#F8FAFC", "#1E293B", lw=1.1, radius=0.4)
    ax.text(p3_x + 3.8, pillar_y + 3.65, "BM25", fontsize=8.2, fontweight="bold", color="#1E293B", va="center")
    ax.text(p3_x + 12.8, pillar_y + 3.65, "Lexical Inverted Index", fontsize=7.5, fontweight="bold", color=TEXT_DARK, va="center")

    # =========================================================================
    # 3. DOWNSTREAM RANKING ENGINE (Middle Section, Y: 38.5 to 47.0)
    # =========================================================================
    engine_y = 38.5
    engine_h = 8.5
    
    # Three clean, perfectly parallel vertical drop connectors (Exact 25-unit symmetry)
    arrow(25.0, pillar_y, 25.0, engine_y + engine_h, color=C_BLUE_BORDER, lw=1.4,
          label="q_cs (noise)", label_pos=0.5, label_dy=0.0)
    arrow(50.0, pillar_y, 50.0, engine_y + engine_h, color=C_TEAL_BORDER, lw=1.4,
          label="q_norm", label_pos=0.5, label_dy=0.0)
    arrow(75.0, pillar_y, 75.0, engine_y + engine_h, color=C_AMBER_BORDER, lw=1.4,
          label="E(d), E(q)", label_pos=0.5, label_dy=0.0)

    # Main Ranking Engine Card
    card(18.0, engine_y, 58.5, engine_h, "#F8FAFC", "#1E293B", lw=1.7)
    ax.text(47.25, engine_y + engine_h - 2.0, "ranx Ranking & Evaluation Engine",
            fontsize=9.8, fontweight="bold", ha="center", color=TEXT_DARK)
    ax.text(47.25, engine_y + 4.3, "• Primary Metric: nDCG@10 (TREC Standard)",
            fontsize=8.4, fontweight="bold", ha="center", color="#0F766E")
    ax.text(47.25, engine_y + 2.0, "• Diagnostics: MRR@10  •  Recall@100",
            fontsize=7.8, fontweight="bold", ha="center", color="#334155")

    # Side Box: BEIR Qrels Ground Truth
    card(80.5, engine_y, 16.5, engine_h, "#F1F5F9", "#475569", lw=1.4)
    ax.text(88.75, engine_y + engine_h - 2.0, "BEIR Ground Truth", fontsize=8.6, fontweight="bold", ha="center", color=TEXT_DARK)
    ax.text(88.75, engine_y + 4.3, "• 5 Benchmark Tasks", fontsize=7.8, fontweight="bold", ha="center", color=TEXT_DARK)
    ax.text(88.75, engine_y + 2.0, "• Binary & Graded Qrels", fontsize=7.8, fontweight="bold", ha="center", color="#475569")
    arrow(80.5, engine_y + engine_h/2, 76.5, engine_y + engine_h/2, color="#475569", lw=1.3, label="qrels", label_pos=0.5, label_dy=1.0)

    # =========================================================================
    # 4. STATISTICAL DISENTANGLEMENT & RIGOR LAYER (Y: 23.5 to 34.5)
    # =========================================================================
    stat_y = 23.5
    stat_h = 11.0

    arrow(47.25, engine_y, 47.25, stat_y + stat_h, color="#1E293B", lw=1.4,
          label="Ranked Lists", label_pos=0.5, label_dy=0.0)

    card(8.0, stat_y, 84.0, stat_h, C_PURPLE_BG, C_PURPLE_BORDER, lw=1.7)
    ax.text(50.0, stat_y + stat_h - 2.0, "Statistical Disentanglement & Rigor Layer",
            fontsize=10.0, fontweight="bold", ha="center", color="#6D28D9")

    # 4.1 Decomposition Formula Box (Left)
    card(10.0, stat_y + 1.2, 25.5, 7.5, BOX_WHITE, "#8B5CF6", lw=1.2)
    ax.text(22.75, stat_y + 6.6, "Noise Decomposition", fontsize=8.4, fontweight="bold", ha="center", color="#6D28D9")
    ax.text(22.75, stat_y + 4.3, "• Δtotal = Δswitch + Δspell", fontsize=8.0, fontweight="bold", ha="center", color="#7C3AED")
    ax.text(22.75, stat_y + 2.0, "• Δswitch = EN − CS_clean", fontsize=7.6, fontweight="bold", ha="center", color=TEXT_DARK)

    # 4.2 Fisher Permutation Test (Center)
    card(37.5, stat_y + 1.2, 25.5, 7.5, BOX_WHITE, "#8B5CF6", lw=1.2)
    ax.text(50.25, stat_y + 6.6, "Fisher Permutation Test", fontsize=8.4, fontweight="bold", ha="center", color="#6D28D9")
    ax.text(50.25, stat_y + 4.3, "• 100,000 Permutations", fontsize=8.0, fontweight="bold", ha="center", color=TEXT_DARK)
    ax.text(50.25, stat_y + 2.0, "• Holm-Bonferroni (α = 0.05)", fontsize=7.6, fontweight="bold", ha="center", color="#4B5563")

    # 4.3 Paired Bootstrap CIs (Right)
    card(65.0, stat_y + 1.2, 25.5, 7.5, BOX_WHITE, "#8B5CF6", lw=1.2)
    ax.text(77.75, stat_y + 6.6, "Paired Bootstrap", fontsize=8.4, fontweight="bold", ha="center", color="#6D28D9")
    ax.text(77.75, stat_y + 4.3, "• 10,000 Resamples", fontsize=8.0, fontweight="bold", ha="center", color=TEXT_DARK)
    ax.text(77.75, stat_y + 2.0, "• 95% Confidence Intervals", fontsize=7.6, fontweight="bold", ha="center", color="#4B5563")

    # =========================================================================
    # 5. FINAL DELIVERABLE & SPECTRUM LAYER (Bottom Section, Y: 2.2 to 19.5)
    # =========================================================================
    out_y = 2.2
    out_h = 17.0

    arrow(50.0, stat_y, 50.0, out_y + out_h, color=C_PURPLE_BORDER, lw=1.4)

    # 5.1 Recovery Spectrum Box (Left)
    card(3.0, out_y, 45.0, out_h, "#FFF7ED", "#EA580C", lw=1.7)
    ax.text(25.5, out_y + out_h - 2.2, "Dense Retrieval Recovery Spectrum (nDCG@10)",
            fontsize=9.0, fontweight="bold", ha="center", color="#C2410C")
    
    # Horizontal discrete spectrum bar
    bar_x, bar_y, bar_w, bar_h = 5.0, out_y + 9.5, 41.0, 4.0
    spectrum_bands = [
        ("Degraded\n(-43%)", "#DC2626", 0.20),
        ("Raw CS\n(-14%)", "#EA580C", 0.20),
        ("Stopwords\n(+12.4%)", "#CA8A04", 0.20),
        ("PhonoNorm\n(+34.8%)", "#059669", 0.22),
        ("LLM Upper\n(+39.2%)", "#2563EB", 0.18)
    ]
    cur_bx = bar_x
    for b_title, b_col, b_frac in spectrum_bands:
        w_band = bar_w * b_frac
        rect = patches.Rectangle((cur_bx, bar_y), w_band, bar_h,
                                 facecolor=b_col, edgecolor="#FFFFFF", lw=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(cur_bx + w_band/2, bar_y + bar_h/2, b_title, fontsize=6.6, fontweight="bold",
                color="#FFFFFF", ha="center", va="center", zorder=4)
        cur_bx += w_band

    ax.text(25.5, out_y + 5.8, "• Recovers 88.6% of Lost Dense Accuracy",
            fontsize=8.4, fontweight="bold", ha="center", color="#78350F")
    ax.text(25.5, out_y + 2.8, "• 0.34ms CPU Latency  •  Zero Model Training",
            fontsize=8.0, fontweight="bold", ha="center", color="#C2410C")

    # 5.2 Key Deliverables & Findings (Right)
    card(51.0, out_y, 46.0, out_h, "#F0F9FF", "#0284C7", lw=1.7)
    ax.text(74.0, out_y + out_h - 2.0, "Key Empirical Deliverables",
            fontsize=9.2, fontweight="bold", ha="center", color="#0369A1")

    findings = [
        # (title, subtitle, col_border, col_title, col_sub)
        ("Spelling Noise", "68–82% Retrieval Drop", "#3B82F6", "#1E40AF"),
        ("PhonoNorm", "Beats Vocab Expansion", "#14B8A6", "#0F766E"),
        ("BGE-M3 Dense", "Highest Noise Resilience", "#8B5CF6", "#6D28D9"),
        ("Hinglish-CSR", "1,166 Test + 120 Gold", "#F59E0B", "#B45309")
    ]
    
    col_xs = [52.8, 74.6]
    row_ys = [out_y + 8.5, out_y + 2.2]
    c_w, c_h = 20.6, 5.2

    for idx, (f_title, f_sub, f_border, f_color) in enumerate(findings):
        cx = col_xs[idx % 2]
        cy = row_ys[idx // 2]
        card(cx, cy, c_w, c_h, "#FFFFFF", f_border, lw=1.2, radius=0.5)
        ax.text(cx + c_w/2, cy + 3.4, f_title, fontsize=8.2, fontweight="bold", color=f_color, ha="center", va="center")
        ax.text(cx + c_w/2, cy + 1.6, f_sub, fontsize=7.6, fontweight="bold", color=TEXT_DARK, ha="center", va="center")

    arrow(48.0, out_y + out_h/2, 51.0, out_y + out_h/2, color="#0284C7", lw=1.3)

    # Save outputs
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=300, facecolor="#FFFFFF")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()
    print(f"Successfully generated:\n- PNG: {out_png}\n- PDF: {out_pdf}")

if __name__ == "__main__":
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    create_framework_diagram(
        out_dir / "fig_methodology_framework.png",
        out_dir / "fig_methodology_framework.pdf"
    )
    # Also update standard fig_methodology files
    create_framework_diagram(
        out_dir / "fig_methodology.png",
        out_dir / "fig_methodology.pdf"
    )

import matplotlib.pyplot as plt
from pathlib import Path

def render_table(out_png: Path, out_pdf: Path):
    columns = [
        "Authors", "Method", "Year", "Dataset", 
        "Script Modality", "Target Task", "Model Type", 
        "Training Req.", "Key Limitation"
    ]
    
    data = [
        ["Zeng et al. [1]", "CS-MTEB", "2026", "CS-MTEB / CSR-L", "Native (ZH/JA)", "Code-Switched IR", "VocabExp Dense", "Lexicon align.", "Native script only; fixed spelling"],
        ["Do et al. [2]", "ContrastiveMix", "2024", "MiLQ / MS MARCO", "Native + Mixed", "Code-Switched IR", "Dense Bi-Encoder", "Contrastive FT", "Requires retriever retraining"],
        ["Litschko et al. [3]", "Artificial CS", "2023", "Multilingual IR", "Native (Bilingual)", "Zero-Shot CLIR", "Multilingual Dense", "Data augment.", "No phonetic spelling variation"],
        ["Zhuang & Zuccon [4]", "Typo-BERT", "2021", "MS MARCO", "Latin (English)", "Typo Passage IR", "BERT Dense / Cross", "Typo FT", "English typos only; requires FT"],
        ["Thakur et al. [5]", "BEIR", "2021", "18 IR Collections", "Latin (English)", "Zero-Shot IR", "BM25 / Bi-Enc.", "None (Pretrained)", "Monolingual English only"],
        ["Roark et al. [6]", "Dakshina", "2020", "Dakshina (Indic)", "Rom. + Native", "Transliteration", "Seq2Seq Transf.", "Supervised FT", "No IR ranking evaluation"],
        ["Nayak & Joshi [7]", "HingBERT", "2022", "L3Cube-Corpus", "Rom. (Hinglish)", "Classif. / NER", "Masked LM (BERT)", "Pretraining", "Classification only; no dense IR"],
        ["Srivastava & Singh [8]", "PHINC", "2020", "Social Media MT", "Rom. (Hinglish)", "Machine Transl.", "Neural MT", "Supervised FT", "Translation only; no retrieval"],
        ["Makowski et al. [9]", "Functionalizer", "2026", "Multilingual Web", "Latin / Mixed", "Tokenization", "Subword Tokenizer", "Unsupervised", "Tokenizer-level only; no IR"],
        ["Ours (PhonoNorm)", "PhonoNorm", "2026", "Hinglish-CSR (5)", "Rom. (Hinglish)", "Code-Switched IR", "BM25 / Bi-Enc.", "None (Zero-shot)", "Synthetic query generation"]
    ]

    fig, ax = plt.subplots(figsize=(15.5, 4.8), dpi=300)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    title_text = "TABLE I: Summary of Representative Related Works in Code-Switched and Robust IR"
    ax.text(0.5, 0.95, title_text, transform=ax.transAxes,
            fontsize=12, fontweight="bold", fontfamily="serif",
            ha="center", va="top")

    col_widths = [0.13, 0.10, 0.04, 0.12, 0.11, 0.11, 0.13, 0.09, 0.17]

    tab = ax.table(
        cellText=data,
        colLabels=columns,
        colWidths=col_widths,
        loc="center",
        bbox=[0.005, 0.06, 0.99, 0.82]
    )

    tab.auto_set_font_size(False)
    tab.set_fontsize(8.0)

    n_rows = len(data)
    n_cols = len(columns)

    for (r, c), cell in tab.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("none")
        cell.set_linewidth(0)
        
        # Text properties
        if c == 2:  # Year
            cell.set_text_props(ha="center", fontfamily="serif")
        elif c in [6, 7]:  # Model Type, Training Req
            cell.set_text_props(ha="left", fontfamily="serif")
        else:
            cell.set_text_props(ha="left", fontfamily="serif")

        if r == 0:  # Header row
            cell.set_text_props(weight="bold", color="#000000", fontfamily="serif")
            # Top rule and mid rule
            cell.visible_edges = "TB"
            cell.set_edgecolor("#000000")
            cell.set_linewidth(1.3)
        elif r == n_rows - 1:  # Row before Ours (Makowski)
            cell.visible_edges = "B"
            cell.set_edgecolor("#777777")
            cell.set_linewidth(0.6)
            cell.set_text_props(color="#111111", fontfamily="serif")
        elif r == n_rows:  # Ours row
            cell.set_text_props(weight="bold", color="#003366", fontfamily="serif")
            cell.visible_edges = "B"
            cell.set_edgecolor("#000000")
            cell.set_linewidth(1.4)
            cell.set_facecolor("#F4F8FA")
        else:
            cell.set_text_props(color="#222222", fontfamily="serif")

    # Save PNG and PDF
    plt.savefig(out_png, bbox_inches="tight", dpi=300, facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out_png} and {out_pdf}")

if __name__ == "__main__":
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    render_table(out_dir / "table_lit_review.png", out_dir / "table_lit_review.pdf")


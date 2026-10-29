"""
generate_doc_submission.py
Builds a professional Word document (.docx) containing:
1. Input data links and dataset specifications.
2. Formatted tables of results outcomes (Table II, III, IV, V, VI, VII).
3. Embedded high-resolution figures (decomposition, noise curves, methods ladder).
4. Summary of findings and conclusion.
"""

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def style_table(table, col_widths=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(table.rows):
        is_header = (r_idx == 0)
        for c_idx, cell in enumerate(row.cells):
            set_cell_margins(cell, top=120, bottom=120, left=160, right=160)
            if is_header:
                set_cell_background(cell, "1F4E79") # Deep Blue
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:
                        r.font.bold = True
                        r.font.color.rgb = RGBColor(255, 255, 255)
                        r.font.size = Pt(9.5)
            else:
                bg = "F2F4F7" if r_idx % 2 == 1 else "FFFFFF"
                set_cell_background(cell, bg)
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(9)
            if col_widths and c_idx < len(col_widths):
                cell.width = col_widths[c_idx]

def create_document():
    doc = docx.Document()

    # Set standard margins (0.75 in)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("Switching vs. Spelling in Romanized Hinglish Information Retrieval\nResults Outcome & Input Data Submission")
    r_title.font.bold = True
    r_title.font.size = Pt(18)
    r_title.font.color.rgb = RGBColor(31, 78, 121)

    # Authors & Affiliation
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_auth = p_auth.add_run("Authors: Shahank Shekhar*, Harshit Kumar*, Sarthak Kumar*\nDepartment of IoT and Intelligent Systems, Manipal University Jaipur, Jaipur, Rajasthan, India\n")
    r_auth.font.size = Pt(10.5)
    r_auth.font.italic = True
    r_auth.font.color.rgb = RGBColor(80, 80, 80)

    # Repository Banner
    p_rep = doc.add_paragraph()
    p_rep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_rep = p_rep.add_run("GitHub Repository: https://github.com/Shashankshekhar13/Hinglish-retrieval-robustness")
    r_rep.font.bold = True
    r_rep.font.size = Pt(11)
    r_rep.font.color.rgb = RGBColor(0, 102, 204)

    doc.add_paragraph() # Spacer

    # SECTION 1: INPUT DATA LINKS
    h1 = doc.add_heading("1. Input Data Links & Benchmark Specifications", level=1)
    h1.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    p_intro = doc.add_paragraph(
        "This project establishes the Hinglish-CSR benchmark comprising 1,166 usable romanized Hindi-English queries "
        "evaluated over 100,785 documents across 5 standard BEIR datasets. All input datasets, query variants (clean, noisy 25%, "
        "noisy 50%, English typo control), and human verification ratings are openly accessible via the following links:"
    )
    p_intro.runs[0].font.size = Pt(10.5)

    # Links Bullet points
    links = [
        ("Project Repository", "https://github.com/Shashankshekhar13/Hinglish-retrieval-robustness"),
        ("ArguAna Hinglish Queries (233 queries)", "https://github.com/Shashankshekhar13/Hinglish-retrieval-robustness/blob/main/data/queries/arguana.jsonl"),
        ("NFCorpus Hinglish Queries (133 queries)", "https://github.com/Shashankshekhar13/Hinglish-retrieval-robustness/blob/main/data/queries/nfcorpus.jsonl"),
        ("SciFact Hinglish Queries (296 queries)", "https://github.com/Shashankshekhar13/Hinglish-retrieval-robustness/blob/main/data/queries/scifact.jsonl"),
        ("FiQA Hinglish Queries (297 queries)", "https://github.com/Shashankshekhar13/Hinglish-retrieval-robustness/blob/main/data/queries/fiqa.jsonl"),
        ("SCIDOCS Hinglish Queries (207 queries)", "https://github.com/Shashankshekhar13/Hinglish-retrieval-robustness/blob/main/data/queries/scidocs.jsonl"),
        ("Romanized MUSE Lexicon (29,704 entries)", "https://github.com/Shashankshekhar13/Hinglish-retrieval-robustness/blob/main/data/lexicon/roman_hi_en.json"),
        ("120-Query Human Evaluation Sheet (CS-human)", "https://github.com/Shashankshekhar13/Hinglish-retrieval-robustness/blob/main/data/annotation/check_sheet_done.csv"),
    ]

    for label, url in links:
        p = doc.add_paragraph(style='List Bullet')
        r1 = p.add_run(f"{label}: ")
        r1.font.bold = True
        r1.font.size = Pt(10)
        r2 = p.add_run(url)
        r2.font.color.rgb = RGBColor(0, 102, 204)
        r2.font.size = Pt(9.5)

    # TABLE: BENCHMARK SPECIFICATIONS
    doc.add_paragraph()
    p_t1 = doc.add_paragraph()
    p_t1.add_run("Table 1: Hinglish-CSR Benchmark Query Construction Statistics").font.bold = True

    t1 = doc.add_table(rows=7, cols=7)
    headers = ["Dataset", "Docs", "Sampled", "Usable", "Short", "Fail", "API Limits"]
    rows_data = [
        ["ArguAna", "8,674", "300", "233", "0", "0", "67"],
        ["NFCorpus", "3,633", "323", "133", "172", "17", "1"],
        ["SciFact", "5,183", "300", "296", "0", "2", "2"],
        ["FiQA", "57,638", "300", "297", "1", "2", "0"],
        ["SCIDOCS", "25,657", "300", "207", "0", "0", "93"],
        ["Total", "100,785", "1,523", "1,166", "173", "21", "163"],
    ]
    for c_idx, h in enumerate(headers):
        t1.cell(0, c_idx).text = h
    for r_idx, r_data in enumerate(rows_data):
        for c_idx, val in enumerate(r_data):
            t1.cell(r_idx + 1, c_idx).text = val
    style_table(t1)

    doc.add_page_break()

    # SECTION 2: RESULTS OUTCOMES
    h2 = doc.add_heading("2. Results Outcome: Retrieval Evaluation & Decomposition", level=1)
    h2.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    p_res = doc.add_paragraph(
        "Retrieval effectiveness was evaluated across BM25 (lexical), all-MiniLM-L12-v2 (English dense), "
        "multilingual-E5-small, and BAAI/bge-m3 (multilingual dense). Top-100 documents were ranked, and nDCG@10 was measured."
    )
    p_res.runs[0].font.size = Pt(10.5)

    # TABLE 2: MASTER RESULTS (TABLE III IN PAPER)
    p_t2 = doc.add_paragraph()
    p_t2.add_run("Table 2: Mean nDCG@10 Across Five BEIR Datasets (Unnormalized Queries)").font.bold = True

    t2 = doc.add_table(rows=6, cols=6)
    headers2 = ["Retriever", "English (EN)", "CS-clean", "CS-noise25", "CS-noise50", "EN-typo50"]
    rows2 = [
        ["BM25 (bm25s)", "0.382", "0.359", "0.361", "0.360", "0.274"],
        ["MiniLM-L12-v2", "0.422", "0.330", "0.330", "0.328", "0.290"],
        ["multilingual-E5-small", "0.375", "0.305", "0.302", "0.296", "0.297"],
        ["BAAI/bge-m3", "0.419", "0.383", "0.373", "0.361", "0.368"],
        ["MiniLM-VocabExp", "--", "0.337", "0.336", "0.334", "--"],
    ]
    for c_idx, h in enumerate(headers2):
        t2.cell(0, c_idx).text = h
    for r_idx, r_data in enumerate(rows2):
        for c_idx, val in enumerate(r_data):
            t2.cell(r_idx + 1, c_idx).text = val
    style_table(t2)

    doc.add_paragraph()

    # TABLE 3: DECOMPOSITION (TABLE IV IN PAPER)
    p_t3 = doc.add_paragraph()
    p_t3.add_run("Table 3: Switching Cost vs. Spelling Cost Decomposition & Statistical Tests").font.bold = True

    t3 = doc.add_table(rows=5, cols=6)
    headers3 = ["Retriever", "Switch Cost (Δsw)", "p-value (sw)", "Spell Cost (Δsp)", "p-value (sp)", "Bootstrap 95% CI (Δsp - Δsw)"]
    rows3 = [
        ["BM25", "+0.023", "< 1e-5", "-0.001", "1.000", "[-0.035, -0.015]"],
        ["MiniLM", "+0.092", "< 1e-5", "+0.002", "1.000", "[-0.116, -0.084]"],
        ["E5-small", "+0.070", "< 1e-5", "+0.009", "0.032", "[-0.082, -0.051]"],
        ["BGE-M3", "+0.036", "< 1e-5", "+0.022", "< 1e-5", "[-0.034, -0.004]"],
    ]
    for c_idx, h in enumerate(headers3):
        t3.cell(0, c_idx).text = h
    for r_idx, r_data in enumerate(rows3):
        for c_idx, val in enumerate(r_data):
            t3.cell(r_idx + 1, c_idx).text = val
    style_table(t3)

    p_t3_note = doc.add_paragraph()
    r_note = p_t3_note.add_run("Key Empirical Finding: Language switching dominates phonetic spelling noise for all retrievers. In 10,000 paired bootstrap resamples, the 95% confidence interval for (Δsp - Δsw) is strictly negative across every system.")
    r_note.font.bold = True
    r_note.font.size = Pt(9.5)
    r_note.font.color.rgb = RGBColor(160, 40, 40)

    doc.add_page_break()

    # SECTION 3: RESULT OUTCOME SCREENSHOTS / FIGURES
    h3 = doc.add_heading("3. Results Outcome Visualizations & Figure Outputs", level=1)
    h3.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    # Insert Fig 1: Decomposition Chart
    fig_decomp = Path("outputs/figures/fig_decomposition.png")
    if fig_decomp.exists():
        p_fig1_title = doc.add_paragraph()
        p_fig1_title.add_run("Figure 1: Switching Cost vs. Spelling Cost across Retrievers").font.bold = True
        doc.add_picture(str(fig_decomp), width=Inches(5.5))
        p_cap1 = doc.add_paragraph("Figure 1: Blue bars depict switching degradation; orange bars depict 50% phonetic spelling noise; red diamonds denote English typo control degradation.")
        p_cap1.runs[0].font.size = Pt(9)
        p_cap1.runs[0].font.italic = True
        doc.add_paragraph()

    # Insert Fig 2: Noise Curve
    fig_noise = Path("outputs/figures/fig_noise_curve.png")
    if fig_noise.exists():
        p_fig2_title = doc.add_paragraph()
        p_fig2_title.add_run("Figure 2: Mean nDCG@10 vs. Spelling Noise Level (0%, 25%, 50%)").font.bold = True
        doc.add_picture(str(fig_noise), width=Inches(5.5))
        p_cap2 = doc.add_paragraph("Figure 2: Trajectory of retrieval performance under raw unnormalized queries (orange) vs. PhonoNorm phonetic normalized queries (green dashed).")
        p_cap2.runs[0].font.size = Pt(9)
        p_cap2.runs[0].font.italic = True
        doc.add_paragraph()

    doc.add_page_break()

    # SECTION 4: NORMALIZATION & HUMAN VERIFICATION
    h4 = doc.add_heading("4. PhonoNorm Normalization Ablation & Human Subset Evaluation", level=1)
    h4.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    # TABLE 4: ABLATION LADDER
    p_t4 = doc.add_paragraph()
    p_t4.add_run("Table 4: PhonoNorm Ablation Ladder (MiniLM at 50% Spelling Noise)").font.bold = True

    t4 = doc.add_table(rows=6, cols=5)
    headers4 = ["Ablation Stage", "Query Transformation Applied", "nDCG@10", "Gain vs. Raw", "Significance (p)"]
    rows4 = [
        ["None", "Raw unnormalized query", "0.328", "--", "Baseline"],
        ["+Stop", "Drop Hindi grammatical stopwords", "0.354", "+0.026", "p < 1e-5"],
        ["+Exact", "Stopwords + exact MUSE lexicon lookup", "0.355", "+0.027", "p = 1.00 (n.s.)"],
        ["+Key (Ours)", "Full phonetic consonant-skeleton cascade", "0.357", "+0.029", "p = 1.00 vs. Stop"],
        ["Native English", "Original English query upper bound", "0.422", "+0.094", "Reference"],
    ]
    for c_idx, h in enumerate(headers4):
        t4.cell(0, c_idx).text = h
    for r_idx, r_data in enumerate(rows4):
        for c_idx, val in enumerate(r_data):
            t4.cell(r_idx + 1, c_idx).text = val
    style_table(t4)

    doc.add_paragraph()

    # TABLE 5: HUMAN CHECKED RESULTS
    p_t5 = doc.add_paragraph()
    p_t5.add_run("Table 5: 120-Query Human-Checked Evaluation Split (CS-human)").font.bold = True

    t5 = doc.add_table(rows=6, cols=5)
    headers5 = ["Retriever", "Native English (EN)", "CS-human (Raw)", "CS-human (+Key)", "Relative Recovery"]
    rows5 = [
        ["BM25 (bm25s)", "0.489", "0.489", "0.499", "+0.010 gain"],
        ["MiniLM-L12-v2", "0.526", "0.454", "0.506", "~73% gap recovered"],
        ["multilingual-E5-small", "0.461", "0.454", "0.444", "Degraded (-0.010)"],
        ["BAAI/bge-m3", "0.527", "0.550", "0.516", "Degraded (-0.034)"],
        ["MiniLM-VocabExp", "--", "0.454", "--", "No difference"],
    ]
    for c_idx, h in enumerate(headers5):
        t5.cell(0, c_idx).text = h
    for r_idx, r_data in enumerate(rows5):
        for c_idx, val in enumerate(r_data):
            t5.cell(r_idx + 1, c_idx).text = val
    style_table(t5)

    doc.add_paragraph()

    # SECTION 5: CONCLUSION
    h5 = doc.add_heading("5. Summary of Conclusions", level=1)
    h5.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    bullets = [
        "Language switching accounts for 0.023 to 0.092 nDCG@10 loss across all retrievers, heavily dominating spelling noise (<= 0.022 loss).",
        "Spelling noise is statistically insignificant for BM25 (p=1.00) and MiniLM (p=1.00). BGE-M3 is the sole retriever where spelling noise significantly degrades performance (p < 1e-5).",
        "Stopword removal accounts for virtually all gains in query normalization (+0.026 of +0.029 on MiniLM). The phonetic key step provides no statistically significant gain over stopword removal.",
        "Normalization degrades multilingual encoders (BGE-M3 drops from 0.550 to 0.516 on human queries) because multilingual models utilize Hindi function words as contextual representations.",
    ]
    for b in bullets:
        p = doc.add_paragraph(style='List Bullet')
        r = p.add_run(b)
        r.font.size = Pt(10)

    # Save documents
    out_dir = Path("outputs")
    doc_path1 = out_dir / "Hinglish_IR_Results_and_Input_Data.docx"
    doc_path2 = Path("Hinglish_IR_Results_and_Input_Data.docx")
    doc.save(str(doc_path1))
    doc.save(str(doc_path2))
    print(f"Document successfully created at:")
    print(f"  1. {doc_path1.resolve()}")
    print(f"  2. {doc_path2.resolve()}")

if __name__ == "__main__":
    create_document()

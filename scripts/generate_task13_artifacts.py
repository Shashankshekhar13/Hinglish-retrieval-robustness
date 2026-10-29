import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Scientific IEEE single-column configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colorblind-safe palette (Okabe-Ito inspired)
COLORS = {
    "switch": "#0072B2",     # Blue
    "spelling": "#D55E00",   # Vermillion
    "en_typo": "#CC79A7",    # Reddish purple
    "key": "#009E73",        # Bluish green
    "none": "#E69F00",       # Amber / Orange
    "stop": "#56B4E9",       # Sky blue
    "exact": "#F0E442",      # Yellow
    "translit": "#999999",   # Gray
    "vocabexp": "#D55E00",   # Vermillion
    "bm25": "#0072B2",       # Blue
    "minilm": "#D55E00",     # Vermillion
    "e5small": "#009E73",    # Bluish green
    "bgem3": "#CC79A7",      # Reddish purple
}

RET_ORDER = ["bm25", "minilm", "e5small", "bgem3"]
RET_LABELS = {
    "bm25": "BM25",
    "minilm": "MiniLM",
    "e5small": "E5-small",
    "bgem3": "BGE-M3",
    "minilm_vocabexp": "MiniLM-VocabExp",
}
DS_ORDER = ["arguana", "nfcorpus", "scifact", "fiqa", "scidocs"]
DS_LABELS = {
    "arguana": "ArguAna",
    "nfcorpus": "NFCorpus",
    "scifact": "SciFact",
    "fiqa": "FiQA",
    "scidocs": "SCIDOCS",
}


# ==========================================
# 1. LaTeX Tables
# ==========================================

def make_table_stats(tables_dir: Path):
    """table_stats.tex: Dataset, domain, corpus docs, evaluated queries, verified queries."""
    corpus_counts = {
        "arguana": 8674,
        "nfcorpus": 3633,
        "scifact": 5183,
        "fiqa": 57638,
        "scidocs": 25657,
    }
    domains = {
        "arguana": "Argumentation / Debate",
        "nfcorpus": "Clinical / Nutrition",
        "scifact": "Scientific Claims",
        "fiqa": "Financial QA",
        "scidocs": "Scientific Literature",
    }
    verified_counts = {
        "arguana": 40,
        "nfcorpus": 40,
        "scifact": 40,
        "fiqa": 0,
        "scidocs": 0,
    }

    # Load evaluated query counts from generation_stats.json
    gen_stats_path = tables_dir / "generation_stats.json"
    if gen_stats_path.exists():
        gen_stats = json.loads(gen_stats_path.read_text(encoding="utf-8"))
        eval_queries = {ds: gen_stats[ds]["ok"] for ds in DS_ORDER}
    else:
        eval_queries = {}
        for ds in DS_ORDER:
            p = Path(f"data/queries/{ds}.jsonl")
            lines = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            eval_queries[ds] = sum(1 for l in lines if l.get("gen_status") == "ok")

    ann_stats_path = tables_dir / "annotation_stats.json"
    ann_summary_str = ""
    if ann_stats_path.exists():
        ann_stats = json.loads(ann_stats_path.read_text(encoding="utf-8"))
        ann_summary_str = f" ({ann_stats.get('natural_pct', 97.5):.1f}\\% natural, {ann_stats.get('meaning_pct', 97.5):.1f}\\% meaning-preserving)"

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dataset statistics across the five evaluated benchmark collections, including domain, corpus size, evaluated query count, and verified query subset size.}",
        r"\label{tab:dataset_stats}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{ll ccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Domain} & \textbf{Corpus Docs} & \textbf{Evaluated Queries} & \textbf{Verified Queries\footnotemark} \\",
        r"\midrule",
    ]

    total_docs = sum(corpus_counts.values())
    total_eval = sum(eval_queries.values())
    total_ver = sum(verified_counts.values())

    for ds in DS_ORDER:
        tex.append(
            f"{DS_LABELS[ds]} & {domains[ds]} & {corpus_counts[ds]:,} & {eval_queries[ds]:,} & {verified_counts[ds]} \\\\"
        )

    tex.append(r"\midrule")
    tex.append(f"\\textbf{{Total / Overall}} & -- & {total_docs:,} & {total_eval:,} & {total_ver} \\\\")
    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\vspace{1mm}",
        rf"\footnotesize{{\footnotetext{{$^1$The verified query subset contains 120 expert-annotated queries across ArguAna, NFCorpus, and SciFact{ann_summary_str}.}}}}",
        r"\end{table}",
    ])

    out_path = tables_dir / "table_stats.tex"
    out_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Generated {out_path}")


def make_table_main(results_df: pd.DataFrame, out_dir: Path):
    """table_main.tex: nDCG@10 across all 5 datasets + Mean for unnormalized conditions."""
    conds = ["EN", "CS-clean", "CS-noise25", "CS-noise50"]
    retrievers = ["bm25", "minilm", "e5small", "bgem3", "minilm_vocabexp"]
    none_df = results_df[results_df["normalizer"] == "none"].copy()
    lookup = none_df.set_index(["dataset", "retriever", "condition"])["ndcg@10"].to_dict()

    # Pre-calculate means across 5 datasets
    mean_lookup = {}
    for ret in retrievers:
        for cond in conds:
            vals = [lookup.get((ds, ret, cond), np.nan) for ds in DS_ORDER]
            valid_vals = [v for v in vals if pd.notna(v)]
            mean_lookup[(ret, cond)] = np.mean(valid_vals) if valid_vals else np.nan

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Retrieval performance (nDCG@10) across five benchmark datasets, models, and query noise conditions under the unnormalized condition.}",
        r"\label{tab:main_results}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l cccc cccc cccc cccc cccc cccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\textbf{ArguAna}} & \multicolumn{4}{c}{\textbf{NFCorpus}} & \multicolumn{4}{c}{\textbf{SciFact}} & \multicolumn{4}{c}{\textbf{FiQA}} & \multicolumn{4}{c}{\textbf{SCIDOCS}} & \multicolumn{4}{c}{\textbf{Mean}} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13} \cmidrule(lr){14-17} \cmidrule(lr){18-21} \cmidrule(lr){22-25}",
        r"\textbf{Retriever} & \textbf{EN} & \textbf{CS-clean} & \textbf{CS-n25} & \textbf{CS-n50} & \textbf{EN} & \textbf{CS-clean} & \textbf{CS-n25} & \textbf{CS-n50} & \textbf{EN} & \textbf{CS-clean} & \textbf{CS-n25} & \textbf{CS-n50} & \textbf{EN} & \textbf{CS-clean} & \textbf{CS-n25} & \textbf{CS-n50} & \textbf{EN} & \textbf{CS-clean} & \textbf{CS-n25} & \textbf{CS-n50} & \textbf{EN} & \textbf{CS-clean} & \textbf{CS-n25} & \textbf{CS-n50} \\",
        r"\midrule",
    ]

    for ret in retrievers:
        row_str = [f"{RET_LABELS.get(ret, ret)}"]
        for ds in DS_ORDER:
            for cond in conds:
                val = lookup.get((ds, ret, cond), np.nan)
                row_str.append(f"{val:.4f}" if pd.notna(val) else "--")
        for cond in conds:
            val = mean_lookup.get((ret, cond), np.nan)
            row_str.append(f"{val:.4f}" if pd.notna(val) else "--")
        tex.append(" & ".join(row_str) + r" \\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\vspace{1mm}",
        r"\footnotesize{\footnotetext{All values represent nDCG@10 scores under the unnormalized query condition.}}",
        r"\end{table*}",
    ])

    out_path = out_dir / "table_main.tex"
    out_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Generated {out_path}")


def make_table_decomp(decomp_df: pd.DataFrame, sig_df: pd.DataFrame, out_dir: Path):
    """table_decomp.tex: Full decomposition table for all datasets + Mean with significance marks."""
    sig_lookup = {}
    for _, r in sig_df.iterrows():
        sig_lookup[(str(r["dataset"]), str(r["retriever"]), str(r["test_id"]))] = bool(r["significant_alpha_0_05"])

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Empirical decomposition of retrieval degradation across five benchmark datasets, isolating cross-lingual switching drop from synthetic spelling degradation and normalizer gains.}",
        r"\label{tab:decomposition}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{ll ccccccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Retriever} & \textbf{EN} & $\Delta_{\text{switch}}$ & $\Delta_{\text{spell50}}$ & $\Delta_{\text{typo50}}$ & \textbf{Gain}$_{\text{stop}}$ & \textbf{Gain}$_{\text{key/stop}}$ & $\Delta_{\text{key50}}$ \\",
        r"\midrule",
    ]

    for ds in DS_ORDER:
        first = True
        for ret in RET_ORDER:
            m = decomp_df[(decomp_df["dataset"] == ds) & (decomp_df["retriever"] == ret)]
            if len(m) == 0:
                continue
            r = m.iloc[0]
            ds_str = DS_LABELS[ds] if first else ""
            first = False

            # Check exploratory significance per dataset
            sig_switch = r"$^{\dagger}$" if sig_lookup.get((ds, ret, "a"), False) else ""
            sig_spell = r"$^{\dagger}$" if sig_lookup.get((ds, ret, "b2"), False) else ""
            sig_stop = r"$^{\dagger}$" if sig_lookup.get((ds, ret, "c"), False) else ""
            sig_key_stop = r"$^{\dagger}$" if sig_lookup.get((ds, ret, "d"), False) else ""

            tex.append(
                f"{ds_str} & {RET_LABELS[ret]} & {r['EN']:.4f} & {r['switch_drop']:+.4f}{sig_switch} & {r['spell_drop_50']:+.4f}{sig_spell} & {r['EN_typo_drop_50']:+.4f} & {r['stop_gain_50']:+.4f}{sig_stop} & {r['key_over_stop_50']:+.4f}{sig_key_stop} & {r['key_noise_invariance']:+.4f} \\\\"
            )
        tex.append(r"\midrule")

    # Mean section
    first = True
    for ret in RET_ORDER:
        m = decomp_df[(decomp_df["dataset"] == "MEAN") & (decomp_df["retriever"] == ret)]
        if len(m) == 0:
            continue
        r = m.iloc[0]
        ds_str = r"\textbf{Mean}" if first else ""
        first = False

        # Check primary family significance for pooled Mean
        sig_switch = r"$^{\dagger}$" if sig_lookup.get(("pooled", ret, "a"), False) else ""
        sig_spell = r"$^{\dagger}$" if sig_lookup.get(("pooled", ret, "b2"), False) else ""
        sig_stop = r"$^{\dagger}$" if sig_lookup.get(("pooled", ret, "c"), False) else ""
        sig_key_stop = r"$^{\dagger}$" if sig_lookup.get(("pooled", ret, "d"), False) else ""

        tex.append(
            f"{ds_str} & {RET_LABELS[ret]} & {r['EN']:.4f} & {r['switch_drop']:+.4f}{sig_switch} & {r['spell_drop_50']:+.4f}{sig_spell} & {r['EN_typo_drop_50']:+.4f} & {r['stop_gain_50']:+.4f}{sig_stop} & {r['key_over_stop_50']:+.4f}{sig_key_stop} & {r['key_noise_invariance']:+.4f} \\\\"
        )

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\vspace{1mm}",
        r"\footnotesize{\footnotetext{$\Delta_{\text{switch}} = \text{EN} - \text{CS}_{\text{clean}}$; $\Delta_{\text{spell50}} = \text{CS}_{\text{clean}} - \text{CS}_{n50}$; $\Delta_{\text{typo50}} = \text{EN} - \text{EN}_{n50}$; $\text{Gain}_{\text{stop}} = \text{CS}_{n50}(\text{stop}) - \text{CS}_{n50}(\text{none})$; $\text{Gain}_{\text{key/stop}} = \text{CS}_{n50}(\text{key}) - \text{CS}_{n50}(\text{stop})$; $\Delta_{\text{key50}} = \text{CS}_{\text{clean}}(\text{key}) - \text{CS}_{n50}(\text{key})$. $^{\dagger}$Statistically significant effect at $\alpha=0.05$ (Holm-corrected across the primary family for pooled results; permutation test for per-dataset results).}}",
        r"\end{table*}",
    ])

    out_path = out_dir / "table_decomp.tex"
    out_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Generated {out_path}")


def make_table_ladder(results_df: pd.DataFrame, sig_df: pd.DataFrame, out_dir: Path):
    """table_ladder.tex: Methods ladder across all 5 datasets + Mean on CS-clean and CS-noise50."""
    norm_ladder = ["none", "stop", "exact", "key", "key_expand", "translit"]
    retrievers = ["bm25", "minilm", "e5small", "bgem3", "minilm_vocabexp"]

    lookup = results_df.set_index(["dataset", "retriever", "condition", "normalizer"])["ndcg@10"].to_dict()

    # Significance lookup
    sig_lookup = {}
    for _, r in sig_df.iterrows():
        sig_lookup[(str(r["dataset"]), str(r["retriever"]), str(r["test_id"]))] = bool(r["significant_alpha_0_05"])

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comparison of normalization methods on code-switched queries under CS-clean and CS-noise50 conditions across all five benchmark datasets (nDCG@10).}",
        r"\label{tab:methods_ladder}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{ll cccccc cccccc}",
        r"\toprule",
        r"& & \multicolumn{6}{c}{\textbf{CS-clean}} & \multicolumn{6}{c}{\textbf{CS-noise50}} \\",
        r"\cmidrule(lr){3-8} \cmidrule(lr){9-14}",
        r"\textbf{Retriever} & \textbf{Normalizer} & \textbf{ArguAna} & \textbf{NFCorpus} & \textbf{SciFact} & \textbf{FiQA} & \textbf{SCIDOCS} & \textbf{Mean} & \textbf{ArguAna} & \textbf{NFCorpus} & \textbf{SciFact} & \textbf{FiQA} & \textbf{SCIDOCS} & \textbf{Mean} \\",
        r"\midrule",
    ]

    for ret in retrievers:
        first = True
        applicable_norms = norm_ladder if ret != "minilm_vocabexp" else ["vocabexp"]
        for norm in applicable_norms:
            r_name = RET_LABELS.get(ret, ret) if first else ""
            first = False
            row_items = [r_name, norm]

            actual_ret = ret
            actual_norm = norm if ret != "minilm_vocabexp" else "none"

            # CS-clean
            clean_vals = []
            for ds in DS_ORDER:
                val = lookup.get((ds, actual_ret, "CS-clean", actual_norm), np.nan)
                clean_vals.append(val)
                row_items.append(f"{val:.4f}" if pd.notna(val) else "--")
            clean_mean = np.nanmean(clean_vals) if any(pd.notna(v) for v in clean_vals) else np.nan
            row_items.append(f"{clean_mean:.4f}" if pd.notna(clean_mean) else "--")

            # CS-noise50
            n50_vals = []
            for ds in DS_ORDER:
                val = lookup.get((ds, actual_ret, "CS-noise50", actual_norm), np.nan)
                n50_vals.append(val)
                if pd.notna(val):
                    is_sig = False
                    if norm == "key":
                        is_sig = sig_lookup.get((ds, ret, "e"), False)
                    elif norm == "stop":
                        is_sig = sig_lookup.get((ds, ret, "c"), False)
                    mark = r"$^{\dagger}$" if is_sig else ""
                    row_items.append(f"{val:.4f}{mark}")
                else:
                    row_items.append("--")
            n50_mean = np.nanmean(n50_vals) if any(pd.notna(v) for v in n50_vals) else np.nan
            is_pooled_sig = False
            if norm == "key":
                is_pooled_sig = sig_lookup.get(("pooled", ret, "e"), False)
            elif norm == "stop":
                is_pooled_sig = sig_lookup.get(("pooled", ret, "c"), False)
            mark_pooled = r"$^{\dagger}$" if is_pooled_sig else ""
            row_items.append(f"{n50_mean:.4f}{mark_pooled}" if pd.notna(n50_mean) else "--")

            tex.append(" & ".join(row_items) + r" \\")
        if ret != retrievers[-1]:
            tex.append(r"\midrule")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\vspace{1mm}",
        r"\footnotesize{\footnotetext{$^{\dagger}$Statistically significant difference versus the unnormalized baseline (none) at $\alpha=0.05$ (Holm-corrected across the primary family for pooled results; unadjusted permutation test for per-dataset results).}}",
        r"\end{table*}",
    ])

    out_path = out_dir / "table_ladder.tex"
    out_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Generated {out_path}")


# ==========================================
# 2. Vector Figures
# ==========================================

def make_fig_decomposition(decomp_df: pd.DataFrame, boot_df: pd.DataFrame, out_dir: Path):
    """fig_decomposition.pdf: Single-column (3.5 in) stacked bar chart with CI error bars."""
    mean_df = decomp_df[decomp_df["dataset"] == "MEAN"].set_index("retriever").loc[RET_ORDER].reset_index()

    switch_drops = mean_df["switch_drop"].values
    spell_drops = mean_df["spell_drop_50"].values
    en_typo_drops = mean_df["EN_typo_drop_50"].values

    # Bootstrap 95% CIs for error bars (1.96 * SE)
    boot_indexed = boot_df.set_index("retriever").loc[RET_ORDER].reset_index()
    switch_err = 1.96 * boot_indexed["bootstrap_se"].values

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(RET_ORDER))
    width = 0.44

    # Stacked bars: switch_drop on bottom, spell_drop on top
    ax.bar(x, switch_drops, width, label="Switching drop",
           color=COLORS["switch"], alpha=0.92, edgecolor="white", linewidth=0.5,
           yerr=switch_err, capsize=2.5, error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#222222"})
    ax.bar(x, spell_drops, width, bottom=switch_drops, label="Spelling drop (η=50%)",
           color=COLORS["spelling"], alpha=0.92, edgecolor="white", linewidth=0.5)

    # Control point for EN typo drop
    ax.scatter(x, en_typo_drops, color=COLORS["en_typo"], marker="D", s=30, zorder=5,
               edgecolors="white", linewidths=0.6, label="EN typo control")

    ax.set_xticks(x)
    ax.set_xticklabels([RET_LABELS[r] for r in RET_ORDER], fontweight="bold")
    ax.set_ylabel("nDCG@10 Degradation", fontweight="bold")
    ax.axhline(0, color="#333333", linewidth=0.6, linestyle="--")
    ax.grid(axis="y", linestyle=":", alpha=0.5, color="#888888")
    ax.set_ylim(-0.015, 0.155)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#444444")

    # Clean, compact top horizontal legend with concise labels
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=3, fontsize=6.2, frameon=True, edgecolor="#cccccc", facecolor="white",
                    framealpha=0.95, columnspacing=0.8, handletextpad=0.3, borderpad=0.25)
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()

    out_pdf = out_dir / "fig_decomposition.pdf"
    out_png = out_dir / "fig_decomposition.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_pdf} and {out_png}")


def make_fig_noise_curve(results_df: pd.DataFrame, out_dir: Path):
    """fig_noise_curve: 4-panel subplots across noise levels."""
    noise_levels = [0, 25, 50]
    cond_map = {0: "CS-clean", 25: "CS-noise25", 50: "CS-noise50"}

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)

    for i, ret in enumerate(RET_ORDER):
        ax = axes[i]
        means_none = []
        means_key = []

        for lvl in noise_levels:
            cond = cond_map[lvl]
            vals_none = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == cond) & (results_df["normalizer"] == "none")]["ndcg@10"].values
            vals_key = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == cond) & (results_df["normalizer"] == "key")]["ndcg@10"].values
            means_none.append(np.mean(vals_none))
            means_key.append(np.mean(vals_key))

        ax.plot(noise_levels, means_none, marker="o", markersize=5.0, markeredgewidth=1.0, markeredgecolor="white",
                linestyle="-", linewidth=1.6, color=COLORS["none"], label="none (unnormalized)", zorder=4)
        ax.plot(noise_levels, means_key, marker="s", markersize=5.0, markeredgewidth=1.0, markeredgecolor="white",
                linestyle="--", linewidth=1.6, color=COLORS["key"], label="key (phonetic normalized)", zorder=4)

        raw_spell_drop = means_none[0] - means_none[2]
        key_drop = means_key[0] - means_key[2]

        title_text = f"{RET_LABELS[ret]}\nraw drop: {raw_spell_drop:+.3f}\nkey drop: {key_drop:+.3f}"
        ax.set_title(title_text, fontsize=7.5, linespacing=1.2, pad=6, fontweight="bold")

        ax.set_xlabel("Spelling Noise", fontsize=7.5)
        if i == 0:
            ax.set_ylabel("Mean nDCG@10", fontsize=7.5, fontweight="bold")
        ax.set_xticks(noise_levels)
        ax.set_xticklabels(["0%", "25%", "50%"], fontsize=7.0)
        ax.set_ylim(0.27, 0.44)
        ax.grid(True, linestyle=":", alpha=0.55, color="#aaaaaa", zorder=1)

        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#444444")

    # Unified top legend
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
                     ncol=2, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95, fontsize=7.2,
                     columnspacing=1.2, handletextpad=0.4, borderpad=0.25)
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    fig.subplots_adjust(top=0.74, wspace=0.15)

    out_pdf = out_dir / "fig_noise_curve.pdf"
    out_png = out_dir / "fig_noise_curve.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_pdf} and {out_png}")


def make_fig_normalizer_effects(results_df: pd.DataFrame, out_dir: Path):
    """fig_normalizer_effects: Forest plot showing normalizer effect sizes (gain vs none) at CS-noise50 with 95% CIs."""
    normalizers = ["stop", "exact", "key", "key_expand", "translit"]
    norm_labels = {
        "stop": "Stopword Removal",
        "exact": "Exact Lexicon",
        "key": "PhonoNorm Key (Ours)",
        "key_expand": "Key + Gloss Expansion",
        "translit": "Devanagari Translit.",
    }

    fig, ax = plt.subplots(figsize=(3.4, 3.7))
    group_centers = [0.0, 3.0, 6.0, 9.0, 12.0]

    # Subtle highlight band for PhonoNorm Key (Ours)
    ax.axhspan(4.5, 7.5, facecolor="#EBF5EE", alpha=0.8, zorder=0, edgecolor="none")

    # Reference zero line
    ax.axvline(0, color="#333333", linestyle="--", linewidth=0.7, alpha=0.85, zorder=2)

    for i, norm in enumerate(normalizers):
        yc = group_centers[i]
        if norm == "translit":
            ret_offsets = {"e5small": -0.35, "bgem3": 0.35}
        else:
            ret_offsets = {"bm25": -1.05, "minilm": -0.35, "e5small": 0.35, "bgem3": 1.05}

        for ret in RET_ORDER:
            if ret not in ret_offsets:
                continue
            deltas = []
            for ds in DS_ORDER:
                v_norm = results_df[(results_df["dataset"] == ds) & (results_df["retriever"] == ret) & (results_df["condition"] == "CS-noise50") & (results_df["normalizer"] == norm)]
                v_none = results_df[(results_df["dataset"] == ds) & (results_df["retriever"] == ret) & (results_df["condition"] == "CS-noise50") & (results_df["normalizer"] == "none")]
                if len(v_norm) > 0 and len(v_none) > 0:
                    deltas.append(v_norm["ndcg@10"].iloc[0] - v_none["ndcg@10"].iloc[0])

            if deltas:
                mean_delta = np.mean(deltas)
                se = np.std(deltas, ddof=1) / np.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
                ci_95 = 1.96 * se
                y_pos = yc + ret_offsets[ret]

                ax.errorbar(mean_delta, y_pos, xerr=ci_95, fmt="o", markersize=4.2,
                            color=COLORS[ret], ecolor=COLORS[ret], elinewidth=1.1, capsize=2.2,
                            markeredgecolor="white", markeredgewidth=0.6,
                            label=RET_LABELS[ret] if norm == "stop" else "", zorder=4)

    ax.set_yticks(group_centers)
    ax.set_yticklabels([norm_labels[n] for n in normalizers], fontsize=7.2, fontweight="medium")

    ax.set_xlabel(r"$\Delta$ nDCG@10 vs. unnormalized (noise 50%)", fontsize=7.8, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5, color="#888888", zorder=1)
    ax.set_xlim(-0.048, 0.065)
    ax.set_ylim(-1.8, 13.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#444444")

    # Clean top legend anchored directly to ax (zero overlap)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=4, fontsize=6.5, frameon=True, edgecolor="#cccccc", facecolor="white",
                    framealpha=0.95, columnspacing=0.8, handletextpad=0.25, borderpad=0.25)
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()

    out_pdf = out_dir / "fig_normalizer_effects.pdf"
    out_png = out_dir / "fig_normalizer_effects.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_pdf} and {out_png}")


def make_fig_methods_ladder(results_df: pd.DataFrame, out_dir: Path):
    """fig_methods_ladder.pdf: 4-panel grouped bar chart replacing the large 14-column table."""
    retrievers = ["bm25", "minilm", "e5small", "bgem3"]
    norm_labels = {
        "none": "Raw",
        "stop": "+Stop",
        "exact": "+Exact",
        "key": "PhonoNorm",
        "key_expand": "+Gloss",
        "translit": "Translit",
    }

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=True)

    for i, ret in enumerate(retrievers):
        ax = axes[i]
        norms_for_ret = ["none", "stop", "exact", "key", "key_expand"]
        if ret in ["e5small", "bgem3"]:
            norms_for_ret.append("translit")
        
        clean_scores = []
        noise_scores = []
        for n in norms_for_ret:
            c_val = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-clean") & (results_df["normalizer"] == n)]["ndcg@10"].mean()
            n_val = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-noise50") & (results_df["normalizer"] == n)]["ndcg@10"].mean()
            clean_scores.append(c_val)
            noise_scores.append(n_val)

        x = np.arange(len(norms_for_ret))
        width = 0.38

        b1 = ax.bar(x - width/2, clean_scores, width, label="CS-clean",
                    color=COLORS["switch"], alpha=0.90, edgecolor="white", linewidth=0.5)
        b2 = ax.bar(x + width/2, noise_scores, width, label="CS-noise50",
                    color=COLORS["spelling"], alpha=0.90, edgecolor="white", linewidth=0.5)

        # Highlight PhonoNorm bar
        ax.bar(3 - width/2, clean_scores[3], width, color=COLORS["switch"], edgecolor="#0F766E", linewidth=1.2)
        ax.bar(3 + width/2, noise_scores[3], width, color=COLORS["spelling"], edgecolor="#0F766E", linewidth=1.2)

        ax.set_title(RET_LABELS[ret], fontsize=8.0, fontweight="bold", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([norm_labels[n] for n in norms_for_ret], rotation=40, ha="right", fontsize=6.5)

        # Highlight PhonoNorm label in bold green
        labels = ax.get_xticklabels()
        if len(labels) > 3:
            labels[3].set_fontweight("bold")
            labels[3].set_color("#0F766E")

        ax.set_ylim(0.25, 0.43)
        ax.set_yticks([0.25, 0.30, 0.35, 0.40])
        ax.grid(axis="y", linestyle=":", alpha=0.55, color="#aaaaaa")

        if i == 0:
            ax.set_ylabel("Mean nDCG@10", fontsize=7.8, fontweight="bold")

        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#444444")

    # Unified top legend
    handles = [b1, b2]
    labels = ["CS-clean", "CS-noise50"]
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05),
                     ncol=2, frameon=True, facecolor="white", edgecolor="#cccccc",
                     framealpha=0.95, fontsize=7.2, columnspacing=1.5, handletextpad=0.4, borderpad=0.25)
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    fig.subplots_adjust(top=0.76, wspace=0.15)
    
    out_pdf = out_dir / "fig_methods_ladder.pdf"
    out_png = out_dir / "fig_methods_ladder.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_pdf} and {out_png}")


def make_table_ladder_compact(results_df: pd.DataFrame, sig_df: pd.DataFrame, out_dir: Path):
    """table_ladder_compact.tex: Single-column IEEE table (6 columns) summarizing the normalization ladder."""
    sig_lookup = {}
    for _, r in sig_df.iterrows():
        sig_lookup[(str(r["dataset"]), str(r["retriever"]), str(r["test_id"]))] = bool(r["significant_alpha_0_05"])

    retrievers = ["bm25", "minilm", "e5small", "bgem3"]
    norm_ladder = ["none", "stop", "exact", "key", "key_expand", "translit"]
    norm_display = {
        "none": "Raw (none)",
        "stop": "+Stopwords",
        "exact": "+Exact Lexicon",
        "key": "PhonoNorm (Ours)",
        "key_expand": "+Gloss Expansion",
        "translit": "Devanagari Translit.",
    }

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Normalization Ladder: Mean nDCG@10 and gain ($\Delta$) over unnormalized baseline across all 5 benchmark collections ($^{\dagger}p < 0.05$).}",
        r"\label{tab:methods_ladder_compact}",
        r"\begin{tabular}{ll cc cc}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{CS-clean}} & \multicolumn{2}{c}{\textbf{CS-noise50}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"\textbf{Retriever} & \textbf{Normalizer} & \textbf{nDCG} & $\mathbf{\Delta}$ & \textbf{nDCG} & $\mathbf{\Delta}$ \\",
        r"\midrule",
    ]

    for ret in retrievers:
        first = True
        base_clean = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-clean") & (results_df["normalizer"] == "none")]["ndcg@10"].mean()
        base_noise = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-noise50") & (results_df["normalizer"] == "none")]["ndcg@10"].mean()

        applicable_norms = [n for n in norm_ladder if (ret in ["e5small", "bgem3"] or n != "translit")]
        for norm in applicable_norms:
            r_name = RET_LABELS[ret] if first else ""
            first = False
            n_name = norm_display[norm]

            c_sub = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-clean") & (results_df["normalizer"] == norm)]
            n_sub = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-noise50") & (results_df["normalizer"] == norm)]

            c_mean = c_sub["ndcg@10"].mean() if len(c_sub) > 0 else np.nan
            n_mean = n_sub["ndcg@10"].mean() if len(n_sub) > 0 else np.nan

            c_delta = c_mean - base_clean if pd.notna(c_mean) and norm != "none" else 0.0
            n_delta = n_mean - base_noise if pd.notna(n_mean) and norm != "none" else 0.0

            is_sig = False
            if norm == "key":
                is_sig = sig_lookup.get(("pooled", ret, "e"), False)
            elif norm == "stop":
                is_sig = sig_lookup.get(("pooled", ret, "c"), False)
            mark = r"$^{\dagger}$" if is_sig else ""

            c_str = f"{c_mean:.4f}" if pd.notna(c_mean) else "--"
            c_d_str = f"{c_delta:+.3f}" if norm != "none" else "--"

            n_str = f"{n_mean:.4f}{mark}" if pd.notna(n_mean) else "--"
            n_d_str = f"{n_delta:+.3f}" if norm != "none" else "--"

            if norm == "key":
                n_name = r"\textbf{" + n_name + r"}"

            tex.append(f"{r_name} & {n_name} & {c_str} & {c_d_str} & {n_str} & {n_d_str} \\\\")
        if ret != retrievers[-1]:
            tex.append(r"\midrule")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_file = out_dir / "table_ladder_compact.tex"
    out_file.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Generated {out_file}")


def make_fig_methods_gain_ladder(results_df: pd.DataFrame, out_dir: Path):
    """fig_methods_gain_ladder.pdf: High-impact 4-panel gain spectrum with 2-decimal numbers and uniform text."""
    retrievers = ["bm25", "minilm", "e5small", "bgem3"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.75), sharey=True)

    norm_names = {"stop": "Stop", "exact": "Exact", "key": "PhonoNorm", "translit": "Translit"}
    C_CLEAN = "#1E40AF"
    C_NOISE = "#D97706"
    TEXT_COLOR = "#111111"

    def fmt_gain(val):
        if pd.isna(val):
            return ""
        if abs(val) < 0.005:
            return "0.00"
        return f"{val:+.2f}"

    for i, ret in enumerate(retrievers):
        ax = axes[i]
        cur_norms = ["stop", "exact", "key"]
        if ret in ["e5small", "bgem3"]:
            cur_norms.append("translit")

        base_clean = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-clean") & (results_df["normalizer"] == "none")]["ndcg@10"].mean()
        base_noise = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-noise50") & (results_df["normalizer"] == "none")]["ndcg@10"].mean()

        clean_deltas = []
        noise_deltas = []

        for n in cur_norms:
            c_sub = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-clean") & (results_df["normalizer"] == n)]
            n_sub = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-noise50") & (results_df["normalizer"] == n)]
            clean_deltas.append(c_sub["ndcg@10"].mean() - base_clean if len(c_sub) > 0 else np.nan)
            noise_deltas.append(n_sub["ndcg@10"].mean() - base_noise if len(n_sub) > 0 else np.nan)

        x = np.arange(len(cur_norms))
        width = 0.36

        # Subtle light gray highlight band behind PhonoNorm column (index 2)
        ax.axvspan(2 - 0.46, 2 + 0.46, facecolor="#F1F5F9", alpha=0.9, zorder=0, edgecolor="none")

        # Bars: Consistent clean/noise colors across all methods!
        ax.bar(x - width/2, clean_deltas, width, label="CS-clean Gain",
               color=C_CLEAN, alpha=0.90, edgecolor="white", linewidth=0.5, zorder=3)
        ax.bar(x + width/2, noise_deltas, width, label="CS-noise50 Gain",
               color=C_NOISE, alpha=0.92, edgecolor="white", linewidth=0.5, zorder=3)

        # Highlight outline on PhonoNorm bars (same fill color, crisp border)
        ax.bar(2 - width/2, clean_deltas[2], width, color=C_CLEAN, edgecolor="#0F172A", linewidth=1.3, zorder=4)
        ax.bar(2 + width/2, noise_deltas[2], width, color=C_NOISE, edgecolor="#7C2D12", linewidth=1.3, zorder=4)

        # Reference zero line
        ax.axhline(0, color="#333333", linestyle="--", linewidth=0.7, alpha=0.85, zorder=2)

        # ALL TEXT IN SAME COLOR (#111111), 2 decimal places
        for idx, (cd, nd) in enumerate(zip(clean_deltas, noise_deltas)):
            if pd.notna(cd):
                y_pos = cd + (0.0032 if cd >= 0 else -0.0055)
                va = "bottom" if cd >= 0 else "top"
                ax.text(idx - width/2, y_pos, fmt_gain(cd), ha="center", va=va, fontsize=6.3,
                        fontweight="bold" if idx == 2 else "normal", color=TEXT_COLOR, zorder=5)
            if pd.notna(nd):
                y_pos = nd + (0.0032 if nd >= 0 else -0.0055)
                va = "bottom" if nd >= 0 else "top"
                ax.text(idx + width/2, y_pos, fmt_gain(nd), ha="center", va=va, fontsize=6.3,
                        fontweight="bold" if idx == 2 else "normal", color=TEXT_COLOR, zorder=5)

        ax.set_title(RET_LABELS[ret], fontsize=8.4, fontweight="bold", pad=8, color=TEXT_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels([norm_names[n] for n in cur_norms], rotation=20, ha="right", fontsize=7.2, color=TEXT_COLOR)

        t_labels = ax.get_xticklabels()
        t_labels[2].set_fontweight("bold")
        t_labels[2].set_color(TEXT_COLOR)

        ax.set_ylim(-0.028, 0.066)
        ax.set_yticks([-0.02, 0.00, 0.02, 0.04, 0.06])
        ax.set_yticklabels(["-0.02", " 0.00", "+0.02", "+0.04", "+0.06"], fontsize=6.8)
        ax.grid(axis="y", linestyle=":", alpha=0.5, color="#888888", zorder=1)

        if i == 0:
            ax.set_ylabel(r"$\Delta$ nDCG@10 vs. Raw", fontsize=8.0, fontweight="bold", color=TEXT_COLOR)

        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#444444")

    # Simple 2-item top legend with consistent colors
    handles = [
        patches.Patch(facecolor=C_CLEAN, edgecolor="white", label="CS-clean Gain"),
        patches.Patch(facecolor=C_NOISE, edgecolor="white", label="CS-noise50 Gain"),
    ]
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05),
                     ncol=2, frameon=True, facecolor="white", edgecolor="#cccccc",
                     framealpha=0.95, fontsize=7.2, columnspacing=1.5, handletextpad=0.4, borderpad=0.25)
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    fig.subplots_adjust(top=0.76, bottom=0.18, wspace=0.15)
    
    out_pdf = out_dir / "fig_methods_gain_ladder.pdf"
    out_png = out_dir / "fig_methods_gain_ladder.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_pdf} and {out_png}")


def make_fig_methods_recovery_gap(results_df: pd.DataFrame, out_dir: Path):
    """fig_methods_recovery_gap.pdf: High-impact concise 3-bar comparison showing noise gap recovery."""
    retrievers = ["bm25", "minilm", "e5small", "bgem3"]
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    x = np.arange(len(retrievers))
    w = 0.25

    raw_noisy = []
    phono_noisy = []
    clean_bound = []

    for ret in retrievers:
        rn = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-noise50") & (results_df["normalizer"] == "none")]["ndcg@10"].mean()
        pn = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-noise50") & (results_df["normalizer"] == "key")]["ndcg@10"].mean()
        cb = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-clean") & (results_df["normalizer"] == "none")]["ndcg@10"].mean()
        raw_noisy.append(rn)
        phono_noisy.append(pn)
        clean_bound.append(cb)

    ax.bar(x - w, raw_noisy, w, label="Raw (CS-noise50)", color="#EF4444", alpha=0.90, edgecolor="white", linewidth=0.5)
    ax.bar(x, phono_noisy, w, label="PhonoNorm (CS-noise50)", color="#0D9488", alpha=0.95, edgecolor="white", linewidth=0.5)
    ax.bar(x + w, clean_bound, w, label="Clean Baseline (CS-clean)", color="#1E40AF", alpha=0.90, edgecolor="white", linewidth=0.5)

    # Clean white numbers inside the bars
    for i in range(len(retrievers)):
        ax.text(x[i] - w, raw_noisy[i] - 0.015, f"{raw_noisy[i]:.3f}", ha="center", va="top", fontsize=6.2, color="white", fontweight="bold")
        ax.text(x[i], phono_noisy[i] - 0.015, f"{phono_noisy[i]:.3f}", ha="center", va="top", fontsize=6.2, color="white", fontweight="bold")
        ax.text(x[i] + w, clean_bound[i] - 0.015, f"{clean_bound[i]:.3f}", ha="center", va="top", fontsize=6.2, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([RET_LABELS[r] for r in retrievers], fontsize=8.2, fontweight="bold")
    ax.set_ylabel("Mean nDCG@10", fontsize=8.0, fontweight="bold")
    ax.set_ylim(0.24, 0.415)
    ax.set_yticks([0.25, 0.30, 0.35, 0.40])
    ax.grid(axis="y", linestyle=":", alpha=0.5, color="#888888")

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#444444")

    # Top legend
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=3, fontsize=6.8, frameon=True, facecolor="white", edgecolor="#cccccc",
                    framealpha=0.95, columnspacing=1.0, handletextpad=0.3, borderpad=0.25)
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    
    out_pdf = out_dir / "fig_methods_recovery_gap.pdf"
    out_png = out_dir / "fig_methods_recovery_gap.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_pdf} and {out_png}")


def make_fig_methods_heatmap(results_df: pd.DataFrame, out_dir: Path):
    """fig_methods_heatmap.pdf: Publication-grade single-column CS-clean gain heatmap with all bold labels."""
    retrievers = ["bm25", "minilm", "e5small", "bgem3"]
    ret_labels = ["BM25", "MiniLM", "E5-small", "BGE-M3"]
    # 4 normalizer methods - without Devanagari Transliteration
    norm_rows = ["stop", "exact", "key", "key_expand"]
    norm_row_labels = ["Stopwords", "Exact Lexicon", "PhonoNorm (Ours)", "Gloss Exp."]

    gain_clean = np.zeros((len(norm_rows), len(retrievers)))

    for c_idx, ret in enumerate(retrievers):
        base_clean = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-clean") & (results_df["normalizer"] == "none")]["ndcg@10"].mean()
        for r_idx, n in enumerate(norm_rows):
            c_sub = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == "CS-clean") & (results_df["normalizer"] == n)]
            gain_clean[r_idx, c_idx] = (c_sub["ndcg@10"].mean() - base_clean)

    # Refined Academic Diverging Palette:
    # Deep Crimson -> Red -> Soft Rose -> Pure White (0.000) -> Soft Blue -> Vibrant Blue -> Royal Blue -> Deep Navy
    cdict_blue = [
        "#991B1B",  # Deep Crimson (-0.025)
        "#DC2626",  # Red (-0.015)
        "#FCA5A5",  # Soft Rose (-0.006)
        "#FFFFFF",  # Pure White (0.000)
        "#BFDBFE",  # Soft Blue (+0.010)
        "#3B82F6",  # Vibrant Blue (+0.025)
        "#1D4ED8",  # Royal Blue (+0.040)
        "#1E3A8A",  # Deep Navy (+0.050)
    ]
    blue_cmap = LinearSegmentedColormap.from_list("academic_blue_red", cdict_blue, N=256)
    norm = TwoSlopeNorm(vmin=-0.025, vcenter=0.000, vmax=0.050)

    # Clean single-column IEEE figure: 4.2 in width
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.set_facecolor("#F8FAFC")
    im = ax.imshow(gain_clean, cmap=blue_cmap, norm=norm, aspect="auto", interpolation="nearest")

    ax.set_title("CS-clean Gain (Δ nDCG@10 vs. Raw)", fontsize=8.4, fontweight="bold", pad=9, color="#111111")
    ax.set_xticks(np.arange(len(retrievers)))
    ax.set_xticklabels(ret_labels, fontweight="bold", fontsize=7.5, color="#111111")

    # Clean white grid lines
    ax.set_xticks(np.arange(-0.5, len(retrievers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(norm_rows), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=3, color="#444444")

    for r in range(len(norm_rows)):
        for c in range(len(retrievers)):
            val = gain_clean[r, c]
            sign = "+" if val > 0 else ""
            txt_color = "#FFFFFF" if (val > 0.022 or val < -0.013) else "#111111"
            fontweight = "bold" if r == 2 else "normal"
            fontsize = 7.2 if r == 2 else 7.0
            ax.text(c, r, f"{sign}{val:.3f}", ha="center", va="center",
                    fontsize=fontsize, color=txt_color, fontweight=fontweight)

    # Highlight PhonoNorm row (index 2) with crisp dark border
    rect = patches.Rectangle((-0.5, 1.5), 4.0, 1.0, fill=False, edgecolor="#0F172A", linewidth=2.0, zorder=5)
    ax.add_patch(rect)

    # ALL row labels BOLD text
    ax.set_yticks(np.arange(len(norm_rows)))
    ax.set_yticklabels(norm_row_labels, fontsize=7.5, color="#111111", fontweight="bold")

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    # Clean vertical colorbar
    cbar_ax = fig.add_axes([0.88, 0.16, 0.025, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Δ nDCG@10 Gain", fontsize=7.2, fontweight="bold", color="#111111")
    cbar.ax.tick_params(labelsize=6.8, color="#444444")
    cbar.set_ticks([-0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    cbar.ax.set_yticklabels(["-0.02", "-0.01", " 0.00", "+0.01", "+0.02", "+0.03", "+0.04", "+0.05"])
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("#333333")

    plt.subplots_adjust(left=0.33, right=0.85, top=0.85, bottom=0.14)
    out_pdf = out_dir / "fig_methods_heatmap.pdf"
    out_png = out_dir / "fig_methods_heatmap.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_pdf} and {out_png}")


def main():
    tables_dir = Path("outputs/tables")
    figures_dir = Path("outputs/figures")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv("outputs/results.csv")
    decomp_df = pd.read_csv("outputs/tables/decomposition.csv")
    sig_df = pd.read_csv("outputs/tables/significance_v2.csv")
    boot_df = pd.read_csv("outputs/tables/bootstrap_spelling_vs_switching.csv")

    # 1. LaTeX Tables
    make_table_stats(tables_dir)
    make_table_main(results_df, tables_dir)
    make_table_decomp(decomp_df, sig_df, tables_dir)
    make_table_ladder(results_df, sig_df, tables_dir)
    make_table_ladder_compact(results_df, sig_df, tables_dir)

    # 2. Vector PDFs
    make_fig_decomposition(decomp_df, boot_df, figures_dir)
    make_fig_noise_curve(results_df, figures_dir)
    make_fig_normalizer_effects(results_df, figures_dir)
    make_fig_methods_ladder(results_df, figures_dir)
    make_fig_methods_gain_ladder(results_df, figures_dir)
    make_fig_methods_recovery_gap(results_df, figures_dir)
    make_fig_methods_heatmap(results_df, figures_dir)


if __name__ == "__main__":
    main()

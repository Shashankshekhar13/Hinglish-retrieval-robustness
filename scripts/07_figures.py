import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scientific IEEE single-column configuration
plt.rcParams.update({
    "font.size": 8,
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 9,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
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
    "none": "#E69F00",       # Orange
    "gray": "#999999",
}


def plot_decomposition(results_df: pd.DataFrame, out_dir: Path):
    """Stacked bar chart of switch_drop vs spell_drop_50, with EN_typo_drop marker."""
    decomp_path = Path("outputs/tables/decomposition.csv")
    if not decomp_path.exists():
        print(f"Decomposition file {decomp_path} not found.")
        return
    df = pd.read_csv(decomp_path)

    # Filter to MEAN rows
    mean_df = df[df["dataset"] == "MEAN"].copy()
    retrievers = ["bm25", "minilm", "e5small", "bgem3"]
    mean_df = mean_df[mean_df["retriever"].isin(retrievers)].set_index("retriever").loc[retrievers].reset_index()

    ret_labels = ["BM25", "MiniLM", "E5-small", "BGE-M3"]
    switch_drops = mean_df["switch_drop"].values
    spell_drops = mean_df["spell_drop_50"].values
    en_typo_drops = mean_df["EN_typo_drop_50"].values

    boot_path = Path("outputs/tables/bootstrap_spelling_vs_switching.csv")
    if boot_path.exists():
        boot_df = pd.read_csv(boot_path).set_index("retriever").loc[retrievers].reset_index()
        switch_err = 1.96 * boot_df["bootstrap_se"].values
    else:
        switch_err = None

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(retrievers))
    width = 0.44

    # Switch drop bar
    ax.bar(x, switch_drops, width, label="Switching drop", color=COLORS["switch"], alpha=0.92,
           edgecolor="white", linewidth=0.5, yerr=switch_err, capsize=2.5,
           error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#222222"})
    # Spell drop stacked on top of switch drop
    ax.bar(x, spell_drops, width, bottom=switch_drops, label="Spelling drop (η=50%)",
           color=COLORS["spelling"], alpha=0.92, edgecolor="white", linewidth=0.5)

    # Overlay marker for EN typo drop control
    ax.scatter(x, en_typo_drops, color=COLORS["en_typo"], marker="D", s=30, zorder=5,
               edgecolors="white", linewidths=0.6, label="EN typo control")

    ax.set_xticks(x)
    ax.set_xticklabels(ret_labels, fontweight="bold")
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

    fig_pdf = out_dir / "fig_decomposition.pdf"
    fig_png = out_dir / "fig_decomposition.png"
    plt.savefig(fig_pdf, bbox_inches="tight")
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_pdf} and {fig_png}")


def plot_noise_curves(results_df: pd.DataFrame, out_dir: Path):
    """nDCG@10 vs noise level (0, 25, 50%) for each retriever, solid=none, dashed=key."""
    retrievers = ["bm25", "minilm", "e5small", "bgem3"]
    ret_labels = {"bm25": "BM25", "minilm": "MiniLM", "e5small": "E5-small", "bgem3": "BGE-M3"}
    noise_levels = [0, 25, 50]
    cond_map = {0: "CS-clean", 25: "CS-noise25", 50: "CS-noise50"}

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)

    for i, ret in enumerate(retrievers):
        ax = axes[i]
        scores_none = []
        scores_key = []

        for lvl in noise_levels:
            cond = cond_map[lvl]
            val_none = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == cond) & (results_df["normalizer"] == "none")]["ndcg@10"].mean()
            val_key = results_df[(results_df["retriever"] == ret) & (results_df["condition"] == cond) & (results_df["normalizer"] == "key")]["ndcg@10"].mean()
            scores_none.append(val_none)
            scores_key.append(val_key)

        # Plot none and key curves
        ax.plot(noise_levels, scores_none, marker="o", markersize=5.0, markeredgewidth=1.0, markeredgecolor="white",
                linestyle="-", linewidth=1.6, color=COLORS["none"], label="none (unnormalized)", zorder=4)
        ax.plot(noise_levels, scores_key, marker="s", markersize=5.0, markeredgewidth=1.0, markeredgecolor="white",
                linestyle="--", linewidth=1.6, color=COLORS["key"], label="key (phonetic normalized)", zorder=4)

        raw_spell_drop = scores_none[0] - scores_none[2]
        key_drop = scores_key[0] - scores_key[2]

        title_text = f"{ret_labels[ret]}\nraw drop: {raw_spell_drop:+.3f}\nkey drop: {key_drop:+.3f}"
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

    fig_pdf = out_dir / "fig_noise_curve.pdf"
    fig_png = out_dir / "fig_noise_curve.png"
    plt.savefig(fig_pdf, bbox_inches="tight")
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_pdf} and {fig_png}")


if __name__ == "__main__":
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.read_csv("outputs/results.csv")
    plot_decomposition(results_df, out_dir)
    plot_noise_curves(results_df, out_dir)
    print("Figures generated successfully.")

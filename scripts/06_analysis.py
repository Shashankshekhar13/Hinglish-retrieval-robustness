import os
import sys
import yaml
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ranx import Qrels, Run, compare
from hcsir.data import load_beir
from hcsir.run_experiments import load_clean_queries


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Applies step-down Holm-Bonferroni correction to a list of p-values."""
    n = len(p_values)
    idx = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    cum_max = 0.0
    for rank, i in enumerate(idx):
        multiplier = n - rank
        val = min(1.0, p_values[i] * multiplier)
        cum_max = max(cum_max, val)
        adjusted[i] = cum_max
    return adjusted


def run_significance_tests():
    print("Loading experiment configuration and datasets...")
    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    datasets = ["arguana", "nfcorpus", "scifact"]

    qrels_by_ds = {}
    for ds_name in datasets:
        ds = load_beir(ds_name, cfg["datasets"][ds_name], cfg["seed"])
        clean = load_clean_queries(ds_name)
        clean_qids = sorted(clean.keys())
        qrels_by_ds[ds_name] = Qrels({qid: ds.qrels[qid] for qid in clean_qids if qid in ds.qrels})

    # Pooled qrels with prefixed QIDs
    pooled_qrels_dict = {}
    for ds_name, qrels in qrels_by_ds.items():
        for qid, docs in qrels.to_dict().items():
            pooled_qrels_dict[f"{ds_name}__{qid}"] = docs
    qrels_by_ds["pooled"] = Qrels(pooled_qrels_dict)

    def load_run(ds_name, ret_name, cond, norm):
        if ds_name != "pooled":
            p = f"outputs/runs/{ds_name}/{ret_name}/{cond}__{norm}.json"
            if not os.path.exists(p):
                return None
            r = Run.from_file(p)
            r.name = f"{cond}__{norm}"
            return r
        else:
            pooled_run_dict = {}
            for d in ["arguana", "nfcorpus", "scifact"]:
                p = f"outputs/runs/{d}/{ret_name}/{cond}__{norm}.json"
                if not os.path.exists(p):
                    return None
                sub_r = Run.from_file(p)
                for qid, docs in sub_r.to_dict().items():
                    pooled_run_dict[f"{d}__{qid}"] = docs
            r = Run(pooled_run_dict)
            r.name = f"{cond}__{norm}"
            return r

    test_definitions = [
        ("a", "EN vs CS-clean (none)", ("EN", "none"), ("CS-clean", "none")),
        ("b1", "CS-clean vs CS-noise25 (none)", ("CS-clean", "none"), ("CS-noise25", "none")),
        ("b2", "CS-clean vs CS-noise50 (none)", ("CS-clean", "none"), ("CS-noise50", "none")),
        ("b3", "CS-noise25 vs CS-noise50 (none)", ("CS-noise25", "none"), ("CS-noise50", "none")),
        ("c", "stop vs none at CS-noise50", ("CS-noise50", "stop"), ("CS-noise50", "none")),
        ("d", "key vs stop at CS-noise50", ("CS-noise50", "key"), ("CS-noise50", "stop")),
        ("e", "key vs none at CS-noise50", ("CS-noise50", "key"), ("CS-noise50", "none")),
    ]

    all_retrievers = ["bm25", "minilm", "e5small", "bgem3", "minilm_vocabexp"]
    tests_run = []

    print("Running paired Fisher randomization tests...")
    for ds_scope in ["arguana", "nfcorpus", "scifact", "pooled"]:
        qrels = qrels_by_ds[ds_scope]
        for ret in all_retrievers:
            for t_id, desc, (c_a, n_a), (c_b, n_b) in test_definitions:
                if ret == "minilm_vocabexp" and not t_id.startswith("b"):
                    continue
                r_a = load_run(ds_scope, ret, c_a, n_a)
                r_b = load_run(ds_scope, ret, c_b, n_b)
                if r_a is None or r_b is None:
                    continue
                rep = compare(
                    qrels,
                    [r_a, r_b],
                    metrics=["ndcg@10"],
                    stat_test="fisher",
                    n_permutations=1000,
                    random_seed=42,
                )
                d = rep.to_dict()
                s_a = d[r_a.name]["scores"]["ndcg@10"]
                s_b = d[r_b.name]["scores"]["ndcg@10"]
                p_val = d[r_a.name]["comparisons"][r_b.name]["ndcg@10"]
                wtl = d[r_a.name]["win_tie_loss"][r_b.name]["ndcg@10"]
                tests_run.append({
                    "dataset": ds_scope,
                    "retriever": ret,
                    "test_id": t_id,
                    "description": desc,
                    "run_a": f"{c_a}__{n_a}",
                    "run_b": f"{c_b}__{n_b}",
                    "ndcg_a": s_a,
                    "ndcg_b": s_b,
                    "diff": s_a - s_b,
                    "wins_a": wtl["W"],
                    "ties": wtl["T"],
                    "wins_b": wtl["L"],
                    "p_value_raw": p_val,
                })

    # Apply Holm correction across all tests
    raw_p = [r["p_value_raw"] for r in tests_run]
    holm_p = holm_bonferroni(raw_p)
    for r, hp in zip(tests_run, holm_p):
        r["p_value_holm"] = hp
        r["significant_alpha_0_05"] = hp < 0.05

    sig_df = pd.DataFrame(tests_run)
    out_sig = Path("outputs/tables/significance.csv")
    out_sig.parent.mkdir(parents=True, exist_ok=True)
    sig_df.to_csv(out_sig, index=False)
    print(f"Saved {len(sig_df)} tests to {out_sig}")
    return sig_df


def generate_table_main(results_df: pd.DataFrame, sig_df: pd.DataFrame):
    """Generates table_main.tex (nDCG@10 across retrievers and conditions)."""
    # Conditions to display
    conds = ["EN", "CS-clean", "CS-noise25", "CS-noise50"]
    retrievers = ["bm25", "minilm", "e5small", "bgem3", "minilm_vocabexp"]
    datasets = ["arguana", "nfcorpus", "scifact"]

    # Filter to none normalizer
    none_df = results_df[results_df["normalizer"] == "none"].copy()
    lookup = none_df.set_index(["dataset", "retriever", "condition"])["ndcg@10"].to_dict()

    # Pre-calculate means across the 3 datasets
    mean_lookup = {}
    for ret in retrievers:
        for cond in conds:
            vals = [lookup.get((ds, ret, cond), np.nan) for ds in datasets]
            valid_vals = [v for v in vals if pd.notna(v)]
            mean_lookup[(ret, cond)] = np.mean(valid_vals) if valid_vals else np.nan

    ret_labels = {
        "bm25": "BM25",
        "minilm": "MiniLM",
        "e5small": "E5-small",
        "bgem3": "BGE-M3",
        "minilm_vocabexp": "MiniLM-VocabExp",
    }

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Retrieval performance (nDCG@10) across datasets, models, and query noise conditions.}",
        r"\label{tab:main_results}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l cccc cccc cccc cccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\textbf{ArguAna}} & \multicolumn{4}{c}{\textbf{NFCorpus}} & \multicolumn{4}{c}{\textbf{SciFact}} & \multicolumn{4}{c}{\textbf{Mean}} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13} \cmidrule(lr){14-17}",
        r"\textbf{Retriever} & \textbf{EN} & \textbf{CS-clean} & \textbf{CS-n25} & \textbf{CS-n50} & \textbf{EN} & \textbf{CS-clean} & \textbf{CS-n25} & \textbf{CS-n50} & \textbf{EN} & \textbf{CS-clean} & \textbf{CS-n25} & \textbf{CS-n50} & \textbf{EN} & \textbf{CS-clean} & \textbf{CS-n25} & \textbf{CS-n50} \\",
        r"\midrule",
    ]

    for ret in retrievers:
        row_str = [f"{ret_labels.get(ret, ret)}"]
        # 3 datasets
        for ds in datasets:
            for cond in conds:
                val = lookup.get((ds, ret, cond), np.nan)
                row_str.append(f"{val:.4f}" if pd.notna(val) else "--")
        # Mean
        for cond in conds:
            val = mean_lookup.get((ret, cond), np.nan)
            row_str.append(f"{val:.4f}" if pd.notna(val) else "--")
        tex.append(" & ".join(row_str) + r" \\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])

    out_file = Path("outputs/tables/table_main.tex")
    out_file.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Saved {out_file}")


def generate_table_methods(results_df: pd.DataFrame, sig_df: pd.DataFrame):
    """Generates table_methods.tex: methods ladder on CS-noise50 (and CS-clean)."""
    # Normalizers ladder
    norm_ladder = ["none", "stop", "exact", "key", "key_expand", "translit"]
    retrievers = ["bm25", "minilm", "e5small", "bgem3", "minilm_vocabexp"]
    datasets = ["arguana", "nfcorpus", "scifact"]

    lookup = results_df.set_index(["dataset", "retriever", "condition", "normalizer"])["ndcg@10"].to_dict()

    # Build significance lookup for key vs none and key vs stop at CS-noise50
    # key: (dataset, retriever, test_id) -> significant_alpha_0_05
    sig_lookup = {}
    for _, r in sig_df.iterrows():
        sig_lookup[(r["dataset"], r["retriever"], r["test_id"])] = r["significant_alpha_0_05"]

    ret_labels = {
        "bm25": "BM25",
        "minilm": "MiniLM",
        "e5small": "E5-small",
        "bgem3": "BGE-M3",
        "minilm_vocabexp": "MiniLM-VocabExp",
    }

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comparison of normalization methods on code-switched queries under CS-noise50 and CS-clean conditions (nDCG@10 and gap recovery percentage). Significance vs none denoted by $\dagger$ (Holm-corrected $p < 0.05$).}",
        r"\label{tab:methods_ladder}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{ll cccc cccc}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{\textbf{CS-clean}} & \multicolumn{4}{c}{\textbf{CS-noise50}} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        r"\textbf{Retriever} & \textbf{Normalizer} & \textbf{ArguAna} & \textbf{NFCorpus} & \textbf{SciFact} & \textbf{Mean} & \textbf{ArguAna} & \textbf{NFCorpus} & \textbf{SciFact} & \textbf{Mean} \\",
        r"\midrule",
    ]

    for ret in retrievers:
        first = True
        applicable_norms = norm_ladder if ret != "minilm_vocabexp" else ["none"]
        for norm in applicable_norms:
            r_name = ret_labels.get(ret, ret) if first else ""
            first = False
            row_items = [r_name, norm]

            # CS-clean values
            clean_vals = []
            for ds in datasets:
                val = lookup.get((ds, ret, "CS-clean", norm), np.nan)
                clean_vals.append(val)
                row_items.append(f"{val:.4f}" if pd.notna(val) else "--")
            clean_mean = np.nanmean(clean_vals) if any(pd.notna(v) for v in clean_vals) else np.nan
            row_items.append(f"{clean_mean:.4f}" if pd.notna(clean_mean) else "--")

            # CS-noise50 values
            n50_vals = []
            for ds in datasets:
                val = lookup.get((ds, ret, "CS-noise50", norm), np.nan)
                n50_vals.append(val)
                if pd.notna(val):
                    # Check significance if key
                    is_sig = False
                    if norm == "key":
                        is_sig = sig_lookup.get((ds, ret, "e"), False)
                    mark = r"$^{\dagger}$" if is_sig else ""
                    row_items.append(f"{val:.4f}{mark}")
                else:
                    row_items.append("--")
            n50_mean = np.nanmean(n50_vals) if any(pd.notna(v) for v in n50_vals) else np.nan
            is_pooled_sig = sig_lookup.get(("pooled", ret, "e"), False) if norm == "key" else False
            mark_pooled = r"$^{\dagger}$" if is_pooled_sig else ""
            row_items.append(f"{n50_mean:.4f}{mark_pooled}" if pd.notna(n50_mean) else "--")

            tex.append(" & ".join(row_items) + r" \\")
        tex.append(r"\midrule" if ret != retrievers[-1] else "")

    # Remove potential empty line before bottomrule
    if tex[-1] == "":
        tex.pop()

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])

    out_file = Path("outputs/tables/table_methods.tex")
    out_file.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Saved {out_file}")


def generate_table_coverage():
    """Generates table_coverage.tex from outputs/norm_stats.csv."""
    p = Path("outputs/norm_stats.csv")
    if not p.exists():
        print(f"Warning: {p} not found")
        return

    df = pd.read_csv(p)
    # Filter to key normalizer on CS conditions
    sub = df[(df["normalizer"] == "key") & (df["condition"].isin(["CS-clean", "CS-noise25", "CS-noise50"]))].copy()

    datasets = ["arguana", "nfcorpus", "scifact"]
    conditions = ["CS-clean", "CS-noise25", "CS-noise50"]

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Resolution statistics of tagged Hindi tokens under phonetic normalization across datasets and spelling noise levels.}",
        r"\label{tab:coverage}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{ll cccc c}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Condition} & \textbf{Exact (\%)} & \textbf{Key (\%)} & \textbf{Coarse (\%)} & \textbf{Unresolved (\%)} & \textbf{Dropped Stop (\%)} \\",
        r"\midrule",
    ]

    for ds in datasets:
        first = True
        for cond in conditions:
            row = sub[(sub["dataset"] == ds) & (sub["condition"] == cond)]
            ds_name = ds if first else ""
            first = False
            if len(row) > 0:
                r = row.iloc[0]
                t_hi = r["tokens_hi"]
                sw = r["stopwords_dropped"]
                sw_pct = (sw / t_hi * 100) if t_hi > 0 else 0.0
                tex.append(
                    f"{ds_name} & {cond} & {r['pct_exact']:.2f} & {r['pct_key']:.2f} & {r['pct_coarse']:.2f} & {r['pct_unresolved']:.2f} & {sw_pct:.2f} \\\\"
                )
        if ds != datasets[-1]:
            tex.append(r"\midrule")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])

    out_file = Path("outputs/tables/table_coverage.tex")
    out_file.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Saved {out_file}")


if __name__ == "__main__":
    sig_df = run_significance_tests()
    results_df = pd.read_csv("outputs/results.csv")
    generate_table_main(results_df, sig_df)
    generate_table_methods(results_df, sig_df)
    generate_table_coverage()
    print("All tables and significance testing completed successfully.")

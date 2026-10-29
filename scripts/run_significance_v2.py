import os
import sys
import yaml
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ranx import Qrels, Run, compare, evaluate
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


def format_p(p: float) -> str:
    """Formats p-value, reporting values below 1e-5 as '< 1e-5'."""
    if p < 1e-5:
        return "< 1e-5"
    return f"{p:.5f}"


def run_v2():
    print("Loading experiment configuration and datasets...")
    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    datasets = ["arguana", "nfcorpus", "scifact", "fiqa", "scidocs"]

    qrels_by_ds = {}
    for ds_name in datasets:
        ds = load_beir(ds_name, cfg["datasets"][ds_name], cfg["seed"])
        clean = load_clean_queries(ds_name)
        clean_qids = sorted(clean.keys())
        qrels_by_ds[ds_name] = Qrels({qid: ds.qrels[qid] for qid in clean_qids if qid in ds.qrels})

    # Pooled qrels
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
            for d in datasets:
                p = f"outputs/runs/{d}/{ret_name}/{cond}__{norm}.json"
                if not os.path.exists(p):
                    return None
                sub_r = Run.from_file(p)
                for qid, docs in sub_r.to_dict().items():
                    pooled_run_dict[f"{d}__{qid}"] = docs
            r = Run(pooled_run_dict)
            r.name = f"{cond}__{norm}"
            return r

    # 1. Primary family: pooled only, 4 retrievers x 5 tests (a, b2, c, d, e) = 20 tests
    primary_tests_defs = [
        ("a", "EN vs CS-clean (none)", ("EN", "none"), ("CS-clean", "none")),
        ("b2", "CS-clean vs CS-noise50 (none)", ("CS-clean", "none"), ("CS-noise50", "none")),
        ("c", "stop vs none at CS-noise50", ("CS-noise50", "stop"), ("CS-noise50", "none")),
        ("d", "key vs stop at CS-noise50", ("CS-noise50", "key"), ("CS-noise50", "stop")),
        ("e", "key vs none at CS-noise50", ("CS-noise50", "key"), ("CS-noise50", "none")),
    ]
    core_retrievers = ["bm25", "minilm", "e5small", "bgem3"]

    print("Running primary family tests (100,000 permutations, pooled)...")
    primary_rows = []
    pooled_qrels = qrels_by_ds["pooled"]

    for ret in core_retrievers:
        for t_id, desc, (c_a, n_a), (c_b, n_b) in primary_tests_defs:
            r_a = load_run("pooled", ret, c_a, n_a)
            r_b = load_run("pooled", ret, c_b, n_b)
            rep = compare(
                pooled_qrels,
                [r_a, r_b],
                metrics=["ndcg@10"],
                stat_test="fisher",
                n_permutations=100000,
                threads=0,
                random_seed=42,
            )
            d = rep.to_dict()
            s_a = d[r_a.name]["scores"]["ndcg@10"]
            s_b = d[r_b.name]["scores"]["ndcg@10"]
            p_val = d[r_a.name]["comparisons"][r_b.name]["ndcg@10"]
            wtl = d[r_a.name]["win_tie_loss"][r_b.name]["ndcg@10"]
            primary_rows.append({
                "family": "primary",
                "dataset": "pooled",
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

    # Apply Holm correction strictly within the 20 primary tests
    raw_p_prim = [r["p_value_raw"] for r in primary_rows]
    holm_p_prim = holm_bonferroni(raw_p_prim)
    for r, hp in zip(primary_rows, holm_p_prim):
        r["p_value_holm"] = hp
        r["p_raw_str"] = format_p(r["p_value_raw"])
        r["p_holm_str"] = format_p(hp)
        r["significant_alpha_0_05"] = hp < 0.05

    # 2. Secondary family: exploratory per-dataset tests and remaining pooled tests
    exploratory_test_defs = [
        ("a", "EN vs CS-clean (none)", ("EN", "none"), ("CS-clean", "none")),
        ("b1", "CS-clean vs CS-noise25 (none)", ("CS-clean", "none"), ("CS-noise25", "none")),
        ("b2", "CS-clean vs CS-noise50 (none)", ("CS-clean", "none"), ("CS-noise50", "none")),
        ("b3", "CS-noise25 vs CS-noise50 (none)", ("CS-noise25", "none"), ("CS-noise50", "none")),
        ("c", "stop vs none at CS-noise50", ("CS-noise50", "stop"), ("CS-noise50", "none")),
        ("d", "key vs stop at CS-noise50", ("CS-noise50", "key"), ("CS-noise50", "stop")),
        ("e", "key vs none at CS-noise50", ("CS-noise50", "key"), ("CS-noise50", "none")),
    ]
    all_retrievers = ["bm25", "minilm", "e5small", "bgem3", "minilm_vocabexp"]

    print("Running exploratory secondary tests (100,000 permutations)...")
    exploratory_rows = []

    # Per-dataset tests
    for ds_scope in datasets:
        qrels = qrels_by_ds[ds_scope]
        for ret in all_retrievers:
            for t_id, desc, (c_a, n_a), (c_b, n_b) in exploratory_test_defs:
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
                    n_permutations=100000,
                    threads=0,
                    random_seed=42,
                )
                d = rep.to_dict()
                s_a = d[r_a.name]["scores"]["ndcg@10"]
                s_b = d[r_b.name]["scores"]["ndcg@10"]
                p_val = d[r_a.name]["comparisons"][r_b.name]["ndcg@10"]
                wtl = d[r_a.name]["win_tie_loss"][r_b.name]["ndcg@10"]
                exploratory_rows.append({
                    "family": "exploratory",
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
                    "p_value_holm": np.nan,
                    "p_raw_str": format_p(p_val),
                    "p_holm_str": "--",
                    "significant_alpha_0_05": p_val < 0.05,
                })

    # Remaining pooled exploratory tests (b1, b3 for core retrievers, and b1, b2, b3 for minilm_vocabexp)
    for ret in all_retrievers:
        for t_id, desc, (c_a, n_a), (c_b, n_b) in exploratory_test_defs:
            if ret in core_retrievers and t_id in ("a", "b2", "c", "d", "e"):
                continue  # Already in primary family
            if ret == "minilm_vocabexp" and not t_id.startswith("b"):
                continue
            r_a = load_run("pooled", ret, c_a, n_a)
            r_b = load_run("pooled", ret, c_b, n_b)
            if r_a is None or r_b is None:
                continue
            rep = compare(
                pooled_qrels,
                [r_a, r_b],
                metrics=["ndcg@10"],
                stat_test="fisher",
                n_permutations=100000,
                threads=0,
                random_seed=42,
            )
            d = rep.to_dict()
            s_a = d[r_a.name]["scores"]["ndcg@10"]
            s_b = d[r_b.name]["scores"]["ndcg@10"]
            p_val = d[r_a.name]["comparisons"][r_b.name]["ndcg@10"]
            wtl = d[r_a.name]["win_tie_loss"][r_b.name]["ndcg@10"]
            exploratory_rows.append({
                "family": "exploratory",
                "dataset": "pooled",
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
                "p_value_holm": np.nan,
                "p_raw_str": format_p(p_val),
                "p_holm_str": "--",
                "significant_alpha_0_05": p_val < 0.05,
            })

    # Combine into full v2 dataframe
    all_tests_df = pd.DataFrame(primary_rows + exploratory_rows)
    out_path = Path("outputs/tables/significance_v2.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_tests_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(all_tests_df)} tests to {out_path} (primary family: {len(primary_rows)}, exploratory: {len(exploratory_rows)}).")

    # 3. Paired bootstrap 95% CI (10,000 resamples) of (CS-clean - CS-noise50) - (EN - CS-clean)
    print("\nRunning paired bootstrap 95% CI (10,000 resamples)...")
    rng = np.random.default_rng(42)
    n_resamples = 10000

    bootstrap_results = []
    for ret in core_retrievers:
        r_en = load_run("pooled", ret, "EN", "none")
        r_clean = load_run("pooled", ret, "CS-clean", "none")
        r_n50 = load_run("pooled", ret, "CS-noise50", "none")

        s_en = evaluate(pooled_qrels, r_en, "ndcg@10", return_mean=False)
        s_clean = evaluate(pooled_qrels, r_clean, "ndcg@10", return_mean=False)
        s_n50 = evaluate(pooled_qrels, r_n50, "ndcg@10", return_mean=False)

        # delta = (CS-clean - CS-noise50) - (EN - CS-clean)
        # delta = spelling_cost - switching_cost
        spelling_cost = s_clean - s_n50
        switching_cost = s_en - s_clean
        delta = spelling_cost - switching_cost

        observed_mean = np.mean(delta)
        observed_spell = np.mean(spelling_cost)
        observed_switch = np.mean(switching_cost)

        # Bootstrap resampling
        n_q = len(delta)
        boot_indices = rng.integers(0, n_q, size=(n_resamples, n_q))
        boot_means = np.mean(delta[boot_indices], axis=1)

        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        boot_se = float(np.std(boot_means, ddof=1))

        # Two-tailed bootstrap p-value against null of delta == 0
        p_boot = 2.0 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0))
        p_boot = min(1.0, max(p_boot, 1.0 / n_resamples))

        bootstrap_results.append({
            "retriever": ret,
            "mean_spelling_cost": observed_spell,
            "mean_switching_cost": observed_switch,
            "mean_delta (spell - switch)": observed_mean,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "bootstrap_se": boot_se,
            "bootstrap_p": p_boot,
        })

    boot_df = pd.DataFrame(bootstrap_results)
    boot_out_path = Path("outputs/tables/bootstrap_spelling_vs_switching.csv")
    boot_df.to_csv(boot_out_path, index=False)
    print(f"Saved bootstrap results to {boot_out_path}")

    return primary_rows, bootstrap_results


if __name__ == "__main__":
    run_v2()

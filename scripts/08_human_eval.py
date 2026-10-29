import os
import sys
import yaml
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ranx import Qrels, Run, evaluate, compare
from hcsir.data import load_beir
from hcsir.lexicon import RomanLexicon
from hcsir.tagger import Tagger
from hcsir.normalize import normalize
from hcsir.retrievers import get_retriever


def load_human_queries(queries_dir: str = "data/queries") -> dict[str, dict[str, dict]]:
    """Loads only the 120 human-verified queries across datasets."""
    datasets = ["arguana", "nfcorpus", "scifact"]
    human_queries = {}

    for ds in datasets:
        p = Path(queries_dir) / f"{ds}.jsonl"
        items = {}
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("gen_status") == "ok" and d.get("CS-human") and str(d["CS-human"]).strip():
                    items[str(d["qid"])] = d
        human_queries[ds] = items
        print(f"Loaded {len(items)} human-verified queries for {ds}")

    return human_queries


def run_human_grid():
    print("Initializing environment for CS-human evaluation...")
    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    datasets_cfg = cfg["datasets"]
    seed = cfg["seed"]

    # Load lexicon and tagger
    lex_path = Path("data/lexicon/roman_hi_en.json")
    if lex_path.exists():
        lex_data = json.loads(lex_path.read_text(encoding="utf-8"))
        lexicon = RomanLexicon(
            exact_map=lex_data.get("exact_map"),
            key_candidates=lex_data.get("key_candidates"),
            coarse_candidates=lex_data.get("coarse_candidates"),
        )
    else:
        lexicon = RomanLexicon.from_pairs([])
    tagger = Tagger(lexicon=lexicon)

    human_queries = load_human_queries()
    retrievers_list = ["bm25", "minilm", "e5small", "bgem3", "minilm_vocabexp"]

    results_rows = []
    runs_cache = {}

    for ds_name in ["arguana", "nfcorpus", "scifact"]:
        print(f"\n--- Evaluating dataset: {ds_name} (40 human queries) ---")
        beir_ds = load_beir(ds_name, datasets_cfg[ds_name], seed)
        ds_queries = human_queries[ds_name]
        qids = sorted(ds_queries.keys())

        # Exact sub-qrels for human queries only
        sub_qrels = Qrels({qid: beir_ds.qrels[qid] for qid in qids if qid in beir_ds.qrels})
        print(f"Sub-qrels size: {len(sub_qrels)} queries")

        # Prepare normalized query sets
        # 1. EN none
        q_en_none = {qid: ds_queries[qid]["EN"] for qid in qids}
        # 2. EN key (control)
        q_en_key = {qid: normalize(ds_queries[qid]["EN"], "key", lexicon, tagger).text for qid in qids}
        # 3. CS-human none
        q_cs_none = {qid: ds_queries[qid]["CS-human"] for qid in qids}
        # 4. CS-human key
        q_cs_key = {qid: normalize(ds_queries[qid]["CS-human"], "key", lexicon, tagger).text for qid in qids}

        query_bundles = [
            ("EN", "none", q_en_none),
            ("EN", "key", q_en_key),
            ("CS-human", "none", q_cs_none),
            ("CS-human", "key", q_cs_key),
        ]

        for ret_name in retrievers_list:
            print(f"Indexing retriever {ret_name} on {ds_name}...")
            retriever_inst = get_retriever(ret_name, cfg=cfg, lexicon=lexicon)
            retriever_inst.index(beir_ds)

            for cond, norm, q_dict in query_bundles:
                if ret_name == "minilm_vocabexp" and (cond != "CS-human" or norm != "none"):
                    continue

                raw_run = retriever_inst.search(q_dict, k=100)
                run = Run(raw_run)
                run.name = f"{cond}__{norm}"
                runs_cache[(ds_name, ret_name, cond, norm)] = run

                metrics = evaluate(sub_qrels, run, ["ndcg@10", "recall@100"])
                results_rows.append({
                    "dataset": ds_name,
                    "retriever": ret_name,
                    "condition": cond,
                    "normalizer": norm,
                    "ndcg@10": float(metrics["ndcg@10"]),
                    "recall@100": float(metrics["recall@100"]),
                    "n_queries": len(sub_qrels),
                })
                print(f"  [{ret_name:15s}] {cond:10s} | {norm:5s} -> nDCG@10: {metrics['ndcg@10']:.4f}")

    df_results = pd.DataFrame(results_rows)
    out_csv = Path("outputs/tables/human_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved human results to {out_csv}")

    # Generate table_human.tex
    generate_table_human(df_results)


def generate_table_human(df: pd.DataFrame):
    """Generates table_human.tex comparing EN vs CS-human (none) vs CS-human (key) on the 120-query verified subset."""
    retrievers = ["bm25", "minilm", "e5small", "bgem3", "minilm_vocabexp"]
    ret_labels = {
        "bm25": "BM25",
        "minilm": "MiniLM",
        "e5small": "E5-small",
        "bgem3": "BGE-M3",
        "minilm_vocabexp": "MiniLM-VocabExp",
    }
    datasets = ["arguana", "nfcorpus", "scifact"]
    lookup = df.set_index(["dataset", "retriever", "condition", "normalizer"])["ndcg@10"].to_dict()

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Retrieval performance (nDCG@10) on the 120-query human-verified subset comparing English queries, unnormalized code-switched queries, and phonetic normalization.}",
        r"\label{tab:human_results}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l ccc ccc ccc ccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{ArguAna}} & \multicolumn{3}{c}{\textbf{NFCorpus}} & \multicolumn{3}{c}{\textbf{SciFact}} & \multicolumn{3}{c}{\textbf{Mean}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}",
        r"\textbf{Retriever} & \textbf{EN} & \textbf{CS-human} & \textbf{+Key} & \textbf{EN} & \textbf{CS-human} & \textbf{+Key} & \textbf{EN} & \textbf{CS-human} & \textbf{+Key} & \textbf{EN} & \textbf{CS-human} & \textbf{+Key} \\",
        r"\midrule",
    ]

    for ret in retrievers:
        row_items = [ret_labels.get(ret, ret)]

        en_vals, cs_vals, key_vals = [], [], []
        for ds in datasets:
            val_en = lookup.get((ds, ret, "EN", "none"), np.nan)
            val_cs = lookup.get((ds, ret, "CS-human", "none"), np.nan)
            val_key = lookup.get((ds, ret, "CS-human", "key"), np.nan)

            en_vals.append(val_en)
            cs_vals.append(val_cs)
            key_vals.append(val_key)

            row_items.append(f"{val_en:.4f}" if pd.notna(val_en) else "--")
            row_items.append(f"{val_cs:.4f}" if pd.notna(val_cs) else "--")
            row_items.append(f"{val_key:.4f}" if pd.notna(val_key) else "--")

        # Means
        m_en = np.nanmean(en_vals) if any(pd.notna(v) for v in en_vals) else np.nan
        m_cs = np.nanmean(cs_vals) if any(pd.notna(v) for v in cs_vals) else np.nan
        m_key = np.nanmean(key_vals) if any(pd.notna(v) for v in key_vals) else np.nan

        row_items.append(f"{m_en:.4f}" if pd.notna(m_en) else "--")
        row_items.append(f"{m_cs:.4f}" if pd.notna(m_cs) else "--")
        row_items.append(f"{m_key:.4f}" if pd.notna(m_key) else "--")

        tex.append(" & ".join(row_items) + r" \\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])

    out_file = Path("outputs/tables/table_human.tex")
    out_file.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Saved {out_file}")


if __name__ == "__main__":
    run_human_grid()

#!/usr/bin/env python3
"""Evaluate saved HarveST clustering labels against one or more truth columns."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import anndata as ad
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def infer_sep(path: str, sep: Optional[str]) -> str:
    if sep is not None:
        return sep
    return "," if path.endswith(".csv") else "\t"


def infer_cluster_columns(columns: Iterable[str]) -> List[str]:
    result = []
    for column in columns:
        name = column.lower()
        if "clust" in name or name == "mclust":
            result.append(column)
    return result


def load_truth(
    adata,
    truth_file: Optional[str],
    truth_id_column: Optional[str],
    truth_label_columns: Optional[List[str]],
    truth_sep: Optional[str],
) -> pd.DataFrame:
    if truth_file is None:
        truth_df = adata.obs.copy()
        if truth_label_columns is None:
            truth_label_columns = ["Ground Truth"] if "Ground Truth" in truth_df else []
        return truth_df.reindex(adata.obs_names)[truth_label_columns]

    sep = infer_sep(truth_file, truth_sep)
    truth_df = pd.read_csv(truth_file, sep=sep)
    id_column = truth_id_column or truth_df.columns[0]
    if id_column not in truth_df.columns:
        raise ValueError(f"Truth id column '{id_column}' not found in {truth_file}")

    if truth_label_columns is None:
        truth_label_columns = [
            column
            for column in truth_df.columns
            if column != id_column and not str(column).startswith("Unnamed")
        ]

    missing = [column for column in truth_label_columns if column not in truth_df.columns]
    if missing:
        raise ValueError(f"Truth label columns not found in {truth_file}: {missing}")

    return truth_df.set_index(id_column).reindex(adata.obs_names)[truth_label_columns]


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    adata = ad.read_h5ad(args.adata)
    cluster_columns = args.cluster_columns or infer_cluster_columns(adata.obs.columns)
    if not cluster_columns:
        raise ValueError("No clustering columns found. Pass --cluster-columns explicitly.")

    missing_clusters = [column for column in cluster_columns if column not in adata.obs.columns]
    if missing_clusters:
        raise ValueError(f"Cluster columns not found in AnnData obs: {missing_clusters}")

    truth = load_truth(
        adata=adata,
        truth_file=args.truth_file,
        truth_id_column=args.truth_id_column,
        truth_label_columns=args.truth_label_columns,
        truth_sep=args.truth_sep,
    )

    rows = []
    for truth_column in truth.columns:
        for cluster_column in cluster_columns:
            eval_df = pd.DataFrame(
                {
                    "truth": truth[truth_column],
                    "cluster": adata.obs[cluster_column],
                },
                index=adata.obs_names,
            ).dropna()
            if eval_df.empty:
                continue

            rows.append(
                {
                    "truth_column": truth_column,
                    "cluster_column": cluster_column,
                    "n_spots": int(eval_df.shape[0]),
                    "n_truth_labels": int(eval_df["truth"].nunique()),
                    "n_cluster_labels": int(eval_df["cluster"].nunique()),
                    "ari": adjusted_rand_score(eval_df["truth"], eval_df["cluster"]),
                    "nmi": normalized_mutual_info_score(eval_df["truth"], eval_df["cluster"]),
                }
            )

    metrics = pd.DataFrame(rows).sort_values(["ari", "nmi"], ascending=False)
    if args.output:
        output = Path(args.output)
    else:
        output = Path(args.adata).resolve().parent / "clustering_metrics_by_truth.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)
    print(metrics.to_string(index=False))
    print(f"Saved metrics to: {output}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved clustering results against multiple truth columns."
    )
    parser.add_argument("--adata", required=True, help="Path to clustering_results.h5ad.")
    parser.add_argument("--truth-file", help="CSV/TSV file containing ground truth labels.")
    parser.add_argument("--truth-id-column", default="ID", help="Spot/barcode id column.")
    parser.add_argument(
        "--truth-label-columns",
        nargs="+",
        help="Truth columns to evaluate. Defaults to all non-id columns in truth file.",
    )
    parser.add_argument("--truth-sep", help="Truth file separator. Defaults by file suffix.")
    parser.add_argument(
        "--cluster-columns",
        nargs="+",
        help="Clustering columns in AnnData obs. Defaults to columns containing 'clust' or 'mclust'.",
    )
    parser.add_argument("--output", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()

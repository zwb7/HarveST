#!/usr/bin/env python3
"""Train and evaluate HarveST from preprocessed matrices."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harvest import Harvest
from harvest.utils import Config


def _as_dict(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config field '{key}' must be a mapping.")
    return value


def build_output_dir(config: Dict[str, Any]) -> str:
    data_cfg = _as_dict(config, "data")
    output_cfg = _as_dict(config, "output")

    source_path = Path(data_cfg.get("original_data_path") or data_cfg["preprocessed_dir"])
    sample_id = data_cfg.get("sample_id") or source_path.name or "unknown_sample"
    output_root = Path(output_cfg.get("root", "./results/train"))
    run_name = output_cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")

    return str(output_root / str(sample_id) / str(run_name))


def resolve_adata_file(data_cfg: Dict[str, Any]) -> Optional[str]:
    if data_cfg.get("adata_file"):
        return str(data_cfg["adata_file"])

    candidate = Path(data_cfg["preprocessed_dir"]) / "adata_processed.h5ad"
    if candidate.exists():
        return str(candidate)
    return None


def save_resolved_config(config: Dict[str, Any], output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "train_config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def run(config_path: str) -> None:
    config = Config.load_config(config_path)
    data_cfg = _as_dict(config, "data")
    model_cfg = dict(_as_dict(config, "model"))
    training_cfg = dict(_as_dict(config, "training"))
    clustering_cfg = dict(_as_dict(config, "clustering"))
    runtime_cfg = _as_dict(config, "runtime")
    output_cfg = _as_dict(config, "output")

    if "preprocessed_dir" not in data_cfg:
        raise ValueError("Config field 'data.preprocessed_dir' is required.")
    if "original_data_path" not in data_cfg:
        raise ValueError("Config field 'data.original_data_path' is required.")

    r_libs_user = runtime_cfg.get("r_libs_user")
    if r_libs_user:
        os.environ["R_LIBS_USER"] = str(r_libs_user)

    output_dir = build_output_dir(config)
    save_resolved_config(config, output_dir)

    harvest = Harvest(
        config=config,
        output_dir=output_dir,
        random_seed=int(runtime_cfg.get("random_seed", 2023)),
        device=runtime_cfg.get("device"),
    )

    preprocessed_data = harvest.load_preprocessed_data(
        matrix_dir=str(data_cfg["preprocessed_dir"]),
        data_path=str(data_cfg["original_data_path"]),
        count_file=str(data_cfg.get("count_file", "filtered_feature_bc_matrix.h5")),
        adata_file=resolve_adata_file(data_cfg),
        n_top_genes=int(data_cfg.get("n_top_genes", 3000)),
        load_ground_truth=bool(data_cfg.get("load_ground_truth", True)),
        truth_file_suffix=str(data_cfg.get("truth_file_suffix", "_truth.txt")),
    )

    n_clusters = int(clustering_cfg.pop("n_clusters", data_cfg.get("n_clusters", 7)))
    results = harvest.cluster(
        n_clusters=n_clusters,
        model_params=model_cfg,
        training_params=training_cfg,
        clustering_params=clustering_cfg,
        plot_results=bool(output_cfg.get("plot_results", True)),
        save_results=bool(output_cfg.get("save_results", True)),
    )

    print("Training and evaluation completed.")
    print(f"Output directory: {output_dir}")
    print(f"Preprocessed keys: {sorted(preprocessed_data.keys())}")
    print(f"Result keys: {sorted(results.keys())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, cluster, and evaluate HarveST from preprocessed matrices."
    )
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Preprocess raw Visium data into reusable HarveST matrices."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

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

    data_path = Path(data_cfg["data_path"])
    sample_id = data_cfg.get("sample_id") or data_path.name or "unknown_sample"
    output_root = Path(output_cfg.get("root", "./results/preprocessed"))
    run_name = output_cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")

    return str(output_root / str(sample_id) / str(run_name))


def save_resolved_config(config: Dict[str, Any], output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "preprocess_config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def run(config_path: str) -> None:
    config = Config.load_config(config_path)
    data_cfg = _as_dict(config, "data")
    preprocessing_cfg = _as_dict(config, "preprocessing")
    runtime_cfg = _as_dict(config, "runtime")

    if "data_path" not in data_cfg:
        raise ValueError("Config field 'data.data_path' is required.")

    output_dir = build_output_dir(config)
    save_resolved_config(config, output_dir)

    harvest = Harvest(
        config=config,
        output_dir=output_dir,
        random_seed=int(runtime_cfg.get("random_seed", 2023)),
        device=runtime_cfg.get("device", "cpu"),
    )

    preprocessed_data = harvest.pre_process(
        data_path=str(data_cfg["data_path"]),
        count_file=str(data_cfg.get("count_file", "filtered_feature_bc_matrix.h5")),
        n_top_genes=int(preprocessing_cfg.get("n_top_genes", 3000)),
        n_bins=int(preprocessing_cfg.get("n_bins", 5)),
        spatial_params=preprocessing_cfg.get("spatial_params"),
        parallel_mi=bool(preprocessing_cfg.get("parallel_mi", True)),
        n_jobs=int(preprocessing_cfg.get("n_jobs", -1)),
        load_ground_truth=bool(data_cfg.get("load_ground_truth", True)),
        truth_file_suffix=str(data_cfg.get("truth_file_suffix", "_truth.txt")),
        truth_file=data_cfg.get("truth_file"),
        truth_id_column=data_cfg.get("truth_id_column"),
        truth_label_column=data_cfg.get("truth_label_column"),
        truth_sep=data_cfg.get("truth_sep"),
    )

    print("Preprocessing completed.")
    print(f"Output directory: {output_dir}")
    print(f"Preprocessed keys: {sorted(preprocessed_data.keys())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw Visium data into reusable HarveST matrices."
    )
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

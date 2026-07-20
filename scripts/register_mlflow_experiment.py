#!/usr/bin/env python3
"""Register MLflow experiment for paper-final ring training.

    uv run python scripts/register_mlflow_experiment.py
    uv run python scripts/register_mlflow_experiment.py --name final_rings_for_paper
"""

from __future__ import annotations

import argparse

import mlflow

from gf_grokking.mlflow_logger import FINAL_RINGS_EXPERIMENT, MLFLOW_TRACKING_URI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=FINAL_RINGS_EXPERIMENT)
    parser.add_argument("--tracking-uri", default=MLFLOW_TRACKING_URI)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    exp = mlflow.set_experiment(args.name)
    print(f"Experiment: {args.name}")
    print(f"  id: {exp.experiment_id}")
    print(f"  tracking_uri: {args.tracking_uri}")


if __name__ == "__main__":
    main()

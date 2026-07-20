"""MLflow logger для экспериментов GF(2^n).

Единый источник истины для метрик, параметров и артефактов:
- backend: sqlite (`mlflow.db`)
- артефакты (модели, чекпойнты, плоты, конфиги) логируются в run через MLflow,
  а не разбросаны по случайным папкам.

Запуск UI на том же сторе:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import torch

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# Эксперименты разнесены по назначению, чтобы отладочные прогоны не смешивались
# с научными результатами:
#   - RESEARCH_EXPERIMENT — настоящие эксперименты с ценными данными.
#   - SMOKE_EXPERIMENT    — проверки работоспособности пайплайна (без науч. ценности).
RESEARCH_EXPERIMENT = "gf_grokking"
SMOKE_EXPERIMENT = "gf_grokking_smoke"
TRANSFORMER_EXPERIMENT = "gf_grokking_transformer"
ALT_GROUP_EXPERIMENT = "gf_grokking_alt_groups"
RING_EXPERIMENT = "gf_grokking_rings"
FINAL_RINGS_EXPERIMENT = "final_rings_for_paper"


def setup_experiment(
    experiment_name: str,
    run_name: str | None = None,
    tags: dict[str, Any] | None = None,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> mlflow.ActiveRun:
    """Инициализирует MLflow эксперимент на едином sqlite-бэкенде."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name, tags=tags or {})


def active_run_id() -> str | None:
    """ID текущего активного run (или None)."""
    run = mlflow.active_run()
    return run.info.run_id if run is not None else None


def log_params(params: dict[str, Any]) -> None:
    """Логгирует гиперпараметры."""
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Логгирует метрики."""
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str | Path, artifact_dir: str | None = None) -> None:
    """Сохраняет существующий файл как артефакт run."""
    if Path(path).exists():
        mlflow.log_artifact(str(path), artifact_dir)


def log_json(data: dict[str, Any], artifact_dir: str, filename: str) -> None:
    """Логгирует dict как JSON-артефакт с корректным именем файла."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        mlflow.log_artifact(str(path), artifact_dir)


def log_model(
    model: torch.nn.Module,
    artifact_dir: str = "model",
    filename: str = "final.pt",
) -> None:
    """Сохраняет state_dict модели как артефакт artifact_dir/filename."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / filename
        torch.save(model.state_dict(), path)
        mlflow.log_artifact(str(path), artifact_dir)


def log_dataset_info(
    n: int,
    poly: int,
    train_size: int,
    test_size: int,
    sample_inputs: torch.Tensor | None = None,
) -> None:
    """Логгирует информацию о датасете."""
    dataset_info: dict[str, Any] = {
        "field": f"GF(2^{n})",
        "num_elements": 1 << n,
        "irreducible_poly": poly,
        "irreducible_poly_bin": bin(poly),
        "train_size": train_size,
        "test_size": test_size,
        "input_dim": (1 << n) * 2,
    }
    if sample_inputs is not None:
        dataset_info["sample_input_shape"] = list(sample_inputs.shape)
    log_json(dataset_info, "dataset", "dataset_info.json")


def log_alt_group_dataset_info(
    group_name: str,
    group_n: int,
    num_elements: int,
    train_size: int,
    test_size: int,
) -> None:
    """Логгирует информацию о датасете A_n."""
    dataset_info: dict[str, Any] = {
        "group": group_name,
        "alternating_n": group_n,
        "num_elements": num_elements,
        "train_size": train_size,
        "test_size": test_size,
        "input_dim": num_elements * 2,
        "input_encoding": "onehot",
    }
    log_json(dataset_info, "dataset", "dataset_info.json")


def end_run(status: str = "FINISHED") -> None:
    """Завершает MLflow run."""
    mlflow.end_run(status=status)


def log_training_artifacts(
    model: torch.nn.Module,
    result: Any,
    run_config: dict[str, Any],
    checkpoint_dir: str | Path,
    elapsed: float,
    *,
    curves_path: Path | None = None,
    extra_metrics: dict[str, float] | None = None,
) -> None:
    """Стандартный набор артефактов после train_model (config, model, plots, checkpoints)."""
    best_test_acc = max(result.test_acc_history) if result.test_acc_history else 0.0
    log_json(run_config, "config", "config.json")
    log_model(model, "model")
    metrics: dict[str, float] = {
        "best_test_acc": best_test_acc,
        "final_train_acc": result.final_train_acc,
        "final_test_acc": result.final_test_acc,
        "best_epoch": float(result.best_epoch),
        "elapsed_seconds": elapsed,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    log_metrics(metrics)
    if curves_path is not None and curves_path.exists():
        log_artifact(curves_path, "plots")
    if result.best_model_path:
        log_artifact(result.best_model_path, "checkpoints")
    last_path = Path(checkpoint_dir) / "last.pt"
    if last_path.exists():
        log_artifact(last_path, "checkpoints")

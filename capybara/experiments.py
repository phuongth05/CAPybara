"""Minimal experiment bookkeeping with explicit, traceable artifacts."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


class ExperimentError(ValueError):
    """Raised when an experiment configuration or prediction is invalid."""


def load_config(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Load a JSON config without adding a runtime dependency.

    YAML is intentionally not parsed implicitly: this keeps local validation
    deterministic until a YAML dependency is explicitly adopted.
    """

    config_path = Path(path)
    if config_path.suffix.lower() != ".json":
        raise ExperimentError(
            f"Only JSON configs are supported by the bootstrap foundation: {config_path}"
        )
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"Invalid JSON config {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentError("The experiment config must contain a JSON object")
    return value


def git_commit(repo_root: os.PathLike[str] | str) -> str | None:
    """Return the checked-out commit, or None for an unborn/non-Git checkout."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def validate_config(config: Mapping[str, Any]) -> None:
    required = ("experiment_id", "model", "dataset", "split", "seed")
    missing = [key for key in required if key not in config]
    if missing:
        raise ExperimentError(f"Missing required config fields: {', '.join(missing)}")
    experiment_id = config["experiment_id"]
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ExperimentError("experiment_id must be a non-empty string")
    if any(char in experiment_id for char in '\\/:*?"<>|'):
        raise ExperimentError("experiment_id contains a path-unsafe character")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ExperimentError("seed must be an integer")


def create_experiment(
    config_path: os.PathLike[str] | str,
    repo_root: os.PathLike[str] | str,
    output_root: os.PathLike[str] | str,
) -> Path:
    """Create an experiment directory and write config/metadata artifacts."""

    config = load_config(config_path)
    validate_config(config)
    repo_root = Path(repo_root)
    output_dir = Path(output_root) / str(config["experiment_id"])
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "experiment_id": config["experiment_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo_root),
        "model": config["model"],
        "checkpoint": config.get("checkpoint", "UNVERIFIED"),
        "upstream_commit": config.get("upstream_commit", "UNVERIFIED"),
        "dataset": config["dataset"],
        "split": config["split"],
        "seed": config["seed"],
        "device": config.get("device", "unspecified"),
        "retrieval_datastore": config.get("retrieval_datastore"),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": config.get("package_versions", {}),
        "config_source": str(Path(config_path).resolve()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def normalize_predictions(predictions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize official prediction records to image_id/caption pairs."""

    normalized = []
    for index, item in enumerate(predictions):
        if "image_id" not in item or "caption" not in item:
            raise ExperimentError(f"Prediction {index} needs image_id and caption")
        caption = item["caption"]
        if not isinstance(caption, str):
            raise ExperimentError(f"Prediction {index} caption must be a string")
        normalized.append({"image_id": str(item["image_id"]), "caption": caption})
    return normalized


def write_predictions(
    output_dir: os.PathLike[str] | str,
    predictions: Iterable[Mapping[str, Any]],
    raw_predictions: Any | None = None,
) -> None:
    """Preserve raw output and write the common normalized prediction schema."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    normalized = normalize_predictions(predictions)
    if raw_predictions is not None:
        (target / "raw_predictions.json").write_text(
            json.dumps(raw_predictions, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (target / "predictions.json").write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def seed_everything(seed: int) -> None:
    """Seed Python's local random sources used by infrastructure code."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

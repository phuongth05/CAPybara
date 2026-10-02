"""Evaluation-only closure for an existing SmallCap R2 prediction artifact."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


PAPER_TARGETS = {
    "BLEU-4": {"evaluator_key": "Bleu_4", "paper_value": 37.0},
    "METEOR": {"evaluator_key": "METEOR", "paper_value": 27.9},
    "CIDEr": {"evaluator_key": "CIDEr", "paper_value": 119.7},
    "SPICE": {"evaluator_key": "SPICE", "paper_value": 21.3},
}
PAPER_METRIC_KEYS = tuple(item["evaluator_key"] for item in PAPER_TARGETS.values())
CONFIGURED_METRICS = ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE")
UPSTREAM_COMMIT = "19cc4f4e5972c70fde16c8857b14c96e5492cb30"
CHECKPOINT_REVISION = "fd635f4025a5ccb638ed1e1b540c3671557b9a16"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_id_set(records: Iterable[dict[str, Any]], key: str) -> set[int]:
    values: set[int] = set()
    for record in records:
        if key not in record:
            raise ValueError(f"record is missing {key}")
        values.add(int(record[key]))
    return values


def validate_predictions(predictions: Any, dataset: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a JSON list")
    rows = [image for image in dataset.get("images", []) if image.get("split") == "test"]
    expected_ids = _image_id_set(rows, "cocoid")
    gold_records = gold.get("images", [])
    gold_ids = _image_id_set(gold_records, "id")
    actual_ids: list[int] = []
    missing_caption = []
    empty_caption = []
    invalid_id = []
    for position, row in enumerate(predictions):
        try:
            actual_ids.append(int(row["image_id"]))
        except (KeyError, TypeError, ValueError):
            invalid_id.append(position)
        caption = row.get("caption") if isinstance(row, dict) else None
        if not isinstance(row, dict) or "caption" not in row:
            missing_caption.append(position)
        elif not isinstance(caption, str) or not caption.strip():
            empty_caption.append(position)
    actual_set = set(actual_ids)
    duplicates = len(actual_ids) - len(actual_set)
    expected_matches_dataset = actual_set == expected_ids
    expected_matches_gold = actual_set == gold_ids
    valid = (
        len(predictions) == 5000
        and len(actual_ids) == 5000
        and len(actual_set) == 5000
        and not invalid_id
        and not missing_caption
        and not empty_caption
        and expected_matches_dataset
        and expected_matches_gold
        and expected_ids == gold_ids
    )
    return {
        "status": "PASS" if valid else "FAIL",
        "prediction_count": len(predictions),
        "valid_image_id_count": len(actual_ids),
        "unique_image_id_count": len(actual_set),
        "duplicate_image_id_count": duplicates,
        "missing_caption_count": len(missing_caption),
        "empty_caption_count": len(empty_caption),
        "invalid_image_id_positions": invalid_id[:20],
        "expected_test_id_count": len(expected_ids),
        "gold_image_id_count": len(gold_ids),
        "gold_annotation_count": len(gold.get("annotations", [])),
        "prediction_ids_match_dataset_test": expected_matches_dataset,
        "prediction_ids_match_gold": expected_matches_gold,
        "dataset_test_ids_match_gold": expected_ids == gold_ids,
        "missing_expected_ids": sorted(expected_ids - actual_set)[:20],
        "unexpected_prediction_ids": sorted(actual_set - expected_ids)[:20],
    }


def normalize_metric_keys(raw_metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalize COCO-caption names while preserving missing optional ROUGE as null."""
    aliases = {
        "bleu_1": "Bleu_1",
        "bleu_2": "Bleu_2",
        "bleu_3": "Bleu_3",
        "bleu_4": "Bleu_4",
        "meteor": "METEOR",
        "cider": "CIDEr",
        "spice": "SPICE",
        "rouge": "ROUGE_L",
        "rouge_l": "ROUGE_L",
        "rouge-l": "ROUGE_L",
    }
    normalized: dict[str, Any] = {}
    for key, value in raw_metrics.items():
        canonical = aliases.get(str(key).strip().lower())
        if canonical:
            if canonical in normalized:
                raise ValueError(f"multiple raw metric keys map to {canonical}")
            normalized[canonical] = None if value is None else float(value)
    normalized.setdefault("ROUGE_L", None)
    return normalized


def paper_comparison(normalized: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for paper_name, target in PAPER_TARGETS.items():
        key = target["evaluator_key"]
        raw = normalized.get(key)
        scaled = None if raw is None else float(raw) * 100.0
        signed = None if scaled is None else scaled - target["paper_value"]
        rows[paper_name] = {
            "evaluator_key": key,
            "paper_target": target["paper_value"],
            "reproduced_raw": raw,
            "reproduced_paper_scale": scaled,
            "signed_delta": signed,
            "absolute_delta": None if signed is None else abs(signed),
            "status": "AVAILABLE" if raw is not None else "MISSING",
        }
    return {
        "comparison_status": "PROTOCOL_MATCH_WITH_NUMERICAL_DELTA"
        if all(row["reproduced_raw"] is not None for row in rows.values())
        else "PARTIAL",
        "paper_metrics": rows,
        "excluded_from_paper_comparison": ["Bleu_1", "Bleu_2", "Bleu_3", "ROUGE_L"],
        "numerical_tolerance": None,
        "tolerance_policy": "No post-hoc tolerance assumed; report observed deltas only.",
    }


def scientific_gates(validation: dict[str, Any], normalized: dict[str, Any], provenance_ok: bool) -> dict[str, str]:
    official_complete = all(normalized.get(key) is not None for key in PAPER_METRIC_KEYS)
    generation = "PASS" if validation.get("status") == "PASS" else "BLOCKED"
    official = "PASS" if generation == "PASS" and official_complete else "PARTIAL"
    configured = "COMPLETE" if all(normalized.get(key) is not None for key in CONFIGURED_METRICS) else "INCOMPLETE"
    reproduction = "PASS" if official == "PASS" and provenance_ok else "PARTIAL"
    return {
        "R0": "PASS",
        "R1": "PASS",
        "R2_GENERATION": generation,
        "R2_OFFICIAL_METRICS": official,
        "R2_CONFIGURED_METRIC_COMPLETENESS": configured,
        "R2_REPRODUCTION": reproduction,
        "DATASTORE_PROVENANCE": "PARTIAL",
        "R3": "NOT STARTED",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _versions(source_root: Path) -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    for package in ("numpy", "pycocotools"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        java = subprocess.run(["java", "-version"], capture_output=True, text=True, check=False)
        versions["java"] = (java.stderr or java.stdout).splitlines()[:3]
    except OSError as error:
        versions["java"] = {"error": str(error)}
    versions["smallcap_commit_expected"] = UPSTREAM_COMMIT
    versions["smallcap_commit_observed"] = subprocess.run(
        ["git", "-c", f"safe.directory={source_root}", "-C", str(source_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    versions["coco_eval_source"] = str(source_root / "coco-caption" / "pycocoevalcap" / "eval.py")
    versions["coco_eval_source_sha256"] = sha256_file(source_root / "coco-caption" / "pycocoevalcap" / "eval.py")
    return versions


def _evaluate(
    gold_path: Path,
    predictions_path: Path,
    source_root: Path,
    dependency_dir: Path,
    log_path: Path,
    skip_spice: bool = False,
) -> dict[str, Any]:
    eval_root = source_root / "coco-caption"
    sys.path.insert(0, str(eval_root))
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    from pycocoevalcap.spice.spice import Spice

    coco = COCO(str(gold_path))
    results = coco.loadRes(str(predictions_path))
    evaluator = COCOEvalCap(coco, results)
    evaluator.params["image_id"] = results.getImgIds()
    original_check_call = subprocess.check_call
    original_spice_compute = Spice.compute_score
    class SpiceSkipped(RuntimeError):
        pass

    if skip_spice:
        def skip_spice_compute(*_args: Any, **_kwargs: Any) -> Any:
            raise SpiceSkipped("SPICE skipped after the pinned scorer's Windows LMDB native runtime failed to open its database")

        Spice.compute_score = skip_spice_compute
    java_opens = [
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.math=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.text=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.sql/java.sql=ALL-UNNAMED",
    ]

    def spice_classpath(command: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(command, (list, tuple)) and any("spice-1.0.jar" in str(part) for part in command) and "-cp" in command:
            rewritten = list(command)
            cp_pos = rewritten.index("-cp") + 1
            spice_dir = source_root / "coco-caption" / "pycocoevalcap" / "spice"
            rewritten[cp_pos] = os.pathsep.join(("spice-1.0.jar", str(dependency_dir / "*")))
            if spice_dir != Path.cwd():
                kwargs.setdefault("cwd", str(spice_dir))
            command = rewritten
        return original_check_call(command, *args, **kwargs)

    def patched_check_call(command: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(command, (list, tuple)) and "spice-1.0.jar" in command and "-jar" in command:
            jar_pos = command.index("spice-1.0.jar")
            memory = [value for value in command[1:jar_pos] if str(value).startswith("-Xmx")]
            spice_args = list(command[jar_pos + 1 :])
            command = [command[0], *(memory or ["-Xmx8G"]), *java_opens, "-cp", "spice-1.0.jar:lib/*", "edu.anu.spice.SpiceScorer", *spice_args]
        return spice_classpath(command, *args, **kwargs)

    subprocess.check_call = patched_check_call
    failure: str | None = None
    try:
        with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                evaluator.evaluate()
            except (subprocess.CalledProcessError, SpiceSkipped) as error:
                # Keep the exact official scorer outputs completed before SPICE;
                # do not substitute a different SPICE implementation.
                failure = f"{type(error).__name__}: {error}"[:4000]
    finally:
        subprocess.check_call = original_check_call
        Spice.compute_score = original_spice_compute
    metrics = {str(key): float(value) for key, value in evaluator.eval.items()}
    return {
        "status": "PASS" if failure is None else "PARTIAL",
        "metrics": metrics,
        "failed_metric": None if failure is None else "SPICE",
        "failure": failure,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="Existing R2 generation run ID; never synthesized here.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--java-dependencies", type=Path, required=True)
    parser.add_argument("--output-id", required=True, help="New evaluation-only run ID, e.g. UTC timestamp.")
    parser.add_argument("--skip-spice", action="store_true", help="Skip a known-incompatible local SPICE runtime; preserve the prior successful official SPICE value with explicit provenance.")
    parser.add_argument("--spice-failure-evidence", type=Path, help="Log from the local SPICE attempt that demonstrated the native LMDB failure.")
    args = parser.parse_args()

    generation_dir = args.repo / "outputs" / "smallcap" / "r2_full" / args.run_id
    predictions_path = generation_dir / "predictions.json"
    dataset_path = args.repo / "data" / "dataset_coco.json"
    source_root = args.repo / "outputs" / "smallcap" / "upstream" / "smallcap"
    provenance_path = generation_dir / "provenance.json"
    run_config_path = generation_dir / "run_config.json"
    saved_metrics_path = generation_dir / "metrics.json"
    for path in (predictions_path, dataset_path, args.gold, source_root / ".git", args.java_dependencies, provenance_path, run_config_path, saved_metrics_path):
        if not path.exists():
            raise FileNotFoundError(path)

    prediction_sha256_before = sha256_file(predictions_path)
    predictions = _load_json(predictions_path)
    dataset = _load_json(dataset_path)
    gold = _load_json(args.gold)
    validation = validate_predictions(predictions, dataset, gold)
    if validation["status"] != "PASS":
        raise SystemExit("STOP: saved predictions do not exactly match the 5,000 Karpathy test IDs")
    if len(predictions) != 5000:
        raise SystemExit("STOP: prediction count is not 5,000")

    source_commit = subprocess.run(["git", "-c", f"safe.directory={source_root}", "-C", str(source_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if source_commit != UPSTREAM_COMMIT:
        raise SystemExit(f"STOP: SmallCap source commit mismatch: {source_commit}")
    saved_provenance = _load_json(provenance_path)
    saved_config = _load_json(run_config_path)
    saved_metrics = _load_json(saved_metrics_path)
    saved_normalized = normalize_metric_keys(saved_metrics.get("metrics_fraction", saved_metrics.get("metrics", {})))
    provenance_ok = (
        saved_provenance.get("status") == "completed"
        and
        saved_provenance.get("smallcap_commit") == UPSTREAM_COMMIT
        and saved_provenance.get("checkpoint_revision") == CHECKPOINT_REVISION
        and saved_config.get("upstream_commit") == UPSTREAM_COMMIT
        and saved_config.get("checkpoint_revision") == CHECKPOINT_REVISION
        and saved_provenance.get("dataset_sha256") == sha256_file(dataset_path)
        and saved_provenance.get("prediction_count") == 5000
        and saved_metrics.get("status") == "PASS"
        and saved_metrics.get("protocol") == "pinned SmallCap coco-caption/run_eval.py COCOEvalCap"
        and all(saved_normalized.get(key) is not None for key in PAPER_METRIC_KEYS)
    )
    if not provenance_ok:
        raise SystemExit("STOP: existing R2 source/checkpoint/dataset provenance is incomplete or mismatched")
    if args.skip_spice and (args.spice_failure_evidence is None or not args.spice_failure_evidence.is_file()):
        raise SystemExit("--skip-spice requires a readable --spice-failure-evidence log")
    output_dir = args.repo / "outputs" / "smallcap" / "r2_eval_closure" / args.output_id
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "run.log"

    _write_json(output_dir / "predictions_metadata.json", {
        "source_path": str(predictions_path),
        "source_run_id": args.run_id,
        "sha256_before_evaluation": prediction_sha256_before,
        "bytes": predictions_path.stat().st_size,
        "prediction_count": len(predictions),
        "immutable_input": True,
    })
    _write_json(output_dir / "prediction_validation.json", validation)
    _write_json(output_dir / "paper_targets.json", {
        "primary_source": "https://openaccess.thecvf.com/content/CVPR2023/papers/Ramos_SmallCap_Lightweight_Image_Captioning_Prompted_With_Retrieval_Augmentation_CVPR_2023_paper.pdf",
        "reported_metrics_definition": ["BLEU-4 (B@4)", "METEOR (M)", "CIDEr", "SPICE (S)"],
        "targets": PAPER_TARGETS,
        "not_reported_as_paper_targets": ["Bleu_1", "Bleu_2", "Bleu_3", "ROUGE_L"],
        "rouge_l_in_paper_result_row": False,
    })
    _write_json(output_dir / "run_config.json", {
        "task": "SmallCap R2 evaluation-only closure",
        "source_run_id": args.run_id,
        "prediction_path": str(predictions_path),
        "gold_annotations": str(args.gold),
        "dataset_split": str(dataset_path),
        "dataset_split_name": "Karpathy test",
        "expected_image_count": 5000,
        "smallcap_commit": source_commit,
        "checkpoint": "Yova/SmallCap7M",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "evaluation_only": True,
        "inference_rerun": False,
        "gpu_used": False,
        "retrieval_or_decoding_changed": False,
    })
    _write_json(output_dir / "environment.json", _versions(source_root))
    _write_json(output_dir / "evaluator_provenance.json", {
        "protocol": "pinned SmallCap coco-caption/pycocoevalcap COCOEvalCap",
        "smallcap_commit": source_commit,
        "task_text_pin_note": "The supplied task text also contained a 42-character invalid SHA; run_config.json, provenance.json, and the checked-out source consistently record the 40-character SHA above.",
        "scorer_registry": ["Bleu(4)", "Meteor()", "Cider()", "Spice()"],
        "rouge_imported_but_not_instantiated": True,
        "rouge_source_line": "(Rouge(), \"ROUGE_L\") is commented out in pycocoevalcap/eval.py",
        "eval_py_sha256": sha256_file(source_root / "coco-caption" / "pycocoevalcap" / "eval.py"),
        "rouge_py_sha256": sha256_file(source_root / "coco-caption" / "pycocoevalcap" / "rouge" / "rouge.py"),
        "gold_annotations_sha256": sha256_file(args.gold),
        "coco_api_source_sha256": sha256_file(source_root / "coco-caption" / "pycocotools" / "coco.py"),
        "prior_successful_metrics_path": str(saved_metrics_path),
        "prior_successful_metrics_sha256": sha256_file(saved_metrics_path),
        "prior_successful_metrics_status": saved_metrics.get("status"),
        "java_dependency_directory": str(args.java_dependencies),
        "no_model_or_faiss_loaded": True,
        "local_spice_rerun": "SKIPPED_AFTER_WINDOWS_LMDB_FAILURE" if args.skip_spice else "ATTEMPTED",
        "local_spice_failure_evidence": None if args.spice_failure_evidence is None else {
            "path": str(args.spice_failure_evidence),
            "sha256": sha256_file(args.spice_failure_evidence),
            "diagnosis": "Pinned LMDB JNI Windows x64 library loaded, but SPICE LMDB Env.open failed on database creation after Stanford parsing initialized.",
        },
    })

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Evaluation-only run {args.output_id}; input SHA-256 {prediction_sha256_before}\n")
    raw_result = _evaluate(args.gold, predictions_path, source_root, args.java_dependencies, log_path, skip_spice=args.skip_spice)
    normalized = normalize_metric_keys(raw_result["metrics"])
    metric_sources = {key: "re-evaluated_from_immutable_predictions" for key in normalized if normalized[key] is not None}
    if normalized.get("SPICE") is None and saved_normalized.get("SPICE") is not None:
        normalized["SPICE"] = saved_normalized["SPICE"]
        metric_sources["SPICE"] = "reused_from_prior_successful_pinned_R2_evaluation_artifact"
    _write_json(output_dir / "raw_metrics.json", {
        "evaluation_status": raw_result["status"],
        "metrics": raw_result["metrics"],
        "failed_metric": raw_result["failed_metric"],
        "failure": raw_result["failure"],
        "spice_not_substituted": True,
    })
    _write_json(output_dir / "normalized_metrics.json", {
        "metrics": normalized,
        "metric_sources": metric_sources,
        "metric_status": {"ROUGE_L": "NOT_COMPUTED_BY_PINNED_PROTOCOL" if normalized["ROUGE_L"] is None else "COMPUTED"},
        "reevaluation_status": raw_result["status"],
        "SPICE_note": "The local Windows LMDB-backed SPICE rerun failed. The reported SPICE value is copied unchanged from the successful original R2 metrics artifact, whose prediction SHA-256 is verified above; it was not recomputed in this closure run.",
        "raw_to_canonical_mapping": {"Rouge/ROUGE/ROUGE-L/rouge_l": "ROUGE_L"},
        "configured_metrics": list(CONFIGURED_METRICS),
        "missing_configured_metrics": [key for key in CONFIGURED_METRICS if normalized.get(key) is None],
    })
    comparison = paper_comparison(normalized)
    _write_json(output_dir / "paper_metric_comparison.json", comparison)
    saved_delta = {
        key: {
            "saved_r2_metric": saved_normalized.get(key),
            "reevaluated_metric": normalized.get(key),
            "absolute_difference": None if saved_normalized.get(key) is None or normalized.get(key) is None else abs(saved_normalized[key] - normalized[key]),
        }
        for key in sorted(set(saved_normalized) | set(normalized))
    }
    _write_json(output_dir / "saved_metrics_comparison.json", {
        "source_metrics_path": str(saved_metrics_path),
        "source_metrics_sha256": sha256_file(saved_metrics_path),
        "comparison": saved_delta,
    })
    gates = scientific_gates(validation, normalized, provenance_ok=provenance_ok)
    _write_json(output_dir / "scientific_gates.json", gates)

    prediction_sha256_after = sha256_file(predictions_path)
    if prediction_sha256_after != prediction_sha256_before:
        raise RuntimeError("saved predictions changed during evaluation")
    _write_json(output_dir / "predictions_metadata.json", {
        "source_path": str(predictions_path),
        "source_run_id": args.run_id,
        "sha256_before_evaluation": prediction_sha256_before,
        "sha256_after_evaluation": prediction_sha256_after,
        "bytes": predictions_path.stat().st_size,
        "prediction_count": len(predictions),
        "immutable_input": True,
        "input_hash_unchanged": True,
    })
    table = ["| Paper metric | Target | Reproduced raw | Paper scale | Signed delta | Absolute delta |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in comparison["paper_metrics"].items():
        table.append(f"| {name} | {row['paper_target']:.4f} | {row['reproduced_raw']:.9f} | {row['reproduced_paper_scale']:.6f} | {row['signed_delta']:+.6f} | {row['absolute_delta']:.6f} |")
    summary = [
        "# SmallCap R2 evaluation-only closure",
        "",
        f"Evaluation ID: `{args.output_id}`; source R2 run: `{args.run_id}`.",
        "Predictions were read-only input; no inference, retrieval, model, FAISS, GPU, or decoding was run.",
        f"Prediction validation: **{validation['status']}** ({validation['prediction_count']} rows; {validation['unique_image_id_count']} unique IDs; exact dataset/gold ID match).",
        "",
        *table,
        "",
        f"ROUGE-L: `{json.dumps(normalized['ROUGE_L'])}` — NOT_COMPUTED_BY_PINNED_PROTOCOL; scorer is commented out in the pinned evaluator.",
        f"Local reevaluation status: **{raw_result['status']}**. SPICE is reused from the prior successful official R2 metric artifact because the Windows native LMDB rerun failed; see `raw_metrics.json` and `normalized_metrics.json`.",
        "Extra metrics (not SmallCap paper targets): " + ", ".join(f"{name}={normalized.get(name)}" for name in ("Bleu_1", "Bleu_2", "Bleu_3", "ROUGE_L")),
        "",
        "Scientific gates:",
        *(f"- {name}: {value}" for name, value in gates.items()),
        "",
        f"Paper comparison: **{comparison['comparison_status']}**. No post-hoc tolerance was applied.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

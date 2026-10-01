#!/usr/bin/env python3
"""Generate and evaluate SmallCap on all 5,000 Karpathy test images.

This is a separate full-evaluation runner. It preserves retrievals and partial
generation checkpoints so a Kaggle runtime interruption can be resumed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capybara.experiments import git_commit, seed_everything
from run_smallcap_smoke import (
    UPSTREAM_URL,
    apply_smallcap_registration,
    environment_info,
    git_status,
    resolve_model,
    run_retrieval,
    validate_retrieval,
)


EXPECTED_UPSTREAM_COMMIT = "19cc4f4e5972c70fde16c8857b14c96e5492cb30"
EXPECTED_MODEL_REVISION = "fd635f4025a5ccb638ed1e1b540c3671557b9a16"
EXPECTED_DATASET_SHA256 = "2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d"


class RunLog:
    def __init__(self, path: Path, append: bool = False) -> None:
        self.path = path
        if not append:
            self.path.write_text("", encoding="utf-8")

    def write(self, message: str) -> None:
        line = f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] {message}\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
        print(message, flush=True)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/smallcap_r2.json")
    parser.add_argument("--smallcap-root", type=Path, required=True)
    parser.add_argument("--dataset-coco", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--retrieval-index", type=Path, required=True)
    parser.add_argument("--retrieval-captions", type=Path, required=True)
    parser.add_argument("--gold-annotations", type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    return parser.parse_args()


def validate_r2_config(config: dict[str, Any]) -> None:
    expected = {
        "experiment_id": "smallcap_coco_karpathy_test_r2_full_001",
        "checkpoint_id": "Yova/SmallCap7M",
        "checkpoint_revision": EXPECTED_MODEL_REVISION,
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "split": "Karpathy test",
        "sample_limit": 5000,
        "retrieval_k": 4,
        "retrieval_encoder": "RN50x64",
        "generation_visual_encoder": "openai/clip-vit-base-patch32",
        "decoder": "gpt2",
        "num_beams": 3,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"R2 config mismatch for {key}: expected {value!r}, got {config.get(key)!r}")


def load_test_rows(dataset_path: Path, gold_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    images = dataset.get("images") if isinstance(dataset, dict) else None
    if not isinstance(images, list):
        raise ValueError("Karpathy dataset must contain an images list")
    test_records = [item for item in images if item.get("split") == "test"]
    if len(test_records) != 5000:
        raise ValueError(f"expected exactly 5000 Karpathy test records, found {len(test_records)}")
    rows = []
    for record in test_records:
        if "cocoid" not in record or "filename" not in record:
            raise ValueError("Karpathy test image is missing cocoid or filename")
        rows.append({
            "image_id": int(record["cocoid"]),
            "file_name": str(record["filename"]).split("_")[-1],
            "references": [" ".join(sentence["tokens"]) for sentence in record.get("sentences", [])],
        })
    ids = [row["image_id"] for row in rows]
    if len(set(ids)) != 5000:
        raise ValueError("Karpathy test image IDs are not unique")

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_images = gold.get("images", [])
    gold_annotations = gold.get("annotations", [])
    gold_ids = {int(item["id"]) for item in gold_images}
    if len(gold_images) != 5000 or gold_ids != set(ids):
        raise ValueError("pinned COCOEvalCap gold image IDs do not match the 5,000 Karpathy test records")
    gold_reference_count = sum(int(item["image_id"]) in gold_ids for item in gold_annotations)
    if gold_reference_count != 25010:
        raise ValueError(f"expected 25,010 test references in gold annotations, found {gold_reference_count}")
    return rows, {
        "test_image_count": len(rows),
        "unique_test_image_ids": len(set(ids)),
        "test_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "gold_annotation_image_count": len(gold_images),
        "gold_reference_count": gold_reference_count,
        "ids_match_gold": True,
    }


def resolve_r2_images(rows: list[dict[str, Any]], images_root: Path) -> list[dict[str, Any]]:
    image_dirs = [
        path for path in images_root.rglob("*")
        if path.is_dir() and path.name in {"train2017", "val2017", "train2014", "val2014"}
    ]
    if images_root.name in {"train2017", "val2017", "train2014", "val2014"}:
        image_dirs.append(images_root)
    priorities = {"val2017": 0, "train2017": 1, "val2014": 2, "train2014": 3}
    image_dirs = sorted(set(path.resolve() for path in image_dirs), key=lambda path: (priorities[path.name], str(path)))
    if not image_dirs:
        raise FileNotFoundError(f"could not find COCO train/val image directories below {images_root}")
    resolved = []
    for row in rows:
        filename = f"{int(row['image_id']):012d}.jpg"
        candidates = [directory / filename for directory in image_dirs if (directory / filename).is_file()]
        if not candidates:
            raise FileNotFoundError(f"missing COCO test image {filename} below {images_root}")
        # Prefer the 2017 release used in the R1 Kaggle harness; if absent,
        # fall back to the 2014 Karpathy source image directories.
        resolved.append({**row, "image_path": str(candidates[0])})
    from PIL import Image

    for row in resolved:
        with Image.open(row["image_path"]) as image:
            image.verify()
    return resolved


def resume_or_create_run(output_root: Path, resume_dir: Path | None) -> tuple[Path, str, bool]:
    if resume_dir is not None:
        run_dir = resume_dir.resolve()
        if not run_dir.is_dir() or not (run_dir / "run_config.json").is_file():
            raise FileNotFoundError(f"resume directory is not a valid R2 run: {run_dir}")
        return run_dir, run_dir.name, True
    base = output_root / "smallcap" / "r2_full"
    base.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{suffix}"
        run_dir = base / run_id
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id, False


def load_partial_predictions(path: Path, expected_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, dict[str, Any]] = {}
    for row in payload:
        image_id = int(row["image_id"])
        if image_id not in expected_ids:
            raise ValueError(f"partial prediction contains unexpected COCO ID {image_id}")
        if image_id in result or not row.get("caption"):
            raise ValueError(f"partial predictions contain duplicate ID or empty caption: {image_id}")
        result[image_id] = row
    return result


def generate_predictions(
    rows: list[dict[str, Any]],
    retrievals: list[dict[str, Any]],
    args: argparse.Namespace,
    config: dict[str, Any],
    run_dir: Path,
    log: RunLog,
    checkpoint_every: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer, CLIPFeatureExtractor

    _, prep_strings, postprocess_preds, _, _, _ = apply_smallcap_registration(args.smallcap_root)
    tokenizer = AutoTokenizer.from_pretrained(config["decoder"], revision=config["decoder_revision"])
    tokenizer.pad_token = "!"
    tokenizer.eos_token = "."
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        config["generation_visual_encoder"], revision=config["generation_visual_encoder_revision"]
    )
    model = AutoModel.from_pretrained(
        config["checkpoint_id"], revision=config["checkpoint_revision"]
    ).eval().cuda()
    resolved_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if resolved_revision and resolved_revision != config["checkpoint_revision"]:
        raise RuntimeError(f"checkpoint revision mismatch: {resolved_revision}")

    template = (args.smallcap_root / "src" / "template.txt").read_text(encoding="utf-8").strip() + " "
    retrieval_by_id = {int(record["coco_id"]): record["retrieval"] for record in retrievals}
    expected_ids = [int(row["image_id"]) for row in rows]
    partial_path = run_dir / "predictions_partial.json"
    completed = load_partial_predictions(partial_path, expected_ids)
    generation = config["generation_parameters"]
    generation_kwargs = {
        "max_new_tokens": int(generation["max_new_tokens"]),
        "no_repeat_ngram_size": int(generation["no_repeat_ngram_size"]),
        "length_penalty": float(generation["length_penalty"]),
        "num_beams": int(generation["num_beams"]),
        "early_stopping": bool(generation["early_stopping"]),
        "eos_token_id": tokenizer.eos_token_id,
    }
    latencies = []
    log.write(f"generation resumes with {len(completed)}/{len(rows)} images already complete")
    for position, row in enumerate(rows, 1):
        image_id = int(row["image_id"])
        if image_id in completed:
            continue
        started = time.perf_counter()
        retrieved = [entry["caption"] for entry in retrieval_by_id[image_id]]
        if len(retrieved) != int(config["retrieval_k"]):
            raise RuntimeError(f"expected k={config['retrieval_k']} retrievals for COCO ID {image_id}")
        decoder_ids = prep_strings(
            "", tokenizer, template=template, retrieved_caps=retrieved,
            k=int(config["retrieval_k"]), is_test=True,
        )
        with Image.open(row["image_path"]) as opened:
            pixels = feature_extractor(opened.convert("RGB"), return_tensors="pt").pixel_values.cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            generated = model.generate(
                pixels,
                decoder_input_ids=torch.tensor([decoder_ids], device="cuda"),
                **generation_kwargs,
            )
        decoded = tokenizer.decode(generated[0])
        caption = postprocess_preds(decoded, tokenizer).strip()
        torch.cuda.synchronize()
        if not caption:
            raise RuntimeError(f"SmallCap generated an empty caption for COCO ID {image_id}")
        completed[image_id] = {"image_id": image_id, "decoded": decoded, "caption": caption}
        latencies.append(time.perf_counter() - started)
        if len(completed) % checkpoint_every == 0 or len(completed) == len(rows):
            ordered_partial = [completed[value] for value in expected_ids if value in completed]
            write_json(partial_path, ordered_partial)
            write_json(run_dir / "generation_progress.json", {
                "completed": len(completed), "total": len(rows), "last_completed_image_id": image_id,
                "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            })
        if position % 25 == 0 or position == len(rows):
            log.write(f"generation progress: {len(completed)}/{len(rows)}; latest COCO ID={image_id}")
    predictions = [completed[value] for value in expected_ids]
    return predictions, {
        "model_revision": resolved_revision or config["checkpoint_revision"],
        "generation_parameters": generation_kwargs,
        "timed_sample_count_this_process": len(latencies),
        "mean_latency_seconds_this_process": statistics.mean(latencies) if latencies else None,
    }


def evaluate_predictions(gold_path: Path, predictions_path: Path, smallcap_root: Path) -> dict[str, Any]:
    eval_root = smallcap_root / "coco-caption"
    sys.path.insert(0, str(eval_root))
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    coco = COCO(str(gold_path))
    coco_res = coco.loadRes(str(predictions_path))
    evaluator = COCOEvalCap(coco, coco_res)
    evaluator.params["image_id"] = coco_res.getImgIds()
    evaluator.evaluate()
    scores = {name: float(value) for name, value in evaluator.eval.items()}
    return {
        "status": "PASS",
        "protocol": "pinned SmallCap coco-caption/run_eval.py COCOEvalCap",
        "gold_annotations": str(gold_path),
        "image_count": len(evaluator.params["image_id"]),
        "metrics_fraction": scores,
        "metrics_percent": {name: value * 100.0 for name, value in scores.items()},
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_r2_config(config)
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be a positive integer")
    gold_path = args.gold_annotations or (args.smallcap_root / "coco-caption" / "annotations" / "captions_testKarpathy.json")
    for path, label in (
        (args.smallcap_root / ".git", "SmallCap Git checkout"),
        (args.dataset_coco, "Karpathy dataset_coco.json"),
        (args.images_root, "COCO images root"),
        (args.retrieval_index, "FAISS index"),
        (args.retrieval_captions, "public caption list"),
        (gold_path, "pinned COCO test gold annotations"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")

    source = git_status(args.smallcap_root)
    if source["commit"] != config["upstream_commit"] or source["dirty"]:
        raise RuntimeError(f"SmallCap source is not clean at the pinned commit: {source}")

    actual_dataset_sha = sha256_file(args.dataset_coco)
    if actual_dataset_sha != EXPECTED_DATASET_SHA256:
        raise ValueError(f"Karpathy dataset SHA-256 mismatch: expected {EXPECTED_DATASET_SHA256}, got {actual_dataset_sha}")
    rows, gold_info = load_test_rows(args.dataset_coco, gold_path)
    rows = resolve_r2_images(rows, args.images_root)
    index, captions, index_info = validate_retrieval(args.retrieval_index, args.retrieval_captions)
    if type(index).__name__ != "IndexFlatIP" or index.d != 1024 or index.ntotal != 565497:
        raise RuntimeError(f"public datastore does not match R2 gate: {type(index).__name__}, d={index.d}, n={index.ntotal}")

    output_dir, run_id, is_resume = resume_or_create_run(args.output_root, args.resume_run)
    log = RunLog(output_dir / "run.log", append=is_resume)
    run_config = {
        **config,
        "run_id": run_id,
        "dataset_coco": str(args.dataset_coco),
        "images_root": str(args.images_root),
        "retrieval_index": str(args.retrieval_index),
        "retrieval_captions": str(args.retrieval_captions),
        "gold_annotations": str(gold_path),
        "smallcap_root": str(args.smallcap_root),
        "checkpoint_every": args.checkpoint_every,
        "capybara_commit": git_commit(REPO_ROOT),
    }
    run_config_path = output_dir / "run_config.json"
    if is_resume:
        old_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        for key in ("experiment_id", "checkpoint_revision", "upstream_commit", "sample_limit", "retrieval_k", "seed"):
            if old_config.get(key) != run_config.get(key):
                raise ValueError(f"resume run config mismatch for {key}")
    else:
        write_json(run_config_path, run_config)
    provenance = {
        "experiment": "smallcap_r2_full",
        "purpose": "full_karpathy_test_reproduction",
        "run_id": run_id,
        "status": "running",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "capybara_commit": run_config["capybara_commit"],
        "smallcap_repository": UPSTREAM_URL,
        "smallcap_commit": source["commit"],
        "smallcap_dirty": source["dirty"],
        "checkpoint": config["checkpoint_id"],
        "checkpoint_revision": config["checkpoint_revision"],
        "dataset": str(args.dataset_coco),
        "dataset_sha256": sha256_file(args.dataset_coco),
        "dataset_test_split": gold_info,
        "selection_order": config["selection_order"],
        "num_samples": len(rows),
        "seed": config["seed"],
        "retrieval": {**index_info, "index_sha256": sha256_file(args.retrieval_index), "captions_sha256": sha256_file(args.retrieval_captions)},
        "retrieval_protocol": {
            "query_encoder": config["retrieval_encoder"],
            "search": "raw exact FAISS top-k from public IndexFlatIP",
            "same_image_filter": False,
            "same_image_filter_rationale": "Karpathy test query IDs are disjoint from verified train+restval datastore source IDs",
        },
        "image_paths_validated": len(rows),
        "resume": is_resume,
    }
    write_json(output_dir / "provenance.json", provenance)
    write_json(output_dir / "selected_ids.json", [
        {"image_id": int(row["image_id"]), "image_path": row["image_path"]} for row in rows
    ])

    try:
        import torch

        seed_everything(int(config["seed"]))
        environment = environment_info(torch)
        if not environment["cuda_available"]:
            raise RuntimeError("CUDA is unavailable; R2 requires a GPU")
        write_json(output_dir / "environment.json", environment)
        model_resolution = resolve_model(config)
        if model_resolution["resolved_revision"] != config["checkpoint_revision"]:
            raise RuntimeError(f"checkpoint revision mismatch: {model_resolution}")
        log.write(f"preflight PASS: {len(rows)} Karpathy test images; GPU={environment['gpu_name']}")
        log.write(f"SmallCap source clean at {source['commit']}; checkpoint revision pinned")

        retrieval_path = output_dir / "retrievals.json"
        if retrieval_path.is_file():
            retrievals = json.loads(retrieval_path.read_text(encoding="utf-8"))
            retrieved_ids = [int(record["coco_id"]) for record in retrievals]
            expected_ids = [int(row["image_id"]) for row in rows]
            if retrieved_ids != expected_ids or any(len(record.get("retrieval", [])) != 4 for record in retrievals):
                raise ValueError("saved retrieval checkpoint does not match this exact test split")
            log.write("loaded verified retrieval checkpoint")
        else:
            retrievals = run_retrieval(rows, index, captions, config, log)
            write_json(retrieval_path, retrievals)
        del index

        raw_predictions, generation_info = generate_predictions(
            rows, retrievals, args, config, output_dir, log, args.checkpoint_every
        )
        predictions = [{"image_id": int(row["image_id"]), "caption": row["caption"]} for row in raw_predictions]
        if len(predictions) != 5000 or len({item["image_id"] for item in predictions}) != 5000:
            raise RuntimeError("R2 prediction set is incomplete or has duplicate image IDs")
        if [item["image_id"] for item in predictions] != [int(row["image_id"]) for row in rows]:
            raise RuntimeError("prediction order/IDs differ from the full Karpathy test split")
        write_json(output_dir / "raw_predictions.json", raw_predictions)
        write_json(output_dir / "predictions.json", predictions)
        write_json(output_dir / "generation_info.json", generation_info)

        metrics = evaluate_predictions(gold_path, output_dir / "predictions.json", args.smallcap_root)
        write_json(output_dir / "metrics.json", metrics)
        provenance.update({
            "status": "completed",
            "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "prediction_count": len(predictions),
            "metrics_status": metrics["status"],
            "metrics": metrics["metrics_fraction"],
        })
        write_json(output_dir / "provenance.json", provenance)
        summary_lines = [
            "# SmallCap R2 full Karpathy test evaluation",
            "",
            "Status: PASS",
            "",
            f"- Images: {len(predictions)}",
            f"- Source commit: `{source['commit']}`",
            f"- Checkpoint: `{config['checkpoint_id']}@{config['checkpoint_revision']}`",
            f"- Datastore: `IndexFlatIP`, {index_info['ntotal']} captions, dimension {index_info['dimension']}",
            "- Metrics:",
        ]
        summary_lines.extend(f"  - {name}: {value:.6f}" for name, value in metrics["metrics_fraction"].items())
        (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        log.write("R2 completed: predictions and official COCO metrics saved")
        print(output_dir)
        print(json.dumps(metrics["metrics_fraction"], indent=2))
        return 0
    except Exception as exc:
        provenance.update({
            "status": "failed",
            "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        })
        write_json(output_dir / "provenance.json", provenance)
        log.write(f"R2 failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

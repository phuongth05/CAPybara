#!/usr/bin/env python3
"""Run the four-sample, provenance-tracked SmallCap R1 smoke test."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara.experiments import git_commit

UPSTREAM_URL = "https://github.com/RitaRamo/smallcap.git"
COMPATIBILITY_PATCHES = [
    "COMPATIBILITY PATCH — no intended methodological change: Transformers 4.36 past_key_values bridge.",
    "COMPATIBILITY PATCH — no intended methodological change: Python 3.12 package bridge from the verified Kaggle harness.",
]


class RunLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.write_text("", encoding="utf-8")

    def write(self, message: str) -> None:
        line = f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] {message}\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
        print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smallcap-root", type=Path, required=True)
    parser.add_argument("--karpathy-annotations", type=Path, required=True)
    parser.add_argument("--retrieval-index", type=Path, required=True)
    parser.add_argument("--retrieval-captions", type=Path, required=True)
    parser.add_argument("--coco-input-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/smallcap_smoke.json")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required {label}: {path}")


def package_versions() -> dict[str, str | None]:
    names = ("torch", "transformers", "tokenizers", "huggingface-hub", "faiss-cpu", "numpy", "Pillow")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_status(repo: Path) -> dict[str, Any]:
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    remotes = subprocess.run(["git", "-C", str(repo), "remote", "-v"], capture_output=True, text=True, check=True).stdout
    return {"commit": git_commit(repo), "dirty": bool(status.strip()), "status_porcelain": status, "remote_v": remotes}


def make_run_dir(output_root: Path) -> tuple[Path, str]:
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / "smallcap" / "r1_smoke" / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"{run_id}_{suffix}"
        run_dir = output_root / "smallcap" / "r1_smoke" / run_id
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        "sample_limit": 4,
        "seed": 2026,
        "retrieval_k": 4,
        "retrieval_encoder": "RN50x64",
        "generation_visual_encoder": "openai/clip-vit-base-patch32",
        "decoder": "gpt2",
        "num_beams": 3,
        "checkpoint_id": "Yova/SmallCap7M",
        "split": "Karpathy test",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"R1 config mismatch for {key}: expected {value!r}, got {config.get(key)!r}")


def load_rows(annotation_path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 5000:
        raise ValueError("Karpathy test annotation must be a list of exactly 5000 records")
    rows = []
    for item in raw:
        match = re.search(r"(\d{12})\.jpg$", str(item.get("image", "")))
        if not match:
            raise ValueError(f"invalid Karpathy image name: {item.get('image')!r}")
        rows.append({"image_id": int(match.group(1)), "references": item.get("caption", [])})
    rows.sort(key=lambda row: row["image_id"])
    random.Random(seed).shuffle(rows)
    selected = rows[:limit]
    if len(selected) != 4 or len({row["image_id"] for row in selected}) != 4:
        raise ValueError("R1 must select exactly four unique COCO IDs")
    return selected


def find_image_dirs(coco_input_root: Path) -> list[Path]:
    dirs = []
    for name in ("train2017", "val2017"):
        dirs.extend(path for path in coco_input_root.rglob(name) if path.is_dir())
    if coco_input_root.name in ("train2017", "val2017"):
        dirs.append(coco_input_root)
    return sorted(set(path.resolve() for path in dirs))


def resolve_and_validate_images(rows: list[dict[str, Any]], coco_input_root: Path) -> list[dict[str, Any]]:
    from PIL import Image

    image_dirs = find_image_dirs(coco_input_root)
    if not image_dirs:
        raise FileNotFoundError(f"could not find train2017/val2017 below {coco_input_root}")
    validated = []
    for row in rows:
        filename = f"{row['image_id']:012d}.jpg"
        candidates = [directory / filename for directory in image_dirs if (directory / filename).is_file()]
        if len(candidates) != 1:
            raise FileNotFoundError(f"expected exactly one image for COCO ID {row['image_id']}, found {candidates}")
        image_path = candidates[0]
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            validated.append({**row, "image_path": str(image_path), "file_exists": True, "opened": True, "rgb_conversion": True, "size": list(rgb.size), "mode": image.mode})
            rgb.close()
    return validated


def environment_info(torch_module: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version, "packages": package_versions(), "torch": torch_module.__version__, "cuda": torch_module.version.cuda, "cuda_available": bool(torch_module.cuda.is_available())}
    try:
        import clip
        info["clip"] = {"module": str(Path(clip.__file__).resolve()), "package_provenance": "OpenAI CLIP commit a9b1bf5920416aaeaec965c25dd9e8f98c864f16"}
    except Exception as exc:
        info["clip"] = {"module": None, "package_provenance": None, "error": f"{type(exc).__name__}: {exc}"}
    if torch_module.cuda.is_available():
        props = torch_module.cuda.get_device_properties(0)
        info.update({"gpu_name": props.name, "gpu_vram_gib": props.total_memory / 2**30, "gpu_capability": list(torch_module.cuda.get_device_capability(0))})
    else:
        info.update({"gpu_name": None, "gpu_vram_gib": None, "gpu_capability": None})
    return info


def resolve_model(config: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    revision = config["checkpoint_revision"]
    config_path = hf_hub_download(config["checkpoint_id"], "config.json", revision=revision)
    return {"model": config["checkpoint_id"], "requested_revision": revision, "config_path": str(config_path), "resolved_revision": revision}


def validate_retrieval(index_path: Path, captions_path: Path) -> tuple[Any, list[str], dict[str, Any]]:
    import faiss

    captions = json.loads(captions_path.read_text(encoding="utf-8"))
    if not isinstance(captions, list) or not all(isinstance(caption, str) for caption in captions):
        raise ValueError("retrieval captions must be a JSON list of strings")
    index = faiss.read_index(str(index_path))
    if index.d <= 0 or index.ntotal <= 0 or not math.isfinite(float(index.d)) or not math.isfinite(float(index.ntotal)):
        raise ValueError("FAISS index has invalid dimension or count")
    if index.ntotal != len(captions):
        raise ValueError(f"FAISS/caption cardinality mismatch: {index.ntotal} != {len(captions)}")
    if len(captions) < 100000:
        raise ValueError(f"caption count is not at the expected COCO datastore scale: {len(captions)}")
    return index, captions, {"ntotal": int(index.ntotal), "dimension": int(index.d), "caption_count": len(captions), "index_type": type(index).__name__, "metadata_nan_check": "dimension/count finite; vector scan not performed"}


def apply_smallcap_registration(smallcap_root: Path) -> tuple[Any, Any, Any, Any, Any, Any]:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    sys.path.insert(0, str(smallcap_root))
    from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
    from src.utils import postprocess_preds, prep_strings
    from src.vision_encoder_decoder import SmallCap, SmallCapConfig

    # COMPATIBILITY PATCH — no intended methodological change.
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        return {"attention_mask": attention_mask, "decoder_attention_mask": decoder_inputs.get("attention_mask"), "decoder_input_ids": decoder_inputs["input_ids"], "encoder_outputs": encoder_outputs, "past_key_values": decoder_inputs.get("past_key_values"), "use_cache": use_cache}

    SmallCap.prepare_inputs_for_generation = prepare_inputs_for_generation
    for name, klass in (("this_gpt2", ThisGPT2Config), ("smallcap", SmallCapConfig)):
        try:
            AutoConfig.register(name, klass)
        except ValueError:
            pass
    for klass_config, klass_model in ((ThisGPT2Config, ThisGPT2LMHeadModel), (SmallCapConfig, SmallCap)):
        try:
            AutoModel.register(klass_config, klass_model)
        except ValueError:
            pass
        try:
            AutoModelForCausalLM.register(klass_config, klass_model)
        except ValueError:
            pass
    return SmallCap, prep_strings, postprocess_preds, AutoModel, AutoModelForCausalLM, AutoConfig


def run_retrieval(rows: list[dict[str, Any]], index: Any, captions: list[str], config: dict[str, Any], log: RunLog) -> list[dict[str, Any]]:
    import faiss
    import torch
    import clip
    from PIL import Image

    retrieval_model, retrieval_preprocess = clip.load(config["retrieval_encoder"], device="cuda")
    retrieval_model.eval()
    records = []
    for row in rows:
        with Image.open(row["image_path"]) as opened:
            image = opened.convert("RGB")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                pixels = retrieval_preprocess(image).unsqueeze(0).cuda()
                embedding = retrieval_model.encode_image(pixels).float().cpu().numpy()
            image.close()
        faiss.normalize_L2(embedding)
        scores, indices = index.search(embedding, int(config["retrieval_k"]))
        retrieval = [{"rank": rank, "index_id": int(index_id), "score": float(score), "caption": captions[int(index_id)]} for rank, (index_id, score) in enumerate(zip(indices[0], scores[0]), 1)]
        if any(not math.isfinite(entry["score"]) for entry in retrieval):
            raise RuntimeError(f"non-finite FAISS score for COCO ID {row['image_id']}")
        if len(retrieval) != 4 or any(not entry["caption"] for entry in retrieval):
            raise RuntimeError(f"retrieval failed for COCO ID {row['image_id']}")
        records.append({"coco_id": row["image_id"], "image_path": row["image_path"], "retrieval": retrieval})
        log.write(f"retrieved 4 captions for COCO ID {row['image_id']}")
    return records


def run_generation(rows: list[dict[str, Any]], retrievals: list[dict[str, Any]], args: argparse.Namespace, config: dict[str, Any], model_resolution: dict[str, Any], log: RunLog) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer, CLIPFeatureExtractor

    SmallCap, prep_strings, postprocess_preds, _, _, _ = apply_smallcap_registration(args.smallcap_root)
    tokenizer = AutoTokenizer.from_pretrained(config["decoder"], revision=config["decoder_revision"])
    tokenizer.pad_token = "!"
    tokenizer.eos_token = "."
    feature_extractor = CLIPFeatureExtractor.from_pretrained(config["generation_visual_encoder"], revision=config["generation_visual_encoder_revision"])
    model = AutoModel.from_pretrained(config["checkpoint_id"], revision=config["checkpoint_revision"]).eval().cuda()
    model_revision = getattr(getattr(model, "config", None), "_commit_hash", None) or model_resolution["resolved_revision"]
    template = (args.smallcap_root / "src" / "template.txt").read_text(encoding="utf-8").strip() + " "
    retrieval_by_id = {record["coco_id"]: record["retrieval"] for record in retrievals}
    predictions = []
    latencies = []
    generation_parameters = {"max_new_tokens": 25, "no_repeat_ngram_size": 0, "length_penalty": 0.0, "min_length": 1, "num_beams": config["num_beams"], "early_stopping": True}
    for row in rows:
        started = time.perf_counter()
        retrieved = [entry["caption"] for entry in retrieval_by_id[row["image_id"]]]
        decoder_ids = prep_strings("", tokenizer, template=template, retrieved_caps=retrieved, k=int(config["retrieval_k"]), is_test=True)
        with Image.open(row["image_path"]) as opened:
            image = opened.convert("RGB")
            pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.cuda()
            image.close()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            generated = model.generate(pixel_values, decoder_input_ids=torch.tensor([decoder_ids], device="cuda"), eos_token_id=tokenizer.eos_token_id, **generation_parameters)
        caption = postprocess_preds(tokenizer.decode(generated[0]), tokenizer).strip()
        torch.cuda.synchronize()
        if not caption:
            raise RuntimeError(f"SmallCap generated an empty caption for COCO ID {row['image_id']}")
        latencies.append(time.perf_counter() - started)
        predictions.append({"image_id": row["image_id"], "caption": caption})
        log.write(f"generated caption for COCO ID {row['image_id']}")
    return predictions, {"model_revision": model_revision, "generation_parameters": generation_parameters, "latency_mean_seconds": sum(latencies) / len(latencies)}


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    run_dir, run_id = make_run_dir(args.output_root)
    log = RunLog(run_dir / "run.log")
    write_json(run_dir / "run_config.json", {"run_id": run_id, **config})
    environment: dict[str, Any] = {}
    source: dict[str, Any] = {}
    model_resolution: dict[str, Any] = {}
    retrieval_info: dict[str, Any] = {}
    selected: list[dict[str, Any]] = []
    retrievals: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    provenance: dict[str, Any] = {"experiment": "smallcap_r1_smoke", "purpose": "engineering_smoke_test", "run_id": run_id, "status": "running", "started_utc": started, "smallcap_repository": UPSTREAM_URL, "smallcap_commit": config["upstream_commit"], "capybara_commit": git_commit(REPO_ROOT), "model": config["checkpoint_id"], "retrieval_encoder": config["retrieval_encoder"], "generation_visual_encoder": config["generation_visual_encoder"], "k": config["retrieval_k"], "beam_size": config["num_beams"], "dataset": config["dataset"], "split": config["split"], "num_samples": config["sample_limit"], "seed": config["seed"], "compatibility_patches": COMPATIBILITY_PATCHES}
    try:
        log.write("preflight: source")
        for path, label in ((args.smallcap_root / "src" / "vision_encoder_decoder.py", "official SmallCap source"), (args.smallcap_root / "src" / "template.txt", "official SmallCap template"), (args.karpathy_annotations, "Karpathy test annotations"), (args.retrieval_index, "FAISS retrieval index"), (args.retrieval_captions, "retrieval captions"), (args.coco_input_root, "COCO input")):
            require(path, label)
        source = git_status(args.smallcap_root)
        if source["commit"] != config["upstream_commit"] or source["dirty"]:
            raise RuntimeError(f"SmallCap source is not a clean pinned checkout: {source}")
        provenance["smallcap_dirty"] = source["dirty"]
        provenance["smallcap_remote_v"] = source["remote_v"]
        log.write(f"source verified: {source['commit']}")

        log.write("preflight: environment")
        import torch
        environment = environment_info(torch)
        write_json(run_dir / "environment.json", environment)
        if not environment["cuda_available"]:
            raise RuntimeError("torch.cuda.is_available() is False; R1 stops without GPU")
        log.write(f"GPU verified: {environment['gpu_name']}")

        log.write("preflight: model resolution")
        model_resolution = resolve_model(config)
        provenance["model_revision"] = model_resolution["resolved_revision"]
        log.write(f"model config resolved: {model_resolution['config_path']}")

        log.write("preflight: retrieval artifacts")
        index, captions, retrieval_info = validate_retrieval(args.retrieval_index, args.retrieval_captions)
        provenance.update({"faiss_ntotal": retrieval_info["ntotal"], "faiss_dimension": retrieval_info["dimension"], "retrieval_caption_count": retrieval_info["caption_count"], "artifact_protocol_status": "ARTIFACT_PROTOCOL_UNVERIFIED"})
        log.write(f"FAISS verified: ntotal={index.ntotal}, d={index.d}, captions={len(captions)}")

        log.write("preflight: deterministic sample and images")
        selected = resolve_and_validate_images(load_rows(args.karpathy_annotations, 4, 2026), args.coco_input_root)
        reference_path = REPO_ROOT / "configs" / "smallcap_r1_selected_ids.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        selected_ids = [row["image_id"] for row in selected]
        if selected_ids != reference["selected_coco_ids"]:
            raise RuntimeError(f"deterministic R1 IDs changed: expected {reference['selected_coco_ids']}, got {selected_ids}")
        write_json(run_dir / "selected_ids.json", [{"coco_id": row["image_id"], "image_path": row["image_path"], "file_exists": row["file_exists"], "opened": row["opened"], "rgb_conversion": row["rgb_conversion"], "size": row["size"], "mode": row["mode"]} for row in selected])
        provenance["selected_coco_ids"] = selected_ids
        log.write(f"selected IDs: {provenance['selected_coco_ids']}")

        log.write("retrieval sanity check")
        retrievals = run_retrieval(selected, index, captions, config, log)
        write_json(run_dir / "retrievals.json", retrievals)

        log.write("SmallCap generation")
        predictions, generation_info = run_generation(selected, retrievals, args, config, model_resolution, log)
        write_json(run_dir / "predictions.json", predictions)
        write_json(run_dir / "raw_predictions.json", predictions)
        provenance["model_revision"] = generation_info["model_revision"]
        provenance["generation_parameters"] = generation_info["generation_parameters"]
        provenance["status"] = "completed"
        provenance["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        provenance["generated_predictions_path"] = str((run_dir / "predictions.json").resolve())
        write_json(run_dir / "provenance.json", provenance)
        write_json(run_dir / "environment.json", environment)
        (run_dir / "summary.md").write_text("# SmallCap R1 smoke test\n\nStatus: PASS\n\nThis is an engineering smoke test. No research metrics were computed.\n\nSelected COCO IDs: " + ", ".join(str(row["image_id"]) for row in selected) + "\n\nGenerated captions:\n" + "\n".join(f"- {item['image_id']}: {item['caption']}" for item in predictions) + "\n", encoding="utf-8")
        log.write("R1 completed: four non-empty captions saved")
        return 0
    except Exception as exc:
        provenance.update({"status": "failed", "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "error": f"{type(exc).__name__}: {exc}"})
        write_json(run_dir / "provenance.json", provenance)
        if environment:
            write_json(run_dir / "environment.json", environment)
        log.write(f"R1 failed: {type(exc).__name__}: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

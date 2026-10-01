#!/usr/bin/env python3
"""Run the evidence-backed SmallCap checkpoint smoke protocol."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
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

from capybara.experiments import create_experiment, git_commit, write_predictions


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


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required {label}: {path}")


def resolve_images(coco_input_root: Path) -> dict[int, Path]:
    """Resolve COCO numeric IDs from train2017/val2017 directories."""

    image_dirs = []
    for name in ("train2017", "val2017"):
        image_dirs.extend(path for path in coco_input_root.rglob(name) if path.is_dir())
    if coco_input_root.name in ("train2017", "val2017"):
        image_dirs.append(coco_input_root)
    image_dirs = sorted(set(path.resolve() for path in image_dirs))
    if not image_dirs:
        raise FileNotFoundError(f"could not find train2017/val2017 below {coco_input_root}")

    resolved: dict[int, Path] = {}
    for directory in image_dirs:
        for image_path in directory.glob("*.jpg"):
            match = re.fullmatch(r"(\d{12})\.jpg", image_path.name)
            if match:
                resolved.setdefault(int(match.group(1)), image_path)
    if not resolved:
        raise FileNotFoundError(f"no twelve-digit COCO JPEGs found below {coco_input_root}")
    return resolved


def load_rows(annotation_path: Path, image_paths: dict[int, Path], limit: int, seed: int) -> list[dict[str, Any]]:
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 5000:
        raise ValueError("Karpathy test annotation must be a list of exactly 5000 records")
    rows = []
    for item in raw:
        match = re.search(r"(\d{12})\.jpg$", str(item.get("image", "")))
        if not match:
            raise ValueError(f"invalid Karpathy image name: {item.get('image')!r}")
        image_id = int(match.group(1))
        if image_id not in image_paths:
            raise FileNotFoundError(f"COCO image missing for image_id={image_id}")
        rows.append({"image_id": image_id, "image_path": str(image_paths[image_id]), "references": item.get("caption", [])})
    rows.sort(key=lambda row: row["image_id"])
    random.Random(seed).shuffle(rows)
    return rows[: min(limit, len(rows))]


def package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "tokenizers", "huggingface-hub", "faiss-cpu", "Pillow")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def write_r1_metadata(output_dir: Path, payload: dict[str, Any]) -> None:
    (output_dir / "r1_metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def run_inference(args: argparse.Namespace, config: dict[str, Any], output_dir: Path) -> None:
    import faiss
    import numpy as np
    import torch
    import clip
    from PIL import Image
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPFeatureExtractor

    source_commit = git_commit(args.smallcap_root)
    expected_commit = config.get("upstream_commit")
    if expected_commit and expected_commit != "UNVERIFIED" and source_commit != expected_commit:
        raise RuntimeError(f"SmallCap source commit mismatch: expected {expected_commit}, got {source_commit}")
    dirty = subprocess.run(["git", "-C", str(args.smallcap_root), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError("SmallCap source checkout is dirty; use a clean pinned checkout")
    if not torch.cuda.is_available():
        raise RuntimeError("SmallCap R1 requires a CUDA GPU; refusing CPU fallback")
    device = torch.device("cuda")

    sys.path.insert(0, str(args.smallcap_root))
    from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
    from src.utils import postprocess_preds, prep_strings
    from src.vision_encoder_decoder import SmallCap, SmallCapConfig

    # COMPATIBILITY PATCH — no intended methodological change.
    # Transformers 4.36 passes past_key_values while this pinned source uses past.
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

    captions = json.loads(args.retrieval_captions.read_text(encoding="utf-8"))
    if not isinstance(captions, list):
        raise ValueError("retrieval captions must be a JSON list indexed by FAISS result ID")
    retrieval_index = faiss.read_index(str(args.retrieval_index))
    if retrieval_index.ntotal != len(captions):
        raise ValueError(f"FAISS/caption cardinality mismatch: {retrieval_index.ntotal} != {len(captions)}")

    tokenizer = AutoTokenizer.from_pretrained(config["decoder"])
    tokenizer.pad_token = "!"
    tokenizer.eos_token = "."
    feature_extractor = CLIPFeatureExtractor.from_pretrained(config["caption_encoder"])
    model = AutoModel.from_pretrained(config["checkpoint_id"]).eval().to(device)
    retrieval_model, retrieval_preprocess = clip.load(config["retrieval_encoder"], device=device)
    retrieval_model.eval()
    template = (args.smallcap_root / "src" / "template.txt").read_text(encoding="utf-8").strip() + " "

    rows = load_rows(args.karpathy_annotations, resolve_images(args.coco_input_root), int(config["sample_limit"]), int(config["seed"]))
    predictions = []
    latencies = []
    for row in rows:
        image = Image.open(row["image_path"]).convert("RGB")
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            retrieval_pixels = retrieval_preprocess(image).unsqueeze(0).to(device)
            embedding = retrieval_model.encode_image(retrieval_pixels).float().cpu().numpy()
        faiss.normalize_L2(embedding)
        _, indices = retrieval_index.search(embedding, int(config["retrieval_k"]))
        retrieved = [captions[int(index)] for index in indices[0]]
        decoder_ids = prep_strings("", tokenizer, template=template, retrieved_caps=retrieved, k=int(config["retrieval_k"]), is_test=True)
        pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            generated = model.generate(pixel_values, decoder_input_ids=torch.tensor([decoder_ids], device=device), max_new_tokens=25, no_repeat_ngram_size=0, length_penalty=0.0, min_length=1, num_beams=int(config["num_beams"]), early_stopping=True, eos_token_id=tokenizer.eos_token_id)
        caption = postprocess_preds(tokenizer.decode(generated[0]), tokenizer)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - started)
        predictions.append({"image_id": row["image_id"], "caption": caption})

    write_predictions(output_dir, predictions, raw_predictions=predictions)
    write_r1_metadata(output_dir, {"status": "completed", "started_utc": config["_started_utc"], "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "source_repository": "https://github.com/RitaRamo/smallcap", "source_commit": source_commit, "source_dirty": False, "checkpoint_id": config["checkpoint_id"], "checkpoint_revision": config.get("checkpoint_revision", "UNVERIFIED"), "dataset": config["dataset"], "split": config["split"], "annotation_path": str(args.karpathy_annotations.resolve()), "annotation_sha256": hashlib.sha256(args.karpathy_annotations.read_bytes()).hexdigest(), "retrieval_index_path": str(args.retrieval_index.resolve()), "retrieval_captions_path": str(args.retrieval_captions.resolve()), "retrieval_k": config["retrieval_k"], "retrieval_encoder": config["retrieval_encoder"], "caption_encoder": config["caption_encoder"], "decoder": config["decoder"], "sample_count": len(predictions), "selected_image_ids": [row["image_id"] for row in rows], "seed": config["seed"], "latency_mean_seconds": sum(latencies) / len(latencies), "generated_predictions_path": str((output_dir / "predictions.json").resolve()), "command": sys.argv, "package_versions": package_versions(), "warnings": ["Engineering smoke test only; no scientific metrics are computed."]})


def main() -> int:
    args = parse_args()
    for path, label in ((args.smallcap_root / "src" / "vision_encoder_decoder.py", "official SmallCap source"), (args.smallcap_root / "src" / "template.txt", "official SmallCap template"), (args.karpathy_annotations, "Karpathy test annotations"), (args.retrieval_index, "official FAISS retrieval index"), (args.retrieval_captions, "official retrieval captions"), (args.coco_input_root, "COCO image input")):
        require(path, label)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["_started_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    output_dir = create_experiment(args.config, REPO_ROOT, args.output_root)
    try:
        run_inference(args, config, output_dir)
    except Exception as exc:
        write_r1_metadata(output_dir, {"status": "failed", "started_utc": config["_started_utc"], "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "command": sys.argv, "error": f"{type(exc).__name__}: {exc}"})
        raise
    print(f"completed SmallCap R1 smoke test: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(3)

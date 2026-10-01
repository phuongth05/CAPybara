"""Audit the SmallCap caption datastore without silently substituting data.

The audit follows the pinned SmallCap ``src/retrieve_caps.py`` implementation:
restval is treated as train, captions are reconstructed from sentence tokens,
GPT-2 tokenization is performed in batches of 512 with padding, and captions
whose padded sequence length exceeds 25 are removed.

The command is deliberately useful in a blocked state.  When the exact
``dataset_coco.json`` or public datastore files are absent it writes an audit
record with explicit ``UNVERIFIED`` fields and an R2 gate of ``R2_BLOCKED``;
it never substitutes the Karpathy test file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "smallcap" / "r1_5_datastore_audit"
OFFICIAL_BATCH_SIZE = 512
OFFICIAL_MAX_TOKENS = 25
OFFICIAL_SOURCE_COMMIT = "19cc4f4e5972c70fde16c8857b14c96e5492cb30"
R1_IDS = [478077, 379529, 87912, 357265]


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def load_dataset_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError("dataset_coco.json must contain an 'images' list")
    records = payload["images"]
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"dataset record {index} is not an object")
        for key in ("cocoid", "split", "sentences"):
            if key not in item:
                raise ValueError(f"dataset record {index} is missing {key!r}")
        if not isinstance(item["sentences"], list):
            raise ValueError(f"dataset record {index} has invalid sentences")
    return records


def dataset_split_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    image_counts: Counter[str] = Counter()
    caption_counts: Counter[str] = Counter()
    for item in records:
        split = str(item["split"])
        image_counts[split] += 1
        caption_counts[split] += len(item["sentences"])
    return {
        "image_counts_by_split": dict(sorted(image_counts.items())),
        "caption_counts_by_split": dict(sorted(caption_counts.items())),
        "total_images": len(records),
        "total_captions": sum(caption_counts.values()),
        "duplicate_image_ids": sorted(
            image_id for image_id, count in Counter(int(item["cocoid"]) for item in records).items() if count > 1
        ),
    }


def reconstruct_train_captions(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reproduce retrieve_caps.load_coco_data's caption construction."""
    captions: list[dict[str, Any]] = []
    for item in records:
        split = "train" if item["split"] == "restval" else item["split"]
        if split != "train":
            continue
        for sentence in item["sentences"]:
            tokens = sentence.get("tokens")
            if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
                raise ValueError(f"invalid sentence tokens for COCO ID {item['cocoid']}")
            captions.append({"image_id": int(item["cocoid"]), "caption": " ".join(tokens)})
    return captions


def load_gpt2_tokenizer(revision: Optional[str] = None) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised in Kaggle, not unit tests
        raise RuntimeError("transformers is required for the exact GPT-2 caption filter") from exc
    kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained("gpt2", **kwargs)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def filter_captions_official(
    captions: Sequence[dict[str, Any]], tokenizer: Any, batch_size: int = OFFICIAL_BATCH_SIZE
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reproduce retrieve_caps.filter_captions exactly, including padded lengths."""
    kept: list[dict[str, Any]] = []
    padded_lengths: list[int] = []
    for start in range(0, len(captions), batch_size):
        batch = captions[start : start + batch_size]
        encoded = tokenizer.batch_encode_plus(
            [item["caption"] for item in batch], return_tensors="np", padding=True
        )["input_ids"]
        encoded = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        if len(encoded) != len(batch):
            raise ValueError("tokenizer returned a different number of rows")
        for item, row in zip(batch, encoded):
            length = len(row)
            padded_lengths.append(length)
            if length <= OFFICIAL_MAX_TOKENS:
                kept.append(dict(item))
    histogram = Counter(str(length) for length in padded_lengths)
    return kept, {
        "batch_size": batch_size,
        "max_tokens": OFFICIAL_MAX_TOKENS,
        "input_count": len(captions),
        "kept_count": len(kept),
        "dropped_count": len(captions) - len(kept),
        "padded_length_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
    }


def filter_captions_unpadded(
    captions: Sequence[dict[str, Any]], tokenizer: Any, batch_size: int = OFFICIAL_BATCH_SIZE
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Diagnostic semantic filter using unpadded GPT-2 lengths.

    This is not the literal pinned source implementation. It is retained
    because the public 565,497-caption artifact may have been built with this
    rule, or with ``attention_mask.sum(1)``.
    """
    kept: list[dict[str, Any]] = []
    lengths: list[int] = []
    for start in range(0, len(captions), batch_size):
        batch = captions[start : start + batch_size]
        encoded = tokenizer.batch_encode_plus(
            [item["caption"] for item in batch], return_tensors="np", padding=False
        )["input_ids"]
        encoded = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        for item, row in zip(batch, encoded):
            length = len(row)
            lengths.append(length)
            if length <= OFFICIAL_MAX_TOKENS:
                kept.append(dict(item))
    histogram = Counter(str(length) for length in lengths)
    return kept, {
        "batch_size": batch_size,
        "max_tokens": OFFICIAL_MAX_TOKENS,
        "input_count": len(captions),
        "kept_count": len(kept),
        "dropped_count": len(captions) - len(kept),
        "unpadded_length_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
    }


def compare_public_captions(
    reconstructed: Sequence[dict[str, Any]], public_captions: Sequence[Any]
) -> dict[str, Any]:
    expected = [item["caption"] for item in reconstructed]
    actual = list(public_captions)
    result: dict[str, Any] = {
        "status": "PASS" if expected == actual else "FAIL",
        "expected_count": len(expected),
        "actual_count": len(actual),
        "exact_sequence_match": expected == actual,
        "expected_canonical_sha256": canonical_json_sha256(expected),
        "actual_canonical_sha256": canonical_json_sha256(actual),
        "expected_duplicate_caption_count": sum(count > 1 for count in Counter(expected).values()),
        "actual_duplicate_caption_count": sum(count > 1 for count in Counter(actual).values()),
    }
    first_mismatch = next(
        (index for index, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]), None
    )
    if first_mismatch is not None:
        result["first_mismatch"] = {
            "index": first_mismatch,
            "expected": expected[first_mismatch],
            "actual": actual[first_mismatch],
        }
    elif len(expected) != len(actual):
        result["first_mismatch"] = {"index": min(len(expected), len(actual)), "reason": "length mismatch"}
    return result


def audit_faiss(index_path: Path, expected_count: Optional[int] = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(index_path), "exists": index_path.is_file()}
    if not index_path.is_file():
        result["status"] = "UNVERIFIED"
        return result
    result["sha256"] = sha256_file(index_path)
    try:
        import faiss

        index = faiss.read_index(str(index_path))
        result.update(
            {
                "status": "PASS",
                "type": type(index).__name__,
                "dimension": int(index.d),
                "ntotal": int(index.ntotal),
                "metric_type": int(index.metric_type),
                "is_index_flat_ip": type(index).__name__ == "IndexFlatIP",
            }
        )
        result["count_matches_captions"] = expected_count is None or int(index.ntotal) == expected_count
        if not result["count_matches_captions"]:
            result["status"] = "FAIL"
    except ImportError:
        result["status"] = "UNVERIFIED"
        result["error"] = "faiss is not installed"
    except Exception as exc:  # pragma: no cover - depends on local artifact
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def query_image_leakage(
    filtered_captions: Sequence[dict[str, Any]], query_ids: Iterable[int]
) -> dict[str, Any]:
    query_set = {int(value) for value in query_ids}
    source_ids = [int(item["image_id"]) for item in filtered_captions]
    overlap = sorted(query_set.intersection(source_ids))
    return {
        "query_ids": sorted(query_set),
        "overlapping_source_image_ids": overlap,
        "overlap_image_count": len(overlap),
        "overlap_caption_count": sum(image_id in query_set for image_id in source_ids),
        "status": "FAIL" if overlap else "PASS",
    }


def audit_r1_retrievals(
    run_path: Optional[Path], filtered_captions: Optional[Sequence[dict[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "UNVERIFIED", "run_path": str(run_path) if run_path else None}
    if run_path is None or not run_path.is_dir():
        result["reason"] = "R1 run directory was not supplied or does not exist"
        return result
    retrieval_path = run_path / "retrievals.json"
    if not retrieval_path.is_file():
        result["reason"] = "retrievals.json is missing"
        return result
    retrievals = json.loads(retrieval_path.read_text(encoding="utf-8"))
    result["query_count"] = len(retrievals) if isinstance(retrievals, list) else None
    result["query_ids"] = [int(row["coco_id"]) for row in retrievals] if isinstance(retrievals, list) else []
    if filtered_captions is None:
        result["reason"] = "source image IDs are unavailable without dataset_coco.json"
        return result
    by_caption = {item["caption"]: int(item["image_id"]) for item in filtered_captions}
    details = []
    for row in retrievals:
        sources = [by_caption.get(item["caption"]) for item in row.get("retrieval", [])]
        details.append({"coco_id": int(row["coco_id"]), "source_image_ids": sources})
    result["details"] = details
    result["status"] = "PASS"
    return result


def audit_embedding_equivalence(
    index_path: Path,
    captions: Sequence[str],
    sample_size: int = 32,
    seed: int = 2026,
    cache_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Compare public FAISS vectors with fresh RN50x64 text embeddings."""
    try:
        import clip
        import faiss
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        return {"status": "UNVERIFIED", "reason": f"missing embedding dependency: {exc}"}
    if not captions or not index_path.is_file():
        return {"status": "UNVERIFIED", "reason": "captions or FAISS index is unavailable"}
    try:
        index = faiss.read_index(str(index_path))
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(captions)), min(sample_size, len(captions))))
        kwargs = {"download_root": str(cache_dir)} if cache_dir else {}
        model, _ = clip.load("RN50x64", device="cpu", **kwargs)
        model.eval()
        with torch.no_grad():
            encoded = model.encode_text(clip.tokenize([captions[index] for index in indices]))
        encoded = encoded.float().cpu().numpy().astype("float32")
        faiss.normalize_L2(encoded)
        public = np.stack([index.reconstruct(int(row)) for row in indices]).astype("float32")
        faiss.normalize_L2(public)
        cosine = np.sum(encoded * public, axis=1)
        return {
            "status": "PASS" if float(np.mean(cosine)) >= 0.999 else "FAIL",
            "encoder": "RN50x64",
            "sample_size": len(indices),
            "seed": seed,
            "indices": indices,
            "mean_cosine": float(np.mean(cosine)),
            "min_cosine": float(np.min(cosine)),
            "max_cosine": float(np.max(cosine)),
            "threshold": 0.999,
        }
    except Exception as exc:  # pragma: no cover - environment/artifact dependent
        return {"status": "UNVERIFIED", "reason": f"{type(exc).__name__}: {exc}"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-coco", type=Path, required=True)
    parser.add_argument("--public-captions", type=Path, required=True)
    parser.add_argument("--public-index", type=Path, required=True)
    parser.add_argument("--smallcap-root", type=Path, required=True)
    parser.add_argument("--r1-run", type=Path)
    parser.add_argument("--gpt2-revision")
    parser.add_argument("--embedding-check", action="store_true")
    parser.add_argument("--embedding-sample-size", type=int, default=32)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    source_commit = git_commit(args.smallcap_root)

    dataset_status: dict[str, Any] = {
        "path": str(args.dataset_coco),
        "exists": args.dataset_coco.is_file(),
        "status": "UNVERIFIED",
        "source_commit": source_commit,
        "required_source_commit": OFFICIAL_SOURCE_COMMIT,
    }
    reconstructed: list[dict[str, Any]] = []
    filtered: Optional[list[dict[str, Any]]] = None
    source_filtered: Optional[list[dict[str, Any]]] = None
    artifact_filtered: Optional[list[dict[str, Any]]] = None
    filter_stats: dict[str, Any] = {"status": "UNVERIFIED"}
    split_stats: dict[str, Any] = {"status": "UNVERIFIED"}
    if args.dataset_coco.is_file():
        records = load_dataset_records(args.dataset_coco)
        split_stats = {"status": "PASS", **dataset_split_stats(records)}
        reconstructed = reconstruct_train_captions(records)
        dataset_status.update({"status": "PASS", "sha256": sha256_file(args.dataset_coco)})
        try:
            tokenizer = load_gpt2_tokenizer(args.gpt2_revision)
            source_filtered, source_stats = filter_captions_official(reconstructed, tokenizer)
            artifact_filtered, artifact_stats = filter_captions_unpadded(reconstructed, tokenizer)
            filtered = artifact_filtered
            filter_stats = {
                "status": "PASS",
                "literal_source": source_stats,
                "artifact_compatible_unpadded": artifact_stats,
                "public_comparison_candidate": "artifact_compatible_unpadded",
            }
        except Exception as exc:
            filter_stats = {"status": "UNVERIFIED", "error": f"{type(exc).__name__}: {exc}"}

    public_status: dict[str, Any] = {"path": str(args.public_captions), "exists": args.public_captions.is_file()}
    public_comparison: dict[str, Any] = {"status": "UNVERIFIED"}
    public_captions: list[Any] = []
    if args.public_captions.is_file():
        public_captions = json.loads(args.public_captions.read_text(encoding="utf-8"))
        if not isinstance(public_captions, list) or not all(isinstance(item, str) for item in public_captions):
            raise ValueError("public captions must be a JSON list of strings")
        public_status.update({"status": "PASS", "sha256": sha256_file(args.public_captions), "count": len(public_captions)})
        if source_filtered is not None and artifact_filtered is not None:
            public_comparison = {
                "status": "PASS"
                if [item["caption"] for item in artifact_filtered] == public_captions
                else "FAIL",
                "literal_source": compare_public_captions(source_filtered, public_captions),
                "artifact_compatible_unpadded": compare_public_captions(artifact_filtered, public_captions),
            }

    faiss_result = audit_faiss(args.public_index, len(public_captions) if public_captions else None)
    leakage = query_image_leakage(filtered, R1_IDS) if filtered is not None else {
        "status": "UNVERIFIED",
        "reason": "dataset_coco.json and exact filtered source IDs are required",
        "query_ids": R1_IDS,
    }
    retrieval_consistency = audit_r1_retrievals(args.r1_run, filtered)
    embedding = (
        audit_embedding_equivalence(
            args.public_index,
            public_captions,
            sample_size=args.embedding_sample_size,
            cache_dir=args.output_root.parent / "clip_cache",
        )
        if args.embedding_check
        else {"status": "UNVERIFIED", "reason": "embedding check not requested"}
    )

    source_behavior = {
        "pinned_smallcap_commit": OFFICIAL_SOURCE_COMMIT,
        "source_file": "src/retrieve_caps.py",
        "caption_construction_lines": "13-27",
        "caption_filter_lines": "29-50",
        "same_image_exclusion_lines": "94-107",
        "same_image_exclusion_applies_to": "offline retrieved_caps_resnet50x64.json construction in retrieve_caps.main",
        "r1_adapter_behavior": "R1 runner searches FAISS and takes raw top-k; it does not call filter_nns",
        "r1_runner_source": "scripts/run_smallcap_smoke.py:229-254",
        "filter_discrepancy": "literal padded source filter and artifact-compatible unpadded filter are both recorded",
    }
    datastore_provenance = {
        "audit": "smallcap_r1_5_datastore_provenance",
        "status": "PASS"
        if all(
            item.get("status") == "PASS"
            for item in (dataset_status, filter_stats, public_status, public_comparison, faiss_result, leakage)
        )
        and public_comparison.get("literal_source", {}).get("status") == "PASS"
        else "INCOMPLETE",
        "smallcap_commit": source_commit,
        "required_smallcap_commit": OFFICIAL_SOURCE_COMMIT,
        "dataset_coco": dataset_status,
        "dataset_split_stats": split_stats,
        "caption_filter": filter_stats,
        "public_captions": public_status,
        "public_caption_comparison": public_comparison,
        "faiss": faiss_result,
        "query_image_leakage": leakage,
        "r1_retrieval_consistency": retrieval_consistency,
        "embedding_equivalence": embedding,
        "official_distribution_status": "OFFICIAL_DISTRIBUTION"
        if public_status.get("status") == "PASS" and faiss_result.get("status") == "PASS"
        else "UNVERIFIED",
        "construction_protocol_status": "VERIFIED"
        if public_comparison.get("literal_source", {}).get("status") == "PASS"
        else "FAILED",
        "caption_content_status": "VERIFIED"
        if public_comparison.get("status") == "PASS"
        else "FAILED",
        "split_leakage_status": "VERIFIED" if leakage.get("status") == "PASS" else "UNVERIFIED",
        "faiss_structure_status": "VERIFIED" if faiss_result.get("status") == "PASS" else "UNVERIFIED",
        "embedding_equivalence_status": (
            "VERIFIED"
            if embedding.get("status") == "PASS"
            else "FAILED"
            if embedding.get("status") == "FAIL"
            else "UNVERIFIED"
        ),
        "byte_level_status": "UNVERIFIED",
        "source_behavior": source_behavior,
        "r1_ids": R1_IDS,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    gate_reasons = []
    if dataset_status["status"] != "PASS":
        gate_reasons.append("exact dataset_coco.json is unavailable or invalid")
    if filter_stats["status"] != "PASS":
        gate_reasons.append("official GPT-2 token filter was not executed")
    if public_comparison.get("status") != "PASS":
        gate_reasons.append("public caption sequence was not proven identical to artifact-compatible reconstruction")
    if public_comparison.get("literal_source", {}).get("status") != "PASS":
        gate_reasons.append("literal pinned source filter does not reproduce the public caption count/sequence")
    if faiss_result.get("status") != "PASS":
        gate_reasons.append("FAISS structure/hash was not fully verified")
    if leakage.get("status") != "PASS":
        gate_reasons.append("query-image leakage was not proven absent")
    if embedding.get("status") != "PASS":
        gate_reasons.append("embedding equivalence is not VERIFIED")
    if retrieval_consistency.get("status") != "PASS":
        gate_reasons.append("R1 retrieval consistency artifact was not supplied")
    gate = {
        "status": "R2_READY" if not gate_reasons else "R2_BLOCKED",
        "reasons": gate_reasons,
        "required_before_r2": [
            "resolve and document the literal-source versus public-artifact filter discrepancy",
            "preserve the R1 Kaggle retrievals.json and complete per-query retrieval consistency",
            "retain the verified dataset, caption, FAISS, leakage, and embedding artifacts",
        ],
    }

    write_json(run_dir / "dataset_split_stats.json", split_stats)
    write_json(run_dir / "caption_filter_stats.json", filter_stats)
    write_json(run_dir / "public_caption_comparison.json", public_comparison)
    write_json(run_dir / "faiss_audit.json", faiss_result)
    write_json(run_dir / "leakage_audit.json", leakage)
    write_json(run_dir / "r1_retrieval_consistency.json", retrieval_consistency)
    write_json(run_dir / "embedding_equivalence.json", embedding)
    write_json(run_dir / "datastore_provenance.json", datastore_provenance)
    write_json(run_dir / "r2_gate.json", gate)
    if filtered is not None:
        write_json(run_dir / "reconstructed_captions.json", filtered)
        write_json(run_dir / "source_padded_reconstructed_captions.json", source_filtered or [])
    else:
        write_json(run_dir / "reconstructed_captions.json", {"status": "UNVERIFIED", "records": []})
    (run_dir / "summary.md").write_text(
        "# SmallCap R1.5 datastore audit\n\n"
        f"Status: **{gate['status']}**\n\n"
        + "\n".join(f"- {reason}" for reason in gate_reasons or ["All requested checks passed."]) + "\n",
        encoding="utf-8",
    )
    print(run_dir)
    print(json.dumps(gate, indent=2))
    return 0 if gate["status"] == "R2_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

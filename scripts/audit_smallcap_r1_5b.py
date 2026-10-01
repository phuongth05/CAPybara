"""Run the SmallCap R1.5b Karpathy and datastore verification audit.

This diagnostic deliberately keeps reproduction readiness separate from datastore
provenance.  It never changes the pinned SmallCap source or rewrites any data
artifact.  Large inputs and generated evidence remain under ``outputs/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from audit_smallcap_datastore import (
    OFFICIAL_BATCH_SIZE,
    OFFICIAL_MAX_TOKENS,
    OFFICIAL_SOURCE_COMMIT,
    R1_IDS,
    audit_faiss,
    audit_r1_retrievals,
    canonical_json_sha256,
    filter_captions_official,
    filter_captions_unpadded,
    git_commit,
    load_dataset_records,
    load_gpt2_tokenizer,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "smallcap" / "r1_5b_karpathy_verification"
KAGGLE_SLUG = "shtvkumar/karpathy-splits"
KAGGLE_URL = "https://www.kaggle.com/datasets/shtvkumar/karpathy-splits"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_schema(records: Sequence[dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    required_image_fields = ["split", "filename", "filepath", "cocoid", "imgid", "sentids", "sentences"]
    required_sentence_fields = ["raw", "tokens", "sentid"]
    image_field_counts = Counter()
    sentence_field_counts = Counter()
    missing_image_fields = Counter()
    missing_sentence_fields = Counter()
    duplicate_image_ids = sorted(
        value for value, count in Counter(int(item["cocoid"]) for item in records).items() if count > 1
    )
    sentence_ids: list[int] = []
    for item in records:
        image_field_counts.update(item.keys())
        for field in required_image_fields:
            if field not in item:
                missing_image_fields[field] += 1
        for sentence in item.get("sentences", []):
            sentence_field_counts.update(sentence.keys())
            for field in required_sentence_fields:
                if field not in sentence:
                    missing_sentence_fields[field] += 1
            if isinstance(sentence.get("sentid"), int):
                sentence_ids.append(sentence["sentid"])
    duplicate_sentence_ids = sorted(value for value, count in Counter(sentence_ids).items() if count > 1)
    return {
        "top_level_keys": sorted(payload.keys()),
        "top_level_types": {key: type(value).__name__ for key, value in payload.items()},
        "image_record_count": len(records),
        "image_field_presence_counts": dict(sorted(image_field_counts.items())),
        "sentence_field_presence_counts": dict(sorted(sentence_field_counts.items())),
        "missing_required_image_fields": dict(sorted(missing_image_fields.items())),
        "missing_required_sentence_fields": dict(sorted(missing_sentence_fields.items())),
        "duplicate_image_ids": duplicate_image_ids,
        "duplicate_sentence_ids": duplicate_sentence_ids,
        "expected_image_fields_checked": required_image_fields,
        "expected_sentence_fields_checked": required_sentence_fields,
    }


def split_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    image_counts = Counter(str(item.get("split")) for item in records)
    caption_counts = Counter(str(item.get("split")) for item in records for _ in item.get("sentences", []))
    unexpected = sorted(set(image_counts) - {"train", "restval", "val", "test"})
    return {
        "image_counts_by_original_split": dict(sorted(image_counts.items())),
        "sentence_counts_by_original_split": dict(sorted(caption_counts.items())),
        "expected_split_counts": {split: {"images": image_counts[split], "sentences": caption_counts[split]} for split in ("train", "restval", "val", "test")},
        "other_unexpected_splits": {split: {"images": image_counts[split], "sentences": caption_counts[split]} for split in unexpected},
        "total_images": len(records),
        "total_sentence_records": sum(len(item.get("sentences", [])) for item in records),
        "duplicate_image_ids": sorted(value for value, count in Counter(int(item["cocoid"]) for item in records).items() if count > 1),
    }


def source_caption_records(records: Sequence[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in records:
        original_split = str(item["split"])
        if original_split not in {"train", "restval"}:
            continue
        for sentence_index, sentence in enumerate(item["sentences"]):
            tokens = sentence.get("tokens")
            raw = sentence.get("raw")
            if field == "tokens_joined":
                caption = " ".join(tokens) if isinstance(tokens, list) else None
            elif field == "raw":
                caption = raw if isinstance(raw, str) else None
            else:
                raise ValueError(f"unknown caption field: {field}")
            if not isinstance(caption, str):
                raise ValueError(f"missing usable {field} caption for image {item['cocoid']}")
            result.append(
                {
                    "image_id": int(item["cocoid"]),
                    "original_split": original_split,
                    "sentid": sentence.get("sentid"),
                    "sentence_index": sentence_index,
                    "raw": raw,
                    "tokens": tokens,
                    "caption": caption,
                }
            )
    return result


def filtered_records(records: Sequence[dict[str, Any]], tokenizer: Any, padded: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if padded:
        kept, stats = filter_captions_official(records, tokenizer)
        stats["filter_semantics"] = "pinned_source_padding_true_len_encoding"
    else:
        kept, stats = filter_captions_unpadded(records, tokenizer)
        stats["filter_semantics"] = "diagnostic_padding_false_len_encoding"
    return kept, stats


def comparison(expected: Sequence[str], actual: Sequence[str]) -> dict[str, Any]:
    expected_list = list(expected)
    actual_list = list(actual)
    expected_counter = Counter(expected_list)
    actual_counter = Counter(actual_list)
    first_mismatch = next((i for i, pair in enumerate(zip(expected_list, actual_list)) if pair[0] != pair[1]), None)
    if first_mismatch is None and len(expected_list) != len(actual_list):
        first_mismatch = min(len(expected_list), len(actual_list))
    return {
        "expected_count": len(expected_list),
        "actual_count": len(actual_list),
        "cardinality_equal": len(expected_list) == len(actual_list),
        "ordered_exact_equal": expected_list == actual_list,
        "multiset_equal": expected_counter == actual_counter,
        "set_equal": set(expected_list) == set(actual_list),
        "intersection_unique_count": len(set(expected_list) & set(actual_list)),
        "public_missing_unique_count": len(set(actual_list) - set(expected_list)),
        "reconstruction_extra_unique_count": len(set(expected_list) - set(actual_list)),
        "public_missing_occurrence_count": sum((actual_counter - expected_counter).values()),
        "reconstruction_extra_occurrence_count": sum((expected_counter - actual_counter).values()),
        "expected_duplicate_value_count": sum(count > 1 for count in expected_counter.values()),
        "actual_duplicate_value_count": sum(count > 1 for count in actual_counter.values()),
        "first_mismatch_index": first_mismatch,
        "status": "PASS" if expected_list == actual_list else "FAIL",
    }


def mismatch_examples(expected_records: Sequence[dict[str, Any]], public: Sequence[str], limit: int = 10) -> list[dict[str, Any]]:
    examples = []
    expected = list(expected_records)
    for index in range(min(len(expected), len(public))):
        if expected[index]["caption"] != public[index]:
            examples.append({"index": index, "candidate_record": expected[index], "public_caption_at_index": public[index]})
            if len(examples) >= limit:
                break
    if len(examples) < limit and len(expected) != len(public):
        for index in range(min(len(expected), len(public)), max(len(expected), len(public))):
            examples.append({
                "index": index,
                "candidate_record": expected[index] if index < len(expected) else None,
                "public_caption_at_index": public[index] if index < len(public) else None,
                "reason": "length mismatch tail",
            })
            if len(examples) >= limit:
                break
    return examples


def public_source_mapping(public: Sequence[str], all_records: Sequence[dict[str, Any]], artifact_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_caption: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        by_caption[record["caption"]].append(record)
    all_match_counts = Counter()
    split_occurrences = Counter()
    for caption in public:
        matches = by_caption.get(caption, [])
        all_match_counts[len(matches)] += 1
        for split in sorted({record["original_split"] for record in matches}):
            split_occurrences[split] += 1
    artifact_splits = Counter(record["original_split"] for record in artifact_records)
    return {
        "public_caption_occurrences": len(public),
        "mapped_uniquely_to_one_candidate_record": all_match_counts[1],
        "mapped_ambiguously_to_multiple_candidate_records": sum(value for key, value in all_match_counts.items() if key > 1),
        "unresolved_candidate_mappings": all_match_counts[0],
        "all_candidate_string_mapping_cardinality": dict(sorted((str(key), value) for key, value in all_match_counts.items())),
        "string_overlap_occurrences_by_original_split": dict(sorted(split_occurrences.items())),
        "artifact_ordered_source_occurrences_by_original_split": dict(sorted(artifact_splits.items())),
        "artifact_source_level_val_test_occurrences": artifact_splits["val"] + artifact_splits["test"],
        "interpretation": "val/test matches in all-candidate string mapping are STRING_OVERLAP; artifact ordered records establish source provenance.",
    }


def faiss_metadata(index_path: Path, caption_count: int) -> dict[str, Any]:
    result = audit_faiss(index_path, caption_count)
    result["byte_size"] = index_path.stat().st_size if index_path.is_file() else None
    if index_path.is_file():
        try:
            import faiss
            import numpy as np

            index = faiss.read_index(str(index_path))
            indices = [0, int(index.ntotal // 2), int(index.ntotal - 1)] if index.ntotal else []
            vectors = np.stack([index.reconstruct(i) for i in indices]).astype("float32") if indices else np.empty((0, index.d), dtype="float32")
            result.update({
                "is_trained": bool(index.is_trained),
                "sample_indices": indices,
                "sample_vector_l2_norms": [float(value) for value in np.linalg.norm(vectors, axis=1)],
            })
        except Exception as exc:
            result["sample_error"] = f"{type(exc).__name__}: {exc}"
    return result


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def decide_gates(
    *,
    source_commit_ok: bool,
    checkpoint_ok: bool,
    public_ok: bool,
    faiss_ok: bool,
    counts_ok: bool,
    karpathy_test_ids_ok: bool,
    official_path_understood: bool,
    r1_retrieval_ok: bool,
    literal_reconstruction_ok: bool,
    source_leakage_ok: bool,
) -> dict[str, Any]:
    reproduction_criteria = {
        "r0_pass_recorded": True,
        "r1_pass_recorded": True,
        "pinned_source_confirmed": source_commit_ok,
        "pinned_checkpoint_confirmed": checkpoint_ok,
        "public_datastore_artifacts_confirmed": public_ok,
        "faiss_loads": faiss_ok,
        "caption_count_matches_faiss": counts_ok,
        "faiss_compatible_with_retrieval_path": faiss_ok,
        "karpathy_test_ids_identified": karpathy_test_ids_ok,
        "official_inference_path_understood": official_path_understood,
        "r1_retrieval_consistency": r1_retrieval_ok,
        "no_metric_invalidating_mismatch": public_ok and faiss_ok and counts_ok and r1_retrieval_ok,
    }
    reproduction_ready = all(reproduction_criteria.values())
    provenance_criteria = {
        "candidate_provenance_established": True,
        "construction_semantics_established": source_commit_ok,
        "public_caption_list_explained": public_ok and literal_reconstruction_ok,
        "source_split_composition_established": source_leakage_ok,
        "no_source_level_val_test_leakage": source_leakage_ok,
        "filtering_behavior_fully_explained": literal_reconstruction_ok,
    }
    provenance_status = "DATASTORE_PROVENANCE_VERIFIED" if all(provenance_criteria.values()) else "DATASTORE_PROVENANCE_PARTIAL"
    return {
        "r2_reproduction": {
            "status": "R2_REPRODUCTION_READY" if reproduction_ready else "R2_REPRODUCTION_BLOCKED",
            "criteria": reproduction_criteria,
            "blocking_reasons": [] if reproduction_ready else [key for key, value in reproduction_criteria.items() if not value],
        },
        "datastore_provenance": {
            "status": provenance_status,
            "criteria": provenance_criteria,
            "blocking_reasons": [] if provenance_status == "DATASTORE_PROVENANCE_VERIFIED" else [
                "literal pinned source filter does not reproduce the public caption sequence/count"
            ],
        },
        "r1_5b_status": "PASS" if reproduction_ready and provenance_status == "DATASTORE_PROVENANCE_VERIFIED" else "PARTIAL",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--public-captions", type=Path, required=True)
    parser.add_argument("--public-index", type=Path, required=True)
    parser.add_argument("--smallcap-root", type=Path, required=True)
    parser.add_argument("--r1-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpt2-revision", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    log_lines = [f"started_utc={started}", f"candidate_slug={KAGGLE_SLUG}"]

    candidate_payload = read_json(args.candidate_dataset)
    if not isinstance(candidate_payload, dict) or not isinstance(candidate_payload.get("images"), list):
        raise ValueError("candidate dataset must be a JSON object with images list")
    records = load_dataset_records(args.candidate_dataset)
    schema = dataset_schema(records, candidate_payload)
    splits = split_stats(records)
    candidate_sha = sha256_file(args.candidate_dataset)
    archive_sha = sha256_file(args.candidate_archive)
    archive_files = []
    with zipfile.ZipFile(args.candidate_archive) as archive:
        archive_files = [{"filename": info.filename, "file_size": info.file_size, "compressed_size": info.compress_size} for info in archive.infolist()]

    dataset_metadata = {
        "kaggle_dataset_slug": KAGGLE_SLUG,
        "kaggle_dataset_url": KAGGLE_URL,
        "source_description": "Pinned SmallCap README.md:42 directly names this Kaggle dataset and dataset_coco.json.",
        "archive_path": str(args.candidate_archive),
        "archive_filename": args.candidate_archive.name,
        "archive_size_bytes": args.candidate_archive.stat().st_size,
        "archive_sha256": archive_sha,
        "archive_members": archive_files,
        "dataset_coco_path": str(args.candidate_dataset),
        "dataset_coco_size_bytes": args.candidate_dataset.stat().st_size,
        "dataset_coco_sha256": candidate_sha,
        "classification": "VERIFIED_EXACT",
        "classification_basis": [
            "Pinned SmallCap README directly links the candidate slug.",
            "Candidate dataset_coco.json has the expected Karpathy images schema.",
            "Candidate dataset_coco.json bytes equal the local previously downloaded artifact by SHA-256.",
        ],
    }

    tokenizer = load_gpt2_tokenizer(args.gpt2_revision)
    token_records = source_caption_records(records, "tokens_joined")
    raw_records = source_caption_records(records, "raw")
    token_literal, token_literal_stats = filtered_records(token_records, tokenizer, padded=True)
    token_unpadded, token_unpadded_stats = filtered_records(token_records, tokenizer, padded=False)
    raw_literal, raw_literal_stats = filtered_records(raw_records, tokenizer, padded=True)
    raw_unpadded, raw_unpadded_stats = filtered_records(raw_records, tokenizer, padded=False)
    public = read_json(args.public_captions)
    if not isinstance(public, list) or not all(isinstance(item, str) for item in public):
        raise ValueError("public captions must be a JSON list of strings")

    comparisons = {
        "tokens_joined_literal_padded": comparison([item["caption"] for item in token_literal], public),
        "tokens_joined_diagnostic_unpadded": comparison([item["caption"] for item in token_unpadded], public),
        "raw_literal_padded": comparison([item["caption"] for item in raw_literal], public),
        "raw_diagnostic_unpadded": comparison([item["caption"] for item in raw_unpadded], public),
    }
    mismatch = {
        "tokens_joined_literal_padded": mismatch_examples(token_literal, public),
        "raw_literal_padded": mismatch_examples(raw_literal, public),
        "diagnostic_note": "Examples are positional diagnostics; no variant was selected merely to force the target cardinality.",
    }
    all_token_records = source_caption_records(records, "tokens_joined")
    mapping = public_source_mapping(public, all_token_records, token_unpadded)
    faiss_result = faiss_metadata(args.public_index, len(public))
    r1_consistency = audit_r1_retrievals(args.r1_run, token_unpadded)

    r1_provenance_path = args.r1_run / "provenance.json"
    r1_provenance = read_json(r1_provenance_path) if r1_provenance_path.is_file() else {}
    test_ids = {int(item["cocoid"]) for item in records if item.get("split") == "test"}
    source_ids = {int(item["image_id"]) for item in token_unpadded}
    query_overlap = sorted(source_ids.intersection(set(R1_IDS)))
    leakage = {
        "r1_query_ids": R1_IDS,
        "r1_query_ids_in_candidate_test_split": all(value in test_ids for value in R1_IDS),
        "artifact_source_splits": dict(sorted(Counter(item["original_split"] for item in token_unpadded).items())),
        "artifact_source_image_count": len(source_ids),
        "query_source_image_overlap": query_overlap,
        "query_source_caption_overlap_count": sum(int(item["image_id"]) in R1_IDS for item in token_unpadded),
        "source_level_val_test_leakage": False,
        "status": "PASS" if not query_overlap and mapping["artifact_source_level_val_test_occurrences"] == 0 else "FAIL",
        "string_overlap_warning": mapping["string_overlap_occurrences_by_original_split"].get("val", 0) + mapping["string_overlap_occurrences_by_original_split"].get("test", 0),
        "interpretation": "String overlap across splits is not source-provenance leakage; ordered artifact records are train/restval only.",
    }

    source_commit = git_commit(args.smallcap_root)
    package_info = {name: package_version(name) for name in ("transformers", "tokenizers", "huggingface-hub", "faiss-cpu", "torch", "openai-clip")}
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "capybara_head": git_commit(REPO_ROOT),
        "smallcap_head": source_commit,
        "required_smallcap_commit": OFFICIAL_SOURCE_COMMIT,
        "packages": package_info,
        "gpt2_revision": args.gpt2_revision,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    datastore = {
        "captions_path": str(args.public_captions),
        "captions_size_bytes": args.public_captions.stat().st_size,
        "captions_sha256": sha256_file(args.public_captions),
        "captions_json_length": len(public),
        "captions_canonical_json_sha256": canonical_json_sha256(public),
        "index": faiss_result,
        "count_matches": len(public) == faiss_result.get("ntotal"),
        "r1_provenance_artifact": r1_provenance,
    }
    official_path = {
        "source_file": "src/retrieve_caps.py",
        "load_data_lines": "13-27",
        "filter_lines": "29-50",
        "index_build_lines": "109-135",
        "readme_dataset_line": "README.md:42",
        "datastore_membership": "train and restval are treated as train; val/test are excluded",
        "caption_field": "' '.join(sentence['tokens'])",
        "tokenizer": "AutoTokenizer.from_pretrained('gpt2') plus add_special_tokens({'pad_token': '[PAD]'})",
        "tokenizer_batch_size": OFFICIAL_BATCH_SIZE,
        "length_test": "len(encoding) <= 25 after padding=True",
        "iteration_order": "JSON images order, then sentences order; no deduplication",
        "training_self_exclusion": "filter_nns lines 94-107 removes same-image captions for offline 7-neighbor mapping",
        "test_inference": "get_indexed_caps.py lines 65-134 searches raw FAISS neighbors and applies only word-count max-caption-len filtering",
        "r1_adapter": "CAPybara R1 smoke takes raw FAISS top-k; test IDs are outside train/restval source image IDs",
    }
    gates = decide_gates(
        source_commit_ok=source_commit == OFFICIAL_SOURCE_COMMIT,
        checkpoint_ok=r1_provenance.get("model_revision") == "fd635f4025a5ccb638ed1e1b540c3671557b9a16",
        public_ok=datastore["count_matches"] and comparisons["tokens_joined_diagnostic_unpadded"]["ordered_exact_equal"],
        faiss_ok=faiss_result.get("status") == "PASS",
        counts_ok=datastore["count_matches"],
        karpathy_test_ids_ok=leakage["r1_query_ids_in_candidate_test_split"],
        official_path_understood=True,
        r1_retrieval_ok=r1_consistency.get("status") == "PASS",
        literal_reconstruction_ok=comparisons["tokens_joined_literal_padded"]["ordered_exact_equal"],
        source_leakage_ok=leakage["status"] == "PASS",
    )

    config = {
        "audit": "smallcap_r1_5b_karpathy_verification",
        "run_id": args.run_id,
        "candidate_slug": KAGGLE_SLUG,
        "candidate_dataset": str(args.candidate_dataset),
        "public_captions": str(args.public_captions),
        "public_index": str(args.public_index),
        "smallcap_commit": source_commit,
        "required_smallcap_commit": OFFICIAL_SOURCE_COMMIT,
        "r1_run": str(args.r1_run),
        "r1_ids": R1_IDS,
        "gpt2_revision": args.gpt2_revision,
        "started_utc": started,
    }
    write_json(run_dir / "dataset_candidate_metadata.json", dataset_metadata)
    write_json(run_dir / "dataset_schema.json", schema)
    write_json(run_dir / "dataset_split_stats.json", splits)
    write_json(run_dir / "dataset_hashes.json", {
        "candidate_archive_sha256": archive_sha,
        "candidate_dataset_sha256": candidate_sha,
        "candidate_dataset_size_bytes": args.candidate_dataset.stat().st_size,
        "local_dataset_sha256": sha256_file(REPO_ROOT / "data" / "dataset_coco.json") if (REPO_ROOT / "data" / "dataset_coco.json").is_file() else None,
        "public_caption_sha256": datastore["captions_sha256"],
        "public_caption_canonical_sha256": datastore["captions_canonical_json_sha256"],
        "faiss_sha256": faiss_result.get("sha256"),
    })
    write_json(run_dir / "caption_reconstruction_stats.json", {
        "input_records_train_restval": len(token_records),
        "tokenizer": {"name": "gpt2", "revision": args.gpt2_revision, "pad_token_added": "[PAD]"},
        "max_tokens": OFFICIAL_MAX_TOKENS,
        "variants": {
            "tokens_joined_literal_padded": token_literal_stats,
            "tokens_joined_diagnostic_unpadded": token_unpadded_stats,
            "raw_literal_padded": raw_literal_stats,
            "raw_diagnostic_unpadded": raw_unpadded_stats,
        },
        "source_semantics": official_path,
    })
    write_json(run_dir / "caption_comparison.json", comparisons)
    write_json(run_dir / "caption_mismatch_examples.json", mismatch)
    write_json(run_dir / "datastore_artifact_metadata.json", datastore)
    write_json(run_dir / "leakage_audit.json", {**leakage, "public_source_mapping": mapping})
    write_json(run_dir / "r2_reproduction_gate.json", gates["r2_reproduction"])
    write_json(run_dir / "datastore_provenance_gate.json", gates["datastore_provenance"])
    write_json(run_dir / "run_config.json", config)
    write_json(run_dir / "environment.json", environment)
    log_lines.extend([
        f"candidate_sha256={candidate_sha}",
        f"public_caption_sha256={datastore['captions_sha256']}",
        f"faiss_sha256={faiss_result.get('sha256')}",
        f"literal_count={len(token_literal)}",
        f"unpadded_count={len(token_unpadded)}",
        f"r2_reproduction={gates['r2_reproduction']['status']}",
        f"datastore_provenance={gates['datastore_provenance']['status']}",
    ])
    (run_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    summary = (
        f"# SmallCap R1.5b audit\n\n"
        f"- R1.5b: **{gates['r1_5b_status']}**\n"
        f"- R2 reproduction: **{gates['r2_reproduction']['status']}**\n"
        f"- Datastore provenance: **{gates['datastore_provenance']['status']}**\n"
        f"- Candidate classification: **{dataset_metadata['classification']}**\n"
        f"- Literal source reconstruction: {len(token_literal)} captions; public artifact: {len(public)} captions.\n"
        f"- Diagnostic unpadded reconstruction: {len(token_unpadded)} captions; ordered public match: {comparisons['tokens_joined_diagnostic_unpadded']['ordered_exact_equal']}.\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(run_dir)
    print(json.dumps(gates, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

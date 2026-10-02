from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from smallcap_r2_closure import (  # noqa: E402
    normalize_metric_keys,
    paper_comparison,
    scientific_gates,
    validate_predictions,
)


def test_rouge_key_aliases_normalize_and_absence_is_null() -> None:
    raw = {"Rouge": 0.42}
    normalized = normalize_metric_keys(raw)
    assert normalized["ROUGE_L"] == 0.42
    assert raw == {"Rouge": 0.42}  # normalization never mutates/presents the raw output as canonical
    assert normalize_metric_keys({"ROUGE-L": 0.42})["ROUGE_L"] == 0.42
    absent = normalize_metric_keys({"Bleu_4": 0.3})
    assert absent["ROUGE_L"] is None
    assert normalize_metric_keys({"ROUGE_L": None})["ROUGE_L"] is None


def test_paper_b4_maps_to_bleu4_not_bleu1() -> None:
    normalized = {"Bleu_1": 0.77, "Bleu_4": 0.367, "METEOR": 0.279, "CIDEr": 1.19, "SPICE": 0.21}
    comparison = paper_comparison(normalized)
    row = comparison["paper_metrics"]["BLEU-4"]
    assert row["evaluator_key"] == "Bleu_4"
    assert row["reproduced_raw"] == 0.367
    assert row["reproduced_raw"] != normalized["Bleu_1"]


def test_paper_comparison_excludes_unreported_metrics() -> None:
    comparison = paper_comparison({"Bleu_1": 0.8, "Bleu_2": 0.6, "Bleu_3": 0.5, "ROUGE_L": 0.4})
    assert set(comparison["paper_metrics"]) == {"BLEU-4", "METEOR", "CIDEr", "SPICE"}
    assert comparison["excluded_from_paper_comparison"] == ["Bleu_1", "Bleu_2", "Bleu_3", "ROUGE_L"]
    assert comparison["paper_metrics"]["BLEU-4"]["reproduced_raw"] is None


def test_scale_delta_and_absolute_delta() -> None:
    comparison = paper_comparison({"Bleu_4": 0.367, "METEOR": 0.279, "CIDEr": 1.19, "SPICE": 0.21})
    row = comparison["paper_metrics"]["BLEU-4"]
    assert row["reproduced_paper_scale"] == 36.7
    assert row["signed_delta"] == 36.7 - 37.0
    assert row["absolute_delta"] == abs(36.7 - 37.0)
    assert comparison["numerical_tolerance"] is None


def test_gate_independence_of_rouge_l_and_datastore_provenance() -> None:
    valid = {"status": "PASS"}
    official = {"Bleu_4": 0.3, "METEOR": 0.2, "CIDEr": 1.0, "SPICE": 0.2, "ROUGE_L": None}
    gates = scientific_gates(valid, official, provenance_ok=True)
    assert gates["R2_OFFICIAL_METRICS"] == "PASS"
    assert gates["R2_CONFIGURED_METRIC_COMPLETENESS"] == "INCOMPLETE"
    assert gates["R2_REPRODUCTION"] == "PASS"
    assert gates["DATASTORE_PROVENANCE"] == "PARTIAL"


def test_prediction_validation_exact_id_set_and_caption_checks() -> None:
    dataset = {"images": [{"split": "test", "cocoid": 10}, {"split": "train", "cocoid": 11}]}
    gold = {"images": [{"id": 10}], "annotations": [{"image_id": 10, "caption": "ref"}]}
    preds = [{"image_id": 10, "caption": "caption"}]
    result = validate_predictions(preds, dataset, gold)
    assert result["prediction_ids_match_dataset_test"] is True
    assert result["prediction_ids_match_gold"] is True
    assert result["status"] == "FAIL"  # synthetic fixture is not the required 5,000-row run

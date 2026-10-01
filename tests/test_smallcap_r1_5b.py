import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_smallcap_r1_5b import (  # noqa: E402
    comparison,
    dataset_schema,
    decide_gates,
    source_caption_records,
    split_stats,
)


class SmallCapR15bTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "dataset": "coco",
            "images": [
                {
                    "split": "train",
                    "filename": "COCO_train2014_000000000001.jpg",
                    "filepath": "",
                    "cocoid": 1,
                    "imgid": 1,
                    "sentids": [11],
                    "sentences": [{"raw": "A cat.", "tokens": ["a", "cat"], "sentid": 11, "imgid": 1}],
                },
                {
                    "split": "restval",
                    "filename": "COCO_train2014_000000000002.jpg",
                    "filepath": "",
                    "cocoid": 2,
                    "imgid": 2,
                    "sentids": [12],
                    "sentences": [{"raw": "A dog.", "tokens": ["a", "dog"], "sentid": 12, "imgid": 2}],
                },
                {
                    "split": "test",
                    "filename": "COCO_test2014_000000000003.jpg",
                    "filepath": "",
                    "cocoid": 3,
                    "imgid": 3,
                    "sentids": [13],
                    "sentences": [{"raw": "A bird.", "tokens": ["a", "bird"], "sentid": 13, "imgid": 3}],
                },
            ],
        }

    def test_schema_and_original_split_counts_are_preserved(self):
        schema = dataset_schema(self.payload["images"], self.payload)
        self.assertEqual(schema["top_level_keys"], ["dataset", "images"])
        self.assertEqual(schema["missing_required_image_fields"], {})
        self.assertEqual(schema["duplicate_sentence_ids"], [])
        stats = split_stats(self.payload["images"])
        self.assertEqual(stats["image_counts_by_original_split"], {"restval": 1, "test": 1, "train": 1})
        self.assertEqual(stats["sentence_counts_by_original_split"], {"restval": 1, "test": 1, "train": 1})

    def test_token_joined_source_mapping_keeps_source_metadata(self):
        mapped = source_caption_records(self.payload["images"], "tokens_joined")
        self.assertEqual([item["caption"] for item in mapped], ["a cat", "a dog"])
        self.assertEqual([item["original_split"] for item in mapped], ["train", "restval"])

    def test_comparison_distinguishes_order_from_multiset(self):
        result = comparison(["a", "b"], ["b", "a"])
        self.assertFalse(result["ordered_exact_equal"])
        self.assertTrue(result["multiset_equal"])
        self.assertTrue(result["set_equal"])

    def test_reproduction_gate_can_be_ready_while_provenance_is_partial(self):
        gates = decide_gates(
            source_commit_ok=True,
            checkpoint_ok=True,
            public_ok=True,
            faiss_ok=True,
            counts_ok=True,
            karpathy_test_ids_ok=True,
            official_path_understood=True,
            r1_retrieval_ok=True,
            literal_reconstruction_ok=False,
            source_leakage_ok=True,
        )
        self.assertEqual(gates["r2_reproduction"]["status"], "R2_REPRODUCTION_READY")
        self.assertEqual(gates["datastore_provenance"]["status"], "DATASTORE_PROVENANCE_PARTIAL")
        self.assertEqual(gates["r1_5b_status"], "PARTIAL")


if __name__ == "__main__":
    unittest.main()

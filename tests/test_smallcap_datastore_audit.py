import sys
import unittest

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_smallcap_datastore import (  # noqa: E402
    canonical_json_sha256,
    compare_public_captions,
    dataset_split_stats,
    filter_captions_official,
    query_image_leakage,
    reconstruct_train_captions,
)


class FakeTokenizer:
    def batch_encode_plus(self, captions, return_tensors, padding):
        assert return_tensors == "np"
        assert padding is True
        lengths = [len(caption.split()) for caption in captions]
        width = max(lengths)
        return {"input_ids": [[1] * length + [0] * (width - length) for length in lengths]}


class SmallCapDatastoreAuditTests(unittest.TestCase):
    def test_restval_is_reclassified_and_only_train_like_captions_are_reconstructed(self):
        records = [
            {"cocoid": 10, "split": "train", "sentences": [{"tokens": ["a", "train"]}]},
            {"cocoid": 11, "split": "restval", "sentences": [{"tokens": ["a", "restval"]}]},
            {"cocoid": 12, "split": "val", "sentences": [{"tokens": ["a", "val"]}]},
        ]
        self.assertEqual(dataset_split_stats(records)["caption_counts_by_split"], {"restval": 1, "train": 1, "val": 1})
        self.assertEqual(
            reconstruct_train_captions(records),
            [{"image_id": 10, "caption": "a train"}, {"image_id": 11, "caption": "a restval"}],
        )

    def test_filter_matches_padded_batch_length_rule(self):
        captions = [
            {"image_id": 1, "caption": "one two"},
            {"image_id": 2, "caption": "one two three"},
            {"image_id": 3, "caption": "short"},
        ]
        kept, stats = filter_captions_official(captions, FakeTokenizer(), batch_size=2)
        self.assertEqual([item["image_id"] for item in kept], [1, 2, 3])
        self.assertEqual(stats["padded_length_histogram"], {"1": 1, "3": 2})

    def test_filter_drops_a_batch_when_padded_width_exceeds_25(self):
        long_caption = " ".join(["word"] * 26)
        captions = [{"image_id": 1, "caption": long_caption}, {"image_id": 2, "caption": "short"}]
        kept, stats = filter_captions_official(captions, FakeTokenizer(), batch_size=2)
        self.assertEqual(kept, [])
        self.assertEqual(stats["dropped_count"], 2)

    def test_public_sequence_comparison_is_order_sensitive(self):
        reconstructed = [{"image_id": 1, "caption": "first"}, {"image_id": 2, "caption": "second"}]
        self.assertEqual(compare_public_captions(reconstructed, ["first", "second"])["status"], "PASS")
        mismatch = compare_public_captions(reconstructed, ["second", "first"])
        self.assertEqual(mismatch["status"], "FAIL")
        self.assertEqual(mismatch["first_mismatch"]["index"], 0)

    def test_query_image_leakage_is_reported(self):
        captions = [{"image_id": 7, "caption": "same"}, {"image_id": 8, "caption": "other"}]
        audit = query_image_leakage(captions, [7, 9])
        self.assertEqual(audit["status"], "FAIL")
        self.assertEqual(audit["overlapping_source_image_ids"], [7])

    def test_canonical_hash_is_stable(self):
        self.assertEqual(canonical_json_sha256(["a", "b"]), canonical_json_sha256(["a", "b"]))
        self.assertNotEqual(canonical_json_sha256(["a", "b"]), canonical_json_sha256(["b", "a"]))


if __name__ == "__main__":
    unittest.main()

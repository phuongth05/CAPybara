import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_smallcap_smoke import load_rows, validate_config


class R1ProtocolTests(unittest.TestCase):
    def test_config_is_exactly_the_four_sample_r1_protocol(self):
        config = json.loads((REPO_ROOT / "configs" / "smallcap_smoke.json").read_text(encoding="utf-8"))
        validate_config(config)
        self.assertEqual(config["sample_limit"], 4)
        self.assertEqual(config["seed"], 2026)
        self.assertEqual(config["retrieval_k"], 4)
        self.assertEqual(config["generation_visual_encoder"], "openai/clip-vit-base-patch32")

    def test_selection_is_deterministic_after_id_sort_and_seeded_shuffle(self):
        with tempfile.TemporaryDirectory() as temp:
            annotation_path = Path(temp) / "coco_karpathy_test.json"
            records = [{"image": f"val2014/COCO_val2014_{image_id:012d}.jpg", "caption": []} for image_id in range(5000)]
            annotation_path.write_text(json.dumps(records), encoding="utf-8")
            first = load_rows(annotation_path, 4, 2026)
            second = load_rows(annotation_path, 4, 2026)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 4)
            self.assertEqual(len({row["image_id"] for row in first}), 4)


if __name__ == "__main__":
    unittest.main()

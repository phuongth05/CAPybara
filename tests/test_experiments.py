import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from capybara.experiments import (
    ExperimentError,
    create_experiment,
    git_commit,
    normalize_predictions,
    write_predictions,
)


class ExperimentTests(unittest.TestCase):
    def test_create_experiment_writes_traceable_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_id": "unit_001",
                        "model": "SmallCap",
                        "dataset": "COCO",
                        "split": "val",
                        "seed": 7,
                    }
                ),
                encoding="utf-8",
            )
            output = create_experiment(config_path, root, root / "outputs")
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["experiment_id"], "unit_001")
            self.assertEqual(metadata["seed"], 7)
            self.assertIn("timestamp_utc", metadata)
            self.assertIsNone(metadata["git_commit"])

    def test_predictions_preserve_raw_and_normalize_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            raw = [{"image_id": 12, "caption": "a caption", "extra": 1}]
            write_predictions(output, raw, raw_predictions=raw)
            normalized = json.loads((output / "predictions.json").read_text())
            self.assertEqual(normalized, [{"image_id": "12", "caption": "a caption"}])
            self.assertEqual(json.loads((output / "raw_predictions.json").read_text()), raw)

    def test_invalid_prediction_is_rejected(self):
        with self.assertRaises(ExperimentError):
            normalize_predictions([{"image_id": 1}])

    def test_unborn_repository_has_no_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            self.assertIsNone(git_commit(root))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Create a traceable experiment output directory from a JSON config."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara.experiments import ExperimentError, create_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    args = parser.parse_args()
    try:
        output = create_experiment(args.config, REPO_ROOT, args.output_root)
    except (ExperimentError, FileExistsError) as exc:
        parser.error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

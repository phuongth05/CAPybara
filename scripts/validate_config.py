#!/usr/bin/env python3
"""Validate a CAPybara experiment config without external ML dependencies."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara.experiments import ExperimentError, load_config, validate_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        validate_config(config)
    except ExperimentError as exc:
        parser.error(str(exc))
    print(f"valid config: {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
set -euo pipefail

smallcap_root="/kaggle/working/smallcap"
dataset_coco="/kaggle/input/datasets/shtvkumar/karpathy-splits/dataset_coco.json"
images_root="/kaggle/input"
retrieval_index="/kaggle/working/CAPybara/datastore/coco_index"
retrieval_captions="/kaggle/working/CAPybara/datastore/coco_index_captions.json"
output_root="/kaggle/working/CAPybara/outputs"
resume_run=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smallcap-root) smallcap_root="$2"; shift 2 ;;
    --dataset-coco) dataset_coco="$2"; shift 2 ;;
    --images-root) images_root="$2"; shift 2 ;;
    --retrieval-index) retrieval_index="$2"; shift 2 ;;
    --retrieval-captions) retrieval_captions="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --resume-run) resume_run="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

python scripts/validate_config.py configs/smallcap_r2.json

args=(
  --config configs/smallcap_r2.json
  --smallcap-root "$smallcap_root"
  --dataset-coco "$dataset_coco"
  --images-root "$images_root"
  --retrieval-index "$retrieval_index"
  --retrieval-captions "$retrieval_captions"
  --output-root "$output_root"
)
if [[ -n "$resume_run" ]]; then
  args+=(--resume-run "$resume_run")
fi

python scripts/run_smallcap_r2.py "${args[@]}"

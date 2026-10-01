#!/usr/bin/env bash
set -euo pipefail

smallcap_root=""
karpathy_annotations=""
retrieval_index=""
retrieval_captions=""
coco_input_root=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smallcap-root) smallcap_root="$2"; shift 2 ;;
    --karpathy-annotations) karpathy_annotations="$2"; shift 2 ;;
    --retrieval-index) retrieval_index="$2"; shift 2 ;;
    --retrieval-captions) retrieval_captions="$2"; shift 2 ;;
    --coco-input-root) coco_input_root="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

repo_url="https://github.com/RitaRamo/smallcap.git"
repo_commit="19cc4f4e5972c70fde16c8857b14c96e5492cb30"
if [[ ! -d "$smallcap_root/.git" ]]; then
  git clone "$repo_url" "$smallcap_root"
fi
git -C "$smallcap_root" fetch --quiet origin "$repo_commit"
git -C "$smallcap_root" checkout --detach "$repo_commit"

# COMPATIBILITY PATCH — no intended methodological change.
# These versions are the locally verified Kaggle bridge for Python 3.12;
# do not replace the upstream model/source or install upstream torch pins.
python -m pip install --no-cache-dir faiss-cpu==1.12.0
python -m pip install --no-cache-dir --only-binary=:all: tokenizers==0.15.2
python -m pip install --no-cache-dir transformers==4.36.2 huggingface-hub==0.20.3 gdown
python -m pip install --no-cache-dir git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16

if [[ ! -f "$karpathy_annotations" ]]; then
  mkdir -p "$(dirname "$karpathy_annotations")"
  python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json', '$karpathy_annotations')"
fi
python -c "import hashlib, pathlib, sys; p=pathlib.Path('$karpathy_annotations'); d=hashlib.md5(p.read_bytes()).hexdigest(); expected='3ff34b0ef2db02d01c37399f6a2a6cd1'; print('Karpathy MD5:', d); sys.exit(0 if d == expected else 'Karpathy MD5 mismatch: '+d)"

mkdir -p "$(dirname "$retrieval_index")" "$(dirname "$retrieval_captions")"
if [[ ! -f "$retrieval_index" ]]; then
  python -m gdown 1ZP5I-xbjaNU7cU48C_ctHd95SaA0jBHe -O "$retrieval_index"
fi
if [[ ! -f "$retrieval_captions" ]]; then
  python -m gdown 1BT0Qc6g40fvtnJ_yY0aipfCuCMgu5qaR -O "$retrieval_captions"
fi

python scripts/run_smallcap_smoke.py \
  --smallcap-root "$smallcap_root" \
  --karpathy-annotations "$karpathy_annotations" \
  --retrieval-index "$retrieval_index" \
  --retrieval-captions "$retrieval_captions" \
  --coco-input-root "$coco_input_root"

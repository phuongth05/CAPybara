# CAPybara

Reproducible infrastructure for retrieval-augmented image captioning (RAIC).

## Current milestone

The first baseline is SmallCap. This repository provides experiment bookkeeping, validation, prediction normalization, and an adapter for the verified official-checkpoint protocol. It does not vendor or modify the official SmallCap implementation.

Status: R0 and the fresh four-image R1 engineering smoke run are PASS; R1.5b verifies the pinned Karpathy candidate and public datastore for reproduction. R2 reproduction is READY, while datastore provenance remains PARTIAL because the literal pinned padded filter does not reproduce the public caption sequence.

## Local validation

From the repository root:

```powershell
python -m unittest discover -s tests -v
python scripts/validate_config.py configs/smallcap_smoke.json
```

The validation command is CPU-only and does not require model, dataset, or datastore artifacts.

## SmallCap smoke test

The Kaggle entry point checks out the pinned official source, installs the verified Python 3.12 compatibility environment, and runs the old evidence-backed protocol: online FAISS retrieval plus online CLIP encoding. It does not use `infer.py` or an HDF5 feature cache.

```bash
bash kaggle/run_smallcap_smoke.sh \
  --smallcap-root /kaggle/working/smallcap \
  --karpathy-annotations /kaggle/working/coco_karpathy_test.json \
  --retrieval-index /kaggle/working/smallcap_datastore/coco_index \
  --retrieval-captions /kaggle/working/smallcap_datastore/coco_index_captions.json \
  --coco-input-root /kaggle/input
```

The script downloads and checks the Karpathy annotation file and official public retrieval artifacts when they are absent. It selects four deterministic Karpathy-test records using seed 2026, performs `k=4` retrieval with `RN50x64`, then generates with `Yova/SmallCap7M` using `openai/clip-vit-base-patch32` as the generation visual encoder and beam size 3. It records retrievals, captions, environment, and provenance metadata. This is engineering validation only; it does not compute research metrics.

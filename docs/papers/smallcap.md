# SmallCap reproduction record

## Executive status

- R0: PASS for a provenance-tracked official-checkpoint baseline.
- R1: READY-FOR-KAGGLE. The old folder records a historical 100-image smoke run, but its raw `results (20).zip` is not present in the old folder, so this repository does not claim a fresh artifact-complete R1 yet.
- R2/R3: out of scope for this milestone.

## Provenance classification

### OFFICIAL

- Repository: https://github.com/RitaRamo/smallcap
- Pinned reproduction commit: `19cc4f4e5972c70fde16c8857b14c96e5492cb30`.
- The archived checkout inside `D:\KIEMCOM\HK3-N3\KLTN\09imagecaptioning\result\results small.zip` contains `.git/config`, the official remote, `HEAD`, packed refs, and clone logs pointing to that commit.
- Checkpoint: `Yova/SmallCap7M`.
- Hugging Face cache manifest in the archived result records checkpoint revision `fd635f4025a5ccb638ed1e1b540c3671557b9a16`.
- Caption encoder: `openai/clip-vit-base-patch32`, cached revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
- Decoder: `gpt2`, cached revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Official retrieval downloads are the FAISS index ID `1ZP5I-xbjaNU7cU48C_ctHd95SaA0jBHe` and caption array ID `1BT0Qc6g40fvtnJ_yY0aipfCuCMgu5qaR`.

These identities agree with the [official repository](https://github.com/RitaRamo/smallcap) and [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Ramos_SmallCap_Lightweight_Image_Captioning_Prompted_With_Retrieval_Augmentation_CVPR2023_paper.html).

### LOCAL-REIMPLEMENTATION

- `notebooks/01_smallcap_kaggle.ipynb` is a local execution harness, not upstream SmallCap code.
- It performs FAISS retrieval online, uses CLIP `RN50x64`, uses CLIP ViT-B/32 for caption conditioning, sets `k=4`, and calls the pinned upstream model directly.
- It uses a Python 3.12/Kaggle environment and records outputs, manifests, metrics, and telemetry.

### VERIFIED-COMPATIBLE

- The old harness applies a runtime `past` → `past_key_values` bridge for Transformers 4.36.2. This does not alter weights or retrieval methodology.
- The old full result contains 5,000 predictions and reports CIDEr `1.195154`, close to the paper-scale value reported in the old analysis. This is evidence that the harness executed successfully, not a fresh CAPybara run.

### UNVERIFIED

- The old top-level directory is not a Git repository, so its working-tree dirty/clean state cannot be recovered.
- The retrieval index/caption array source split and query-caption exclusion policy are not proven by the available artifacts.
- The raw 100-image smoke archive referenced as `results (20).zip` is absent from the old folder.

## Artifact inventory from old evidence

| Artifact | Exact path | Exists | Size/provenance | Used by | R1? |
|---|---|---:|---|---|---:|
| Old SmallCap harness | `D:\KIEMCOM\HK3-N3\KLTN\09imagecaptioning\notebooks\01_smallcap_kaggle.ipynb` | Yes | 22,874 bytes; local notebook | Kaggle execution | No |
| Archived official source + `.git` | `D:\KIEMCOM\HK3-N3\KLTN\09imagecaptioning\result\results small.zip` → `smallcap/` | Yes | 2,565,477,667-byte archive; pinned commit proven inside | model imports/template | Yes |
| FAISS retrieval index | same archive → `smallcap_datastore/coco_index` | Yes | 2,316,275,757 bytes; opaque FAISS binary | `faiss.read_index` | Yes |
| Retrieval captions | same archive → `smallcap_datastore/coco_index_captions.json` | Yes | 31,252,824 bytes; JSON array with 565,497 entries | FAISS ID lookup | Yes |
| Karpathy test annotations | same archive → `coco_karpathy_test.json` | Yes | 1,735,468 bytes; JSON list of 5,000 records | deterministic sample selection | Yes |
| HF cache manifest | same archive → `__huggingface_repos__.json` | Yes | 538 bytes; model/tokenizer revisions | checkpoint/encoder provenance | Yes |
| Full prediction bundle | `D:\KIEMCOM\HK3-N3\KLTN\09imagecaptioning\result\smallcap-full-results.zip` | Yes | 74,675 bytes; 5,000 predictions plus metadata | historical verification | No |
| COCO images | external Kaggle dataset, resolved under `/kaggle/input` | Not local | old notebook searches `train2017`/`val2017` | image encoding | Yes |

The checkpoint weights themselves are not stored in the old folder; the old notebook downloads the pinned Hugging Face model.

## Recovered execution protocol

The old notebook clones `https://github.com/RitaRamo/smallcap.git`, checks out `19cc4f4`, downloads the two official retrieval artifacts, downloads the Karpathy test list, and loads `Yova/SmallCap7M`. It then:

1. sorts Karpathy records by numeric COCO ID;
2. shuffles with seed `2026` and selects the configured profile size;
3. computes online `RN50x64` image embeddings;
4. retrieves `k=4` captions from the FAISS index;
5. encodes the image with CLIP ViT-B/32;
6. generates with GPT-2/SmallCap, beam size 3, `max_new_tokens=25`;
7. preserves predictions, metrics, telemetry, and a manifest.

The old notebook's default is `PROFILE="full"`; its historical smoke report describes 100 images. CAPybara uses the same protocol with `sample_limit=4` for a minimal engineering R1 run.

## Leakage and protocol findings

### Facts

- The retrieval array is precomputed and indexed by FAISS result ID.
- The notebook uses the same retrieval index for Karpathy-test queries and does not construct a new datastore locally.
- The old notebook does not prove whether query-image captions or test references occur in the retrieval array.
- No HDF5 feature cache is used by the recovered R1 path; both retrieval and caption image encodings are computed online.

### Unresolved

- Retrieval source split and deduplication policy.
- Whether the official datastore contains captions from query images.
- Whether the released FAISS index includes validation/test references.
- Exact mapping between the official datastore construction and the paper's leakage assumptions.

These remain scientific risks; no “no leakage” claim is made.

## Dependency findings

### Published upstream requirements

The archived upstream `requirements.txt` specifies Python 3.9-era packages including `torch==1.12.1`, `torchvision==0.13.1`, `transformers==4.21.1`, `tokenizers==0.12.1`, `faiss-gpu==1.7.2`, and `h5py==3.7.0`.

### Verified local compatibility environment

The old manifest records Python 3.12.13, PyTorch 2.10.0+cu128, Transformers 4.36.2, Tokenizers 0.15.2, Hugging Face Hub 0.20.3, faiss-cpu 1.12.0, pycocoevalcap 1.2, and sentence-transformers 5.4.1 on a Tesla T4. CAPybara reproduces this compatibility strategy without replacing PyTorch/CUDA.

The modern environment is a compatibility patch, not the original upstream environment. It must remain explicitly labeled as such.

## CAPybara R1 adapter

`scripts/run_smallcap_smoke.py` now uses the recovered protocol and refuses CPU fallback, dirty source checkouts, source-commit mismatches, missing images, mismatched FAISS/caption cardinality, or invalid Karpathy annotations. It writes `config.json`, `metadata.json`, `r1_metadata.json`, `raw_predictions.json`, and normalized `predictions.json`.

The generated R1 result is an engineering smoke test only. It must not be used for BLEU, CIDEr, SPICE, or model-comparison claims.

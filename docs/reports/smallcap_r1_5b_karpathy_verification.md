# SmallCap R1.5b — Karpathy verification, datastore mismatch diagnosis, and R2 gate closure

Audit run: `outputs/smallcap/r1_5b_karpathy_verification/20261001T152000Z/`  
No training, robustness experiment, or 5,000-image inference was run.

## 1. Repository state

**VERIFIED FACT**

- CAPybara HEAD before this R1.5b change: `817667e46114401f701b673f7b75e5338ded6252`.
- Pinned SmallCap checkout: `19cc4f4e5972c70fde16c8857b14c96e5492cb30`, clean.
- Existing R1 artifact: `outputs/smallcap/r1_smoke/20261001T041625Z_from_download/`, completed with four deterministic test IDs and `retrievals.json`.
- Existing public artifacts: `data/dataset_coco.json`, `datastore/coco_index_captions.json`, and `datastore/coco_index`.
- The R1.5b run is additive; previous R1/R1.5 audit directories were not overwritten.

## 2. Candidate dataset provenance

**VERIFIED FACT**

The candidate is Kaggle dataset [`shtvkumar/karpathy-splits`](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits). The pinned SmallCap README names this exact dataset and filename at `README.md:42`.

- Archive: `karpathy-splits.zip`, 40,837,470 bytes.
- Archive SHA-256: `ad262c22a46e6c76a6f3bcfddf6a8c5f27e787c152cd24380d01a01c196bee89`.
- Archive members: `dataset_coco.json`, `dataset_flickr30k.json`, `dataset_flickr8k.json`.
- Candidate `dataset_coco.json`: 144,186,139 bytes; SHA-256 `2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d`.

**OFFICIAL DISTRIBUTION**

The candidate is classified `VERIFIED_EXACT` for the expected SmallCap input because the pinned README directly links it and its bytes match the local dataset artifact by SHA-256. This classification does not imply that the public datastore was rebuilt by the currently checked-out literal code.

## 3. Dataset schema and split counts

**VERIFIED FACT**

The JSON has top-level keys `dataset` and `images`. It contains 123,287 image records. Every image record has `split`, `filename`, `filepath`, `cocoid`, `imgid`, `sentids`, and `sentences`; every sentence has `raw`, `tokens`, `sentid`, and `imgid`. There are no duplicate image IDs, duplicate sentence IDs, missing required fields, or unexpected splits.

| Original split | Images | Sentence records |
|---|---:|---:|
| train | 82,783 | 414,113 |
| restval | 30,504 | 152,634 |
| val | 5,000 | 25,010 |
| test | 5,000 | 25,010 |
| Total | 123,287 | 616,767 |

## 4. Pinned SmallCap construction semantics

**VERIFIED_SOURCE_BEHAVIOR**

From pinned `src/retrieve_caps.py`:

- `load_coco_data`, lines 13–27, changes `restval` to `train`, includes only train-like records, iterates JSON image order then sentence order, and constructs captions with `' '.join(sentence['tokens'])`.
- `filter_captions`, lines 29–50, uses `AutoTokenizer.from_pretrained('gpt2')`, adds `[PAD]`, batches 512 captions, calls `batch_encode_plus(..., return_tensors='np', padding=True)`, and accepts `len(encoding) <= 25`.
- There is no caption deduplication and no raw-caption substitution in the pinned source.
- `filter_nns`, lines 94–107, excludes same-image captions for the offline seven-neighbor mapping.
- `src/retrieve_caps.py`, lines 109–135, builds the normalized `IndexFlatIP` datastore from the filtered caption sequence.
- The GPT-2 revision used here is `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

**OFFICIAL DISTRIBUTION**

The official test-time retrieval path in `src/get_indexed_caps.py`, lines 65–134, searches the public FAISS datastore and applies the maximum-caption-length check; it is distinct from training-time `filter_nns` self-exclusion.

## 5. Caption reconstruction result

**VERIFIED FACT**

Train+restval provides 566,747 input captions.

- Literal pinned source (`tokens` joined, `padding=True`, `len(encoding) <= 25`): **177,152** accepted and 389,595 rejected.
- Diagnostic unpadded token-length variant (`tokens` joined, `padding=False`): **565,497** accepted and 1,250 rejected.
- Raw-caption diagnostics do not explain the public artifact: raw/unpadded retains 564,664 and raw/padded retains 77,824.

No preprocessing was changed to force 565,497. The mismatch is specifically between the literal padded batch-width test and the public sequence; the unpadded token-joined diagnostic reproduces the target cardinality.

## 6. Public caption artifact comparison

**VERIFIED FACT**

The recomputed public `datastore/coco_index_captions.json` values are:

- Size: 31,252,824 bytes.
- Raw SHA-256: `5cf206a68ae18a10667928fd916d077a3374f361d83e10a5482d2f71ada041a6`.
- JSON length: 565,497.
- Canonical JSON SHA-256: `dd515fca4f99d7de4938b4c7e5f582cdb11192dca0cb13e36d70fa8ca67c663e`.

This resolves the earlier hash ambiguity: the correct raw hash ends in `041a6`, not `0416`.

- Literal padded reconstruction: cardinality, ordered, multiset, and set comparisons all **FAIL**; first mismatch index is 0.
- Unpadded token-joined diagnostic: cardinality, ordered, multiset, and set comparisons all **PASS**; first mismatch is null.
- Raw-caption variants fail both cardinality and content comparisons.

The unpadded variant is explanatory for the public content, but it is not promoted to “literal official construction”; the pinned source still says `padding=True`.

## 7. Leakage/source mapping

**VERIFIED FACT**

The public-order-compatible reconstruction contains only train/restval source records: 413,192 train captions and 152,305 restval captions. The four R1 query IDs are all in candidate `test`; source image overlap is zero and query-source caption overlap is zero.

All 565,497 public caption strings map to candidate records. 530,905 occurrences map to one candidate record and 34,592 are ambiguous because identical strings occur in multiple candidate records. The audit distinguishes those string matches from source provenance; no val/test source records occur in the ordered artifact-compatible reconstruction.

## 8. R1 retrieval protocol assessment

**OFFICIAL DISTRIBUTION / INFERENCE**

The R1 ZIP retrieval audit remains **PASS**: the frozen query order is preserved, all four queries have four retrieved captions, all 16/16 captions map to candidate source records, and unresolved count is 0.

R1 raw top-4 retrieval is classified `OFFICIAL_PROTOCOL_COMPATIBLE` for the Karpathy test run: the public datastore membership is train+restval, the test query IDs are absent from its source image IDs, and official test retrieval searches the datastore directly. The separate `filter_nns` self-exclusion is a training/offline-neighbor concern, not evidence of test leakage here.

## 9. FAISS artifact verification

**VERIFIED FACT**

The public `datastore/coco_index` is 2,316,275,757 bytes with SHA-256 `12cb69e401eac7d29adeb9c36576790f4619b5bf70029a037cb90266f1099fc7`. It loads as `IndexFlatIP`, metric type `0`, dimension `1024`, `ntotal=565497`, and `is_trained=true`. Caption count equals FAISS `ntotal`. Deterministic sampled vector norms were `1.0`, `0.9999999404`, and `1.0`.

## 10. Tests and validation

**VERIFIED FACT**

- Full unit suite: **17 tests passed**.
- Python compilation: `scripts/audit_smallcap_r1_5b.py` and its tests passed.
- Config validation: `python scripts/validate_config.py configs/smallcap_smoke.json` passed.
- No full R2 inference, training, or robustness experiment was executed.
- CAPybara commit after this audit: `44d0ab10cb897745ca252ff82a8daa6af1941346`.

## 11. Scientific gates

R1: PASS  
R1.5b: PARTIAL  
R2_REPRODUCTION: READY  
DATASTORE_PROVENANCE: PARTIAL  
R3: NOT STARTED

R2 reproduction is ready because the pinned source/checkpoint, public datastore hashes, FAISS structure, caption count, test IDs, inference path, and R1 retrieval consistency are verified. Datastore provenance remains partial because the literal pinned padded filter does not reproduce the public caption list; this unresolved issue does not invalidate use of the officially distributed pretrained artifacts for reproduction.

Audit output directory: `outputs/smallcap/r1_5b_karpathy_verification/20261001T152000Z/`.

## 12. Next action

R2 reproduction is ready. A dedicated 5,000-image runner and config were added after this audit. From the active Kaggle notebook, run:

```bash
%cd /kaggle/working/CAPybara
!bash kaggle/run_smallcap_r2.sh --smallcap-root /kaggle/working/smallcap --dataset-coco /kaggle/input/datasets/shtvkumar/karpathy-splits/dataset_coco.json --images-root /kaggle/input
```

The R2 runner checkpoints generated captions every 50 images and can resume using `--resume-run /kaggle/working/CAPybara/outputs/smallcap/r2_full/<RUN_ID>`.

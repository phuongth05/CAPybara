# SmallCap R2 — evaluation closure and reproduction report

## 1. Experiment identity

- Experiment: SmallCap7M, MS COCO Karpathy test, 5,000 images.
- Generation run: `outputs/smallcap/r2_full/20261001T171104Z/`.
- Evaluation-only closure: `outputs/smallcap/r2_eval_closure/20261002T044229Z/`.
- CAPybara HEAD at R2 generation: `20df4b635e0f49f9e2e4b1a6408b81840d0944dd`.
- No inference, model loading, retrieval, FAISS access, decoding, or GPU work occurred in this closure.

## 2. Source and checkpoint provenance

- SmallCap checkout recorded by the R2 `run_config.json` and `provenance.json`, and verified against the local checkout: `19cc4f4e5972c70fde16c8857b14c96e5492cb30` (40-character Git SHA).
- The supplied task text also included `19cc4f4e5972c70fde16e1b5407b14c96e5492cb30`, a 42-character invalid SHA. It is not the SHA recorded by the run; no run metadata was rewritten to that value.
- Checkpoint: `Yova/SmallCap7M`, revision `fd635f4025a5ccb638ed1e1b540c3671557b9a16`.
- Dataset SHA-256: `2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d`.
- Datastore provenance remains an independent partial issue; see section 13.

## 3. Saved prediction validation

The immutable input was `outputs/smallcap/r2_full/20261001T171104Z/predictions.json`.

- SHA-256 before and after closure: `2dfbb295c3be8fe57baa893e6b3d3d08a5f53170d442b55775497769c199c400` (unchanged).
- 5,000 prediction records; 5,000 unique image IDs; no duplicate IDs.
- No missing caption fields, empty captions, or invalid IDs.
- Prediction IDs exactly match both the 5,000 `split == test` IDs in `data/dataset_coco.json` and the 5,000 IDs in the pinned gold references.
- Gold file: pinned checkout `coco-caption/annotations/captions_testKarpathy.json`; 25,010 reference annotations; SHA-256 `fcaa4f4e80717fef6508ea583fe0f87dffc9c0c634db25343cdc20eebc23759a`.
- Machine-readable evidence: `outputs/smallcap/r2_eval_closure/20261002T043653Z/prediction_validation.json` and `predictions_metadata.json`.

## 4. Evaluator provenance

The evaluator is the `COCOEvalCap` implementation in the SmallCap-pinned checkout, not a substitute caption metric package. Its source is `outputs/smallcap/upstream/smallcap/coco-caption/pycocoevalcap/eval.py`, SHA-256 `b5d3d3d60c4e96fac551dada47230d47dff299292e348571769688a1ea2b5dd3`.

Its scorer registry instantiates `Bleu(4)`, `Meteor()`, `Cider()`, and `Spice()`. BLEU returns orders 1–4. The COCO-caption `Rouge` class is imported but its registration line, `(Rouge(), "ROUGE_L")`, is commented out. The same pinned reference data and PTB tokenization path were used for the local BLEU/METEOR/CIDEr reevaluation.

The original R2 `metrics.json` and `provenance.json` record a successful official evaluation, including SPICE, on the same saved run; the recovery notebook uses `evaluate_predictions` on that saved prediction path and records the pinned source, scorer/runtime dependency hashes, and `reused_saved_predictions: true`. The original `run.log` retains an earlier failed SPICE attempt, not the later successful recovery-cell output; the successful metric is recorded in structured `metrics.json` and `provenance.json`. The local Windows closure independently reproduced BLEU-1–4, METEOR, and CIDEr exactly. A local SPICE rerun was attempted but the Windows LMDB JNI native runtime failed while opening its database; evidence is retained at `outputs/smallcap/r2_eval_closure/20261002T043126Z/run.log`. Therefore the closure preserves the prior successful SPICE score unchanged and labels its source explicitly; it does not claim SPICE was recomputed on Windows.

ROUGE-L diagnosis is **C**: the CAPybara config requests `ROUGE_L`, but the pinned evaluator does not instantiate the ROUGE scorer. It is not an output-key normalization issue. The official SmallCap paper path intentionally reports the four metrics below, not ROUGE-L.

## 5. Primary-paper metric verification

The primary CVPR 2023 paper’s Table 1 columns are `Model`, parameter count, `B@4`, `M`, `CIDEr`, and `S`. The SmallCap 7M row is `37.0`, `27.9`, `119.7`, and `21.3`, respectively. The paper explicitly defines its standard evaluation metrics as BLEU-4 (B@4), METEOR (M), CIDEr, and SPICE (S), using the COCO evaluation package: [SmallCap, CVPR 2023 primary paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ramos_SmallCap_Lightweight_Image_Captioning_Prompted_With_Retrieval_Augmentation_CVPR_2023_paper.pdf).

**B@4 in the SmallCap paper means BLEU-4, not BLEU-1.** The target mapping is `B@4 → Bleu_4`.

## 6. Reproduced raw metrics

Values are raw evaluator fractions/scores, not paper-scale values. The first six values below were freshly evaluated from the immutable predictions in the closure. SPICE is the unchanged value from the prior successful official R2 evaluation artifact, because the pinned SPICE LMDB runtime could not be rerun on Windows.

| Metric | Raw value | Source in this closure |
|---|---:|---|
| Bleu_1 | 0.7713822802077761 | Re-evaluated |
| Bleu_2 | 0.6170201446188212 | Re-evaluated |
| Bleu_3 | 0.47742656659218013 | Re-evaluated |
| Bleu_4 | 0.3670115581011909 | Re-evaluated |
| METEOR | 0.27901066152881876 | Re-evaluated |
| CIDEr | 1.1951543812549756 | Re-evaluated |
| SPICE | 0.21261598647828214 | Prior successful official R2 metric artifact; not recomputed in the Windows closure |
| ROUGE_L | `null` | Not computed by the pinned evaluator |

The fresh local BLEU/METEOR/CIDEr values are exactly equal to the corresponding values in the original `metrics.json`. Details are in `raw_metrics.json`, `normalized_metrics.json`, and `saved_metrics_comparison.json` under the evaluation-only directory.

## 7. Paper-scale comparison and deltas

Paper-scale value is raw × 100. Signed delta is reproduced paper-scale minus the paper target. No tolerance was assumed or invented.

| Metric | Paper target | Reproduced raw | Reproduced paper-scale | Signed delta | Absolute delta | Status |
|---|---:|---:|---:|---:|---:|---|
| BLEU-4 | 37.0 | 0.3670115581011909 | 36.70115581011909 | -0.29884418988091 | 0.29884418988091 | Protocol match with numerical delta |
| METEOR | 27.9 | 0.27901066152881876 | 27.901066152881874 | +0.001066152881876 | 0.001066152881876 | Protocol match with numerical delta |
| CIDEr | 119.7 | 1.1951543812549756 | 119.51543812549757 | -0.184561874502435 | 0.184561874502435 | Protocol match with numerical delta |
| SPICE | 21.3 | 0.21261598647828214 | 21.261598647828215 | -0.038401352171785 | 0.038401352171785 | Protocol match with numerical delta; prior successful run value |

Overall comparison label: **PROTOCOL_MATCH_WITH_NUMERICAL_DELTA**. This is not an `EXACT_MATCH`; no post-hoc numerical threshold was used.

## 8. EXTRA / NOT A SMALLCAP PAPER TARGET

- BLEU-1: `0.7713822802077761`
- BLEU-2: `0.6170201446188212`
- BLEU-3: `0.47742656659218013`
- ROUGE-L: `null` — not emitted by the pinned evaluator and not reported as a SmallCap paper target.

These metrics are excluded from paper-target deltas. In particular, the paper’s B@4 `37.0` is compared only to reproduced BLEU-4 `36.70115581011909`, never to BLEU-1.

## 9. ROUGE-L status

- Pinned evaluator key mapping: no emitted Rouge key exists because the scorer is not instantiated.
- `normalized_metrics.json` records `"ROUGE_L": null` with `NOT_COMPUTED_BY_PINNED_PROTOCOL`.
- CAPybara configured-metric completeness remains **INCOMPLETE** because `configs/smallcap_r2.json` lists ROUGE-L. This does not invalidate the four-metric SmallCap paper evaluation.
- No value was inferred, manually inserted, or substituted.

## 10. Reproduction semantics

- Label: **PROTOCOL_MATCH_WITH_NUMERICAL_DELTA**.
- The saved test IDs, checkpoint revision, recorded SmallCap source, gold references, and four official metric values are available and traceable.
- Numerical differences are reported as observed; no acceptance tolerance was preregistered or introduced after seeing results.
- Local closure status is **PARTIAL** for independent reevaluation only: SPICE was not rerun on Windows. The complete official metric set remains available from the successful original R2 evaluation artifact.

## 11. Scientific gates

R0: PASS
R1: PASS
R2_GENERATION: PASS
R2_OFFICIAL_METRICS: PASS
R2_CONFIGURED_METRIC_COMPLETENESS: INCOMPLETE
R2_REPRODUCTION: PASS
DATASTORE_PROVENANCE: PARTIAL
R3: NOT STARTED

R2 reproduction PASS is independent of datastore provenance. It does not assert that the distributed datastore’s original construction has been fully reconstructed.

## 12. Evaluator versions and runtime

- SmallCap: recorded/observed commit `19cc4f4e5972c70fde16c8857b14c96e5492cb30`.
- Vendored `COCOEvalCap`, PTBTokenizer, BLEU, METEOR, CIDEr, SPICE source: same pinned checkout; COCO evaluation source hashes are in `evaluator_provenance.json`.
- Original Kaggle inference environment: Python 3.12.13, Transformers 4.36.2, PyTorch 2.10.0+cu128; generation used Tesla T4. This closure used no GPU.
- Local reevaluation environment: Python 3.12.7, NumPy 1.26.4, pycocotools distribution 2.0.11, OpenJDK 17.0.16.
- The SPICE runtime dependency versions and successful Kaggle JAR hashes are preserved in the original R2 `provenance.json`; local Windows SPICE failed at LMDB database open and its output was not substituted.

## 13. Datastore provenance limitation

`DATASTORE_PROVENANCE` remains **PARTIAL**. The existing R1.5b audit found that the literal padded caption filter in the pinned source does not reconstruct the public caption list, although the public datastore distribution, index dimensions/count, test leakage checks, and R1 retrieval consistency were audited. Matching R2 metrics does not resolve that independent construction-provenance discrepancy. See [R1.5b audit](smallcap_r1_5b_karpathy_verification.md).

## 14. Next action

Keep the original 5,000 predictions and R2 artifacts immutable. If independent SPICE reevaluation is required, run the pinned evaluator in the same Linux/Kaggle-compatible environment with the recorded Java dependencies and compare against `0.21261598647828214`; do not alter predictions or substitute a different implementation. ROUGE-L should remain null unless the project deliberately adds and documents a separate unified-evaluator protocol. Resolve datastore construction provenance independently before changing its gate.

## 15. Tests and validation

- Full unit suite: **23 passed** (`python -m pytest -q`).
- Config validation: **PASS** (`python scripts/validate_config.py configs/smallcap_r2.json`).
- Python compilation: **PASS** for the closure script, its tests, and `scripts/run_smallcap_r2.py`.
- `git diff --check`: run before commit; no dataset, model, FAISS, or prediction file is staged.

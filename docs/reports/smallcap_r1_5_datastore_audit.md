# SmallCap R1.5 Datastore Provenance and Leakage Audit

Audit run: `outputs/smallcap/r1_5_datastore_audit/20261001T050000Z_blocked/`  
Decision: **R2_BLOCKED**.  
This report intentionally distinguishes an unavailable fact from a failed scientific check.

## 1. Scope and decision

R1.5 audits the caption datastore used by the four-image R1 smoke run. It does not start R2, full evaluation, training, or metric computation. The exact `dataset_coco.json` used to build the datastore was not found in the CAPybara workspace or the archived SmallCap result package, so the exact reconstruction and leakage verdict cannot be completed.

## 2. Pinned inputs and source identity

The pinned upstream SmallCap source is commit `19cc4f4e5972c70fde16c8857b14c96e5492cb30`. The archived source is at `D:\KIEMCOM\HK3-N3\KLTN\09imagecaptioning\result\results small.zip` under `smallcap/`. The R1 run recorded CAPybara commit `a046d4da6b22f2fa1ff1b9a4881c7661630f750f`, model `Yova/SmallCap7M` revision `fd635f4025a5ccb638ed1e1b540c3671557b9a16`, and GPT-2 revision `607a...` in the verified harness metadata.

## 3. Official datastore construction logic

The pinned `smallcap/src/retrieve_caps.py` reconstructs captions from `dataset_coco.json['images']`: `restval` is changed to `train`, only train-like records contribute captions, and each caption is `' '.join(sentence['tokens'])` (source lines 13–27). It then tokenizes in batches of 512 with GPT-2 plus a `[PAD]` token and keeps rows whose padded encoded length is at most 25 (source lines 29–50). The audit implementation is [audit_smallcap_datastore.py](../../scripts/audit_smallcap_datastore.py) and follows these details, including the padded-batch behavior.

## 4. Dataset split and caption counts

Status: **UNVERIFIED**. `dataset_coco.json` was absent from the current CAPybara tree and from the archived result package; a recursive search found no matching file. Therefore exact train/restval/val/test image counts, raw caption counts, and post-filter counts are intentionally `UNVERIFIED`, not replaced with counts from the Karpathy test file.

The blocked run stores the machine-readable placeholder and gate at [dataset_split_stats.json](../../outputs/smallcap/r1_5_datastore_audit/20261001T050000Z_blocked/dataset_split_stats.json) and [r2_gate.json](../../outputs/smallcap/r1_5_datastore_audit/20261001T050000Z_blocked/r2_gate.json) when artifacts are available locally; `outputs/` is gitignored by protocol.

## 5. Caption reconstruction and GPT-2 filter

Status: **UNVERIFIED** because the exact dataset and tokenizer execution inputs were unavailable. The deterministic implementation is covered by focused tests for restval handling, sentence-token joining, batch padding, and the `<=25` rule. It writes `reconstructed_captions.json` and `caption_filter_stats.json` on a complete run.

## 6. Public caption artifact

The archive contains `smallcap_datastore/coco_index_captions.json`, a JSON list of **565,497** caption strings, with file size **31,252,824** bytes and raw SHA-256 `5cf206a68ae18a10667928fd916d077a3374f361d83e10a5482d2f71ada0416`. Its canonical JSON SHA-256 (UTF-8, compact separators, `ensure_ascii=False`) is `dd515fca4f99d7de4938b4c7e5f582cdb11192dca0cb13e36d70fa8ca67c663e`. These values describe the archived public artifact; they do not prove it was generated from the missing dataset file.

## 7. FAISS index structure

The archive contains `smallcap_datastore/coco_index`, size **2,316,275,757** bytes (ZIP entry CRC `0x0f53f7ce`); its direct SHA-256 was not computed in the blocked audit. The R1 runtime recorded `ntotal=565,497` and dimension `1024`, matching the public caption count. The pinned source constructs an `IndexFlatIP` after L2 normalization (source lines 83–92 and 123–135), but the actual archived index type and metric remain **UNVERIFIED** until `faiss.read_index` is run on the exact file.

## 8. Source image IDs and leakage audit

Status: **UNVERIFIED**. The public captions JSON contains strings only, not source image IDs. Without the exact source dataset and reconstruction, the audit cannot map each public caption back to its COCO image ID, cannot count train/restval source IDs, and cannot test overlap with the four R1 query IDs. No overlap claim is made.

## 9. Query-image exclusion behavior

The official `retrieve_caps.py` includes `filter_nns` (source lines 94–107), which skips candidates whose source image ID equals the query image ID and retains seven neighbors for the offline `retrieved_caps_resnet50x64.json` mapping. However, the CAPybara R1 runner uses raw FAISS top-k results in `scripts/run_smallcap_smoke.py` lines 229–254 and does not call `filter_nns`. Consequently, R1's four retrieved captions are not proven to have same-image captions removed. This is a protocol risk, not evidence that leakage occurred.

## 10. R1 retrieval consistency

R1 itself passed as an engineering smoke test for IDs `[478077, 379529, 87912, 357265]`, with four non-empty predictions and the recorded FAISS count/dimension above. R1 retrieval records contain caption text and scores but no source image IDs, so R1.5 cannot perform the requested source-ID leakage check or independent retrieval consistency comparison against a reconstructed datastore. The result is therefore **UNVERIFIED**, while the original R1 status remains PASS.

## 11. R2 gate

**R2_BLOCKED.** The blockers are: missing exact `dataset_coco.json`; no execution of the official GPT-2 filter on that file; no exact sequence comparison between reconstructed and public captions; no direct FAISS type/hash verification; and no proof that R1 query IDs are excluded from retrieved source IDs. The gate must not be bypassed by using `coco_karpathy_test.json` as a substitute.

## 12. Reproduction command and next action

After obtaining the exact `dataset_coco.json`, run the audit with the pinned public artifacts and the saved R1 retrieval directory:

```powershell
python scripts/audit_smallcap_datastore.py `
  --dataset-coco <exact-dataset-coco.json> `
  --public-captions <coco_index_captions.json> `
  --public-index <coco_index> `
  --smallcap-root <clean-smallcap-checkout-at-19cc4f4e> `
  --r1-run <r1-run-directory> `
  --gpt2-revision <verified-gpt2-revision> `
  --output-root outputs/smallcap/r1_5_datastore_audit
```

Only a complete `R2_READY` gate should permit the next milestone. The focused local test suite currently passes **12/12**.

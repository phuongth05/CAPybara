# Báo cáo audit datastore SmallCap R1.5

Audit đầy đủ: `outputs/smallcap/r1_5_datastore_audit/20261001T081500Z_final/`
Kết luận: **R2_BLOCKED**. Không chạy R2 hoặc full inference.

## 1. Phạm vi và quyết định

Audit kiểm tra dataset, reconstruction caption, public caption array, leakage split, cấu trúc FAISS, embedding RN50x64 và consistency của R1. R1 engineering smoke vẫn **PASS**; R1.5 chưa thể mở R2 vì literal filter trong source pin không tái tạo public artifact và chưa có `retrievals.json` của Kaggle R1 trong workspace.

## 2. Provenance đầu vào

Source SmallCap: commit `19cc4f4e5972c70fde16c8857b14c96e5492cb30`. Dataset được tải bằng Kaggle CLI từ [`vuthetam/mscoco-2014`](https://www.kaggle.com/datasets/vuthetam/mscoco-2014), file `dataset_coco.json`, kích thước `144186139` bytes, SHA-256 `2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d`. GPT-2 revision dùng trong audit là `607a30d783dfa663caf39e06633721c8d4cfcd7e`; CLIP source commit là `a9b1bf5920416aaeaec965c25dd9e8f98c864f16`.

## 3. VERIFIED FACT — split và caption counts

`dataset_coco.json` có 123,287 ảnh và 616,767 captions:

| Split | Ảnh | Captions |
|---|---:|---:|
| train | 82,783 | 414,113 |
| restval | 30,504 | 152,634 |
| val | 5,000 | 25,010 |
| test | 5,000 | 25,010 |

Không có duplicate `cocoid`.

## 4. VERIFIED FACT — reconstruction theo source pin

`src/retrieve_caps.py` lines 13–27 đổi `restval` thành `train`, lấy caption train-like và ghép `' '.join(sentence['tokens'])`. Tổng đầu vào train+restval là **566,747** captions. Audit giữ nguyên thứ tự annotation và ghi cả hai reconstruction: `reconstructed_captions.json` và `source_padded_reconstructed_captions.json`.

## 5. VERIFIED FACT — discrepancy của caption filter

Source pin lines 29–50 dùng `padding=True` nhưng kiểm tra `len(encoding)`, nên literal reproduction giữ **177,152** captions. Khi dùng độ dài GPT-2 không padding — tương đương kiểm tra độ dài thật hoặc `attention_mask.sum(1)` — kết quả là **565,497** captions, loại **1,250**. Đây là chẩn đoán, không phải thay đổi source upstream.

## 6. OFFICIAL DISTRIBUTION — public captions

`datastore/coco_index_captions.json` có **565,497** captions, raw SHA-256 `5cf206a68ae18a10667928fd916d077a3374f361d83e10a5482d2f71ada041a6`. Reconstruction unpadded có canonical SHA-256 `dd515fca4f99d7de4938b4c7e5f582cdb11192dca0cb13e36d70fa8ca67c663e`, đúng bằng public array; so sánh sequence là **EXACT_MATCH**. Literal padded reconstruction không match: 177,152 vs 565,497, canonical hash `5a9f47af7743c07b8a457a14922d0b573ea17fc72f3c3703549f9a9fc21e3853`.

## 7. VERIFIED FACT — FAISS structure

`datastore/coco_index` đọc được bằng FAISS và có:

- type: `IndexFlatIP`;
- `ntotal=565497`;
- `d=1024`;
- metric type `0`;
- SHA-256 `12cb69e401eac7d29adeb9c36576790f4619b5bf70029a037cb90266f1099fc7`.

Count FAISS khớp chính xác public caption array.

## 8. VERIFIED FACT — split leakage

Caption source IDs của reconstruction unpadded chỉ đến từ train+restval. Bốn R1 IDs `[478077, 379529, 87912, 357265]` không overlap source IDs: **0 image, 0 caption**. Vì vậy không phát hiện caption từ val/test hoặc caption cùng query image trong datastore candidate list.

## 9. INFERENCE — query-image exclusion logic

Source `retrieve_caps.py` lines 94–107 có `filter_nns`, loại candidate có cùng `image_id` và giữ 7 neighbor cho offline mapping. Tuy nhiên R1 runner tại `scripts/run_smallcap_smoke.py:229-254` lấy raw FAISS top-4 và không gọi `filter_nns`. Trong audit này, việc không overlap source ID làm R1 query-image leakage **được loại trừ theo split**, nhưng hành vi filtering của runner vẫn được ghi nhận để không nhầm với official offline mapping.

## 10. VERIFIED FACT — embedding equivalence

Chọn deterministic 32 caption với seed 2026, encode bằng CLIP `RN50x64`, rồi so với vector public trong FAISS sau L2 normalization:

- mean cosine: `0.9999978542`;
- min cosine: `0.9999919534`;
- max cosine: `0.9999994039`.

Embedding equivalence: **VERIFIED**.

## 11. R1 retrieval consistency

ZIP R1 đã cung cấp `retrievals.json`. Audit xác nhận đúng 4 query IDs theo thứ tự cố định, đủ 4 rank mỗi query, 16/16 captions map được về source image IDs, và unresolved caption count bằng `0`. Trạng thái retrieval consistency: **VERIFIED**.

## 12. R2 gate và hành động tiếp theo

**R2_BLOCKED** vì:

1. literal source filter (`padding=True`, `len(encoding)`) không tái tạo public artifact; public artifact khớp unpadded/attention-mask semantics.

Artifact machine-readable nằm tại [r2_gate.json](../../outputs/smallcap/r1_5_datastore_audit/20261001T081500Z_final/r2_gate.json), [datastore_provenance.json](../../outputs/smallcap/r1_5_datastore_audit/20261001T081500Z_final/datastore_provenance.json), và [embedding_equivalence.json](../../outputs/smallcap/r1_5_datastore_audit/20261001T081500Z_final/embedding_equivalence.json). Cần quyết định/ghi nhận rõ filter semantics trước khi chuyển sang R2.

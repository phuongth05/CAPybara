# CAPybara research rules

This repository supports reproducible retrieval-augmented image-captioning research.

- Reproduce and verify published baselines before proposing or implementing `Ours`.
- Treat third-party baseline source as read-only. Use adapters, wrappers, and launch scripts instead of patching upstream code.
- Do not change dataset splits, retrieval datastore contents, or evaluation protocols without explicit approval. Record any ambiguity.
- Prevent retrieval leakage: record datastore source/split and whether query images or captions are excluded.
- Every experiment needs an ID, config, metadata, Git commit, seed, preserved raw predictions, normalized predictions, and traceable metrics.
- Validate configuration, data paths, and a small smoke test before GPU execution.
- Document discrepancies between official and unified evaluation; never silently change a scientific protocol.
- Keep baseline integrations separate from future `Ours` code.
- Do not commit datasets, checkpoints, features, or other large artifacts.
- Mark unavailable facts `UNVERIFIED` rather than guessing.

The current milestone is SmallCap R1.5 datastore provenance and leakage clearance. Do not begin R2, training, full evaluation, robustness experiments, or other baselines until the R1.5 gate is explicitly resolved.

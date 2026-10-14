`policy.json` is the tested development candidate: a 50-input, 64-unit residual
MLP exported as finite JSON weights for standard-library inference. It is loaded
and validated before initialization. `evaluation.json` contains the independent
seed-53M comparison against retained `policy-v2.json`; `neural-kaggle-evaluation.json`
records the earlier seed-52M final test. The v2 and v1 checkpoints and their reports
remain available for rollback. The selected checkpoint is staged for the October 14 nightly test round.
See `../provenance.md` for training sources and public-input restrictions.

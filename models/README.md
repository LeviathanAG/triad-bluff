`policy.json` is the conservative final policy: 50 own/public inputs, 128 ReLU
units (64 frozen incumbent units plus 64 fitted residual units), and finite JSON
weights for standard-library CPU inference. Its SHA256 is
`2aaeaaf13084a254ed443d4ffcc4288ab32ffe2b284d5bd9023c4a78594b0cf7`.
The runtime validates it before initialization. `policy-neural64.json` preserves
the exact previous NN64 policy; `policy-v2.json` and `policy-v1.json` preserve
older rollbacks. Historical `evaluation.json` and `neural-kaggle-evaluation.json`
report the NN64 seed53M/52M tests, rather than this conservative policy's results.
The conservative95M/99M tests and complete four-policy90M evidence are retained
in `../training/`; `conservative_candidate` identifies the active policy in the
90M raw rows. See `../provenance.md` for final selection, real opponent-style
tradeoffs, training sources, own/public feature restrictions and resource checks.

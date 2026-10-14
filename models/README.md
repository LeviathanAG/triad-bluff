`policy.json` is the retained 30-feature learned linear Bluff checkpoint. It is
loaded once when the agent starts, before initialization responds. `evaluation.json`
contains its independent paired comparison against `policy-v1.json`; the older
checkpoint and its evaluation reports remain available for rollback. See
`../provenance.md` for training sources, evidence, and inference restrictions.

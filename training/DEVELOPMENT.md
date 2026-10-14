# Development experiments

These sources and notebooks are development artifacts. The deployed checkpoint
remains the tested 30-feature v2 model in `../models/policy.json`. No experimental
model is promoted because supervised loss or model-selection score improved.

Frozen private Kaggle notebooks preserve their exact embedded source:

- `kaggle-bluff-refinement/`: v3 public/own pile deductions, 40-generation CPU
  search; train/selection/final seeds 24M/25M/26M.
- `kaggle-bluff-fast-adaptation/`: v4, 42 features with call-risk interactions,
  80-generation CPU search, read-only observation shortcut and parallel validation.
  Seeds 31M/32M/33M. The first four final seed blocks were observed in a small
  training-code smoke check; promotion requires a separate entirely fresh test.
- `kaggle-bluff-counterfactual/`: canonical Kaggle handle
  `xdantalianx/triad-bluff-counterfactual-value-data`; 3,000 source games,
  two uniform reservoir samples per decision phase, four paired rollout trials
  per action, up to ten feature-distinct actions. Features are 50 own/public v5
  values. Whole source games must stay together in supervised train/validation
  splits; seeds and return labels are never policy inputs.

The current trainer additionally supports v5, 50 features with pooled public
caller outcomes and Beta summary uncertainty. These summaries are learned model
inputs, not calibrated certainty guarantees. Later sealed votes remain unknown.
Local v5 training uses disjoint seeds 41M/42M/43M; any selected candidate must also
pass a fresh paired actual-game comparison against retained v2.

`bluff_training_observation.py` is training-only. It calls the SDK's unchanged
filtered observation function while temporarily detaching histories, restores
the state before any action or engine step, and shares only public history and
the requested player's private history with verified read-only actors. Across
10,344 decisions, observations and actor decisions matched and histories were
unmodified. A separate 32-game comparison gave identical rewards and 4.78x
faster sampled game evaluation. This shortcut is absent from the inference image.

Counterfactual rollouts copy the dealt simulator state and actor memories only
for training rewards. Initial hidden hands stay fixed within a sampled state;
future stochastic actor streams share trial seeds across candidate actions.
Bluff has no future draw deck. The exported matrix contains filtered features,
returns, trial returns, state offsets, and source-game IDs for splitting only.
Phase/hand-size coverage histograms check that endgame examples are represented.

The measured regressions in `bluff-regression-diagnostics.json` motivate these
experiments: roughly 73% of v2's bluffs were caught against the two frequent-call
stress regimes, while its average predicted any-call risk was 53–55%. Dataset
and model-selection opponents are development proxies, not competition entrants.

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

V5 completed an independent test on fresh seeds 44M (2,048 games per regime).
The original fixed league showed -0.00098 points, 95% paired interval
[-0.02979, 0.02686]. Both frequent-call stress weaknesses also lacked a supported
improvement. V5 is retained here as an unsuccessful experiment, not deployed.

The counterfactual dataset completed with 12,000 states, 67,757 actions and four
paired trials per action from 3,000 source games. Both phases have 6,000 states;
own-hand coverage is 2–46 cards. Integrity, source-game splits, finite features,
return bounds and paired-trial means were validated. The dataset itself is held
privately in Kaggle and local training outputs; only its verification is committed.

`kaggle-bluff-residual/` is the exact version-1 source of private T4 job
`xdantalianx/triad-bluff-residual`. A 64-unit residual MLP compares two training
objectives: gap-weighted sigmoid ranking and raw paired-point-difference MSE.
MSE retains zero-gap pairs; paired-return standard errors control its weights.
Each loss tests three regularizers and three residual strengths, yielding 18
neural candidates. Training uses whole-source-game 80/20 splits and train-only
normalization. Actual-game selection seeds are 51M, final seeds 52M, while the
initial three-epoch pipeline smoke uses 61M/62M. Retained v2 is included in
selection and wins ties. Any selected model still needs a separate fresh paired
comparison and complete runtime checks before parent review and promotion.

`bluff_mlp_policy.py` exports finite standard-library inference weights; PyTorch
is used only for training. Its training-only NumPy scorer recomputes close score
ties with the exact scalar implementation. An 810-decision, 12-game zero-residual
check preserved the previous feature policy's actions and matched both scorers.
No neural model is currently deployed.

The residual GPU job completed. Its selection chose ranking loss, regularizer
0.01 and residual strength 0.5; point-difference regression did not win the
actual-game selection. Fresh Kaggle final seeds 52M gave +0.09473 league points
with a paired 95% interval [0.04004, 0.14941], 1,024 games. An additional fixed
independent comparison on seeds 53M is pending. The selected neural checkpoint
is backed up under `bluff-residual-result/` as experimental training material.
A 1,463-decision/12-game replay check matched fresh stdlib scoring to training
NumPy scoring across 4/8 players and 1/2 decks; maximum replay was 15.81ms.
Main and the packaged runtime remain the retained v2 model.

The next neural-incumbent iteration is private and experimental. CPU collection
`xdantalianx/triad-bluff-neural-incumbent-data` v1 is running on fresh source games
64M, using the selected neural policy both in source play and later own rollout
actions. The separate fit-only GPU notebook and actual-game-only CPU selector
are prepared but not pushed. GPU training exports 18 candidates across both
losses; CPU selection uses 65M, fixed-candidate final evaluation 66M and later
independent comparison 67M. Long gameplay therefore consumes no GPU reservation.
The frozen neural baseline is exactly converted to the new normalizer, then a
new 64-unit residual yields a 128-unit stdlib scorer. Eight-game collection and
CPU-selection smokes passed; maximum exact baseline conversion error was
3.33e-16. No new trained checkpoint exists yet.

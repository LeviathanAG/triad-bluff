# Provenance

Team: TungMasterSoloClear (TECHTEAM11600).

The protocol adapter, schemas, Docker template, and development evaluator originate
from Sir-Nike/bluffSDK commit 96da8042c285af4e4f43655f2ad99592bd49b8e6. Existing vendor
attribution is retained. The authoritative game evaluator is a training and testing
dependency; the runtime contains no evaluator or hidden-hand access.

The action scorer in `src/learned_policy.py` and its cross-entropy optimization
procedure were developed for this entry. `models/policy.json` contains 18 learned
floating-point weights. It is a non-language linear model, with no pretrained
external model, LLM, or inference service. Its features use the current player's
cards and filtered public claim/challenge history. Unchallenged claims are never
counted as proven truthful observations.

Training ran in the private Kaggle notebook xdantalianx/triad-bluff-training,
version 1: seed 20261014, 25 generations, population 32, 64 games per candidate,
and independent opponent mixtures. Base SDK scoring and the 1,000-turn cap were
used. Rollouts stop after our confirmed finish because its placement cannot
change. Training uses the simulator only outside the runtime.

`models/evaluation.json` records a separate 13,000,000-seed test range with all
four seats rotated, 1,024 games per opponent regime, and paired bootstrap intervals
over seed blocks. Against the tested diverse league, mean gain over the original
SDK starter was 0.1885 points/game (95% interval 0.1240–0.2607). This measures those
local opponents, not official entrants. The stochastic-bluffer regime showed a
small uncertain regression; no universal winning claim is made.

Runtime reconstruction from complete history ensures identical decisions after
restart. No seeds, opponent private cards, training libraries, account credentials,
or remote downloads are used during inference.

An additional fresh test against the SDK's supplied adaptive competitive strategy
used 1,024 seat-rotated games. The trained policy scored 4.6016 versus 2.75
for the SDK starter; paired gain interval [1.7705078125, 1.9287353515624996].
Runtime and container checks: 32 Python tests passed, one optional organizer test
skipped, smoke/contract checks passed, three local and three restricted Docker
games completed without faults or fallbacks. Stateless reconstruction matched
the training policy across 1,832 decisions in 16 multi-seat/deck games.

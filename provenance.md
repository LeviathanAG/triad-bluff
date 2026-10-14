# Provenance

Team: TungMasterSoloClear (TECHTEAM11600).

The protocol adapter, schemas, Docker template, and development evaluator originate
from Sir-Nike/bluffSDK commit 96da8042c285af4e4f43655f2ad99592bd49b8e6. Existing vendor
attribution is retained. The authoritative evaluator is used for training and
 testing; inference contains no evaluator or opponent private-hand access.

The entry's non-language linear scorer has 30 learned floating-point weights in
`models/policy.json`. `src/learned_policy.py` extends the original 18-feature scorer
in `src/learned_policy_v1.py`. Its features use the player's own cards, own private
placements, public claims, publicly revealed challenged batches, public hand sizes,
and observed challenge decisions. Unchallenged claims are never treated as verified
truth. Rank counts inferred from revealed pickups are conservative lower bounds:
every unknown placement decrements every rank bound. Later sealed challenge votes
are never inferred from a selected earlier challenger.

Training ran in private Kaggle notebook `xdantalianx/triad-bluff-league`, version 1,
on CPU: seed 20261016, 30 generations, population 32, eight four-seat blocks per
candidate, 64 held-out validation blocks. A 200-turn training cap saved compute;
selection and final evaluation used the real 1,000-turn cap. The opponent league
included SDK baselines, manual policies, stochastic bluffers, and archived policies.
`training/train_bluff_league_original.py` records the exact trainer embedded in
that notebook. The current `training/train_bluff_league.py` adds the supplied
adaptive rival and configurable disjoint seed ranges for further experiments.
Those later trainer changes did not produce the currently retained checkpoint.

`models/evaluation.json` records independent tests starting at seed 20,000,000:
1,024 games per regime with all four seats rotated. Both policies faced the same
fixed opponent pool, which included the previous learned checkpoint and the SDK's
supplied adaptive strategy. The new model scored 4.0020 versus 3.7314 points/game
for the previous learned model in the diverse league. Paired gain was 0.2705
(95% bootstrap interval 0.1797–0.3652, resampling four-seat seed blocks). Gains were
positive against the tested adaptive, truthful, starter and stochastic-bluffer
regimes, tied at five points against the aggressive caller, and positive against
three copies of the previous learned model. These results measure local opponents,
not unknown competition entrants. The previous checkpoint and its reports are
retained as `models/policy-v1.json`, `models/evaluation-v1.json`, and
`models/competitive-evaluation-v1.json`.

Inference reconstructs the complete filtered history each decision, giving the
same decisions in persistent and restarted processes. No seeds, hidden hands,
training libraries, account credentials, or remote downloads enter inference.
The optional training-only v3 pile deduction module is not deployed in this
checkpoint. Runtime replay matched the training policy across 1,726 decisions
in 16 games with four/eight players and one/two decks; all actions were legal.
The SDK suite passed 32 tests with one optional organizer integration skipped;
smoke and protocol contract checks passed. Three local and three restricted
Docker games completed without faults or fallbacks.

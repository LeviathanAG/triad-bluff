# Provenance

Team: TungMasterSoloClear (TECHTEAM11600).

This development branch contains a tested neural candidate awaiting parent
confidence review. The default main branch remains the retained linear v2.

The protocol adapter, schemas, Docker template and evaluator originate from
Sir-Nike/bluffSDK commit 96da8042c285af4e4f43655f2ad99592bd49b8e6. Attribution
is retained. Organizer dependency-accounting update 2e13997 is merged; it changes
no game mechanics. The entry uses the CPU template and claims no base exemption.

The scorer is a non-language residual network: 50 inputs, one 64-unit ReLU layer,
linear baseline plus scalar residual. `models/policy.json` stores finite floating
weights, train-only normalization and training metadata. Runtime uses Python's
standard library, with no NumPy/PyTorch, network calls or remote model downloads.

Features derive exclusively from the player's own hand, own private placements
and public history: claims, revealed challenged batches, hand sizes, observed
selected/earlier challenge outcomes and conservative rank bounds. Unknown
placements decrement every relevant rank bound; unchallenged claims are not
verified truth. Later sealed votes remain unknown. Beta summaries are modeling
features, not calibrated certainty claims. The additional pile deduction challenges
only when disjoint known own/public cards leave insufficient rank copies.

Private CPU job `xdantalianx/triad-bluff-counterfactual-value-data` version 1
collected 3,000 source games (seed family 50M): 12,000 phase-stratified whole-game
reservoir states, 67,757 feature-distinct candidate actions, four paired trials
per action. Own hand sizes span 2–46; both phases have 6,000 states. Rollouts copy
the simulator and actor memories, retain dealt hands, and share future stochastic
actor streams across candidate actions. Hidden hands influence training rewards
only, never policy inputs. Bluff has no future draw deck. Source-game IDs enforce
supervised splits and do not enter inference features.

Private T4 job `xdantalianx/triad-bluff-residual` version 1 fit 18 candidates using
2,400 training/600 validation source games. It compared gap-weighted sigmoid
ranking and paired-point-difference MSE (including zero-gap pairs), three residual
regularizers and three strengths. Actual games selected ranking loss, regularizer
0.01, residual strength 0.5 and supervised epoch 97. Normalization uses training
rows only. The three-epoch pipeline smoke used seeds 61M/62M, actual selection
used 51M, and the fixed selected candidate's Kaggle final test used 52M. Exact
notebook sources, provenance and JSON integrity checks are retained in `training/`.

The Kaggle final test had 1,024 games per regime: league gain over v2 was +0.09473
points/game, paired 95% bootstrap interval [0.04004, 0.14941]. The independent
fixed-candidate comparison in `models/evaluation.json` uses entirely fresh 53M
seeds, 2,048 games per regime and all four seats. Both policies face identical
fixed opponents, including retained v2 and the SDK adaptive strategy. Gains were:

- diverse league +0.13477, interval [0.09717, 0.17188];
- adaptive/starter +0.05273, interval [0.03564, 0.07080];
- truthful +0.04688, interval [0.02881, 0.06592];
- stochastic bluffer +0.09521, interval [0.05420, 0.13574];
- three retained-v2 rivals +0.24072, interval [0.19873, 0.28516];
- aggressive caller tied at five points.

The two frequent-call stress regimes previously problematic for v2 gave +0.02344
[0.00879, 0.03809] and +0.01709 [-0.00293, 0.03809]. Intervals resample four-seat
seed blocks. These are development proxies, not other teams; no competition win
or universal optimality is claimed. The original v2 and v1 checkpoints and their
reports remain in `models/` for rollback. Historical v2 provenance is preserved
in `training/PROVENANCE-V2.md`. V5 linear refinement failed its independent gate
and was not adopted.

Inference reconstructs complete filtered history on every decision. Across 1,463
decisions in 12 games with four/eight players and one/two decks, fresh stdlib
scoring exactly matched persistent training NumPy decisions; all actions were
legal and inferred own/public bounds sound. Maximum full replay was 15.81ms.
No seeds, hidden hands or credentials enter inference. The training-only read-only
observation shortcut and NumPy batching are absent from deployed source.

The SDK suite passed 33 tests with one optional organizer integration skipped;
smoke and contract checks passed. Three local protocol games and three restricted
Docker games completed with zero faults/fallbacks. The Docker runs used one CPU,
1 GiB RAM, no network, a read-only filesystem and USER 10000:10000. Maximum observed
Docker decision latency was 327.78ms, below the five-second action limit. The
complete uncompressed image is 129,925,120 bytes, well below 1.5 GiB; source/assets
are about 1.1 MB. No registry publication or official submission has occurred.

A supplemental entirely fresh 58M grid varied bluff probabilities 0.1/0.35/0.65/0.95
and call probabilities 0/0.1/0.35/0.8, with 512 games per regime. No statistically
supported regression was observed versus v2. The largest gain was +0.51367
[0.41992, 0.61133] at bluff 0.95/call 0; the sole negative mean was -0.00391
[-0.08984, 0.08203]. This grid was not used to select the checkpoint; its complete
report is `training/bluff-residual-adversarial-grid.json`.

# Bluff final release provenance

Team **TungMasterSoloClear**, ID **TECHTEAM11600**.

This conservative128 policy is prepared for the user-authorized October15 noon
IST final source push. The release is a direct child of held main
9d48b9ddd37ceb42b7892b39ef4fa6c84eb0452f, using the exact runtime and model
reviewed at bfc7f75ba0eedd90d0b0be9ff927df680bcbe0ae. The documentary manifest
now names the intended main branch. No earlier final promotion is implied.

models/policy.json SHA256:
2aaeaaf13084a254ed443d4ffcc4288ab32ffe2b284d5bd9023c4a78594b0cf7.
models/policy-neural64.json is the exact historical incumbent rollback, SHA256:
ba6ed2a7aa7f03637c02225fb96c71e4798fb5073f3bf5ff2a793038719a1b08.
The independently reviewed balanced alternative remains separately preserved at
8d94e3b383d75dc1e36387f22dba6eb00e0a93d6; its checkpoint is not the active model.

The network takes50 own/public inputs, with64 frozen incumbent units and64
learned residual units. Standard-library CPU inference needs no training library,
network access or runtime downloads. Inputs are own cards, own private placements,
public history, hand sizes and conservative public card-count deductions; hidden
hands, unobserved votes and opponent-style IDs never enter inference. Hidden
simulator state supplies training rewards only. The ranking residual used3000
source games, four paired return trials per action, whole-game phase reservoirs,
whole-source-game train/validation splits, and neural64 source/future own actions.
Its conservative strength.15 was selected from a fixed.05/.10/.15/.25 grid at93M
using five fixed opponent compositions and a mean-regression guard. Fresh94M
final and95M independent games were separate from that selection.

95M2048games/regime versus NN64: league+.15381 CI[.11084,.19727], NN64
selfplay+.16504[.12549,.20363],v2 selfplay+.20020[.16698,.23340]. Competitive
and starter means0; truthful+.00293 and bluffer-.00635 had intervals crossing0.
The weak bluff.1/call.35 style had-.01611[-.03174,-.00049]. A fixed99M
replication with8192games tied exactly at4.93701171875 points/game, gain0
[-.007446,.007568]. Both reports remain preserved; neither establishes universal
non-regression or victory against unknown final opponents.

The independent90M four-way gate compared the fixed balanced candidate, held
NN64, retainedv2 and this conservative policy. It used512 four-seat seed blocks,
2048games per policy per regime,12 regimes and98304 total games. It retained the
frozen88M opponent pool. Raw block_points/block_wins, exact ordered seeds,
checkpoint hashes and game configuration are preserved in training/balanced90-*;
the raw field conservative_candidate identifies this final model. Paired95%
bootstrap intervals resample four-seat seed blocks, not independent seat games.

Using exactly the nine equally weighted regimes fixed before balanced fitting,
conservative versus NN64 averaged+.121311 points/game CI[.106391,.136393];
balanced versus NN64+.120822[.105956,.135581]. Balanced minus conservative
was-.000488[-.014865,.013836], so this proxy mixture does not distinguish them.
Conservative had smaller adverse observed means versus NN64 in these regimes.
The final choice favors that broader stability. On90M conservative minus balanced:

- mixed league+.008789[-.034668,.053223], uncertain;
- weak bluff.65/call.35+.038574[.015625,.062500];
- conservative selfplay+.192383[.136230,.249512];
- NN64 selfplay-.268066[-.312988,-.223633];
- v2 selfplay-.285645[-.322266,-.250977].

These are real tradeoffs. The unknown official opponent mixture is not inferred.
Public organizer standings belong to the previous NN64 snapshot: round1 rank1/5,
800games3572points671wins; round2 rank1/6,640games2638points414wins. Different
lineups and schedules prevent causal comparison of round averages. Only public
aggregates were used; no competitor private source, traces or organizer messaging.

Training-only hidden-world augmentation was rejected after violating truthful
source-policy support. No such labels entered this checkpoint. State-balanced
and tree experiments remain separate. Runtime/card rules are pinned to SDK
96da8042c285af4e4f43655f2ad99592bd49b8e6; dependency update2e13997 changes no
game mechanics. Default4-player/1-deck rewards5/3/2/1 and cap ties sharing remaining
positions were used. SDK attribution is retained.

The unchanged exact runtime/model already passed33 SDK tests with1optional skip,
1481 training NumPy versus stdlib replay decisions over12games (4/8players,
1/2decks),167 persistent/restarted decisions across4seats, and3 restricted Docker
games with zero faults/fallbacks, maximum316.45ms. Final image identity, new
restricted smoke and source/image cap evidence are recorded separately for this
release. Required limits remain CPU1,RAM1GiB,scratch1GiB,GPUfalse,VRAM0,
startup60s/action5s,linux/amd64,USER10000:10000,offline read-only runtime.
No dependency-base exemption is claimed. Registry destination remains unknown;
local OCI fingerprints do not imply a published pullable registry image.

Final release revalidation:33 tests passed/1optional skip; smoke and contract
checks passed. All36 src/model files and the manifest match the built container
byte-for-byte. Three fresh restricted final-image games (122M correctness-only
seeds) had zero faults/fallbacks, maximum action388.72ms. The executable runtime and JSON model weights
remain identical to reviewed conservative128; the models README now correctly
describes128 units and distinguishes historicalNN64 reports. Training/restart evidence is
retained. See training/bluff-final-tests.json and bluff-final-byte-identity.json.

Organizer manifest update: the required top-level teamName is exactly
TungMasterSoloClear, matching the website team and existing team_name identity.
The rebuilt image contains the exact source manifest bytes; one additional
restricted game completed with zero faults/fallbacks and maximum359.25ms.
No executable code or model weights changed. Private organizer replay traces
were kept outside the submission repository and were not used for training.

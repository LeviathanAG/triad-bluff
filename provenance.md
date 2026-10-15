# Provenance

Team: TungMasterSoloClear (TECHTEAM11600).

This private review candidate uses a 128-unit residual scorer: the frozen 64-unit
NN64 network plus 64 fitted units, 50 own/public inputs, standard-library CPU
inference. The SDK runtime, filtered information handling, Dockerfile and limits
are unchanged from main 9d48b9ddd37ceb42b7892b39ef4fa6c84eb0452f. Game source
is Sir-Nike/bluffSDK 96da8042c285af4e4f43655f2ad99592bd49b8e6; the merged
2e13997 update affects dependency accounting only. No GPU exemption is claimed.
Historical NN64 provenance is in training/PROVENANCE-NEURAL64.md and its exact
rollback checkpoint is models/policy-neural64.json.

Private free CPU balanced-incumbent-data v1 collected 3,500 whole source games
at 87M, seven opponent styles, 14,000 phase-stratified reservoir states, 77,879
actions and six paired future trials. Hidden simulator state determines rewards
only. Exported features use own cards, own private placements and public history.
NPZ SHA256: b952b79928c065ec07a3eb2537a501b2c0d042c42262dbe188fbb56c025879cd.
The split is 2,800/700 whole games. A free T4 fit compared 18 ranking/value,
regularization and strength candidates. Fixed 88M actual-game selection used nine
regimes, 128 four-seat seed blocks, an eligible NN64 baseline and a predeclared
maximum .025 observed-mean regression guard. It selected value loss, penalty .01,
residual strength .25, epoch105. All zero-gap point-regression pairs were retained.
Checkpoint SHA256: 8253987d638222d8f1314baa5e7045f1b5ab58c5b224506be011078e21d330ba.

Fresh 89M final games supported mixed-league and learned-opponent improvements.
A fixed independent 90M gate then compared candidate, held NN64, retained v2 and
conservative128 against the same frozen opponent pool: 512 four-seat seed blocks,
2,048 games per policy per regime, 12 regimes, 98,304 total games. Full raw block
arrays and exact sources are retained. Paired bootstrap95% intervals resample
seed blocks, not individual seat games. Versus NN64 the candidate gained:

- mixed league +.109863 [.064929,.153320] points/game;
- NN64 selfplay +.473145 [.431140,.517090];
- v2 selfplay +.481445 [.438464,.524915];
- learned mixture +.310547 [.257813,.364746];
- conservative128 selfplay +.202148 [.142078,.261719].

The stochastic-bluffer comparison was -.014160 [-.057129,.030273]; the
bluff.65/call.35 comparison was -.009766 [-.031738,.011731]. No universal
improvement is claimed. Against conservative128, mixed league was -.008789
[-.053223,.034668], NN64 selfplay +.268066 [.223633,.312988] and v2 selfplay
+.285645 [.250977,.322266], but bluff.65/call.35 was -.038574 [-.062500,-.015625]
and conservative128 selfplay -.192383 [-.249512,-.136230]. These concrete
tradeoffs remain part of final selection. Public standings use tonight's NN64;
these proxy tests do not identify other teams' policies or guarantee a win.

No physically consistent hidden-world augmentation labels were used: a local
prototype violated truthful source-policy support and was explicitly rejected.
State-balanced follow-up and rejected tree objectives remain separate experiments.
No competitor private data or code was accessed. Credentials never enter source,
features or inference. No network access or training-library dependency is needed
at runtime. CPU1/RAM1GiB/scratch1GiB, GPUfalse/VRAM0, UID10000:10000, startup60s,
action5s remain the admission limits. Registry destination remains unknown;
local image identity does not imply a published pullable image.

Admission verification: 33 SDK tests passed, one optional organizer integration
skipped; smoke and protocol-contract checks passed. Across 1,474 decisions in
12 games (4/8 players, 1/2 decks), fresh standard-library full-history inference
matched training NumPy actions exactly, with sound own/public card bounds and
maximum18.14ms. A separate four-seat transport check compared170 decisions
between persistent and freshly initialized processes exactly, both phases,
null checkpoints, maximum initialization30.92ms and action17.90ms.

Three fresh restricted Docker games (one CPU, RAM1GiB, read-only filesystem,
no network, tmpfs scratch1GiB, no capabilities, no privilege gain) completed
with zero faults/fallbacks. Maximum observed action399.47ms. Container model,
rollback and manifest identities, Linux/amd64, UID10000:10000, exec entrypoint
and all seven resource labels were checked. This is a private review snapshot;
main remains held and no published image registry destination is available.

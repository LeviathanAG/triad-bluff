# Bluff private conservative candidate

Team **TungMasterSoloClear**, ID **TECHTEAM11600**.

This private candidate is rooted in tonight's main `9d48b9ddd37ceb42b7892b39ef4fa6c84eb0452f`. It is not promoted
or a final submission. `models/policy.json` is the 128-unit CPU residual network
SHA256 `2aaeaaf13084a254ed443d4ffcc4288ab32ffe2b284d5bd9023c4a78594b0cf7`; `models/policy-neural64.json` preserves tonight's neural64 scorer.

The network uses50 own/public features.64 frozen incumbent units plus64 learned
units implement standard-library inference. The added ranking residual was fit
on3000 source games using four paired simulator return trials, whole-game
phase reservoirs, whole-source-game train/validation splitting, and no hidden
hands or opponent-style IDs as neural inputs. Hidden state supplies training
rewards only. Source and future own rollout actions used the neural64 incumbent.

The conservative strength.15 was selected from a fixed.05/.10/.15/.25 grid on
fresh93M seeds using five fixed rival compositions and a mean regression guard.
Fresh94M final games and95M independent games were separate from selection.
95M2048games/regime: mixed league+.15381 points/game CI[.11084,.19727], neural64
selfplay+.16504 CI[.12549,.20363], linearv2 selfplay+.20020 CI[.16698,.23340].
Competitive/starter mean0, truthful+.00293 and bluffer-.00635 had intervals
crossing0. One weak frequent-calling style had-.01611 CI[-.03174,-.00049].
The exact same fixed checkpoint was replicated on fresh99M8192games in that
style: both policies4.93701171875 points/game, gain0 CI[-.007446,.007568].
Both reports are preserved. The replication does not establish universal
non-regression or victory against unknown final opponents.

Tonight's organizer public test result belongs to the retained incumbent:
rank1/5,800games3572points4.465/game and671wins83.875%. Individual rival
histories/private submissions were not accessed. SDK default rewards5/3/2/1
were used for four-player comparisons; cap ties share remaining rewards.

Runtime evidence for this exact model and unchanged runtime/Dockerfile:
33pytest passed,1optional skip;1481 training-versus-stdlib replay decisions on
12games with4/8players and1/2decks;167 persistent-versus-restarted decisions
across four seats;3 restricted CPU1/RAM1GiB/offline/read-only Docker games with
zero faults/fallbacks and max301.17ms. Labels and non-root entrypoint remain
CPU1,RAM1GiB,scratch1GiB,GPUfalse,VRAM0,linux/amd64,USER10000:10000.
The candidate source manifest and model hash match. New immutable image/source
size evidence is recorded separately after building this exact commit.
No shared-base exemption is claimed. Registry destination remains unknown;
local OCI fingerprints are not published pullable registry references.

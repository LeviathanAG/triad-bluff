"""Create one private candidate commit directly atop the frozen nightly source."""
import hashlib,io,json,os,subprocess,tarfile,tempfile
from pathlib import Path
root=Path(__file__).resolve().parent.parent;repo=root/'bluffSDK'
base='9d48b9ddd37ceb42b7892b39ef4fa6c84eb0452f';branch='codex/conservative-candidate'
def git(*args,env=None,input=None):return subprocess.check_output(['git','-C',str(repo),*args],env=env,input=input)
model=(root/'training/bluff-conservative-ranking-output/bluff-policy-mlp.json').read_bytes();sha=hashlib.sha256(model).hexdigest()
manifest=json.loads(git('show',base+':submission.json'));manifest['model']['sha256']=sha;manifest['model']['hidden_units']=128;manifest['source_branch']=branch
readme=git('show',base+':README.md').decode().replace('independently tested 64-unit neural','independently tested conservative 128-unit neural')
readme=readme.replace('See `provenance.md` for training, validation and resource evidence.','This private candidate has not been promoted to main. The winning neural64 rollback is `models/policy-neural64.json`.\nSee `provenance.md` for training, validation and resource evidence.')
provenance=f'''# Bluff private conservative candidate

Team **TungMasterSoloClear**, ID **TECHTEAM11600**.

This private candidate is rooted in tonight's main `{base}`. It is not promoted
or a final submission. `models/policy.json` is the 128-unit CPU residual network
SHA256 `{sha}`; `models/policy-neural64.json` preserves tonight's neural64 scorer.

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
'''
files={'models/policy.json':model,'models/policy-neural64.json':git('show',base+':models/policy.json'),'submission.json':(json.dumps(manifest,indent=2)+'\n').encode(),'README.md':readme.encode(),'provenance.md':provenance.encode()}
for name in ['bluff-conservative-independent95.json','bluff-conservative-independent95-verification.json','bluff-conservative-weakspot-replication.json','bluff-conservative-protocol-restart.json','bluff-conservative-replay-verification.json','bluff-conservative-docker-verification.json','bluff-organizer-test-leaderboard.json']:
    files['training/'+name]=(root/'training'/name).read_bytes()
with tempfile.TemporaryDirectory(prefix='bluff-candidate-index-') as temp:
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/'index'))
    git('read-tree',base,env=env)
    for path,data in files.items():
        blob=git('hash-object','-w','--stdin',input=data).decode().strip()
        git('update-index','--add','--cacheinfo',f'100644,{blob},{path}',env=env)
    tree=git('write-tree',env=env).decode().strip()
    commit=git('commit-tree',tree,'-p',base,input=b'Stage conservative Bluff128 candidate with independent evidence\n').decode().strip()
git('update-ref','refs/heads/'+branch,commit)
dest=root/'training/bluff-conservative-source';dest.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(git('archive',commit))) as archive:archive.extractall(dest,filter='data')
report={'candidate_commit':commit,'branch':branch,'parent':base,'model_sha256':sha,'hidden_units':128,'source_directory':str(dest),'main_unchanged':git('rev-parse','main').decode().strip(),'promoted':False}
(root/'training/bluff-conservative-staged-source.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

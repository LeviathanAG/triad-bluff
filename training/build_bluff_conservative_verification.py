"""Freeze independent CPU gate for the93M-selected conservative checkpoint."""
import json,base64,zlib
from pathlib import Path
root=Path(__file__).resolve().parent
names=['bluff_policy.py','bluff_policy_v2.py','bluff_policy_v3.py','bluff_policy_v4.py','bluff_policy_v5.py','bluff_mlp_policy.py','bluff_training_observation.py','train_bluff_league.py','collect_bluff_counterfactual.py','evaluate_bluff_checkpoint.py','select_bluff_residual_cpu.py','evaluate_bluff_threeway.py']
source='import os, subprocess,base64,zlib\nfrom pathlib import Path\nos.environ["OPENBLAS_NUM_THREADS"]="1"\nos.environ["BLUFF_FAST_OBSERVATIONS"]="1"\nwork=Path("/kaggle/working/training");work.mkdir(exist_ok=True)\n'
source+='sources='+repr({name:(root/name).read_text() for name in names})+'\nfor name,text in sources.items(): (work/name).write_text(text)\n'
source+='sdk=Path("/kaggle/working/bluffSDK")\nsubprocess.run(["git","clone","--quiet","https://github.com/Sir-Nike/bluffSDK",str(sdk)],check=True)\nsubprocess.run(["git","-C",str(sdk),"checkout","--quiet","96da8042c285af4e4f43655f2ad99592bd49b8e6"],check=True)\nos.environ["BLUFF_SDK"]=str(sdk)\n(sdk/"models").mkdir(exist_ok=True)\n'
source+=f'(sdk/"models/policy-v1.json").write_text({(root.parent/"bluffSDK/models/policy-v1.json").read_text()!r})\n'
paths={'candidate.json':root/'bluff-conservative-ranking-output/bluff-policy-mlp.json','incumbent.json':root.parent/'bluffSDK/models/policy.json','v2.json':root.parent/'bluffSDK/models/policy-v2.json','selection.json':root/'bluff-conservative-ranking-output/selection.json'}
for name,path in paths.items():
    encoded=base64.b64encode(zlib.compress(path.read_bytes())).decode()
    source+=f'(work/{name!r}).write_bytes(zlib.decompress(base64.b64decode({encoded!r})))\n'
source+='subprocess.run(["python",str(work/"evaluate_bluff_threeway.py"),"--candidate",str(work/"candidate.json"),"--incumbent",str(work/"incumbent.json"),"--v2",str(work/"v2.json"),"--blocks","512","--seed-start","95000000","--workers","4","--opponent-pool",str(work/"selection.json"),"--output","/kaggle/working/independent.json"],check=True,timeout=21600)\n'
dest=root/'kaggle-bluff-conservative-verify';dest.mkdir(exist_ok=True)
notebook={'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':source.splitlines(keepends=True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
(dest/'bluff-conservative-verify.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
metadata={'id':'xdantalianx/triad-bluff-conservative-ranking-verify','title':'Triad Bluff Conservative Ranking Verify','code_file':'bluff-conservative-verify.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':[]}
(dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(dest)

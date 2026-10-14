"""Freeze a private CPU data job with balanced whole-game rival compositions."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
names=['bluff_policy.py','bluff_policy_v2.py','bluff_policy_v3.py','bluff_policy_v4.py','bluff_policy_v5.py','bluff_mlp_policy.py','bluff_training_observation.py','bluff_training_snapshot.py','train_bluff_league.py','collect_bluff_counterfactual.py']
source='import os, subprocess\nfrom pathlib import Path\nos.environ["OPENBLAS_NUM_THREADS"]="1"\nos.environ["BLUFF_FAST_OBSERVATIONS"]="1"\nos.environ["BLUFF_FAST_SNAPSHOTS"]="1"\nwork=Path("/kaggle/working/training");work.mkdir(exist_ok=True)\n'
source+='sources='+repr({name:(root/name).read_text() for name in names})+'\nfor name,text in sources.items(): (work/name).write_text(text)\n'
for name,path in [('incumbent.json',root.parent/'bluffSDK/models/policy.json'),('previous.json',root.parent/'bluffSDK/models/policy-v2.json')]:
    source+=f'(work/{name!r}).write_text({path.read_text()!r})\n'
source+='sdk=Path("/kaggle/working/bluffSDK")\nsubprocess.run(["git","clone","--quiet","https://github.com/Sir-Nike/bluffSDK",str(sdk)],check=True)\nsubprocess.run(["git","-C",str(sdk),"checkout","--quiet","96da8042c285af4e4f43655f2ad99592bd49b8e6"],check=True)\nos.environ["BLUFF_SDK"]=str(sdk)\n'
source+='subprocess.run(["python",str(work/"collect_bluff_counterfactual.py"),"--incumbent",str(work/"incumbent.json"),"--previous",str(work/"previous.json"),"--games","3500","--samples-per-phase","2","--trials","6","--max-actions","10","--workers","4","--seed-start","87000000","--source-compositions","balanced","--output","/kaggle/working/output"],check=True,timeout=21600)\n'
dest=root/'kaggle-bluff-balanced-data';dest.mkdir(exist_ok=True)
notebook={'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':source.splitlines(keepends=True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
(dest/'bluff-balanced-data.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
metadata={'id':'xdantalianx/triad-bluff-balanced-incumbent-data','title':'Triad Bluff Balanced Incumbent Data','code_file':'bluff-balanced-data.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':[]}
(dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(dest)

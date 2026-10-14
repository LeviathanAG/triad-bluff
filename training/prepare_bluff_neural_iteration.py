"""Prepare fit-only GPU and actual-game-only CPU stages; never pushes jobs."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
FILES=['bluff_policy.py','bluff_policy_v2.py','bluff_policy_v3.py','bluff_policy_v4.py','bluff_policy_v5.py','bluff_mlp_policy.py','bluff_training_observation.py','train_bluff_league.py']


def notebook(directory,slug,title,name,sources,code,gpu,kernel_source):
    directory.mkdir(exist_ok=True)
    prefix='''import os, subprocess\nfrom pathlib import Path\nos.environ["OPENBLAS_NUM_THREADS"]="1"\nos.environ["BLUFF_FAST_OBSERVATIONS"]="1"\nwork=Path("/kaggle/working/training");work.mkdir(exist_ok=True)\n'''+f'sources={sources!r}\n'+'for name,source in sources.items():\n    (work/name).write_text(source)\n'+'''sdk=Path("/kaggle/working/bluffSDK")\nsubprocess.run(["git","clone","--quiet","https://github.com/Sir-Nike/bluffSDK",str(sdk)],check=True)\nsubprocess.run(["git","-C",str(sdk),"checkout","--quiet","96da8042c285af4e4f43655f2ad99592bd49b8e6"],check=True)\nos.environ["BLUFF_SDK"]=str(sdk)\n'''
    source=prefix+code;compile(source,name,'exec')
    data={'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':source.splitlines(keepends=True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':5}
    (directory/name).write_text(json.dumps(data,indent=2)+'\n')
    meta={'id':slug,'title':title,'code_file':name,'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':gpu,'enable_tpu':False,'enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':[kernel_source]}
    (directory/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


def main():
    sources={name:(ROOT/name).read_text() for name in FILES+['train_bluff_residual.py']}
    previous=(ROOT.parent/'bluffSDK/models/policy-v2.json').read_text()
    code=f'(work/"previous.json").write_text({previous!r})\n'+'''data=next(Path("/kaggle/input").rglob("counterfactual.npz"))\nfor epochs,destination in ((3,"/kaggle/temp/bluff-fit-smoke"),(120,"/kaggle/working/output")):\n    subprocess.run(["python",str(work/"train_bluff_residual.py"),"--data",str(data),"--previous",str(work/"previous.json"),"--width","64","--epochs",str(epochs),"--batch","4096","--workers","4","--device","cuda","--seed","20261021","--loss-types","ranking,value","--skip-game-evaluation","--output",destination],check=True,timeout=7200)\n'''
    notebook(ROOT/'kaggle-bluff-neural-fit','xdantalianx/triad-bluff-neural-incumbent-fit','Triad Bluff Neural Incumbent Fit','bluff-neural-fit.ipynb',sources,code,True,'xdantalianx/triad-bluff-neural-incumbent-data')
    sources={name:(ROOT/name).read_text() for name in FILES+['collect_bluff_counterfactual.py','evaluate_bluff_checkpoint.py','select_bluff_residual_cpu.py']}
    v1=(ROOT.parent/'bluffSDK/models/policy-v1.json').read_text()
    code=f'(work/"linear.json").write_text({previous!r})\n(work/"previous.json").write_text({v1!r})\n'+'''summary=next(Path("/kaggle/input").rglob("fit-summary.json"))\nsubprocess.run(["python",str(work/"select_bluff_residual_cpu.py"),"--candidates",str(summary.parent),"--linear-backup",str(work/"linear.json"),"--previous-v1",str(work/"previous.json"),"--validation-blocks","128","--test-blocks","256","--validation-seed-start","65000000","--test-seed-start","66000000","--workers","4","--output","/kaggle/working/output"],check=True,timeout=21600)\n'''
    notebook(ROOT/'kaggle-bluff-neural-select','xdantalianx/triad-bluff-neural-incumbent-select','Triad Bluff Neural Incumbent Select','bluff-neural-select.ipynb',sources,code,False,'xdantalianx/triad-bluff-neural-incumbent-fit')


if __name__=='__main__':main()

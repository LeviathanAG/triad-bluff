"""Exercise a named private CPU image under exact admission resource limits."""
import argparse,json,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'bluffSDK/local_evaluator/src'))
from triad_bluff.local_runner import run_commands
from triad_bluff import engine
parser=argparse.ArgumentParser();parser.add_argument('--image',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed-start',type=int,required=True)
args=parser.parse_args();reports=[]
for sample in range(3):
    commands={seat:['docker','run','--rm','-i','--name',f'bluff-private-{uuid.uuid4().hex}', '--cpus','1','--memory','1g','--memory-swap','1g','--pids-limit','128','--network','none','--read-only','--tmpfs','/tmp:rw,nosuid,nodev,size=1073741824','--cap-drop','ALL','--security-opt','no-new-privileges','--user','10000:10000',args.image] for seat in range(4)}
    report=run_commands(commands,seed=args.seed_start+sample,config=engine.GameConfig(),game_id=f'private-check-{sample}',max_restarts=0)
    assert not report.faults,report.faults
    assert not any(report.fallbacks.values()),report.fallbacks
    reports.append({'points':report.state.outcome.points,'fallbacks':report.fallbacks,'faults':report.faults,'max_action_seconds':max((v for times in report.latencies.values() for v in times),default=0.)})
args.output.write_text(json.dumps({'image':args.image,'seed_start':args.seed_start,'games':3,'restricted_cpu_and_memory':True,'faults':0,'fallbacks':0,'reports':reports},indent=2)+'\n')
print(json.dumps({'image':args.image,'games':3,'faults':0,'fallbacks':0}))

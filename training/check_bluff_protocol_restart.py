"""Exact persistent-versus-restarted SDK transport check for staged runtime."""
from pathlib import Path
import sys,json,time,argparse,hashlib
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'bluffSDK/local_evaluator/src'));sys.path.insert(0,str(ROOT/'training'))
from triad_bluff import engine,protocol
from triad_bluff.local_runner import AgentProcess
from train_bluff_league import make_policy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,default=ROOT/'bluffSDK')
    parser.add_argument('--python',type=Path,default=ROOT/'bluffSDK/.venv/bin/python')
    parser.add_argument('--output',type=Path,default=ROOT/'training/bluff-tonight-protocol-restart.json')
    parser.add_argument('--seed-start',type=int,default=79_000_000)
    args=parser.parse_args()
    repo=args.repo.resolve();command=[str(args.python.resolve()),str(repo/'src/agent.py')]
    total=0;phases=set();max_init=max_act=0.
    for sample in range(4):
        seat=sample;seed=args.seed_start+sample;game=f'restart-{seed}'
        state=engine.new_game(engine.GameConfig(),seed)
        rivals=[make_policy({'kind':'bluffer','bluff_rate':.65,'call_rate':.35},seed*10+s) for s in range(4)]
        persistent=AgentProcess(command,cwd=str(repo))
        try:
            begin=time.perf_counter();ready=persistent.request(protocol.build_initialize(game,seat,'init-persistent',1000,config=state.config.to_dict()),60)
            max_init=max(max_init,time.perf_counter()-begin);assert protocol.validate_agent_response(ready,'ready','init-persistent') is None;assert ready['agent_state'] is None
            while state.outcome is None and seat not in state.finished_set:
                decision=engine.next_decision(state);observation=engine.observe(state,decision.seat)
                if decision.seat==seat:
                    req=f'act-{total}';message=protocol.build_act(game,req,observation,None);start=time.perf_counter();reply=persistent.request(message,5);max_act=max(max_act,time.perf_counter()-start)
                    assert protocol.validate_agent_response(reply,'action',req) is None
                    fresh=AgentProcess(command,cwd=str(repo))
                    try:
                        start=time.perf_counter();ready=fresh.request(protocol.build_initialize(game,seat,'init-fresh',1000,config=state.config.to_dict()),60);max_init=max(max_init,time.perf_counter()-start)
                        assert protocol.validate_agent_response(ready,'ready','init-fresh') is None
                        alternate=fresh.request(message,5);assert protocol.validate_agent_response(alternate,'action',req) is None
                        assert alternate['action']==reply['action'];assert alternate['agent_state'] is None and reply['agent_state'] is None
                    finally:fresh.close()
                    action=reply['action'];total+=1;phases.add(observation['phase'])
                else:action=rivals[decision.seat].act(observation)
                assert engine.step(state,decision,engine.decode_action(state,decision,action)).ok
        finally:persistent.close()
    report={'games':4,'seed_start':args.seed_start,'four_seats_rotated':True,'decisions_compared':total,'phases':sorted(phases),'persistent_vs_fresh_process_actions_exact_equal':True,'checkpoint':None,'all_actions_legal':True,'max_initialize_seconds':max_init,'max_persistent_action_seconds':max_act,'model_sha256':hashlib.sha256((repo/'models/policy.json').read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()

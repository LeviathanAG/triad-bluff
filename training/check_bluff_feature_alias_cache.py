"""Exact old/new training actor comparison; no runtime or statistical policy change."""
import json,os,time
from pathlib import Path
os.environ['BLUFF_FAST_OBSERVATIONS']='1'
from train_bluff_league import engine
from bluff_mlp_policy import FastPolicy
from bluff_training_observation import observe_read_only

class PreviousFastPolicy(FastPolicy):
    def act(self,observation):
        self.update(observation);candidates=self.candidates(observation)
        matrix=self.np.asarray([vector for _,vector in candidates],dtype=self.np.float64)
        scores=matrix@self.base_array+self.output_bias
        scores+=self.np.maximum(0.,((matrix-self.mean_array)/self.scale_array)@self.matrix.T+self.hidden_bias)@self.output_array
        largest=float(scores.max());band=1e-8*max(1.,abs(largest))
        finalists=[index for index,value in enumerate(scores) if value>=largest-band]
        chosen=max(finalists,key=lambda index:self.score_features(candidates[index][1]))
        return candidates[chosen][0]

root=Path(__file__).resolve().parent
models={'neural64':root.parent/'bluffSDK/models/policy.json','conservative128':root/'bluff-conservative-ranking-output/bluff-policy-mlp.json'}
reports=[]
for name,path in models.items():
    document=json.loads(path.read_text());old_calls=new_calls=0;old_time=new_time=0.;decisions=0
    for index,(players,decks) in enumerate(((4,1),(4,1),(4,1),(4,1),(8,1),(8,2))):
        old=[PreviousFastPolicy(document) for _ in range(players)];new=[FastPolicy(document) for _ in range(players)]
        for actor in old:
            original=actor.score_features
            nonlocal_marker=[0]
            actor.counter=nonlocal_marker
            # Bind this actor's mutable counter separately from loop locals.
            def count(vector,original=original,counter=nonlocal_marker):
                counter[0]+=1;return original(vector)
            actor.score_features=count
        for actor in new:
            original=actor.score_features;counter=[0];actor.counter=counter
            def count(vector,original=original,counter=counter):
                counter[0]+=1;return original(vector)
            actor.score_features=count
        state=engine.new_game(engine.GameConfig(players=players,decks=decks,max_turns=1000),118000000+index)
        while state.outcome is None:
            decision=engine.next_decision(state);obs=observe_read_only(state,decision.seat)
            start=time.perf_counter();before=old[decision.seat].act(obs);old_time+=time.perf_counter()-start
            start=time.perf_counter();after=new[decision.seat].act(obs);new_time+=time.perf_counter()-start
            assert before==after
            assert engine.step(state,decision,engine.decode_action(state,decision,after)).ok
            decisions+=1
        old_calls+=sum(actor.counter[0] for actor in old);new_calls+=sum(actor.counter[0] for actor in new)
    reports.append({'model':name,'games':6,'decisions':decisions,'all_actions_exact':True,'old_exact_score_calls':old_calls,'new_exact_score_calls':new_calls,'old_act_seconds':old_time,'new_act_seconds':new_time,'observed_full_act_speedup':old_time/new_time})
report={'development_only':True,'published_runtime_unchanged':True,'decision_local_cache_only':True,'source_seeds_118M_correctness_only':True,'families':reports,'scope':'Exact whole-game training actor parity and local timing; timing is workload-specific, not a claimed cloud job speedup.'}
(root/'bluff-feature-alias-cache-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

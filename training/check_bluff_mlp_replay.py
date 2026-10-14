"""Compare training batching and fresh stdlib inference using filtered SDK inputs."""
import json
from pathlib import Path
import time
from collections import Counter
from train_bluff_league import engine, make_policy
from bluff_mlp_policy import Policy, FastPolicy
from bluff_training_observation import observe_read_only


def main():
    document=json.loads(Path('training/kaggle-bluff-residual-output/output/bluff-policy-mlp.json').read_text())
    count=games=0;maximum=0.; total=0.; hands=Counter()
    for players,decks in ((4,1),(4,2),(8,1),(8,2)):
        for sample in range(3):
            seed=55_000_000+players*100+decks*10+sample
            state=engine.new_game(engine.GameConfig(players=players,decks=decks,max_turns=1000),seed)
            actors=[FastPolicy(document) if seat%3==0 else make_policy({'kind':('competitive','bluffer','random')[seat%3],'bluff_rate':.65,'call_rate':.35},seed*10+seat) for seat in range(players)]
            while state.outcome is None:
                decision=engine.next_decision(state); observation=observe_read_only(state,decision.seat)
                actor=actors[decision.seat]; action=actor.act(observation)
                if isinstance(actor,FastPolicy):
                    start=time.perf_counter(); fresh=Policy(document); replay=fresh.act(observation)
                    elapsed=time.perf_counter()-start;maximum=max(maximum,elapsed);total+=elapsed
                    assert replay==action, 'training scorer and fresh stdlib inference disagree'
                    assert fresh.known==actor.known and fresh.pile_known==actor.pile_known
                    for seat,bounds in fresh.known.items():
                        actual=Counter(card.rank for card in state.hands[seat])
                        assert all(count<=actual[rank] for rank,count in bounds.items())
                    count+=1;hands[len(observation['hand'])]+=1
                transition=engine.step(state,decision,engine.decode_action(state,decision,action));assert transition.ok
            games+=1
    report={'games':games,'decisions':count,'configurations':[[4,1],[4,2],[8,1],[8,2]],'seed_family':55000000,'checks':'training NumPy actions match fresh stdlib full-history replay; sound own/public bounds; all decisions legal','max_full_replay_seconds':maximum,'mean_full_replay_seconds':total/max(1,count),'own_hand_size_histogram':dict(sorted(hands.items()))}
    Path('training/bluff-residual-runtime-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()

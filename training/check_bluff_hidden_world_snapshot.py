"""Small private augmentation proof; not a strength gate or posterior test."""
import argparse,copy,hashlib,json,os,pickle,random
from collections import Counter
from pathlib import Path
os.environ['BLUFF_FAST_OBSERVATIONS']='1'
from train_bluff_league import engine,make_policy
from collect_bluff_counterfactual import SOURCE_REGIMES,source_specs,checkpoint_spec,advance,rollout
from bluff_policy_v5 import Policy as Features,FEATURE_NAMES
from bluff_training_observation import observe_read_only
from bluff_hidden_world_snapshot import resample_snapshot

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--label-probe',action='store_true');args=parser.parse_args()
incumbent=json.loads((root.parent/'bluffSDK/models/policy.json').read_text())
previous=json.loads((root.parent/'bluffSDK/models/policy-v2.json').read_text())
weights=incumbent['base_weights']+[0.]*(len(FEATURE_NAMES)-len(incumbent['base_weights']))
counts=Counter();rows=[];label_rows=[]
for index,regime in enumerate(SOURCE_REGIMES):
    seed=115000000+index;own=seed%4;specs=source_specs(regime,random.Random(seed),incumbent,previous);specs[own]=checkpoint_spec(incumbent)
    actors=[make_policy(spec,seed*10+seat) for seat,spec in enumerate(specs)]
    state=engine.new_game(engine.GameConfig(max_turns=1000),seed);snapshots={phase:[] for phase in ('PLACE','CHALLENGE')};eligible=Counter();rng=random.Random(seed^12345)
    while state.outcome is None and own not in state.finished_set:
        decision=engine.next_decision(state)
        if decision.seat==own:
            obs=observe_read_only(state,own);feature=Features(weights);feature.update(obs)
            if len(feature.candidates(obs))>1:
                phase=obs['phase'];eligible[phase]+=1;slot=rng.randrange(eligible[phase])
                if slot<3:
                    snapshot=copy.deepcopy((state,actors))
                    if len(snapshots[phase])<3:snapshots[phase].append(snapshot)
                    else:snapshots[phase][slot]=snapshot
        advance(state,actors)
    for phase,pieces in snapshots.items():
        for number,snapshot in enumerate(pieces):
            original,_=snapshot;original_bytes=pickle.dumps(original)
            obs=observe_read_only(original,own);feature=Features(weights);feature.update(obs);expected=feature.candidates(obs)
            expected_action=make_policy(specs[own],1).act(obs)
            if args.label_probe and number==0:
                preferred=next(item for item in expected if item[0]==expected_action)
                alternative=next(item for item in expected if tuple(item[1])!=tuple(preferred[1]))
                actions=[preferred[0],alternative[0]];conditional=[];augmented=[];unseen=[];truths=[]
                for trial in range(6):
                    trial_seed=seed*100000+number*1000+trial
                    world,info=resample_snapshot(snapshot,own,specs,trial_seed^987654321)
                    truths.append(all(card.rank==world[0].pending_claim['claimed_rank'] for card in world[0].pending_batch) if phase=='CHALLENGE' else None)
                    conditional.append([rollout(snapshot,own,action,trial_seed) for action in actions])
                    augmented.append([rollout(world,own,action,trial_seed) for action in actions])
                    unseen.append(info['unknown_physical_cards'])
                import numpy as np
                first=np.asarray(conditional);second=np.asarray(augmented)
                label_rows.append({'source_seed':seed,'regime':regime,'phase':phase,'trials':6,
                                   'conditional_returns':conditional,'augmented_returns':augmented,
                                   'unknown_card_counts':unseen,
                                   'source_pending_truthful':all(card.rank==original.pending_claim['claimed_rank'] for card in original.pending_batch) if phase=='CHALLENGE' else None,
                                   'augmented_pending_truthful':truths,
                                   'conditional_paired_gap_std':float((first[:,0]-first[:,1]).std(ddof=1)),
                                   'augmented_paired_gap_std':float((second[:,0]-second[:,1]).std(ddof=1))})
            for trial in range(4):
                world,info=resample_snapshot(snapshot,own,specs,seed*10000+number*10+trial)
                alternate,alternate_actors=world
                actual_obs=observe_read_only(alternate,own);assert actual_obs==obs
                f=Features(weights);f.update(actual_obs);assert f.candidates(actual_obs)==expected
                assert alternate_actors[own].act(actual_obs)==expected_action
                for seat,actor in enumerate(alternate_actors):
                    if hasattr(actor,'known'):
                        for other,rank_counts in actor.known.items():
                            actual=Counter(card.rank for card in alternate.hands[other])
                            assert all(count<=actual[rank] for rank,count in rank_counts.items())
                    if hasattr(actor,'pile_known'):
                        pile={(card.rank,card.suit) for card in alternate.pile}
                        assert all((card['rank'],card['suit']) in pile for card in actor.pile_known.values())
                for _ in range(50):
                    if alternate.outcome is not None or own in alternate.finished_set:break
                    advance(alternate,alternate_actors);engine._check_invariants(alternate)
                    counts['future_legal_decisions']+=1
                assert pickle.dumps(original)==original_bytes
                counts['worlds']+=1;counts['feature_vectors']+=len(expected)
                counts['changed_worlds']+=info['changed_identity_positions']>0
                rows.append({'regime':regime,'phase':phase,'source_seed':seed,**info})
            counts['states']+=1
report={'training_only_prototype':True,'whole_own_observation_features_and_neural_actions_exact':True,'original_snapshots_unchanged':True,'all_rebuilt_actor_known_rank_bounds_and_pile_cards_sound':True,'deck_pending_and_future_action_invariants_verified':True,'counts':dict(counts),'rows':rows,'historical_policy_likelihood_weighted':False,'source_sha256':hashlib.sha256((root/'bluff_hidden_world_snapshot.py').read_bytes()).hexdigest(),'scope':'One-deck augmentation correctness/feasibility only; not a calibrated posterior, dataset, game-strength or admission result.'}
(root/'bluff-hidden-world-prototype.json').write_text(json.dumps(report,indent=2)+'\n')
if args.label_probe:
    probe={'training_only_label_probe':True,'states':len(label_rows),'action_count':2,'paired_trials_per_state':6,
           'source_seeds':list(range(115000000,115000007)),'feature_vectors_unchanged':True,
           'historical_policy_likelihood_weighted':False,'rows':label_rows,
           'known_truthful_style_pending_violations':sum(not value for row in label_rows if row['phase']=='CHALLENGE' and row['regime'] in ('competitive','truthful') for value in row['augmented_pending_truthful']),
           'collection_admission':'NOT_VALIDATED: pending claims can violate deterministic source-policy support; do not merge these probe labels into training.',
           'scope':'Conditional versus public-consistent latent-world label variability, not policy-performance evidence.'}
    (root/'bluff-hidden-world-label-probe.json').write_text(json.dumps(probe,indent=2)+'\n')
print(json.dumps({key:value for key,value in report.items() if key!='rows'}))

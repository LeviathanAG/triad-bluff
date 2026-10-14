"""Training-only aggregate action telemetry for the measured stress weaknesses."""
import concurrent.futures
import json
import multiprocessing
import random
from collections import Counter
from pathlib import Path
from train_bluff_league import engine, make_policy
from bluff_training_observation import observe_read_only


def diagnose(job):
    candidate, seeds, bluff_rate = job
    metrics = Counter()
    for seed in seeds:
        rival = {'kind': 'bluffer', 'bluff_rate': bluff_rate, 'call_rate': .35}
        for seat in range(4):
            actors = [None] * 4
            actors[seat] = make_policy(candidate, seed * 10 + 9)
            for offset in range(1, 4):
                actors[(seat + offset) % 4] = make_policy(rival, seed * 10 + offset)
            state = engine.new_game(engine.GameConfig(), seed)
            own_claims = {}
            own_turn = 0
            while state.outcome is None and seat not in state.finished_set:
                decision = engine.next_decision(state)
                observation = observe_read_only(state, decision.seat)
                action = actors[decision.seat].act(observation)
                if decision.seat == seat:
                    if observation['phase'] == 'PLACE':
                        own_turn += 1
                        cards = {card['card_ref']: card for card in observation['hand']}
                        truthful = all(cards[ref]['rank'] == action['claimed_rank'] for ref in action['card_refs'])
                        label = 'truthful' if truthful else 'bluff'
                        metrics['own_' + label + '_placements'] += 1
                        metrics['own_' + label + '_cards'] += len(action['card_refs'])
                        if not truthful:
                            metrics['own_bluff_turn_sum'] += own_turn
                            if hasattr(actors[seat], 'caller_probability'):
                                risk = actors[seat].caller_probability(observation, len(action['card_refs']), len(action['card_refs']) == len(observation['hand']))
                                metrics['own_bluff_predicted_any_call_sum'] += risk
                    elif action['type'] == 'challenge':
                        metrics['own_challenge_votes'] += 1
                previous = len(state.public_history)
                assert engine.step(state, decision, engine.decode_action(state, decision, action)).ok
                for event in state.public_history[previous:]:
                    if event['type'] == 'claim' and event['seat'] == seat:
                        own_claims[event['claim_id']] = (truthful, own_turn)
                    elif event['type'] in ('claim_accepted', 'challenge_resolved'):
                        own = own_claims.pop(event['claim_id'], None)
                        if own is not None and event['type'] == 'challenge_resolved':
                            metrics['own_truthful_challenged' if own[0] else 'own_bluff_caught'] += 1
                            metrics['own_bluff_caught_turn_sum'] += 0 if own[0] else own[1]
                        if event['type'] == 'challenge_resolved' and event['challenger'] == seat:
                            metrics['own_selected_call_success' if not event['truthful'] else 'own_selected_call_failure'] += 1
            points = state.config.position_points[state.finished_order.index(seat)] if seat in state.finished_order else state.outcome.points[seat]
            metrics['points'] += points
            metrics['games'] += 1
    return dict(metrics)


def main():
    current = json.loads(Path('bluffSDK/models/policy.json').read_text())
    previous = json.loads(Path('bluffSDK/models/policy-v1.json').read_text())
    policies = [{'kind': 'v2', 'weights': current['weights']}, {'kind': 'v1', 'weights': previous['weights']}]
    seeds = list(range(38_000_000, 38_000_000 + 128))
    chunks = [seeds[i:i + 16] for i in range(0, len(seeds), 16)]
    jobs = [(policy, chunk, bluff) for bluff in (.1, .65) for policy in policies for chunk in chunks]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(diagnose, jobs))
    report = []
    for index, bluff in enumerate((.1, .65)):
        for policy_index, kind in enumerate(('v2', 'v1')):
            aggregate = Counter()
            start = (index * 2 + policy_index) * len(chunks)
            for part in results[start:start + len(chunks)]:
                aggregate.update(part)
            report.append({'kind': kind, 'bluff_rate': bluff, 'call_rate': .35, 'telemetry': dict(aggregate)})
    payload = {'seed_start': seeds[0], 'note': 'Training-only diagnostics; model inputs remain own/public only.', 'results': report}
    Path('training/bluff-regression-diagnostics.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()

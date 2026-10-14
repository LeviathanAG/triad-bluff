"""Paired offline policy-improvement returns; deployment features are own/public.

The dealt simulator hands are held fixed within each labeled state. Opponent
random choices are averaged using common trial seeds across candidate actions.
Whole-game, phase-stratified reservoir sampling covers later/endgame decisions.
There is no draw deck to shuffle in Bluff. Hidden simulator state is confined to
training returns and never appears in the exported feature vectors.
"""
import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import os
from pathlib import Path
import random
import time
from collections import Counter
import numpy as np
from train_bluff_league import engine, make_policy
from bluff_policy_v5 import Policy as Features, FEATURE_NAMES
from bluff_training_observation import observe_read_only


def reward(state, seat):
    if seat in state.finished_order:
        return float(state.config.position_points[state.finished_order.index(seat)])
    return float(state.outcome.points[seat])


def advance(state, actors):
    decision = engine.next_decision(state)
    observation = observe_read_only(state, decision.seat)
    action = actors[decision.seat].act(observation)
    if not engine.step(state, decision, engine.decode_action(state, decision, action)).ok:
        raise RuntimeError('rollout actor produced an illegal action')


def rollout(snapshot, seat, selected, trial_seed):
    state, actors = copy.deepcopy(snapshot)
    # Preserve each actor's public/private memory at the sampled decision while
    # integrating future stochastic choices. Each action gets the same streams.
    for player, actor in enumerate(actors):
        if hasattr(actor, 'rng'):
            actor.rng = random.Random(trial_seed * 10 + player)
    decision = engine.next_decision(state)
    if decision.seat != seat:
        raise RuntimeError('sampled decision is for a different player')
    if not engine.step(state, decision, engine.decode_action(state, decision, selected)).ok:
        raise RuntimeError('counterfactual action is not legal at sampled state')
    while state.outcome is None and seat not in state.finished_set:
        advance(state, actors)
    return reward(state, seat)


def pool_specs(incumbent, previous):
    return [{'kind': 'v2', 'weights': incumbent['weights']},
            {'kind': 'v1', 'weights': previous['weights']},
            *({'kind': kind} for kind in ('starter', 'competitive', 'random', 'aggressive', 'heuristic')),
            *({'kind': 'bluffer', 'bluff_rate': bluff, 'call_rate': call}
              for bluff, call in ((.1, .0), (.1, .35), (.35, .1), (.65, .35), (.95, .0), (.95, .35)))]


def collect_job(job):
    seed, incumbent, previous, samples_per_phase, trials, max_actions = job
    rng = random.Random(seed ^ 0x424C554646)
    own = seed % 4
    specs = [rng.choice(pool_specs(incumbent, previous)) for _ in range(4)]
    specs[own] = {'kind': 'v2', 'weights': incumbent['weights']}
    actors = [make_policy(spec, seed * 10 + player) for player, spec in enumerate(specs)]
    state = engine.new_game(engine.GameConfig(max_turns=1000), seed)
    reservoirs = {'PLACE': [], 'CHALLENGE': []}
    eligible = Counter()
    feature_weights = incumbent['weights'] + [0.] * (len(FEATURE_NAMES) - len(incumbent['weights']))
    while state.outcome is None and own not in state.finished_set:
        decision = engine.next_decision(state)
        if decision.seat == own:
            observation = observe_read_only(state, own)
            policy = Features(feature_weights); policy.update(observation)
            if len(policy.candidates(observation)) > 1:
                phase = observation['phase']
                eligible[phase] += 1
                index = rng.randrange(eligible[phase])
                if index < samples_per_phase:
                    snapshot = copy.deepcopy((state, actors))
                    if len(reservoirs[phase]) < samples_per_phase:
                        reservoirs[phase].append(snapshot)
                    else:
                        reservoirs[phase][index] = snapshot
        advance(state, actors)
    rows = []
    for phase in ('PLACE', 'CHALLENGE'):
        for snapshot in reservoirs[phase]:
            original, _ = snapshot
            observation = observe_read_only(original, own)
            policy = Features(feature_weights); policy.update(observation)
            candidates = policy.candidates(observation)
            baseline = make_policy(specs[own], seed * 10 + own).act(observation)
            preferred = next(item for item in candidates if item[0] == baseline) if any(item[0] == baseline for item in candidates) else candidates[0]
            # Identical model inputs cannot be distinguished by the scorer;
            # avoid spending rollout compute on exact feature aliases.
            representatives = {tuple(preferred[1]): preferred}
            for candidate in candidates:
                representatives.setdefault(tuple(candidate[1]), candidate)
            selected = list(representatives.values())
            if len(selected) > max_actions:
                selected = [preferred] + rng.sample([item for item in selected if item is not preferred], max_actions - 1)
            trial_values = [[rollout(snapshot, own, action, seed * 100_000 + len(rows) * 1000 + trial)
                             for trial in range(trials)] for action, _ in selected]
            features = [vector for _, vector in selected]
            targets = [sum(values) / trials for values in trial_values]
            rows.append((features, targets, trial_values, seed, phase, len(observation['hand'])))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--games', type=int, default=2000)
    parser.add_argument('--samples-per-phase', type=int, default=2)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--max-actions', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed-start', type=int, default=50_000_000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.games, args.samples_per_phase, args.trials, args.workers) < 1 or args.max_actions < 2:
        parser.error('positive counts and at least two candidate actions required')
    args.output.mkdir(parents=True, exist_ok=True)
    incumbent = json.loads(args.incumbent.read_text()); previous = json.loads(args.previous.read_text())
    features, targets, trial_targets, offsets, game_ids = [], [], [], [0], []
    phases = Counter(); hands = Counter(); started = time.time()
    jobs = [(seed, incumbent, previous, args.samples_per_phase, args.trials, args.max_actions)
            for seed in range(args.seed_start, args.seed_start + args.games)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for index, rows in enumerate(pool.map(collect_job, jobs)):
            for vectors, values, trials, seed, phase, hand in rows:
                features.extend(vectors); targets.extend(values); trial_targets.extend(trials)
                offsets.append(len(features)); game_ids.append(seed); phases[phase] += 1; hands[hand] += 1
            if (index + 1) % 100 == 0:
                print(json.dumps({'games': index + 1, 'states': len(game_ids), 'actions': len(features),
                                  'seconds': round(time.time() - started, 1)}), flush=True)
    np.savez_compressed(args.output / 'counterfactual.npz', features=np.asarray(features, dtype=np.float32),
                        targets=np.asarray(targets, dtype=np.float32), trial_targets=np.asarray(trial_targets, dtype=np.float32),
                        offsets=np.asarray(offsets, dtype=np.int64), game_ids=np.asarray(game_ids, dtype=np.int64))
    metadata = {'method': 'paired one-step policy improvement, phase-stratified whole-game reservoirs',
                'input_features': FEATURE_NAMES, 'feature_policy': 'v5', 'games': args.games,
                'states': len(game_ids), 'actions': len(features), 'trials': args.trials,
                'samples_per_phase': args.samples_per_phase, 'max_actions': args.max_actions,
                'seed_start': args.seed_start, 'incumbent': incumbent, 'opponent_pool': pool_specs(incumbent, previous),
                'sdk_game_commit': '96da8042c285af4e4f43655f2ad99592bd49b8e6',
                'hidden_state_use': 'training returns only; dealt hands fixed within state, common future actor seeds across actions',
                'future_draw_pile_shuffle': False, 'own_hand_size_histogram': dict(sorted(hands.items())),
                'phase_histogram': dict(phases), 'seconds': round(time.time() - started, 1)}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({key: value for key, value in metadata.items() if key not in ('incumbent', 'input_features', 'opponent_pool')}), flush=True)


if __name__ == '__main__':
    main()

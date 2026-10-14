"""Train against diverse Bluff policies with paired seat blocks and held-out tests."""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
SDK = Path(os.environ.get('BLUFF_SDK', ROOT.parent / 'bluffSDK'))
sys.path.insert(0, str(SDK / 'local_evaluator/src'))
from triad_bluff import engine
from triad_bluff.baselines import HeuristicAgent, RandomAgent, TruthfulAgent, AggressiveChallengeAgent
from bluff_policy import Policy as V1, INITIAL_WEIGHTS as V1_INITIAL, RANKS
from bluff_policy_v2 import Policy as V2, INITIAL_WEIGHTS, FEATURE_NAMES


class Starter(TruthfulAgent):
    def act(self, observation):
        if observation['phase'] == 'CHALLENGE':
            return super().act(observation)
        from collections import defaultdict
        groups = defaultdict(list)
        for card in observation['hand']:
            groups[card['rank']].append(card)
        rank, cards = max(groups.items(), key=lambda item: (len(item[1]), item[0]))
        return {'type': 'place', 'card_refs': [c['card_ref'] for c in cards[:4]], 'claimed_rank': rank}


class StochasticBluffer(TruthfulAgent):
    def __init__(self, seed, bluff_rate, call_rate):
        self.rng = random.Random(seed)
        self.bluff_rate, self.call_rate = bluff_rate, call_rate

    def act(self, observation):
        truthful = super().act(observation)
        if observation['phase'] == 'CHALLENGE':
            if truthful['type'] == 'challenge':
                return truthful
            finishing = observation['pending_claim'].get('provisional_finish', False)
            probability = min(.95, self.call_rate + .2 * finishing)
            return {'type': 'challenge' if self.rng.random() < probability else 'accept'}
        hand = observation['hand']
        if len(hand) > 1 and self.rng.random() < self.bluff_rate:
            from collections import Counter
            held = Counter(c['rank'] for c in hand)
            count = min(4, len(hand)) if len(hand) <= 4 else self.rng.choice((2, 3, 4))
            chosen = sorted(hand, key=lambda c: (held[c['rank']], RANKS.index(c['rank'])))[:count]
            # Owning copies of the claimed rank makes impossible calls less likely.
            claimed = max(RANKS, key=lambda rank: held[rank])
            return {'type': 'place', 'card_refs': [c['card_ref'] for c in chosen], 'claimed_rank': claimed}
        return truthful


def make_policy(spec, seed):
    kind = spec['kind']
    if kind == 'v1':
        return V1(spec['weights'])
    if kind == 'v2':
        return V2(spec['weights'])
    if kind == 'bluffer':
        return StochasticBluffer(seed, spec['bluff_rate'], spec['call_rate'])
    if kind == 'starter':
        return Starter()
    if kind == 'truthful':
        return TruthfulAgent()
    if kind == 'random':
        return RandomAgent(seed)
    if kind == 'aggressive':
        return AggressiveChallengeAgent(seed)
    return HeuristicAgent(seed)


def score_game(seed, seat, candidate, rivals, horizon=1000):
    actors = [None] * 4
    actors[seat] = make_policy(candidate, seed * 10 + 9)
    for offset, rival in enumerate(rivals, 1):
        actors[(seat + offset) % 4] = make_policy(rival, seed * 10 + offset)
    state = engine.new_game(engine.GameConfig(max_turns=horizon), seed)
    while state.outcome is None and seat not in state.finished_set:
        decision = engine.next_decision(state)
        observation = engine.observe(state, decision.seat)
        payload = actors[decision.seat].act(observation)
        action = engine.decode_action(state, decision, payload)
        if not engine.step(state, decision, action).ok:
            raise RuntimeError('policy produced an illegal action')
    if seat in state.finished_order:
        placement = state.finished_order.index(seat)
        return state.config.position_points[placement], float(placement == 0)
    return state.outcome.points[seat], 0.


def evaluate_job(job):
    candidate, seeds, pool, mode, *options = job
    horizon = options[0] if options else 1000
    points, wins = [], []
    for seed in seeds:
        rng = random.Random(seed ^ 0x424C554646)
        if mode == 'league':
            rivals = [rng.choice(pool) for _ in range(3)]
        elif mode == 'bluffer':
            rivals = [{'kind': 'bluffer', 'bluff_rate': .35, 'call_rate': .1}] * 3
        else:
            rivals = [{'kind': mode}] * 3
        outcomes = [score_game(seed, seat, candidate, rivals, horizon) for seat in range(4)]
        points.append(sum(p for p, _ in outcomes) / 4.)
        wins.append(sum(w for _, w in outcomes) / 4.)
    return {'points': float(np.mean(points)), 'win_rate': float(np.mean(wins)),
            'games': 4 * len(seeds), 'block_points': points, 'block_wins': wins}


def summarize(result):
    return {k: v for k, v in result.items() if not k.startswith('block_')}


def paired_interval(candidate, reference, seed=20261014):
    differences = np.asarray(candidate['block_points']) - np.asarray(reference['block_points'])
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(differences), (5000, len(differences)))].mean(axis=1)
    return {'mean_gain': float(differences.mean()), 'ci95': [float(v) for v in np.quantile(means, [.025, .975])],
            'unit': 'points per game', 'sampling_unit': 'four-seat seed block'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--incumbent', type=Path)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--population', type=int, default=32)
    parser.add_argument('--train-blocks', type=int, default=8)
    parser.add_argument('--train-horizon', type=int, default=200)
    parser.add_argument('--validation-blocks', type=int, default=64)
    parser.add_argument('--test-blocks', type=int, default=256)
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--output', type=Path, default=ROOT / 'bluff-league-output')
    args = parser.parse_args()
    if min(args.iterations, args.population, args.train_blocks, args.validation_blocks, args.test_blocks, args.workers, args.train_horizon) < 1 or args.train_horizon > 1000:
        parser.error('counts must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    weights = json.loads(args.incumbent.read_text())['weights'] if args.incumbent else V1_INITIAL
    incumbent = {'kind': 'v2' if len(weights) == len(FEATURE_NAMES) else 'v1', 'weights': weights}
    mean = np.asarray(weights + INITIAL_WEIGHTS[len(weights):], dtype=float)
    best, sigma = mean.copy(), np.full(len(mean), 1.5)
    rng = np.random.default_rng(args.seed)
    cautious = list(INITIAL_WEIGHTS); cautious[2] -= 8.; cautious[4] -= 8.
    daring = list(INITIAL_WEIGHTS); daring[2] += 4.; daring[5] += 4.
    reference = [{'kind': k} for k in ('starter', 'heuristic', 'truthful', 'aggressive', 'random')] + [
        incumbent, {'kind': 'v2', 'weights': INITIAL_WEIGHTS}, {'kind': 'v2', 'weights': cautious},
        {'kind': 'v2', 'weights': daring}] + [
        {'kind': 'bluffer', 'bluff_rate': b, 'call_rate': c} for b,c in ((.15,.03),(.35,.1),(.5,.35))]
    archive = list(reference)
    selection_seeds = list(range(8_000_000, 8_000_000 + args.validation_blocks))
    history, started = [], time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        baseline = list(pool.map(evaluate_job, [(incumbent, selection_seeds, reference, 'league')]))[0]
        initial = list(pool.map(evaluate_job, [({'kind': 'v2', 'weights': best.tolist()}, selection_seeds, reference, 'league')]))[0]
        selected_score = initial['points']
        for generation in range(args.iterations):
            if generation and generation % 10 == 0:
                mean, sigma = best.copy(), np.full(len(mean), 1.5)
            population = rng.normal(mean, sigma, (args.population, len(mean)))
            probes = max(1, args.population // 5)
            population[-probes:] = rng.normal(best, 3., (probes, len(mean)))
            population = np.clip(population, -20., 20.)
            population[0] = mean
            if args.population > 1:
                population[1] = best
            seeds = list(range(50_000 + generation * args.train_blocks, 50_000 + (generation + 1) * args.train_blocks))
            # Fast training probes use a shorter cap; model selection and final
            # reporting always use the real 1,000-turn cap. Short-cap results are
            # approximate training rewards, never evidence of event performance.
            scores = list(pool.map(evaluate_job, [({'kind':'v2','weights':w.tolist()},seeds,archive,'league',args.train_horizon) for w in population]))
            ranked = sorted(range(len(scores)), key=lambda i: (scores[i]['points'], scores[i]['win_rate']), reverse=True)
            elite = population[ranked[:max(2, args.population // 5)]]
            mean = .35 * mean + .65 * elite.mean(axis=0)
            sigma = np.maximum(.3, .35 * sigma + .65 * elite.std(axis=0))
            candidate = population[ranked[0]]
            spec = {'kind':'v2','weights':candidate.tolist()}
            validation = list(pool.map(evaluate_job, [(spec,selection_seeds,reference,'league')]))[0]
            if validation['points'] > selected_score:
                best, selected_score = candidate.copy(), validation['points']
            if generation % 4 == 3:
                archive.append(spec)
                archive = reference + archive[len(reference):][-12:]
            record = {'generation':generation+1,'train':summarize(scores[ranked[0]]),'validation':summarize(validation),
                      'paired_validation_gain':paired_interval(validation,baseline),'selected_points':selected_score,
                      'archive_size':len(archive),'elapsed_seconds':round(time.time()-started,1)}
            history.append(record); print(json.dumps(record),flush=True)
            V2(best).save(args.output/'bluff-policy-v2.json',{'method':'archive-league cross-entropy policy optimization',
                'seed':args.seed,'generations':generation+1,'population':args.population,
                'train_blocks':args.train_blocks,'train_horizon':args.train_horizon,'evaluation_horizon':1000,'validation_blocks':args.validation_blocks,
                'selected_validation_points':selected_score,'incumbent_validation_points':baseline['points'],
                'sdk_commit':'96da8042c285af4e4f43655f2ad99592bd49b8e6'})
            (args.output/'training-progress.json').write_text(json.dumps(history,indent=2)+'\n')
            (args.output/'archive.json').write_text(json.dumps(archive,indent=2)+'\n')
        seeds = list(range(9_000_000,9_000_000+args.test_blocks))
        chosen = {'kind':'v2','weights':best.tolist()}
        modes = ('league','starter','truthful','bluffer','aggressive')
        results = list(pool.map(evaluate_job,[(spec,seeds,reference,mode) for mode in modes for spec in (chosen,incumbent)]))
        report = {mode:{'candidate':summarize(results[2*i]),'incumbent':summarize(results[2*i+1]),
                        'paired_gain':paired_interval(results[2*i],results[2*i+1])} for i,mode in enumerate(modes)}
        (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'held_out_evaluation':report}),flush=True)


if __name__ == '__main__':
    main()

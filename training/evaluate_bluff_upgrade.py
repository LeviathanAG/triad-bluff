"""Fresh paired comparison against the retained learned Bluff checkpoint."""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
from train_bluff_league import evaluate_job, paired_interval, summarize
from evaluate_bluff_checkpoint import combine
from bluff_policy_v2 import INITIAL_WEIGHTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--incumbent', type=Path, default=Path('bluffSDK/models/policy.json'))
    parser.add_argument('--blocks', type=int, default=256)
    parser.add_argument('--seed-start', type=int, default=20_000_000)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fast-observations', action='store_true')
    parser.add_argument('--stress-regimes', action='store_true')
    args = parser.parse_args()
    if args.fast_observations:
        os.environ['BLUFF_FAST_OBSERVATIONS'] = '1'
    candidate_doc = json.loads(args.candidate.read_text())
    incumbent_doc = json.loads(args.incumbent.read_text())
    def spec(document):
        if document.get('format') == 'triad-bluff-residual-mlp-v1':
            return {'kind': 'mlp', 'model': document}
        return {'kind': document.get('training', {}).get('policy_kind', 'v4' if len(document['weights']) > len(INITIAL_WEIGHTS) else 'v2') if len(document['weights']) >= len(INITIAL_WEIGHTS) else 'v1',
                'weights': document['weights']}
    candidate, incumbent = spec(candidate_doc), spec(incumbent_doc)
    # Fixed pool is identical for both own-policy comparisons. It includes the
    # retained learned checkpoint and the SDK's independently supplied rival.
    reference = [{'kind': k} for k in ('starter', 'heuristic', 'truthful', 'aggressive', 'random', 'competitive')]
    reference += [incumbent, {'kind': 'v2', 'weights': INITIAL_WEIGHTS}]
    reference += [{'kind': 'bluffer', 'bluff_rate': b, 'call_rate': c}
                  for b, c in ((.15, .03), (.35, .1), (.5, .35))]
    modes = ['league', 'competitive', 'starter', 'truthful', 'bluffer', 'aggressive', 'incumbent_selfplay']
    special = {'incumbent_selfplay': [incumbent]}
    if args.stress_regimes:
        for label, bluff, call in (('weakspot_bluff01_call035', .1, .35), ('weakspot_bluff065_call035', .65, .35)):
            modes.append(label); special[label] = [{'kind': 'bluffer', 'bluff_rate': bluff, 'call_rate': call}]
    seeds = list(range(args.seed_start, args.seed_start + args.blocks))
    chunks = [seeds[i:i + 32] for i in range(0, len(seeds), 32)]
    jobs = [(policy, chunk, special.get(mode, reference), 'league' if mode in special else mode)
            for mode in modes for policy in (candidate, incumbent) for chunk in chunks]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(evaluate_job, jobs))
    report = {}
    for index, mode in enumerate(modes):
        count = len(chunks)
        a = combine(results[2 * index * count:(2 * index + 1) * count])
        b = combine(results[(2 * index + 1) * count:(2 * index + 2) * count])
        report[mode] = {'candidate': summarize(a), 'incumbent': summarize(b),
                        'paired_gain': paired_interval(a, b)}
    payload = {'seed_start': args.seed_start, 'opponent_pool': reference,
               'candidate_training': candidate_doc['training'],
               'incumbent_training': incumbent_doc['training'], 'evaluation': report}
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload), flush=True)


if __name__ == '__main__':
    main()

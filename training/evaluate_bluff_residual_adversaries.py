"""Stress-test retained learned checkpoints across a fixed bluff/call grid."""
import concurrent.futures
import json
import os
os.environ["BLUFF_FAST_OBSERVATIONS"] = "1"
import multiprocessing
from pathlib import Path
from train_bluff_league import evaluate_job, paired_interval, summarize
from evaluate_bluff_checkpoint import combine


def main():
    current = json.loads(Path('bluffSDK/models/policy.json').read_text())
    previous = json.loads(Path('bluffSDK/models/policy-v2.json').read_text())
    policies = [{'kind': 'mlp', 'model': current},
                {'kind': 'v2', 'weights': previous['weights']}]
    regimes = [(bluff, call) for bluff in (.1, .35, .65, .95)
               for call in (.0, .1, .35, .8)]
    seeds = list(range(58_000_000, 58_000_000 + 128))
    chunks = [seeds[i:i + 32] for i in range(0, len(seeds), 32)]
    jobs = [(policy, chunk, [{'kind': 'bluffer', 'bluff_rate': bluff, 'call_rate': call}], 'league')
            for bluff, call in regimes for policy in policies for chunk in chunks]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(evaluate_job, jobs))
    reports = []
    count = len(chunks)
    for index, (bluff, call) in enumerate(regimes):
        candidate = combine(results[(2 * index) * count:(2 * index + 1) * count])
        incumbent = combine(results[(2 * index + 1) * count:(2 * index + 2) * count])
        reports.append({'bluff_rate': bluff, 'call_rate': call,
                        'neural_candidate': summarize(candidate), 'retained_v2': summarize(incumbent),
                        'paired_gain': paired_interval(candidate, incumbent)})
    report = {'seed_start': seeds[0], 'sampling_unit': 'four-seat seed block',
              'note': 'Fixed development stress opponents; fresh grid not used to select the neural candidate.',
              'regimes': reports}
    Path('training/bluff-residual-adversarial-grid.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

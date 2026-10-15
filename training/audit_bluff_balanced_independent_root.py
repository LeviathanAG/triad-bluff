"""Read-only independent audit of frozen 90M game arrays; no policy selection."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / 'kaggle-bluff-balanced-verify-output/output'


def main():
    raw = json.loads((DIRECTORY / 'independent-blocks.json').read_text())
    report = json.loads((DIRECTORY / 'independent.json').read_text())
    assert raw['seed_ids'] == list(range(90000000, 90000512))
    assert report['seed_start'] == 90000000
    assert raw['game_config'] == dict(players=4, decks=1, max_turns=1000,
                                     position_points=[5, 3, 2, 1])
    assert raw['sdk_game_commit'] == '96da8042c285af4e4f43655f2ad99592bd49b8e6'
    assert raw['checkpoint_file_sha256'] == [
        '8253987d638222d8f1314baa5e7045f1b5ab58c5b224506be011078e21d330ba',
        'ba6ed2a7aa7f03637c02225fb96c71e4798fb5073f3bf5ff2a793038719a1b08',
        '606284b35375d3e305adb8cd37ef2e479881e3fc85f838dbe10d79d5171df3a5',
        '2aaeaaf13084a254ed443d4ffcc4288ab32ffe2b284d5bd9023c4a78594b0cf7',
    ]
    expected_order = ['candidate', 'incumbent_neural', 'retained_v2',
                      'conservative_candidate']
    assert raw['checkpoint_order'] == expected_order
    assert len(raw['rows']) == 12
    assert set(raw['rows']) == set(report['evaluation'])
    observed = {}
    for regime, rows in raw['rows'].items():
        assert list(rows) == expected_order
        for name, row in rows.items():
            summary = report['evaluation'][regime][name]
            points = np.asarray(row['block_points'])
            wins = np.asarray(row['block_wins'])
            assert points.shape == wins.shape == (512,)
            assert np.isfinite(points).all() and np.isfinite(wins).all()
            assert ((1 <= points) & (points <= 5)).all()
            assert ((0 <= wins) & (wins <= 1)).all()
            assert row['games'] == summary['games'] == 2048
            for key, value in [('points', points.mean()), ('win_rate', wins.mean())]:
                assert np.isclose(row[key], value, rtol=0, atol=1e-12)
                assert np.isclose(summary[key], value, rtol=0, atol=1e-12)
        observed[regime] = {}
        for comparator, label in [('incumbent_neural', 'neural'),
                                  ('retained_v2', 'v2'),
                                  ('conservative_candidate', 'conservative')]:
            difference = (np.asarray(rows['candidate']['block_points']) -
                          np.asarray(rows[comparator]['block_points']))
            # Reproduce the predeclared bootstrap, without retuning its seed.
            rng = np.random.default_rng(20261014)
            means = difference[rng.integers(0, 512, (5000, 512))].mean(axis=1)
            interval = np.quantile(means, [.025, .975])
            quoted = report['evaluation'][regime]['paired_gain_vs_' + label]
            assert np.isclose(quoted['mean_gain'], difference.mean(), rtol=0, atol=1e-12)
            assert np.allclose(quoted['ci95'], interval, rtol=0, atol=1e-12)
            observed[regime][label] = dict(mean_gain=float(difference.mean()),
                                          ci95=interval.tolist())
    result = dict(
        raw_arrays_verified=True, sampling_unit='four-seat seed block',
        blocks_per_regime=512, games_per_policy_per_regime=2048,
        total_games=512 * 4 * 4 * 12,
        checkpoints=raw['checkpoint_file_sha256'], results=observed,
        files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [DIRECTORY / 'independent.json',
                                DIRECTORY / 'independent-blocks.json']},
        scope='Frozen independent gate audit. No deployment or new selection performed.')
    (ROOT / 'bluff-balanced90-root-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

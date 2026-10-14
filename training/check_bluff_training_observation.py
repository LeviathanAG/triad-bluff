"""Verify filtered observation equality, actor purity and equivalent decisions."""
import json
import time
from pathlib import Path
from train_bluff_league import engine, make_policy
from bluff_training_observation import observe_read_only


def main():
    v2 = json.loads(Path('bluffSDK/models/policy.json').read_text())['weights']
    v1 = json.loads(Path('bluffSDK/models/policy-v1.json').read_text())['weights']
    specs = [{'kind': 'v1', 'weights': v1}, {'kind': 'v2', 'weights': v2},
             {'kind': 'v3', 'weights': v2}, {'kind': 'competitive'},
             {'kind': 'bluffer', 'bluff_rate': .95, 'call_rate': .35},
             {'kind': 'random'}, {'kind': 'truthful'}, {'kind': 'aggressive'}]
    standard_seconds = shortcut_seconds = 0.
    decisions = 0
    for players, decks, games in ((4, 1, 12), (8, 2, 6)):
        for seed in range(games):
            state = engine.new_game(engine.GameConfig(players=players, decks=decks, max_turns=1000), 29_000_000 + seed)
            standard = [make_policy(specs[(seat + seed) % len(specs)], seed * 10 + seat) for seat in range(players)]
            shortcut = [make_policy(specs[(seat + seed) % len(specs)], seed * 10 + seat) for seat in range(players)]
            while state.outcome is None:
                decision = engine.next_decision(state)
                start = time.perf_counter(); a = engine.observe(state, decision.seat)
                standard_seconds += time.perf_counter() - start
                start = time.perf_counter(); b = observe_read_only(state, decision.seat)
                shortcut_seconds += time.perf_counter() - start
                assert a == b, 'shortcut changed a filtered observation field'
                expected = standard[decision.seat].act(a)
                actual = shortcut[decision.seat].act(b)
                assert expected == actual, 'shortcut changed an actor decision'
                assert a['public_history'] == state.public_history
                assert a['private_history'] == state.private_history[decision.seat]
                assert engine.step(state, decision, engine.decode_action(state, decision, actual)).ok
                decisions += 1
    report = {'games': 18, 'decisions': decisions,
              'observation_seconds_standard': standard_seconds,
              'observation_seconds_shortcut': shortcut_seconds,
              'observation_speedup': standard_seconds / shortcut_seconds,
              'verified_actor_kinds': [spec['kind'] for spec in specs],
              'checks': 'filtered fields equal, actor decisions equal, histories unmodified, actions legal'}
    Path('training/bluff-observation-efficiency.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

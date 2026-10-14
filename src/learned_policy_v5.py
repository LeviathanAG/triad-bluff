"""Public caller pooling and uncertainty features for a learned Bluff scorer.

Beta summaries are modeling features, not calibrated certainty claims. Observed
selected/earlier caller outcomes are pooled; later sealed votes stay unknown.
"""
import json
import math
from pathlib import Path
from learned_policy_v4 import Policy as V4, FEATURE_NAMES as V4_NAMES, INITIAL_WEIGHTS as V4_INITIAL

EXTRA_PLACE = ['bluff_pooled_caller_probability', 'bluff_pooled_caller_uncertainty',
               'bluff_specific_caller_uncertainty', 'bluff_sparse_own_outcomes',
               'truthful_pooled_caller_probability_batch', 'bluff_pooled_caller_probability_pile']
EXTRA_CHALLENGE = ['global_revealed_bluff_uncertainty', 'count_revealed_bluff_uncertainty']
FEATURE_NAMES = V4_NAMES + EXTRA_PLACE + EXTRA_CHALLENGE
INITIAL_WEIGHTS = V4_INITIAL + [0.] * 8


def beta_summary(samples, successes, prior_success=1., prior_total=5.):
    a = successes + prior_success
    b = samples - successes + prior_total - prior_success
    return a / (a + b), math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.)))


class Policy(V4):
    def __init__(self, weights):
        if len(weights) != len(FEATURE_NAMES) or not all(math.isfinite(float(w)) for w in weights):
            raise ValueError('invalid v5 model weights')
        super().__init__(weights[:len(V4_NAMES)])
        self.weights = list(map(float, weights))

    def caller_summaries(self, observation, count=None, finish=None):
        escape_mean = escape_upper = 1.
        for seat in observation['active_seats']:
            if seat == observation['seat']:
                continue
            if count is None:
                samples = calls = 0
                for (caller, _, _), record in self.calls.items():
                    if caller == seat:
                        samples += record[0]; calls += record[1]
            else:
                samples, calls = self.calls.get((seat, count, finish), (0, 0))
            mean, deviation = beta_summary(samples, calls)
            escape_mean *= 1. - mean
            escape_upper *= 1. - min(1., mean + deviation)
        return 1. - escape_mean, escape_mean - escape_upper

    def candidates(self, observation):
        candidates = super().candidates(observation)
        if observation['phase'] == 'CHALLENGE':
            if len(candidates) == 1:
                return [(candidates[0][0], [0.] * len(self.weights))]
            claim = observation['pending_claim']
            samples, bluffs = self.global_bluffs.get(claim['seat'], (0, 0))
            _, global_uncertainty = beta_summary(samples, bluffs, .6, 5.)
            samples, bluffs = self.stats.get((claim['seat'], claim['count']), (0, 0))
            _, count_uncertainty = beta_summary(samples, bluffs, .6, 5.)
            return [(action, vector[:len(V4_NAMES)] + [0.] * len(EXTRA_PLACE) +
                     ([global_uncertainty, count_uncertainty] if action['type'] == 'challenge' else [0., 0.]))
                    for action, vector in candidates]
        by_ref = {card['card_ref']: card for card in observation['hand']}
        pooled, pooled_uncertainty = self.caller_summaries(observation)
        sparse = float(self.our_outcomes[0] <= 2)
        pile = min(observation['pile_size'], 40) / 20.
        risks = {}
        results = []
        for action, vector in candidates:
            count = len(action['card_refs'])
            if count not in risks:
                risks[count] = self.caller_summaries(observation, count, count == len(observation['hand']))
            _, specific_uncertainty = risks[count]
            bluff = float(any(by_ref[ref]['rank'] != action['claimed_rank'] for ref in action['card_refs']))
            extra = [bluff * pooled, bluff * pooled_uncertainty, bluff * specific_uncertainty,
                     bluff * sparse, (1. - bluff) * pooled * count / 4., bluff * pooled * pile]
            results.append((action, vector[:len(V4_NAMES)] + extra + [0.] * len(EXTRA_CHALLENGE)))
        return results

    def save(self, path, metadata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({'format': 'triad-bluff-linear-v5',
            'features': FEATURE_NAMES, 'weights': self.weights, 'training': metadata}, indent=2) + '\n')

"""Linear Bluff scorer with learned interactions for public call-risk adaptation."""
from collections import Counter
import json
import math
from pathlib import Path
from learned_policy_v3 import Policy as V3
from learned_policy import FEATURE_NAMES as V2_NAMES, INITIAL_WEIGHTS as V2_INITIAL

EXTRA_PLACE = ['bluff_call_probability_batch', 'bluff_call_probability_pile',
               'bluff_call_probability_finish', 'bluff_nearest_opponent_finish',
               'truthful_call_probability_batch', 'bluff_remaining_truthful_turns']
EXTRA_CHALLENGE = ['global_revealed_bluff_finishing', 'count_revealed_bluff_pile',
                   'global_revealed_bluff_pile', 'count_revealed_bluff_small_own_hand',
                   'global_revealed_bluff_count', 'opponent_hand_pace_relative_to_own']
FEATURE_NAMES = V2_NAMES + EXTRA_PLACE + EXTRA_CHALLENGE
INITIAL_WEIGHTS = V2_INITIAL + [0.] * 12


class Policy(V3):
    def __init__(self, weights):
        if len(weights) != len(FEATURE_NAMES) or not all(math.isfinite(float(w)) for w in weights):
            raise ValueError('invalid v4 model weights')
        super().__init__(weights[:len(V2_NAMES)])
        self.weights = list(map(float, weights))

    def candidates(self, observation):
        candidates = super().candidates(observation)
        hand = observation['hand']
        pile = min(observation['pile_size'], 40) / 20.
        if observation['phase'] == 'CHALLENGE':
            if len(candidates) == 1:
                return [(candidates[0][0], [0.] * len(self.weights))]
            claim = observation['pending_claim']
            samples, bluffs = self.global_bluffs.get(claim['seat'], (0, 0))
            global_rate = (bluffs + .6) / (samples + 5.)
            samples, bluffs = self.stats.get((claim['seat'], claim['count']), (0, 0))
            count_rate = (bluffs + .6) / (samples + 5.)
            extra = [global_rate * float(claim.get('provisional_finish', False)),
                     count_rate * pile, global_rate * pile,
                     count_rate * float(len(hand) <= 3), global_rate * claim['count'] / 4.,
                     (len(hand) - observation['hand_sizes'][claim['seat']]) / max(1, len(hand))]
            return [(action, vector[:len(V2_NAMES)] + [0.] * len(EXTRA_PLACE) +
                     (extra if action['type'] == 'challenge' else [0.] * len(EXTRA_CHALLENGE)))
                    for action, vector in candidates]
        by_ref = {card['card_ref']: card for card in hand}
        held = Counter(card['rank'] for card in hand)
        nearest = min(observation['hand_sizes'][seat] for seat in observation['active_seats']
                      if seat != observation['seat'])
        results = []
        risks = {}
        for action, vector in candidates:
            count = len(action['card_refs']); finish = count == len(hand)
            if count not in risks:
                risks[count] = self.caller_probability(observation, count, finish)
            risk = risks[count]
            bluff = float(any(by_ref[ref]['rank'] != action['claimed_rank'] for ref in action['card_refs']))
            remaining = held.copy()
            for ref in action['card_refs']:
                remaining[by_ref[ref]['rank']] -= 1
            turns = sum((n + 3) // 4 for n in remaining.values() if n > 0)
            extra = [bluff * risk * count / 4., bluff * risk * pile,
                     bluff * risk * float(finish), bluff / max(1, nearest),
                     (1. - bluff) * risk * count / 4., bluff * turns / 13.]
            results.append((action, vector[:len(V2_NAMES)] + extra + [0.] * len(EXTRA_CHALLENGE)))
        return results

    def save(self, path, metadata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({'format': 'triad-bluff-linear-v4',
            'features': FEATURE_NAMES, 'weights': self.weights, 'training': metadata}, indent=2) + '\n')

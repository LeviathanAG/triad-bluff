"""Bluff v2 model with additional, sound public/own pile rank deductions.

Older unchallenged batches are never assumed truthful. These deductions use
only own placement cards and the last publicly revealed batch before pickup.
"""
from collections import Counter
from learned_policy import Policy as V2, FEATURE_NAMES, INITIAL_WEIGHTS


class Policy(V2):
    def candidates(self, observation):
        if observation['phase'] == 'CHALLENGE':
            claim = observation['pending_claim']
            rank, count, placer = claim['claimed_rank'], claim['count'], claim['seat']
            # Known old pile cards, current own cards and lower bounds on other
            # players' hands are disjoint, so their sum is a conservative count.
            pile_count = Counter(card['rank'] for card in self.pile_known.values())[rank]
            held_count = sum(card['rank'] == rank for card in observation['hand'])
            other_count = sum(counts[rank] for seat, counts in self.known.items()
                              if seat not in (observation['seat'], placer))
            copies = 4 * observation['game_config']['decks']
            if count > copies - held_count - other_count - pile_count:
                return [({'type': 'challenge'}, [0.] * len(self.weights))]
        return super().candidates(observation)

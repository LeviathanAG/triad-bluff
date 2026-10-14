"""Learned Bluff scorer with conservative rank bounds and public caller evidence.

Inference reads only own hand/private history and public events. Unknown discarded
cards decrement every rank bound, ensuring no hidden-hand identity is assumed.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from bluff_policy import Policy as V1, PLACE_FEATURES, CHALLENGE_FEATURES, INITIAL_WEIGHTS as V1_INITIAL, RANKS

EXTRA_PLACE = ['bluff_any_call_probability', 'bluff_impossible_for_caller',
               'truthful_turns_removed', 'bluff_known_pile_fraction',
               'truthful_finish', 'bluff_no_impossibility_call_possible']
EXTRA_CHALLENGE = ['placer_global_revealed_bluff_rate', 'known_outside_rank_fraction',
                   'known_pile_fraction', 'claim_fraction_of_original_hand',
                   'finishing_large_batch', 'known_placer_rank_before_claim']
FEATURE_NAMES = PLACE_FEATURES + CHALLENGE_FEATURES + EXTRA_PLACE + EXTRA_CHALLENGE
INITIAL_WEIGHTS = [.5, 1., -.5, -2., -8., -3., -1., .2, .5] + V1_INITIAL[len(PLACE_FEATURES):] + [-6., -12., 3., -1., 4., 3., 1., 1., -.5, .5, .5, -.5]


def identity(card):
    return card.get('deck', 0), card['rank'], card['suit']


class Policy(V1):
    def __init__(self, weights):
        if len(weights) != len(FEATURE_NAMES) or not all(math.isfinite(float(w)) for w in weights):
            raise ValueError('invalid model weights')
        super().__init__(weights[:len(V1_INITIAL)])
        self.weights = list(map(float, weights))
        self._reset_evidence()

    def _reset_evidence(self):
        self.folded_history = []
        self.known = defaultdict(Counter)
        self.pile_known = {}
        self.calls = defaultdict(lambda: [0, 0])
        self.global_bluffs = defaultdict(lambda: [0, 0])
        self.pending_meta = {}
        self.active = []
        self.before_claim = {}

    def update(self, observation):
        history = observation.get('public_history', [])
        if len(history) < len(self.folded_history) or history[:len(self.folded_history)] != self.folded_history:
            self.processed = 0
            self.claims = {}
            self.stats = defaultdict(lambda: [0, 0])
            self.our_outcomes = [0, 0]
            self._reset_evidence()
        super().update(observation)
        own_places = {event['claim_id']: event['cards'] for event in observation.get('private_history', []) if event.get('type') == 'own_placement'}
        for event in history[len(self.folded_history):]:
            kind, cid = event.get('type'), event.get('claim_id')
            if kind == 'game_start':
                self.active = list(event['active_seats'])
            elif kind == 'claim':
                placer, count = event['seat'], event['count']
                self.before_claim[cid] = dict(self.known[placer])
                # The rank in an unverified claim is not evidence of actual cards.
                for rank in list(self.known[placer]):
                    self.known[placer][rank] = max(0, self.known[placer][rank] - count)
                priority = sorted((s for s in self.active if s != placer), key=lambda s: (s - placer) % observation['game_config']['players'])
                self.pending_meta[cid] = (placer, count, priority, bool(event.get('provisional_finish')))
                for card in own_places.get(cid, []):
                    self.pile_known[identity(card)] = card
            elif kind in ('claim_accepted', 'challenge_resolved'):
                placer, count, priority, finish = self.pending_meta.pop(cid)
                selected = event.get('challenger') if kind == 'challenge_resolved' else None
                # The first selected challenger is revealed. Earlier seats must
                # have accepted; later choices remain unknown and are not counted.
                for seat in priority:
                    record = self.calls[(seat, count, finish)]
                    record[0] += 1
                    record[1] += int(seat == selected)
                    if seat == selected:
                        break
                if kind == 'challenge_resolved':
                    record = self.global_bluffs[placer]
                    record[0] += 1
                    record[1] += int(not event['truthful'])
                    for card in event['revealed_batch']:
                        self.pile_known[identity(card)] = card
            elif kind == 'pickup':
                recipient = event['recipient']
                self.known[recipient].update(card['rank'] for card in self.pile_known.values())
                self.pile_known = {}
            elif kind == 'finish':
                self.active = [s for s in self.active if s != event['seat']]
        self.folded_history = list(history)

    def caller_probability(self, observation, count, finish):
        escape = 1.
        for seat in observation['active_seats']:
            if seat == observation['seat']:
                continue
            samples, calls = self.calls.get((seat, count, finish), (0, 0))
            # Beta prior avoids certainties from small or selectively observed samples.
            escape *= 1. - (calls + 1.) / (samples + 5.)
        return 1. - escape

    def candidates(self, observation):
        hand = observation['hand']
        held = Counter(c['rank'] for c in hand)
        copies = 4 * observation.get('game_config', {}).get('decks', 1)
        if observation['phase'] == 'CHALLENGE':
            claim = observation['pending_claim']
            placer, rank, count = claim['seat'], claim['claimed_rank'], claim['count']
            outside = held[rank] + sum(counts[rank] for seat, counts in self.known.items() if seat not in (observation['seat'], placer))
            if count > copies - outside:
                return [({'type': 'challenge'}, [0.] * len(self.weights))]
            base = super().candidates(observation)
            samples, bluffs = self.global_bluffs.get(placer, (0, 0))
            extra = [(bluffs + .6) / (samples + 5.), outside / copies,
                     len(self.pile_known) / max(1, observation['pile_size']),
                     count / max(1, observation['hand_sizes'][placer] + count),
                     float(claim.get('provisional_finish', False)) * count / 4.,
                     self.before_claim.get(claim['claim_id'], {}).get(rank, 0) / copies]
            return [(a, v + [0.] * len(EXTRA_PLACE) + (extra if a['type'] == 'challenge' else [0.] * len(EXTRA_CHALLENGE))) for a, v in base]
        groups = defaultdict(list)
        for card in hand:
            groups[card['rank']].append(card)
        actions = [action for action, _ in super().candidates(observation)]
        # Full truthful batches plus partial truthful batches and mixed shedding
        # batches of up to four cards, including a possible finishing bluff.
        for rank, cards in groups.items():
            for count in range(1, min(4, len(cards)) + 1):
                actions.append({'type': 'place', 'card_refs': [c['card_ref'] for c in cards[:count]], 'claimed_rank': rank})
        ordered = sorted(hand, key=lambda c: (held[c['rank']], RANKS.index(c['rank'])))
        for count in range(2, min(4, len(hand)) + 1):
            for rank in RANKS:
                actions.append({'type': 'place', 'card_refs': [c['card_ref'] for c in ordered[:count]], 'claimed_rank': rank})
        by_ref, seen, result = {c['card_ref']: c for c in hand}, set(), []
        pile = min(observation.get('pile_size', 0), 40) / 20.
        call_rate = (self.our_outcomes[1] + 1.) / (self.our_outcomes[0] + 4.)
        for action in actions:
            refs, rank = set(action['card_refs']), action['claimed_rank']
            key = tuple(sorted(refs)), rank
            if key in seen:
                continue
            seen.add(key)
            truthful = all(by_ref[r]['rank'] == rank for r in refs)
            bluff = float(not truthful)
            remaining = Counter(c['rank'] for c in hand if c['card_ref'] not in refs)
            finish = not remaining
            vector = [len(refs), float(truthful), bluff, bluff * pile, bluff * call_rate,
                      float(bluff and finish), len(remaining) / 13., max(remaining.values(), default=0) / 4., held[rank] / 4.]
            impossible = any(self.known.get(s, {}).get(rank, 0) + len(refs) > copies for s in observation['active_seats'] if s != observation['seat'])
            extra = [bluff * self.caller_probability(observation, len(refs), finish),
                     bluff * impossible, sum((n + 3) // 4 for n in held.values()) - sum((n + 3) // 4 for n in remaining.values()),
                     bluff * len(self.pile_known) / max(1, observation['pile_size']),
                     float(truthful and finish), bluff * float(held[rank] >= len(refs))]
            result.append((action, vector + [0.] * len(CHALLENGE_FEATURES) + extra + [0.] * len(EXTRA_CHALLENGE)))
        return result

    def save(self, path, metadata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({'format': 'triad-bluff-linear-v2', 'features': FEATURE_NAMES,
                                        'weights': self.weights, 'training': metadata}, indent=2) + '\n')

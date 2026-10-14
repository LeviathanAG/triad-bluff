"""Compact learned Bluff action scorer using own cards and public evidence."""
from collections import Counter, defaultdict
import json
from pathlib import Path

RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
PLACE_FEATURES = ["batch_size", "truthful", "bluff", "bluff_pile", "bluff_call_rate",
                  "bluff_finish", "remaining_groups", "remaining_largest_group", "claimed_rank_held"]
CHALLENGE_FEATURES = ["bias", "revealed_bluff_rate", "rank_held", "claim_count", "pile_size",
                      "opponent_finishing", "own_small_hand", "opponent_small_hand", "samples"]
INITIAL_WEIGHTS = [4., 5., -2., -4., -8., -8., -1., .2, .5,
                   -3.5, 7., .7, -.3, -.5, 1., -1., .5, 0.]


class Policy:
    def __init__(self, weights):
        if len(weights) != len(INITIAL_WEIGHTS):
            raise ValueError("wrong feature count")
        self.weights = [float(v) for v in weights]
        self.processed = 0
        self.claims = {}
        self.stats = defaultdict(lambda: [0, 0])
        self.our_outcomes = [0, 0]

    def update(self, observation):
        history = observation.get("public_history", [])
        if len(history) < self.processed:
            self.processed, self.claims, self.stats, self.our_outcomes = 0, {}, defaultdict(lambda: [0, 0]), [0, 0]
        for event in history[self.processed:]:
            kind = event.get("type")
            identity = event.get("claim_id")
            if kind == "claim":
                self.claims[identity] = (event.get("seat"), event.get("count", 1))
            elif kind in ("claim_accepted", "challenge_resolved"):
                placer, count = self.claims.pop(identity, (event.get("placer", event.get("seat")), event.get("count", 1)))
                if placer == observation["seat"]:
                    self.our_outcomes[0] += 1
                    self.our_outcomes[1] += int(kind == "challenge_resolved")
                if kind == "challenge_resolved" and isinstance(event.get("truthful"), bool):
                    stats = self.stats[(placer, count)]
                    stats[0] += 1
                    stats[1] += int(not event["truthful"])
        self.processed = len(history)

    def candidates(self, observation):
        hand = observation["hand"]
        held = Counter(c["rank"] for c in hand)
        pile = min(observation.get("pile_size", 0), 40) / 20.
        if observation["phase"] == "CHALLENGE":
            claim = observation["pending_claim"]
            count, rank, placer = claim["count"], claim["claimed_rank"], claim["seat"]
            copies = 4 * observation.get("game_config", {}).get("decks", 1)
            if count > copies - held[rank]:
                return [({"type": "challenge"}, [0.] * len(self.weights))]
            samples, bluffs = self.stats[(placer, count)]
            posterior = (bluffs + .6) / (samples + 5.)
            sizes = observation["hand_sizes"]
            vector = [1., posterior, held[rank] / copies, count / 4., pile,
                      float(claim.get("provisional_finish", False)), float(len(hand) <= 3),
                      1. / max(1, sizes[placer]), min(samples, 20) / 20.]
            return [({"type": "accept"}, [0.] * len(self.weights)),
                    ({"type": "challenge"}, [0.] * len(PLACE_FEATURES) + vector)]
        groups = defaultdict(list)
        for card in hand:
            groups[card["rank"]].append(card)
        actions = []
        # Safe truthful groups are always available. Constructed bluff candidates
        # use only currently owned references; the count is the actual batch size.
        for rank, cards in groups.items():
            actions.append({"type": "place", "card_refs": [c["card_ref"] for c in cards[:4]], "claimed_rank": rank})
        ordered = sorted(hand, key=lambda c: (held[c["rank"]], RANKS.index(c["rank"])))
        for count in (2, 3):
            if len(ordered) >= count:
                for rank in groups:
                    actions.append({"type": "place", "card_refs": [c["card_ref"] for c in ordered[:count]], "claimed_rank": rank})
        call_rate = (self.our_outcomes[1] + 1.) / (self.our_outcomes[0] + 4.)
        by_ref = {c["card_ref"]: c for c in hand}
        result = []
        for action in actions:
            refs = set(action["card_refs"])
            rank = action["claimed_rank"]
            truthful = all(by_ref[ref]["rank"] == rank for ref in refs)
            remaining = Counter(c["rank"] for c in hand if c["card_ref"] not in refs)
            vector = [len(refs), float(truthful), float(not truthful), float(not truthful) * pile,
                      float(not truthful) * call_rate, float(not truthful and len(refs) == len(hand)),
                      len(remaining) / 13., max(remaining.values(), default=0) / 4., held[rank] / 4.]
            result.append((action, vector + [0.] * len(CHALLENGE_FEATURES)))
        return result

    def act(self, observation):
        self.update(observation)
        return max(self.candidates(observation), key=lambda candidate:
                   sum(w * value for w, value in zip(self.weights, candidate[1])))[0]

    def save(self, path, metadata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"format": "triad-bluff-linear-v1", "features": PLACE_FEATURES + CHALLENGE_FEATURES,
            "weights": self.weights, "training": metadata}, indent=2) + "\n")

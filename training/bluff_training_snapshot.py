"""Training-only simulator clone sharing verified immutable prior records.

The pinned SDK freezes physical Card/GameConfig values and only appends to
public/private histories and debug traces. Verified read-only actors never
change those prior records. Mutable containers, current claims, RNG state and
actor evidence still deepcopy normally. Full simulator state remains label-only.
"""
import copy


def clone_snapshot(snapshot):
    state, _ = snapshot
    if not type(state.config).__dataclass_params__.frozen:
        raise TypeError('snapshot sharing requires the pinned frozen GameConfig')
    memo = {id(state.config): state.config}
    for cards in (*state.hands, state.pile, state.pending_batch):
        for card in cards:
            if not type(card).__dataclass_params__.frozen:
                raise TypeError('snapshot sharing requires immutable physical cards')
            memo[id(card)] = card
    for records in (state.public_history, *state.private_history, state.trace):
        for record in records:
            memo[id(record)] = record
    return copy.deepcopy(snapshot, memo)

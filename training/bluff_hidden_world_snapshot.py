"""Training-only public-consistent latent-card augmentation; not a posterior.

One deck only. All card identities ever observed by our seat or publicly revealed
stay fixed. Other identities are permuted coherently throughout the simulated
private card histories and current ownership. Future actors still see only their
own/public observations. Historical opponent-policy likelihoods and sealed
votes are not reweighted, so uniform augmentation is not an exact belief model.
"""
import copy
import random
from train_bluff_league import engine,make_policy
from bluff_training_observation import observe_read_only


def card_identities(value):
    if isinstance(value,dict):
        if 'rank' in value and 'suit' in value:
            yield (value['rank'],value['suit'])
        for child in value.values():yield from card_identities(child)
    elif isinstance(value,(list,tuple)):
        for child in value:yield from card_identities(child)


def resample_snapshot(snapshot,own,specifications,world_seed):
    original,_=snapshot
    if original.config.decks!=1:
        raise ValueError('physical identity augmentation is currently one-deck only')
    if len(specifications)!=original.config.players:
        raise ValueError('one original actor specification per seat required')
    state=copy.deepcopy(original)
    cards=[card for hand in original.hands for card in hand]+list(original.pile)
    by_uid={card.uid:card for card in cards}
    by_identity={(card.rank,card.suit):card for card in cards}
    assert len(cards)==len(by_uid)==len(by_identity)==52
    fixed=set(card_identities(original.public_history))|set(card_identities(original.private_history[own]))
    fixed.update((card.rank,card.suit) for card in original.hands[own])
    unknown=sorted((card for card in cards if (card.rank,card.suit) not in fixed),key=lambda card:card.uid)
    shuffled=unknown.copy();random.Random(world_seed).shuffle(shuffled)
    replacements={card.uid:replacement for card,replacement in zip(unknown,shuffled)}
    mapping={card.uid:replacements.get(card.uid,card) for card in cards}
    identity_mapping={(card.rank,card.suit):mapping[card.uid] for card in cards}
    state.hands=[[mapping[card.uid] for card in hand] for hand in original.hands]
    state.pile=[mapping[card.uid] for card in original.pile]
    state.pending_batch=[mapping[card.uid] for card in original.pending_batch]
    def transform(value):
        if isinstance(value,dict):
            result={key:transform(child) for key,child in value.items()}
            if 'rank' in result and 'suit' in result:
                replacement=identity_mapping[(value['rank'],value['suit'])]
                result.update(replacement.to_public_dict())
            return result
        if isinstance(value,list):
            result=[transform(child) for child in value]
            if result and all(isinstance(child,dict) and 'rank' in child and 'suit' in child for child in result):
                result.sort(key=lambda card:by_identity[(card['rank'],card['suit'])].sort_key())
            return result
        if isinstance(value,tuple):return tuple(transform(child) for child in value)
        return value
    state.private_history=transform(original.private_history)
    # Trace hN references encode the original hidden hands' canonical ordering.
    # They are evaluator diagnostics, not an observation or rollout dependency.
    # Remove the historical diagnostic trace rather than present it as a replay
    # of augmented private hands. Future transitions create a new branch trace.
    state.trace=[]
    engine._check_invariants(state)
    assert observe_read_only(state,own)==observe_read_only(original,own)
    actors=[make_policy(spec,world_seed*10+seat) for seat,spec in enumerate(specifications)]
    for seat,actor in enumerate(actors):
        if hasattr(actor,'update'):actor.update(observe_read_only(state,seat))
    return (state,actors),{'unknown_physical_cards':len(unknown),
                           'changed_identity_positions':sum(card.uid!=mapping[card.uid].uid for card in cards),
                           'fixed_own_public_identities':len(fixed),
                           'historical_policy_likelihood_weighted':False}

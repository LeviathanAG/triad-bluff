"""Training-only observation shortcut for verified read-only simulator actors.

All fields still come from the SDK's filtered observe(). The engine is not
advanced while histories are temporarily detached. Only the requested seat's
private history is returned. Never import this helper into an inference image.
"""
from triad_bluff import engine


def observe_read_only(state, seat):
    public = state.public_history
    private = state.private_history[seat]
    try:
        state.public_history = []
        state.private_history[seat] = []
        observation = engine.observe(state, seat)
    finally:
        state.public_history = public
        state.private_history[seat] = private
    observation['public_history'] = public
    observation['private_history'] = private
    return observation

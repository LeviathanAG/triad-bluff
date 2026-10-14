"""Frozen residual neural scorer; reconstruct filtered evidence after every restart."""

import json
from pathlib import Path
from bluff_agent.strategy import accept, challenge, place
from learned_residual import Policy

_MODEL = json.loads((Path(__file__).resolve().parent.parent / "models" / "policy.json").read_text())
# Validate the entire finite parameter schema before initialization responds.
Policy(_MODEL)

def select_action(observation, agent_state=None):
    if observation["phase"] == "FINISHED":
        return accept(), None
    # Complete own/public history is authoritative across process restarts.
    return Policy(_MODEL).act(observation), None

__all__ = ["accept", "challenge", "place", "select_action"]

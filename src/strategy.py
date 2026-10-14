"""Frozen learned action ranker; reconstruct public evidence after every restart."""

import json
import math
from pathlib import Path

from bluff_agent.strategy import accept, challenge, place
from learned_policy import FEATURE_NAMES, Policy

_MODEL = json.loads((Path(__file__).resolve().parent.parent / "models" / "policy.json").read_text())
if _MODEL.get("format") != "triad-bluff-linear-v2" or _MODEL.get("features") != FEATURE_NAMES:
    raise ValueError("Incompatible Bluff policy checkpoint")
_WEIGHTS = _MODEL["weights"]
if len(_WEIGHTS) != len(FEATURE_NAMES) or not all(math.isfinite(float(w)) for w in _WEIGHTS):
    raise ValueError("Invalid Bluff policy parameters")


def select_action(observation, agent_state=None):
    if observation["phase"] == "FINISHED":
        return accept(), None
    # The complete filtered history is authoritative. Reconstructing it makes
    # inference identical in persistent and restarted processes, without hidden
    # opponent state or a mutable checkpoint that can diverge from the history.
    return Policy(_WEIGHTS).act(observation), None

__all__ = ["accept", "challenge", "place", "select_action"]

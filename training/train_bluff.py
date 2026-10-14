"""Optimize Bluff policy weights on mixed real SDK games, then test separate seeds."""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
import random
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
SDK = Path(os.environ.get("BLUFF_SDK", ROOT.parent / "bluffSDK"))
sys.path.insert(0, str(SDK / "local_evaluator" / "src"))
from triad_bluff.baselines import HeuristicAgent, RandomAgent, TruthfulAgent, AggressiveChallengeAgent
from triad_bluff import engine
from triad_bluff.engine import GameConfig
from bluff_policy import INITIAL_WEIGHTS, Policy


def evaluate_job(job):
    weights, seeds, regime = job
    points, wins = [], 0
    classes = [HeuristicAgent, RandomAgent, TruthfulAgent, AggressiveChallengeAgent]
    for index, seed in enumerate(seeds):
        seat = index % 4
        opponent_rng = random.Random(seed ^ 0x5452494144)
        agents = []
        for other in range(4):
            if other == seat:
                agents.append(Policy(weights))
            else:
                name = opponent_rng.choices(range(len(classes)), weights=(3, 1, 4, 1))[0] if regime == "mixed" else ("heuristic", "random", "truthful", "aggressive").index(regime)
                cls = classes[name]
                agents.append(cls() if cls is TruthfulAgent else cls(seed * 10 + other))
        state = engine.new_game(GameConfig(max_turns=1000), seed)
        # Once our finish is confirmed, its reward is fixed. Continuing the
        # remaining opponents cannot change it and can be very costly with
        # three random bluffers. This only shortens training simulation.
        while state.outcome is None and seat not in state.finished_order:
            decision = engine.next_decision(state)
            observation = engine.observe(state, decision.seat)
            payload = agents[decision.seat].act(observation)
            transition = engine.step(state, decision, engine.decode_action(state, decision, payload))
            if not transition.ok:
                raise RuntimeError("training policy produced an illegal action")
        if seat in state.finished_order:
            place = state.finished_order.index(seat)
            points.append(state.config.position_points[place])
            wins += int(place == 0)
        else:
            points.append(state.outcome.points[seat])
            wins += int(bool(state.outcome.placements) and state.outcome.placements[0] == seat)
    return {"points": float(np.mean(points)), "win_rate": wins / len(seeds), "games": len(seeds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--eval-games", type=int, default=128)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", type=Path, default=ROOT / "bluff-output")
    args = parser.parse_args()
    if min(args.iterations, args.population, args.games, args.eval_games, args.workers) < 1:
        parser.error("counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    mean = np.asarray(INITIAL_WEIGHTS, dtype=float)
    sigma = np.full(len(mean), 1.5)
    best = mean.copy()
    best_validation = -float("inf")
    history = []
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for generation in range(args.iterations):
            population = rng.normal(mean, sigma, (args.population, len(mean)))
            population[0] = mean
            if args.population > 1:
                population[1] = best
            seeds = list(range(10_000 + generation * args.games, 10_000 + (generation + 1) * args.games))
            scores = list(pool.map(evaluate_job, [(w.tolist(), seeds, "mixed") for w in population]))
            ranked = sorted(range(len(scores)), key=lambda i: (scores[i]["points"], scores[i]["win_rate"]), reverse=True)
            elite = population[ranked[:max(2, args.population // 4)]]
            mean = .3 * mean + .7 * elite.mean(axis=0)
            sigma = np.maximum(.15, .3 * sigma + .7 * elite.std(axis=0))
            candidate = population[ranked[0]]
            validation = list(pool.map(evaluate_job, [(candidate.tolist(), list(range(2_000_000, 2_000_000 + args.eval_games)), "mixed")]))[0]
            if validation["points"] > best_validation:
                best, best_validation = candidate.copy(), validation["points"]
            entry = {"generation": generation + 1, "training": scores[ranked[0]], "validation": validation,
                     "elapsed_seconds": round(time.time() - started, 1)}
            history.append(entry)
            print(json.dumps(entry), flush=True)
            Policy(best).save(args.output / "bluff-policy.json", {"method": "cross-entropy policy optimization",
                "seed": args.seed, "generations": generation + 1, "population": args.population,
                "games_per_candidate": args.games, "validation": best_validation,
                "sdk_commit": "96da8042c285af4e4f43655f2ad99592bd49b8e6"})
            (args.output / "training-progress.json").write_text(json.dumps(history, indent=2) + "\n")
        seeds = list(range(3_000_000, 3_000_000 + args.eval_games))
        regimes = ("heuristic", "random", "truthful", "aggressive", "mixed")
        jobs = [(w, seeds, regime) for regime in regimes for w in (best.tolist(), INITIAL_WEIGHTS)]
        scores = list(pool.map(evaluate_job, jobs))
        report = {regime: {"learned": scores[2*i], "initial": scores[2*i+1]} for i, regime in enumerate(regimes)}
        (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"held_out_evaluation": report}), flush=True)


if __name__ == "__main__":
    main()

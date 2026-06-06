#!/usr/bin/env python3
"""Timed regression benchmarks for RobotMath2030."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def bench_operator() -> dict:
    from robotmath.neural_operators import OperatorTrainConfig, make_operator_dataset, train_deeponet
    from robotmath.neural_operators.fno_mass_spring import fno_trajectory_mse, train_fno
    from robotmath.physics.mass_spring import MassSpringParams, simulate

    train_u, times, train_y = make_operator_dataset(n_samples=256, seed=0)
    test_u, _, test_y = make_operator_dataset(n_samples=64, seed=1)
    cfg = OperatorTrainConfig(epochs=80, seed=0)

    t0 = time.perf_counter()
    deeponet, _ = train_deeponet(train_u, times, train_y, cfg)
    deeponet_train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    fno, _ = train_fno(train_u, times, train_y, cfg)
    fno_train_s = time.perf_counter() - t0

    from robotmath.neural_operators import operator_trajectory_mse

    deeponet_mse = operator_trajectory_mse(deeponet, test_u, times, test_y)
    fno_mse = fno_trajectory_mse(fno, test_u, times, test_y)

    params = MassSpringParams()
    steps = len(times) - 1
    n = 200
    t0 = time.perf_counter()
    for u in test_u[:n]:
        simulate(params, float(u[0]), float(u[1]), steps)
    sim_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    deeponet.predict_trajectory(test_u[:n], times)
    infer_s = time.perf_counter() - t0

    return {
        "deeponet_test_mse": deeponet_mse,
        "fno_test_mse": fno_mse,
        "deeponet_train_seconds": deeponet_train_s,
        "fno_train_seconds": fno_train_s,
        "simulator_batch_seconds": sim_s,
        "deeponet_batch_seconds": infer_s,
        "batch_size": n,
    }


def bench_world_model_replan() -> dict:
    from miniworlds.grid_world import GridWorld
    from robotmath.world_models import WorldModelConfig, rollout_env, train_world_model

    env = GridWorld(layout="easy")
    t0 = time.perf_counter()
    model, _ = train_world_model(env.collect_random_dataset(800, 0), WorldModelConfig(epochs=40))
    train_s = time.perf_counter() - t0

    goal = env.reset()
    goal_obs = env.observation()
    _, closed = rollout_env(env, model, goal_obs, replan=True)
    _, open_ok = rollout_env(env, model, goal_obs, replan=False, horizon=4)

    return {
        "train_seconds": train_s,
        "closed_loop_success": closed,
        "open_loop_success": open_ok,
    }


def main() -> int:
    results = {
        "operator": bench_operator(),
        "world_model": bench_world_model_replan(),
    }
    print(json.dumps(results, indent=2))
    ok = (
        results["operator"]["deeponet_test_mse"] < 0.08
        and results["operator"]["fno_test_mse"] < 0.08
        and results["world_model"]["closed_loop_success"] is True
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

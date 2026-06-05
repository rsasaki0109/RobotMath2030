# Chapter 02 — Tiny Lie Graph Optimizer

## Robotics context

Pose graph SLAM represents robot poses as **nodes** and relative measurements (odometry, loop closures) as **edges**.
Optimization finds poses that best explain all edges — this is the backbone of modern mapping backends (GTSAM, g2o, Ceres).

## What this demo shows

- A 4-pose square loop with odometry drift and a loop closure
- **Lie residual** optimization (correct linearization on SE(2))
- **Euclidean residual** optimization (naive — often slower or wrong)

## Failure cases

| Mistake | Symptom |
|---------|---------|
| Euclidean (dx, dy, dθ) residual | Slow convergence or wrong minimum at large angles |
| No fixed node | Whole graph floats (gauge freedom) |
| Bad loop init | Local minimum, map "folds" |

## Run

```bash
python chapters/02_tiny_lie_graph_optimizer/demo.py
```

## Next

→ [Chapter 03 — Retraction vs projection](../03_retraction_vs_projection/)


# RobotMath2030

**RobotMath2030 is an executable visual guide to the mathematics behind future robotics, Physical AI, world models, and humanoids.**

Not a textbook. Not a paper reimplementation repo.
Learn **why** the math appears in robot maps, poses, policies, contact, and world models — with tiny Python code and interactive visualizations.

> Read this **after** [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics), **before** GTSAM / Ceres / LeRobot production stacks.

[日本語 README](README_ja.md)

## Demos at a glance

| Lie pose graph | Sinkhorn point clouds | Multimodal actions |
|:--:|:--:|:--:|
| ![Pose graph loop closure](assets/animations/pose_graph_loop_closure.gif) | ![Sinkhorn OT](assets/animations/sinkhorn_point_clouds.gif) | ![Diffusion policy 2D](assets/animations/diffusion_policy_2d.gif) |
| Ch.02 — SE(2) SLAM | Ch.05 — optimal transport | Ch.07 — diffusion policy |

Regenerate GIFs: `pip install -e ".[torch]" && python scripts/render_all_gifs.py`

## Why RobotMath2030?

When you read robotics papers you keep hitting terms like:

`Wasserstein distance` · `SE(3)-equivariant policy` · `score-based policy` · `latent world model` · `differentiable contact` · `manifold optimization`

Standard math courses teach definitions. They rarely explain **which robot situation** forces that math — or **what breaks** when you treat pose as a Euclidean vector.

RobotMath2030 fills that gap:

```
math definition  →  minimal runnable code  →  robotics context  →  failure cases
```

## First demos

| Chapter | What you learn |
|---------|----------------|
| [01 — Pose is not a vector](chapters/01_pose_is_not_vector/) | SE(2) composition, exp/log, Euler-angle failure |
| [02 — Tiny Lie graph optimizer](chapters/02_tiny_lie_graph_optimizer/) | Pose graph SLAM in ~50 lines; Euclidean vs Lie residuals |
| [05 — Sinkhorn for point clouds](chapters/05_sinkhorn_point_clouds/) | Soft correspondence for maps and scans; OT vs naive matching |
| [06 — Wasserstein map evaluation](chapters/06_wasserstein_map_evaluation/) | Map drift / ghost obstacles; L2 grid MSE vs W2 |
| [07 — Diffusion policy 2D](chapters/07_diffusion_policy_2d/) | Multimodal trajectories; mean regression vs diffusion |
| [09 — Differentiable physics](chapters/09_differentiable_physics/) | Mass-spring ID; hard contact breaks gradients |
| [10 — Tiny world model](chapters/10_tiny_world_model/) | Latent dynamics + imagination MPC; open-loop drift |

```bash
pip install -e ".[torch]"
python chapters/10_tiny_world_model/demo.py
```

## 3-month roadmap

```
Month 1  Geometry of State      Lie groups · pose graphs · manifold optimization
Month 2  Geometry of Distribution  Sinkhorn OT · diffusion · natural gradient
Month 3  Geometry of Dynamics     Differentiable physics · tiny world model
```

See [docs/roadmap.md](docs/roadmap.md) for the full plan.

## Repository structure

```
robotmath/     Tiny reference implementations (Lie, OT, diffusion, …)
chapters/      Runnable lessons with concept.yaml metadata
miniworlds/    Synthetic worlds for demos
docs/          Concept maps and references
tests/         Math property tests
```

## Install

```bash
git clone https://github.com/rsasaki0109/RobotMath2030.git
cd RobotMath2030
pip install -e ".[dev]"
pytest
```

Core dependencies: **NumPy**, **Matplotlib**. **PyTorch** is required for Ch.07+ (`pip install -e ".[torch]"`).

## Positioning

| Project | Role |
|---------|------|
| PythonRobotics | Classical robotics algorithms in Python |
| **RobotMath2030** | **Math OS for reading future robotics papers** |
| GTSAM / Ceres / Sophus | Production geometry & optimization |
| LeRobot | Robot learning models, data, tools |

## License

MIT — see [LICENSE](LICENSE).

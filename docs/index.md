# RobotMath2030

**Executable visual guide to the mathematics behind future robotics, Physical AI, world models, and humanoids.**

Not a textbook. Not a paper reimplementation repo.

Each chapter connects:

```
math definition  →  minimal runnable code  →  robotics context  →  failure cases
```

## Start here

| Path | Description |
|------|-------------|
| [Chapter guide](chapters.md) | All 12 demos with run commands |
| [Math map](math_map.md) | Concept connections for paper reading |
| [Roadmap](roadmap.md) | MVP + Phase 2 status |

## Three pillars

**Geometry of State** — Lie groups, pose graphs, manifold optimization, RMPs

**Geometry of Distribution** — optimal transport, diffusion / flow matching, natural gradient, equivariance

**Geometry of Dynamics** — differentiable physics, tiny world models

## Quick install

```bash
git clone https://github.com/rsasaki0109/RobotMath2030.git
cd RobotMath2030
pip install -e ".[dev]"
pytest
pip install -e ".[torch]"   # Ch.07+ and Ch.12
python scripts/smoke_all_chapters.py
```

## Positioning

Read this **after** [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics),
**before** GTSAM / Ceres / LeRobot production stacks.

[GitHub repository](https://github.com/rsasaki0109/RobotMath2030) · [日本語 README](https://github.com/rsasaki0109/RobotMath2030/blob/main/README_ja.md)

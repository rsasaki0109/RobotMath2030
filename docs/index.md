# RobotMath2030

**Executable visual guide to the mathematics behind future robotics, Physical AI, world models, and humanoids.**

Each chapter connects:

```
math definition  →  minimal runnable code  →  robotics context  →  failure cases
```

!!! tip "Start here"
    **[15 Tiny Math Engines](landing.md)** — demo GIFs, all chapters by pillar, Colab links.

## Quick links

| Path | Description |
|------|-------------|
| [15 Tiny Math Engines](landing.md) | Landing page with GIF gallery |
| [Chapter guide](chapters.md) | All 15 demos with run commands |
| [Math map](math_map.md) | Concept connections for paper reading |
| [Roadmap](roadmap.md) | MVP through Phase 3 status |

## Three pillars

**Geometry of State** — Lie groups, pose graphs, manifold optimization, RMPs

**Geometry of Distribution** — optimal transport, diffusion / flow matching, natural gradient, equivariance, topology

**Geometry of Dynamics** — differentiable physics, world models, neural operators (DeepONet / FNO)

## Quick install

```bash
git clone https://github.com/rsasaki0109/RobotMath2030.git
cd RobotMath2030
pip install -e ".[dev]"
pytest
pip install -e ".[torch]"   # Ch.07+ and most of Ch.12–15
python scripts/smoke_all_chapters.py
```

## Positioning

Read this **after** [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics),
**before** GTSAM / Ceres / LeRobot production stacks.

[GitHub repository](https://github.com/rsasaki0109/RobotMath2030) · [日本語 README](https://github.com/rsasaki0109/RobotMath2030/blob/main/README_ja.md)

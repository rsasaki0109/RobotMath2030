# AGENTS

## Purpose

RobotMath2030 is an executable visual guide to the mathematics behind future robotics:
Physical AI, world models, Lie groups, optimal transport, diffusion policies, and more.

This is **not** a generic math textbook or a paper reimplementation repo.
Each chapter connects: **math → minimal code → robotics context → failure cases**.

## Scope

- Target readers: robotics engineers reading modern papers (Diffusion Policy, world models, SE(3)-equivariant nets, etc.)
- Positioning: the math layer **after** PythonRobotics, **before** GTSAM / Ceres / LeRobot production stacks
- Keep implementations tiny and readable (micrograd-style), not production libraries

## Repository layout

```
robotmath/     Reusable minimal reference implementations
chapters/      Learning content: README, demo.py, concept.yaml, assets
miniworlds/    Synthetic environments for demos
benchmarks/    Small diagnostic benchmarks
docs/          Concept maps, roadmap, references
tests/         Property tests for math correctness
```

## Operating rules

- Prefer NumPy + Matplotlib for core chapters; PyTorch only where needed (diffusion, world models)
- Every chapter must include `concept.yaml` with math, robotics, code paths, and failure cases
- Visual demos should run in under 30 seconds on CPU without GPU or robot hardware
- Separate **facts** (what the code does) from **inference** (what breaks on real robots) in chapter READMEs
- Do not expand scope into full OT libraries, SLAM frameworks, or robot learning stacks

## Chapter format

Each chapter under `chapters/` should contain:

1. `README.md` — intuition, robotics connection, failure cases
2. `demo.py` — runnable visualization or experiment
3. `concept.yaml` — machine-readable metadata for humans and AI agents
4. `assets/` — figures and GIFs (optional at first)

## MVP pillars (first 3 months)

1. **Geometry of State** — Lie groups, pose graphs, manifold optimization
2. **Geometry of Distribution** — optimal transport, diffusion / flow matching, information geometry
3. **Geometry of Dynamics** — differentiable physics, tiny latent world models

## Delivery standards

- English README is primary; Japanese README (`README_ja.md`) is supplementary
- Keep dependencies minimal: `numpy`, `matplotlib`; optional `torch` for later chapters
- CI should smoke-test imports and core math properties
- Never commit secrets, private datasets, or device-specific dumps

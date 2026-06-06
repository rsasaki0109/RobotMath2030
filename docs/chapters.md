# 10+2 Tiny Math Engines

All twelve chapters are runnable locally. Each follows:

**math → minimal code → robotics context → failure cases**

Machine-readable index: [concept_index.yaml](concept_index.yaml)

## Geometry of State (Month 1)

| # | Chapter | Run |
|---|---------|-----|
| 01 | [Pose is not a vector](chapters/01_pose_is_not_vector.md) | `python chapters/01_pose_is_not_vector/demo.py` |
| 02 | [Tiny Lie graph optimizer](chapters/02_tiny_lie_graph_optimizer.md) | `python chapters/02_tiny_lie_graph_optimizer/demo.py` |
| 03 | [Retraction vs projection](chapters/03_retraction_vs_projection.md) | `python chapters/03_retraction_vs_projection/demo.py` |
| 04 | [Riemannian motion policy (2D)](chapters/04_riemannian_motion_policy.md) | `python chapters/04_riemannian_motion_policy/demo.py` |

**You learn:** Lie groups, pose graphs, manifold updates, task-space motion metrics.

## Geometry of Distribution (Month 2)

| # | Chapter | Run |
|---|---------|-----|
| 05 | [Sinkhorn for point clouds](chapters/05_sinkhorn_point_clouds.md) | `python chapters/05_sinkhorn_point_clouds/demo.py` |
| 06 | [Wasserstein map evaluation](chapters/06_wasserstein_map_evaluation.md) | `python chapters/06_wasserstein_map_evaluation/demo.py` |
| 07 | [Diffusion policy 2D](chapters/07_diffusion_policy_2d.md) | `python chapters/07_diffusion_policy_2d/demo.py` |
| 08 | [Flow matching vs diffusion](chapters/08_flow_matching_vs_diffusion.md) | `python chapters/08_flow_matching_vs_diffusion/demo.py` |

Ch.07–08 require PyTorch: `pip install -e ".[torch]"`.

**You learn:** optimal transport, map comparison, multimodal policies, flow matching.

## Geometry of Dynamics (Month 3)

| # | Chapter | Run |
|---|---------|-----|
| 09 | [Differentiable physics](chapters/09_differentiable_physics.md) | `python chapters/09_differentiable_physics/demo.py` |
| 10 | [Tiny world model + planning](chapters/10_tiny_world_model.md) | `python chapters/10_tiny_world_model/demo.py` |

**You learn:** parameter ID through physics, contact gradient failure, latent dynamics, imagination MPC.

## Phase 2 — Distribution geometry extensions

| # | Chapter | Run |
|---|---------|-----|
| 11 | [Information geometry](chapters/11_information_geometry.md) | `python chapters/11_information_geometry/demo.py` |
| 12 | [SE(3)-equivariant preview](chapters/12_se3_equivariant_preview.md) | `python chapters/12_se3_equivariant_preview/demo.py` |

Ch.12 requires PyTorch.

**You learn:** natural gradients on policy manifolds, SE(3) symmetry for 3D perception.

## Phase 2 — Dynamics & operators

| # | Chapter | Run |
|---|---------|-----|
| 13 | [Neural operators](chapters/13_neural_operators.md) | `python chapters/13_neural_operators/demo.py` |

**You learn:** DeepONet-style operators, amortized simulation vs integrator loops.

Timed benchmarks: `python benchmarks/run_benchmarks.py`

## Suggested reading orders

**Paper-first (Physical AI):** 07 → 10 → 09 → 05 → 02 → 01

**SLAM / mapping engineer:** 01 → 02 → 03 → 05 → 06 → 04

**Full curriculum:** 01 through 12 in order

## Colab quick start

Open a notebook on [GitHub](https://github.com/rsasaki0109/RobotMath2030/tree/main/notebooks) or run any demo in Colab:

```python
!git clone https://github.com/rsasaki0109/RobotMath2030.git
%cd RobotMath2030
!pip install -q -e ".[torch]"
!MPLBACKEND=Agg python chapters/05_sinkhorn_point_clouds/demo.py
```

## Smoke test all chapters

```bash
pip install -e ".[all]"
python scripts/smoke_all_chapters.py
```

See [roadmap.md](roadmap.md) for star milestones and future extensions.

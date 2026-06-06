# Chapter 13 — Neural Operators (DeepONet Preview)

## Robotics context

Robot simulators integrate dynamics step-by-step. **Neural operators** learn a map
between function spaces — e.g., initial conditions → entire trajectories — so you can
**amortize** rollout cost in MPC, digital twins, or surrogate modeling.

DeepONet separates:

```
G[u](t) ≈ branch(u) · trunk(t)
```

`u` = initial state (branch), `t` = query time (trunk).

## What this demo shows

1. Random mass-spring initial conditions `(x₀, v₀)`
2. Train a tiny DeepONet on simulated position trajectories
3. Compare held-out MSE vs semi-implicit Euler integrator
4. Batch inference timing — operator vs looped simulation

## Failure cases

| Wrong idea | What breaks |
|------------|-------------|
| MLP on flattened (u, t) grid | Does not generalize across time resolutions |
| Operator trained on one IC | Not an operator — just a curve fit |
| No physics coverage in training | Extrapolates badly outside demo envelope |
| Replace simulator everywhere | Contact / hybrid dynamics need hybrids (Ch.09) |

## Run

Requires PyTorch:

```bash
pip install -e ".[torch]"
python chapters/13_neural_operators/demo.py
```

## Related

- Chapter 09 — differentiable physics (parameter ID, contact)
- Chapter 10 — world models (latent dynamics alternative)
- FNO / DeepONet literature

## Next

→ [Chapter 14 — Topology / TDA](../14_topology_tda/README.md)

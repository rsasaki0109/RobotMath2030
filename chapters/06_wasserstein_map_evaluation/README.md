# Chapter 06 — Wasserstein Map Evaluation

## Robotics context

SLAM produces **occupancy maps** over time. After loop closure or multi-session mapping you ask:

> How different are these two maps?

Pixel L2 on the full grid is common — but sparse maps dilute obstacle errors because most cells are empty.
**Wasserstein distance** compares maps as **distributions of occupied mass**.

## What this demo shows

1. Reference occupancy map vs drifted / ghost-obstacle variants
2. **L2 grid MSE** can look deceptively small on sparse maps
3. **W2 via Sinkhorn** tracks meaningful map changes and visualizes transport

## Failure cases

| Wrong idea | What breaks |
|------------|-------------|
| MSE over entire grid | Empty cells dominate; obstacle shift looks tiny |
| Pixel L2 without alignment | Penalizes drift as massive error without interpretability |
| Count occupied cells only | Misses *where* mass moved |

## Run

```bash
python chapters/06_wasserstein_map_evaluation/demo.py
```

## Related

- Chapter 05 — Sinkhorn point cloud matching
- POT — Python Optimal Transport

## Next

→ [Chapter 07 — Diffusion policy 2D](../07_diffusion_policy_2d/)

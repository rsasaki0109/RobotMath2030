# Chapter 05 — Sinkhorn for Point Clouds and Maps

## Robotics context

Robot perception often compares **distributions**, not ordered lists:

- LiDAR scan vs map
- Two occupancy maps after loop closure
- Demo trajectories vs policy rollouts

L2 on raw coordinates fails when point correspondence is unknown.
**Optimal transport** finds a soft matching that minimizes transport cost.

## What this demo shows

1. Two misaligned L-shaped point clouds (simulated partial scan)
2. **Sinkhorn** soft correspondence vs naive index matching
3. Transport cost convergence as entropic OT stabilizes

## Failure cases

| Wrong idea | What breaks |
|------------|-------------|
| Match point i to point i | Order differs across scans |
| Nearest neighbor only | Ambiguous symmetry, outliers |
| ε → 0 without stabilization | `exp(-C/ε)` underflows |
| ε too large | Everything matches everything |

## Run

```bash
python chapters/05_sinkhorn_point_clouds/demo.py
```

## Related production tools

- [POT](https://pythonot.github.io/) — Python optimal transport
- [GeomLoss](https://www.kernel-operations.io/geomloss/) — GPU Sinkhorn losses

## Next

→ Chapter 06 — Wasserstein map evaluation (planned)

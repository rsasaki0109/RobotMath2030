# Chapter 15 — Fourier Neural Operator vs DeepONet

## Robotics context

Chapter 13 showed **DeepONet** — branch encodes initial conditions, trunk encodes time.
**Fourier Neural Operators** learn in the **spectral domain** on a discretized field (here, time),
mixing global modes with fixed-size filters. Both amortize simulation; the inductive bias differs.

## What this demo shows

1. Same mass-spring dataset as Ch.13
2. Train tiny **FNO1D** and **DeepONet** with matched epochs
3. Compare test MSE, held-out trajectory, batch inference time

## Failure cases

| Wrong idea | What breaks |
|------------|-------------|
| Too few Fourier modes | Underfits fast oscillations / stiff segments |
| Flatten (u, t) grid into MLP | Loses resolution invariance story of FNO |
| Train on one trajectory | Not an operator — same pitfall as Ch.13 |
| Replace integrator everywhere | Contact / hybrid dynamics still need physics (Ch.09) |

## Run

Requires PyTorch:

```bash
pip install -e ".[torch]"
python chapters/15_fourier_neural_operator/demo.py
```

## Related

- Chapter 13 — DeepONet operator preview
- Chapter 09 — differentiable physics baseline
- FNO literature (Li et al.)

## Next

→ [Benchmark suite](../../benchmarks/README.md) · [docs site](https://rsasaki0109.github.io/RobotMath2030/)

# Benchmarks

Lightweight timed diagnostics for RobotMath2030 — not a production benchmark suite.

## Chapter smoke test

```bash
pip install -e ".[all]"
python scripts/smoke_all_chapters.py
```

Expected runtime: ~1–2 minutes on CPU.

## Timed regression suite

```bash
pip install -e ".[all]"
python benchmarks/run_benchmarks.py
```

Reports JSON metrics for:

| Benchmark | Metrics |
|-----------|---------|
| **Operator (Ch.13–15)** | DeepONet + FNO test MSE, train time, batch simulator vs operator inference |
| **World model (Ch.10)** | Train time, closed-loop vs open-loop success |

Exit code 0 when thresholds pass (DeepONet/FNO MSE < 0.08, closed-loop replan succeeds).

## Property tests

```bash
pytest -q
```

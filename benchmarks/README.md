# Benchmarks

Lightweight diagnostics for RobotMath2030 — not a production benchmark suite.

## Chapter smoke test

Runs every `chapters/*/demo.py` headless (`MPLBACKEND=Agg`):

```bash
pip install -e ".[all]"
python scripts/smoke_all_chapters.py
```

Expected runtime: ~1–2 minutes on CPU (PyTorch chapters dominate).

## Property tests

Math and behavior checks live under `tests/` and run in CI:

```bash
pytest -q
```

## Future

Phase 2 may add timed regression baselines (Sinkhorn iterations, diffusion sample collision rate, world-model replan success).

#!/usr/bin/env python3
"""Headless smoke test for all chapter demos."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHAPTERS: list[tuple[str, bool]] = [
    ("01_pose_is_not_vector", False),
    ("02_tiny_lie_graph_optimizer", False),
    ("03_retraction_vs_projection", False),
    ("04_riemannian_motion_policy", False),
    ("05_sinkhorn_point_clouds", False),
    ("06_wasserstein_map_evaluation", False),
    ("07_diffusion_policy_2d", True),
    ("08_flow_matching_vs_diffusion", True),
    ("09_differentiable_physics", True),
    ("10_tiny_world_model", True),
    ("11_information_geometry", False),
    ("12_se3_equivariant_preview", True),
    ("13_neural_operators", True),
    ("14_topology_tda", False),
    ("15_fourier_neural_operator", True),
]


def run_demo(slug: str) -> subprocess.CompletedProcess[str]:
    demo = ROOT / "chapters" / slug / "demo.py"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        [sys.executable, str(demo)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    need_torch = any(torch for _, torch in CHAPTERS)
    if need_torch:
        try:
            import torch  # noqa: F401
        except ImportError:
            print("PyTorch required for full smoke test. Install: pip install -e '.[torch]'")
            return 1

    failed: list[str] = []
    for slug, _ in CHAPTERS:
        result = run_demo(slug)
        if result.returncode == 0:
            print(f"OK  {slug}")
        else:
            print(f"FAIL {slug}")
            if result.stdout:
                print(result.stdout[-800:])
            if result.stderr:
                print(result.stderr[-800:])
            failed.append(slug)

    if failed:
        print(f"\n{len(failed)} chapter demo(s) failed.")
        return 1

    print(f"\nAll {len(CHAPTERS)} chapter demos passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

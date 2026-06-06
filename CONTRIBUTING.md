# Contributing to RobotMath2030

Thanks for your interest in RobotMath2030.

## Project goal

Each chapter follows:

**math → minimal code → robotics context → failure cases**

Please avoid:

- Production-scale reimplementations of GTSAM, POT, or robot learning stacks
- Large dependency additions without a strong chapter need
- Paper reimplementations without teaching value

## Development setup

```bash
git clone https://github.com/rsasaki0109/RobotMath2030.git
cd RobotMath2030
python -m venv .venv && source .venv/bin/activate   # optional
pip install -e ".[all]"
pytest -q
python scripts/smoke_all_chapters.py
```

Environment notes:

- Core chapters: NumPy + Matplotlib only
- Torch chapters: `pip install -e ".[torch]"` (Ch.07–10, 12–13, 15)
- Docs: `pip install -e ".[docs]"` then `python scripts/sync_chapter_docs.py && mkdocs serve`

## Before opening a PR

1. Run `pytest -q`
2. Run `python scripts/smoke_all_chapters.py` (set `MPLBACKEND=Agg` in CI)
3. If you changed chapter READMEs: `python scripts/sync_chapter_docs.py`
4. If you changed demo visuals: `python scripts/render_all_gifs.py`
5. Keep PRs focused on **one chapter** or **one tiny engine**

## New chapter checklist

- [ ] `chapters/NN_slug/README.md` — robotics context + failure-case table
- [ ] `chapters/NN_slug/demo.py` — runs headless on CPU in < ~60 s (torch chapters may take longer in CI)
- [ ] `chapters/NN_slug/concept.yaml` — metadata for agents and docs
- [ ] `tests/test_*.py` — property tests for core math behavior
- [ ] Entry in `docs/concept_index.yaml` + `scripts/smoke_all_chapters.py`
- [ ] `python scripts/sync_chapter_docs.py` + `mkdocs.yml` nav if needed

## Repository layout

| Path | Role |
|------|------|
| `robotmath/` | Tiny reusable math engines |
| `miniworlds/` | Synthetic worlds for demos |
| `chapters/` | Runnable lessons |
| `tests/` | Property tests |
| `scripts/smoke_all_chapters.py` | Headless demo runner (CI) |
| `benchmarks/run_benchmarks.py` | Timed regression thresholds (CI) |
| `docs/` | MkDocs site sources |

## Good first issues

Ideas that fit the project scope:

- Add a **failure-case panel** to an existing demo (before/after plot)
- Improve a **Colab notebook** (clearer markdown, faster default epochs)
- Add a **property test** for edge cases (angle wrap, OT marginals, etc.)
- Fix broken **internal links** after chapter renames
- Japanese summary paragraph in a chapter README (keep English primary)

Open a GitHub issue with the `good first issue` label if you want maintainers to triage scope first.

## Code style

- Match surrounding files (minimal diff, no drive-by refactors)
- Prefer pure NumPy unless the chapter topic requires PyTorch
- No secrets, device IDs, or private data in commits

## Questions

Open a [GitHub issue](https://github.com/rsasaki0109/RobotMath2030/issues) for larger design questions before a big PR.

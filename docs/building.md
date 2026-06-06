# Building the docs site

The site uses [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

## Local preview

```bash
pip install -e ".[docs]"
python scripts/sync_chapter_docs.py   # copy chapter READMEs → docs/chapters/
mkdocs serve
```

Open http://127.0.0.1:8000

## Production build

```bash
mkdocs build --strict
```

Output: `site/`

## CI / GitHub Pages

Push to `main` runs the [`docs` workflow](https://github.com/rsasaki0109/RobotMath2030/blob/main/.github/workflows/docs.yml).

Enable **Settings → Pages → Build and deployment → GitHub Actions** once per repository.

Published URL: https://rsasaki0109.github.io/RobotMath2030/

## Editing chapter pages

Edit the source under `chapters/<slug>/README.md`, then re-run `scripts/sync_chapter_docs.py`.
This also copies demo GIFs from `assets/animations/` into `docs/assets/animations/` for the landing page.
Do not hand-edit files in `docs/chapters/` (they are generated).

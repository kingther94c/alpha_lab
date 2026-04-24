# AGENTS.md

Instructions for any coding agent working in this repo.

## First steps

1. Read `README.md` and `CLAUDE.md` before planning changes.
2. Skim `src/alpha_lab/` to see which submodule already owns the concept you're touching.

## Rules

- **Prefer extending existing modules** over creating new ones. Only add a new file when the concept clearly doesn't fit anywhere existing.
- **Reusable logic must live in `src/alpha_lab/`.** Do not leave reusable code trapped inside notebooks.
- **Do not create a `helpers.py` / `utils.py` grab-bag.** Use the existing submodules (`utils/`, `analytics/`, `features/`, etc.).
- **Never modify files under `data/raw/`.** Treat raw data as read-only.
- **Never commit** secrets, `.env`, or anything under `data/private/`.
- **Keep functions small and pure** where practical. Add docstrings on public functions. Type annotations where they help.
- **Don't overengineer.** No config systems, plugin registries, or abstract base classes unless justified by real reuse.
- **Backtrader is optional.** Don't centralize the design around any single backtest framework.

## Making changes

- Keep each change focused. If you spot unrelated issues, note them in your summary rather than fixing them inline.
- When making a non-obvious choice (algorithm, library, placement), add a one-line note in your response explaining the tradeoff.
- Add a test only when the change introduces or relies on a core reusable helper.
- Run `pytest -q` and `ruff check .` before declaring a task done.

## What "done" looks like

- Code lives in the right submodule.
- Notebook is thin; logic lives in `src/alpha_lab/`.
- No stray debug prints, no dead code, no speculative abstractions.
- Tests (if any) pass.

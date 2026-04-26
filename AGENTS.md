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
- **Always clear notebook outputs before committing notebooks.**
- **Avoid future information in backtests and signals.** Signals, weights, filters, normalizers, model fits, and risk estimates must use only data available at the decision timestamp. If a notebook intentionally uses forward returns or full-sample statistics to study a relationship, label it explicitly as research/diagnostic and do not present it as tradable performance.
- **Trade after signal formation.** For reusable backtests, lag target weights or otherwise make the execution timing explicit. Do not let same-close signals earn same-close returns unless the notebook clearly documents that this is a non-tradable diagnostic.
- **Keep functions small and pure** where practical. Add docstrings on public functions. Type annotations where they help.
- **Don't overengineer.** No config systems, plugin registries, or abstract base classes unless justified by real reuse.
- **Backtrader is optional.** Don't centralize the design around any single backtest framework.

## Agent skills

Use repo-relevant skills deliberately:

- **GitHub skills**: use when asked to inspect PRs/issues, fix failing CI, address review comments, or publish local changes. Keep the commit scope tight and never include secrets, private data, raw data, or unrelated notebook output churn.
- **OpenAI docs skill**: use for OpenAI API / model / prompt-upgrade work. Rely on official OpenAI docs rather than remembered API details.
- **Google Drive / Sheets / Docs skills**: use only when the user explicitly references connected Drive files or wants data copied from external Sheets/Docs into the research workflow. Keep reusable analysis code in `src/alpha_lab/`, not in Drive-only artifacts.
- **Playwright/browser skills**: use only for browser automation, web-app verification, or inspecting pages that cannot be handled with normal HTTP/data loaders. Do not use browser automation for ordinary notebook or pandas debugging.
- **Notebook/research work**: no special external skill is required by default. Keep notebooks thin, clear outputs before commits, and move reusable data loading, optimization, analytics, and backtest logic into `src/alpha_lab/`.

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

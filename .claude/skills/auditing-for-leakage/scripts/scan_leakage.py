#!/usr/bin/env python3
"""Heuristic lookahead / leakage scanner for alpha_lab notebooks and modules.

A smoke detector, not a proof: it flags *patterns* commonly associated with future-information
leaks (negative shifts, centered windows, backfill, full-sample normalization, random splits, …)
so a human can confirm intent. A clean scan is necessary, not sufficient - pair it with the manual
checklist in SKILL.md.

Usage (project interpreter; stdlib only, no deps):
    python scan_leakage.py path/to/notebook.ipynb [more.ipynb ...] [module.py ...]

Exit code is 1 if any BLOCKER-severity pattern is found, else 0.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

BLOCKER, WARN, INFO = "BLOCKER", "WARN", "INFO"
_RANK = {BLOCKER: 0, WARN: 1, INFO: 2}

# (severity, id, regex, message, fix)
RULES = [
    (BLOCKER, "neg_shift", r"\.shift\(\s*-\s*\d",
     "Negative shift pulls future data into the present (forward-looking signal).",
     "Signals may only use shift(k>=0); put forward returns on the *target* via data.align.forward_returns."),
    (BLOCKER, "neg_pctchange", r"\.pct_change\(\s*(periods\s*=\s*)?-\s*\d",
     "Negative-period pct_change reads the future.",
     "Use a positive period; never negative for a signal input."),
    (BLOCKER, "center_true", r"center\s*=\s*True",
     "Centered rolling window includes future points around t.",
     "Use center=False (trailing window)."),
    (BLOCKER, "bfill", r"\.bfill\(|fillna\(\s*method\s*=\s*['\"](b?fill|backfill)",
     "Backfill copies future values backward into earlier timestamps.",
     "Use ffill, or leave NaN and handle only at the edge."),

    (WARN, "fullsample_zscore", r"-\s*[\w\.]+\.mean\(\)\s*\)\s*/\s*[\w\.]+\.std\(\)",
     "Looks like full-sample z-score (x - x.mean())/x.std() - a hidden lookahead.",
     "Use rolling/expanding stats: features.transforms.zscore(..., window=...)."),
    (WARN, "fullsample_quantile", r"\.quantile\(",
     "Full-sample .quantile() as a threshold peeks at the whole distribution.",
     "Use an expanding/rolling quantile computed through t."),
    (WARN, "scaler_fit", r"(StandardScaler|MinMaxScaler|RobustScaler|QuantileTransformer|PowerTransformer)\(",
     "Sklearn scaler - fitting on the full sample leaks evaluation-period statistics.",
     "Fit on the train fold only (ml.cv / data.holdout); transform out-of-sample."),
    (WARN, "fit_transform", r"\.fit_transform\(",
     "fit_transform over the whole array leaks if it spans the evaluation window.",
     "Fit on train, transform on test, inside a time-ordered split."),
    (WARN, "train_test_split", r"train_test_split\(",
     "Random train/test split breaks time order (uses future rows to train).",
     "Use a time-ordered/holdout split: data.holdout / ml.cv."),
    (WARN, "interpolate", r"\.interpolate\(",
     "Interpolation can blend future into past on signal inputs.",
     "Avoid on signal inputs; prefer ffill or leaving NaN."),

    (INFO, "rolling", r"\.rolling\(",
     "rolling window - confirm trailing (default center=False) and min_periods does not peek.",
     "Usually leak-safe; just verify."),
    (INFO, "resample", r"\.resample\(",
     "resample - verify label/closed convention (label='right' can stamp a bar with future-of-bar data) and that you trade after the bar closes.",
     "Usually leak-safe; just verify."),
    (INFO, "ffill", r"\.ffill\(|fillna\(\s*method\s*=\s*['\"]ffill",
     "ffill - leak-safe, but confirm you are not filling across a gap you would trade through.",
     "Usually fine; verify."),
    (INFO, "shift0", r"\.shift\(\s*0\s*\)",
     "shift(0) is a no-op - confirm execution is actually lagged by >=1 period.",
     "Lag weights before applying returns (run_backtest lags by 1)."),
    (INFO, "forward_returns", r"forward_returns\(",
     "forward_returns - fine to *score* a signal; never feed it back into the signal.",
     "Keep it on the evaluation side only."),
    (INFO, "fullsample_corr", r"\.(corr|cov)\(\)",
     "Full-sample corr/cov - fine as a diagnostic, a lookahead if it feeds weights.",
     "Use rolling/expanding if it feeds weights."),
]
_COMPILED = [(sev, rid, re.compile(rx), msg, fix) for sev, rid, rx, msg, fix in RULES]


def iter_code_lines(path):
    """Yield (location_label, line_text) for code lines in a .ipynb or .py file."""
    if path.endswith(".ipynb"):
        try:
            nb = json.loads(_read(path))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ! could not parse {path}: {e}", file=sys.stderr)
            return
        cell_no = 0
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            cell_no += 1
            src = cell.get("source", [])
            lines = src if isinstance(src, list) else src.splitlines(keepends=True)
            for i, line in enumerate(lines, 1):
                yield f"cell {cell_no}, line {i}", line.rstrip("\n")
    else:
        for i, line in enumerate(_read(path).splitlines(), 1):
            yield f"line {i}", line


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def scan_file(path):
    findings = []
    for loc, line in iter_code_lines(path):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        for sev, rid, rx, msg, fix in _COMPILED:
            if rx.search(line):
                findings.append((sev, rid, loc, line.strip(), msg, fix))
    findings.sort(key=lambda f: (_RANK[f[0]], f[2]))
    return findings


def main():
    ap = argparse.ArgumentParser(description="Heuristic lookahead/leakage scanner for alpha_lab.")
    ap.add_argument("paths", nargs="+", help=".ipynb or .py files to scan")
    args = ap.parse_args()

    totals = {BLOCKER: 0, WARN: 0, INFO: 0}
    for path in args.paths:
        findings = scan_file(path)
        print(f"\nLeakage audit: {path}")
        if not findings:
            print("  (no suspect patterns - run the manual checklist anyway)")
            continue
        for sev, rid, loc, code, msg, fix in findings:
            totals[sev] += 1
            print(f"  [{sev}] {loc}: {msg}  ({rid})")
            print(f"      > {code}")
            print(f"      fix: {fix}")

    n_b, n_w, n_i = totals[BLOCKER], totals[WARN], totals[INFO]
    print(f"\nSummary: {n_b} blocker, {n_w} warn, {n_i} info across {len(args.paths)} file(s).")
    if n_b:
        print("Headline performance is NOT trustworthy until the blockers are fixed and the study re-run.")
    elif n_w:
        print("No blockers, but review the warnings and complete the manual checklist before trusting results.")
    else:
        print("No automated suspects - the manual checklist is now the real audit.")
    sys.exit(1 if n_b else 0)


if __name__ == "__main__":
    main()

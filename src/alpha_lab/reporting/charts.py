"""Plotly charts with alpha_lab's default styling (``configs/reporting.yaml``).

Each function returns a ``plotly.graph_objects.Figure`` so callers can further
customize, display inline, or write to HTML via ``reporting/render.py`` later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from alpha_lab.analytics.returns import cumulative_returns, drawdown
from alpha_lab.utils.config import load_config

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _plotly_go():
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotly is required for alpha_lab.reporting charts. Install project dependencies with "
            '`pip install -e ".[dev]"`.'
        ) from exc
    return go


def _layout_defaults() -> dict:
    try:
        cfg = load_config("reporting").get("chart", {})
    except FileNotFoundError:
        cfg = {}
    return {
        "width": cfg.get("width", 900),
        "height": cfg.get("height", 500),
        "template": cfg.get("template", "plotly_white"),
    }


def equity_curve(returns: pd.Series | pd.DataFrame, name: str = "strategy") -> go.Figure:
    """Wealth index (starts at 1). Accepts a Series or DataFrame of streams."""
    go = _plotly_go()
    eq = cumulative_returns(returns)
    fig = go.Figure()
    if isinstance(eq, pd.Series):
        fig.add_scatter(x=eq.index, y=eq.values, name=name, mode="lines")
    else:
        for col in eq.columns:
            fig.add_scatter(x=eq.index, y=eq[col].values, name=str(col), mode="lines")
    fig.update_layout(title="Equity curve", yaxis_title="Wealth", **_layout_defaults())
    return fig


def drawdown_chart(returns: pd.Series) -> go.Figure:
    """Filled drawdown area."""
    go = _plotly_go()
    dd = drawdown(returns)
    fig = go.Figure()
    fig.add_scatter(x=dd.index, y=dd.values, fill="tozeroy", mode="lines", name="drawdown")
    fig.update_layout(title="Drawdown", yaxis_title="Drawdown", **_layout_defaults())
    return fig


def heatmap_monthly(returns: pd.Series) -> go.Figure:
    """Year × month heatmap of compounded monthly returns."""
    go = _plotly_go()
    from alpha_lab.backtest.metrics import monthly_table

    tbl = monthly_table(returns)
    month_cols = [c for c in tbl.columns if c != "YTD"]
    z = tbl[month_cols].values
    text = [[f"{v * 100:.1f}%" if pd.notna(v) else "" for v in row] for row in z]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=month_cols,
            y=tbl.index.astype(str),
            colorscale="RdYlGn",
            zmid=0,
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y} %{x}: %{text}<extra></extra>",
        )
    )
    fig.update_layout(title="Monthly returns", **_layout_defaults())
    return fig

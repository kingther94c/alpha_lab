import pandas as pd

from alpha_lab.reporting.charts import heatmap_monthly


def test_heatmap_monthly_includes_ytd_column_without_annotations():
    idx = pd.bdate_range("2023-01-03", "2024-12-31")
    returns = pd.Series(0.001, index=idx)

    fig = heatmap_monthly(returns)

    assert fig.data[0].x[-1] == "YTD"
    assert fig.data[0].texttemplate == "%{text}"
    assert len(fig.layout.annotations) == 0


def test_heatmap_monthly_uses_dynamic_height_for_long_histories():
    idx = pd.bdate_range("2000-01-03", "2026-12-31")
    returns = pd.Series(0.0001, index=idx)

    fig = heatmap_monthly(returns)

    assert fig.layout.height >= 120 + 26 * 27
    assert len(fig.layout.annotations) == 0

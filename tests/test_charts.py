import pandas as pd

from alpha_lab.reporting.charts import heatmap_monthly


def test_heatmap_monthly_includes_underlined_ytd_column():
    idx = pd.bdate_range("2023-01-03", "2024-12-31")
    returns = pd.Series(0.001, index=idx)

    fig = heatmap_monthly(returns)

    assert fig.data[0].x[-1] == "YTD"
    ytd_annotations = [annotation for annotation in fig.layout.annotations if annotation.x == "YTD"]
    assert len(ytd_annotations) == 2
    assert all(str(annotation.text).startswith("<u>") for annotation in ytd_annotations)

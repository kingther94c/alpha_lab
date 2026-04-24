"""Regression helpers — starting with rolling OLS.

TODO: multi-asset batch rolling OLS, robust regressions, exposure attribution.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools import add_constant


def rolling_ols(
    y: pd.Series,
    X: pd.Series | pd.DataFrame,
    window: int,
    *,
    add_intercept: bool = True,
) -> pd.DataFrame:
    """Rolling OLS of *y* on *X* over a fixed window.

    Returns a DataFrame of rolling coefficients (one column per regressor, plus
    ``const`` if ``add_intercept``).
    """
    X_ = X.to_frame() if isinstance(X, pd.Series) else X
    if add_intercept:
        X_ = add_constant(X_, has_constant="add")
    model = RollingOLS(y, X_, window=window).fit()
    return model.params

import pandas as pd
import pytest

from alpha_lab.analytics.risk import cvar


def test_cvar_averages_tail_observations():
    returns = pd.Series([-0.10, -0.05, 0.00, 0.05, 0.10])

    assert cvar(returns, q=0.4) == pytest.approx(-0.075)


def test_cvar_rejects_invalid_quantile():
    with pytest.raises(ValueError, match="q must be between 0 and 1"):
        cvar(pd.Series([0.01]), q=1.0)

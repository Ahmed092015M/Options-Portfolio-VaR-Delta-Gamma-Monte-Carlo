"""
Volatility forecasts used by the risk model (not the pricing volatility, which is the VIX).

Sections
--------
1. EWMA     ewma_vol     (Step 4.1)

The GJR-GARCH-t forecast is in ``garch``.
"""
import numpy as np
import pandas as pd

__all__ = ["ewma_vol"]


# =========================================================================== #
# 1. EWMA                                                                     #
# =========================================================================== #
def ewma_vol(returns, lam=0.94, init_window=250):
    """EWMA volatility forecast: sigma2_t = lam sigma2_{t-1} + (1 - lam) r_{t-1}^2 (uses data up to t-1).

    Parameters
    ----------
    returns : pd.Series
        Daily returns (decimal) or any other daily change (e.g. VIX log changes).
    lam : float
        Decay factor (RiskMetrics: 0.94, i.e. a half-life of about 11 days).
    init_window : int
        The first variance is the mean square of the first ``init_window`` returns.

    Returns
    -------
    pd.Series
        Daily volatility forecast for each day (decimal), same index as ``returns``.
    """
    x = np.asarray(returns, dtype=float)
    var = np.empty_like(x)
    var[0] = np.mean(x[:init_window] ** 2)
    for t in range(1, len(x)):
        var[t] = lam * var[t - 1] + (1 - lam) * x[t - 1] ** 2      # yesterday's return only
    return pd.Series(np.sqrt(var), index=returns.index, name="ewma_vol")

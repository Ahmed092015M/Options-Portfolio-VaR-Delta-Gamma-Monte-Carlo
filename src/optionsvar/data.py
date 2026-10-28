"""
Market data: S&P 500 (Yahoo Finance), VIX and the 3-month T-bill rate (FRED).

Sections
--------
1. Downloads        load_fred, load_yahoo           (Steps 1.2 - 1.4)
2. One calendar     align_market_data               (Step 1.5)

The pricing inputs of the book are the index level S, the implied volatility
sigma = VIX / 100 and the continuously compounded rate r = ln(1 + y / 100),
where y is the 3-month T-bill yield in % per year.
"""
import numpy as np
import pandas as pd

__all__ = ["FRED_URL", "load_fred", "load_yahoo", "align_market_data"]

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="


# =========================================================================== #
# 1. Downloads                                                                #
# =========================================================================== #
def load_fred(code, start="1990-01-01"):
    """Download one FRED series as a daily pd.Series without missing values.

    FRED writes missing days (holidays) as "."; they are read as NaN and dropped.

    Parameters
    ----------
    code : str
        FRED series code, e.g. "VIXCLS" (VIX) or "DGS3MO" (3-month T-bill, % per year).
    start : str
        First date kept (YYYY-MM-DD).

    Returns
    -------
    pd.Series
        Index = date, name = code, float values.
    """
    df = pd.read_csv(FRED_URL + code, na_values=".")
    s = pd.Series(df.iloc[:, 1].values, index=pd.to_datetime(df.iloc[:, 0]), name=code)
    s.index.name = "date"
    return s.loc[start:].dropna().astype(float)


def load_yahoo(ticker, start="1990-01-01"):
    """Download the daily close of one Yahoo Finance ticker as a pd.Series.

    ``yfinance`` is imported here, not at the top of the module, so that the rest
    of the package (and the unit tests) work without it.

    Parameters
    ----------
    ticker : str
        Yahoo ticker, e.g. "^GSPC" (S&P 500 price index).
    start : str
        First date requested (YYYY-MM-DD).

    Returns
    -------
    pd.Series
        Index = date (time zone removed), name = ticker; closing level.
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):          # newer yfinance: one column per ticker
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"
    return close.dropna().astype(float).rename(ticker)


# =========================================================================== #
# 2. One calendar                                                             #
# =========================================================================== #
def align_market_data(spx, vix, rate):
    """Put the index, VIX and the T-bill rate on the trading days common to the index and VIX.

    The rate is published on fewer days (bond-market holidays); on a missing day the
    last known value is used (forward fill), so no future information enters.

    Parameters
    ----------
    spx : pd.Series
        S&P 500 closes.
    vix : pd.Series
        VIX closes (% per year).
    rate : pd.Series
        3-month T-bill yield (% per year), e.g. FRED DGS3MO.

    Returns
    -------
    pd.DataFrame
        Index = date; columns 'S' (index level), 'VIX', 'sigma' (VIX / 100),
        'rate_pct' (T-bill, last known value) and 'r' (continuously compounded, decimal).
    """
    days = spx.index.intersection(vix.index)                   # days when both closed
    df = pd.DataFrame({"S": spx.loc[days], "VIX": vix.loc[days]})
    all_days = rate.index.union(days)
    df["rate_pct"] = rate.reindex(all_days).ffill().loc[days]  # last known rate, no look-ahead
    df["sigma"] = df["VIX"] / 100
    df["r"] = np.log(1 + df["rate_pct"] / 100)
    return df[["S", "VIX", "sigma", "rate_pct", "r"]]

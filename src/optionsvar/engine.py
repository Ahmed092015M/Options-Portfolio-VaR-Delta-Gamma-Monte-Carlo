"""
Out-of-sample VaR / ES engines for the options book.

Sections
--------
1. One day, index risk only        rolling_option_var        (Steps 7.1 - 7.2, 8.3)
2. Ten days, non-overlapping       block_hday_var            (Step 8.5)
3. One day, index + implied vol    rolling_option_var_vega   (Steps 9.1 - 9.2)

Timing (no look-ahead): the row dated t (the date of return t) uses the book built on
day t-1 (index, rate and VIX of t-1), the volatility forecast for day t (built from
returns up to t-1), and historical scenarios from the days before t.

Size: the engines price the 10-contract book at S_{t-1} and multiply every output by
size = notional / (100 x 10 x S_{t-1}), i.e. the constant-notional book of Step 3.4.

Output: a DataFrame indexed by date with MultiIndex columns (book, method, measure),
measures 'VaR' (99%), 'VaR97.5' and 'ES' (97.5%), all positive losses in USD.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from .book import MATURITY, MULTIPLIER, N_CONTRACTS, book_greeks, book_legs
from .garch import simulate_gjr_paths
from .var_methods import full_reval_pnl

__all__ = ["BOOKS", "rolling_option_var", "block_hday_var", "rolling_option_var_vega"]

BOOKS = ["long straddle", "short puts", "call spread", "total"]
MEASURES = ("VaR", "VaR97.5", "ES")

Z99, Z975 = norm.ppf(0.99), norm.ppf(0.975)
K_ES = norm.pdf(Z975) / 0.025                     # normal ES 97.5% = K_ES x std (2.338)


def _tail(x):
    """VaR 99%, VaR 97.5% and ES 97.5% (positive losses) of scenario P&Ls."""
    q1, q25 = np.quantile(x, [0.01, 0.025])
    return -q1, -q25, -x[x <= q25].mean()


def _normal(s):
    """VaR 99%, VaR 97.5% and ES 97.5% of a normal P&L with mean 0 and std s."""
    return Z99 * s, Z975 * s, K_ES * s


def _put(row, book, method, values):
    """Store the three measures of one (book, method) in a result row."""
    for m, v in zip(MEASURES, values):
        row[(book, method, m)] = v


def _frame(records, index):
    """Assemble result rows into a DataFrame with (book, method, measure) columns."""
    out = pd.DataFrame(records, index=index)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["book", "method", "measure"])
    return out


# =========================================================================== #
# 1. One day, index risk only                                                 #
# =========================================================================== #
def rolling_option_var(mkt, ewma, q=0.015, notional=5_000_000, start=1000, end=None,
                       fhs_window=1000, n_sim=20_000, seed=0, nu=None):
    """Daily out-of-sample VaR/ES of each sub-book with DN, DG, MC (normal), FHS and optionally MC-t.

    Methods (implied volatility held fixed in every scenario):

    - DN   : delta-normal, z |delta| S sigma_t
    - DG   : exact quantile of theta + delta dS + 0.5 gamma dS^2, dS normal
    - MC   : full revaluation on normal returns z x sigma_t
    - FHS  : full revaluation on the last ``fhs_window`` standardised returns x sigma_t
    - MC-t : full revaluation on standardised Student-t returns (only if ``nu`` is given)

    The same normal draws are used every day (common random numbers), so day-to-day
    changes in VaR come from the market, not from simulation noise.

    Parameters
    ----------
    mkt : pd.DataFrame
        Columns 'S', 'sigma', 'r'; index = date.
    ewma : pd.Series
        Daily volatility forecast of index returns (EWMA or GJR), indexed by the return dates (mkt.index[1:]).
    q : float
        Dividend yield.
    notional : float
        USD notional per N_CONTRACTS (constant-size book).
    start, end : int
        First and last+1 position in the return series to forecast.
    fhs_window : int
        Past returns used by FHS.
    n_sim : int
        Normal draws for DG and MC.
    seed : int
        Random seed.
    nu : pd.Series, optional
        Student-t degrees of freedom for each return date; if given, adds method 'MC-t'
        (t shocks built from the same uniforms every day, t quantiles cached by nu rounded to 0.5).

    Returns
    -------
    pd.DataFrame
        Index = forecast date; columns (book, method, measure).
    """
    ret = mkt["S"].pct_change().iloc[1:]
    R, SIG = ret.values, ewma.reindex(ret.index).values
    z = np.random.default_rng(seed).standard_normal(n_sim)
    u = norm.cdf(z)                                              # the same uniforms every day, for t shocks
    NU = None if nu is None else nu.reindex(ret.index).ffill().values
    t_cache = {}
    end = len(R) if end is None else end
    records = []
    for t in range(start, end):
        prev = mkt.iloc[t]                                       # day t-1 market (mkt has one more row than ret)
        S, r, sig, sd = prev["S"], prev["r"], prev["sigma"], SIG[t]
        size = notional / (MULTIPLIER * N_CONTRACTS * S)
        legs = book_legs(S)
        G = book_greeks(legs, S, MATURITY, r, q, sig)
        mc = full_reval_pnl(legs, S, r, q, sig, z * sd)
        zf = R[t - fhs_window:t] / SIG[t - fhs_window:t]         # standardised past returns
        fh = full_reval_pnl(legs, S, r, q, sig, zf * sd)
        if NU is not None:
            k = min(round(NU[t] * 2) / 2, 100.0)                 # nu rounded to 0.5
            if k not in t_cache:
                t_cache[k] = student_t.ppf(u, k) * np.sqrt((k - 2) / k)   # t shocks with variance 1
            mct = full_reval_pnl(legs, S, r, q, sig, t_cache[k] * sd)
        row = {}
        dS = z * S * sd                                          # index moves in points
        for b in BOOKS:
            g = G.loc[b]
            _put(row, b, "DN", _normal(abs(g["delta"]) * S * sd))
            _put(row, b, "DG", _tail(g["theta"] + g["delta"] * dS + 0.5 * g["gamma"] * dS ** 2))
            _put(row, b, "MC", _tail(mc[b]))
            _put(row, b, "FHS", _tail(fh[b]))
            if NU is not None:
                _put(row, b, "MC-t", _tail(mct[b]))
        records.append({k: v * size for k, v in row.items()})
    return _frame(records, ret.index[start:end])


# =========================================================================== #
# 2. Ten days, non-overlapping blocks                                         #
# =========================================================================== #
def block_hday_var(mkt, vol, params, oos_1d, q=0.015, notional=5_000_000, h=10, start=2000,
                   n_sim=20_000, seed=0):
    """h-day VaR/ES on non-overlapping blocks: DN, DG, sqrt(h) x FHS 1-day, MC normal, MC GJR paths.

    - DN, DG, MC normal : the daily volatility scaled by sqrt(h), theta x h
    - MC GJR            : h-day returns simulated along GJR-t paths (latest parameters), full revaluation
    - sqrt10 FHS        : sqrt(h) x the saved one-day FHS forecast (the regulatory shortcut)

    Blocks start every h days so the realised h-day P&Ls do not overlap and the
    breaches are independent under a correct model (needed by the tests).

    Parameters
    ----------
    mkt : pd.DataFrame
        Columns 'S', 'sigma', 'r'; index = date.
    vol : pd.Series
        GJR daily volatility forecast, indexed by return dates.
    params : pd.DataFrame
        GJR estimates (omega, alpha, gamma, beta, nu), indexed by the first day they apply to.
    oos_1d : pd.DataFrame
        One-day forecasts containing the 'FHS' method (output of ``rolling_option_var``).
    q, notional : float
        Dividend yield and USD notional per N_CONTRACTS.
    h : int
        Horizon in days.
    start : int
        First return position; blocks every h days from there.
    n_sim : int
        Draws for DG and the two Monte Carlo methods.
    seed : int
        Random seed (each block uses seed + block number for the GJR paths).

    Returns
    -------
    pd.DataFrame
        Index = first day of each block; columns (book, method, measure).
    """
    ret = mkt["S"].pct_change().iloc[1:]
    SIG = vol.reindex(ret.index).values
    z = np.random.default_rng(seed).standard_normal(n_sim)
    sq = np.sqrt(h)
    records, idx = [], []
    for b_no, t in enumerate(range(start, len(ret) - h + 1, h)):
        d = ret.index[t]
        prev = mkt.iloc[t]
        S, r, sig, sd = prev["S"], prev["r"], prev["sigma"], SIG[t]
        size = notional / (MULTIPLIER * N_CONTRACTS * S)
        p = params.loc[:d].iloc[-1][["omega", "alpha", "gamma", "beta", "nu"]].values.astype(float)
        legs = book_legs(S)
        G = book_greeks(legs, S, MATURITY, r, q, sig)
        mcn = full_reval_pnl(legs, S, r, q, sig, z * sd * sq, dt=h / 252)
        mcg = full_reval_pnl(legs, S, r, q, sig, simulate_gjr_paths(p, sd, h, n_sim, seed + b_no), dt=h / 252)
        row = {}
        dS = z * S * sd * sq
        for b in BOOKS:
            g = G.loc[b]
            _put(row, b, "DN", _normal(abs(g["delta"]) * S * sd * sq))
            _put(row, b, "DG", _tail(h * g["theta"] + g["delta"] * dS + 0.5 * g["gamma"] * dS ** 2))
            _put(row, b, "MC normal", _tail(mcn[b]))
            _put(row, b, "MC GJR", _tail(mcg[b]))
        rec = {k: v * size for k, v in row.items()}
        for b in BOOKS:                                          # sqrt(h) x the saved one-day FHS (already scaled)
            for m in MEASURES:
                rec[(b, "sqrt10 FHS", m)] = sq * oos_1d.loc[d, (b, "FHS", m)]
        records.append(rec)
        idx.append(d)
    return _frame(records, pd.DatetimeIndex(idx))


# =========================================================================== #
# 3. One day, index + implied volatility (vega)                               #
# =========================================================================== #
def rolling_option_var_vega(mkt, vol_idx, vol_vix, q=0.015, notional=5_000_000, start=2000, end=None,
                            fhs_window=1000):
    """Daily VaR/ES with joint FHS of index returns and VIX log changes (full revaluation, vega included).

    Each past day w gives one scenario with BOTH moves of that day, each rescaled by
    its own volatility ratio, so the index-vol dependence (correlation -0.7, VIX up on
    almost every crash day) is kept without modelling it:

        index return  = R_w / sigma_idx_w x sigma_idx_t
        VIX log change = V_w / sigma_vix_w x sigma_vix_t,   dsigma = sigma_{t-1} (e^{change} - 1)

    Parameters
    ----------
    mkt : pd.DataFrame
        Columns 'S', 'VIX', 'sigma', 'r'; index = date.
    vol_idx : pd.Series
        Daily volatility forecast of index returns (e.g. GJR), indexed by return dates.
    vol_vix : pd.Series
        Daily volatility forecast of VIX log changes (e.g. EWMA), indexed by return dates.
    q, notional : float
        Dividend yield and USD notional per N_CONTRACTS.
    start, end : int
        First and last+1 return position to forecast.
    fhs_window : int
        Past days used.

    Returns
    -------
    pd.DataFrame
        Index = forecast date; columns (book, 'FHS-vega', measure).
    """
    ret = mkt["S"].pct_change().iloc[1:]
    dlv = np.log(mkt["VIX"]).diff().iloc[1:]
    R, V = ret.values, dlv.values
    SI, SV = vol_idx.reindex(ret.index).values, vol_vix.reindex(ret.index).values
    end = len(R) if end is None else end
    records = []
    for t in range(start, end):
        prev = mkt.iloc[t]
        S, r, sig = prev["S"], prev["r"], prev["sigma"]
        size = notional / (MULTIPLIER * N_CONTRACTS * S)
        w = slice(t - fhs_window, t)
        ret_s = R[w] / SI[w] * SI[t]                             # index scenario returns
        dlv_s = V[w] / SV[w] * SV[t]                             # VIX log-change scenarios, same days
        dsig = sig * (np.exp(dlv_s) - 1)                         # implied-vol change of each scenario
        pl = full_reval_pnl(book_legs(S), S, r, q, sig, ret_s, dsigma=dsig)
        row = {}
        for b in BOOKS:
            _put(row, b, "FHS-vega", _tail(pl[b]))
        records.append({k: v * size for k, v in row.items()})
    return _frame(records, ret.index[start:end])

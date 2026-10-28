"""
GJR-GARCH(1,1) with standardised Student-t shocks for S&P 500 returns (the project's extension).

Sections
--------
1. Model           gjr_variance, gjr_t_loglik         (Step 8.1, Equations 13-14)
2. Estimation      fit_gjr_t, rolling_gjr_vol         (Steps 8.1 - 8.2)
3. Simulation      simulate_gjr_paths                 (Step 8.4)

Model (returns r_t in %, mean zero):
    r_t = sigma_t z_t,   z_t ~ standardised Student-t(nu) (variance 1)
    sigma2_t = omega + (alpha + gamma 1[r_{t-1} < 0]) r_{t-1}^2 + beta sigma2_{t-1}
gamma > 0 is the leverage effect: falls raise volatility more than rises.
Stationary if alpha + gamma / 2 + beta < 1 (half of the days are falls).
"""
import numpy as np
import pandas as pd
from scipy import optimize, special

__all__ = ["gjr_variance", "gjr_t_loglik", "fit_gjr_t", "rolling_gjr_vol", "simulate_gjr_paths"]

START = (0.02, 0.01, 0.15, 0.88, 7.0)                    # typical equity-index values (returns in %)


# =========================================================================== #
# 1. Model                                                                    #
# =========================================================================== #
def gjr_variance(r, omega, alpha, gamma, beta, s2_0=None):
    """Conditional variance path: s2_t = omega + (alpha + gamma 1[r_{t-1} < 0]) r_{t-1}^2 + beta s2_{t-1}.

    Parameters
    ----------
    r : array
        Returns (in %, mean zero).
    omega, alpha, gamma, beta : float
        GJR parameters.
    s2_0 : float, optional
        Starting variance (default: sample variance).

    Returns
    -------
    np.ndarray
        s2_t for each day, using returns up to t-1.
    """
    r = np.asarray(r, dtype=float)
    s2 = np.empty(len(r))
    s2[0] = r.var() if s2_0 is None else s2_0
    for t in range(1, len(r)):
        s2[t] = omega + (alpha + gamma * (r[t - 1] < 0)) * r[t - 1] ** 2 + beta * s2[t - 1]
    return s2


def gjr_t_loglik(params, r):
    """Log-likelihood of GJR-GARCH(1,1) with standardised Student-t shocks.

    log f(r_t) = ln G((nu+1)/2) - ln G(nu/2) - 0.5 ln(pi (nu-2)) - 0.5 ln s2_t
                 - (nu+1)/2 ln(1 + r_t^2 / ((nu-2) s2_t))

    Returns -1e10 for invalid parameters (negative variance or nu <= 2), so the
    optimiser is pushed back into the valid region.
    """
    omega, alpha, gamma, beta, nu = params
    r = np.asarray(r, dtype=float)
    s2 = gjr_variance(r, omega, alpha, gamma, beta)
    if np.any(s2 <= 0) or nu <= 2:
        return -1e10
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return float(np.sum(c - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(r ** 2 / (s2 * (nu - 2)))))


# =========================================================================== #
# 2. Estimation                                                               #
# =========================================================================== #
def fit_gjr_t(r, start=START):
    """Maximum-likelihood fit of GJR-GARCH(1,1)-t (SLSQP with bounds and constraints).

    Constraints: alpha + gamma / 2 + beta < 0.9999 (stationarity) and
    alpha + gamma >= 0 (the variance cannot fall after a large down day).

    Parameters
    ----------
    r : array
        Returns in % (mean zero).
    start : tuple
        Starting values (omega, alpha, gamma, beta, nu).

    Returns
    -------
    params : np.ndarray
        (omega, alpha, gamma, beta, nu).
    loglik : float
    success : bool
    """
    r = np.asarray(r, dtype=float)
    bounds = [(1e-6, 5.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0), (2.05, 100.0)]
    cons = [{"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2] / 2 - p[3]},   # stationarity
            {"type": "ineq", "fun": lambda p: p[1] + p[2]}]                         # alpha + gamma >= 0
    res = optimize.minimize(lambda p: -gjr_t_loglik(p, r), start, method="SLSQP",
                            bounds=bounds, constraints=cons)
    return res.x, -res.fun, bool(res.success)


def rolling_gjr_vol(returns, window=1000, refit_every=20, start=None):
    """Out-of-sample GJR-GARCH-t daily volatility forecasts, re-estimated every ``refit_every`` days.

    For each block of ``refit_every`` days starting at position j, the model is fitted on
    the ``window`` returns before j (warm start from the previous fit), and the variance
    recursion is run through the block: the forecast for day t uses returns up to t-1 only.

    Parameters
    ----------
    returns : pd.Series
        Daily returns (decimal).
    window : int
        Number of past days used for each estimation.
    refit_every : int
        Days between re-estimations.
    start : int, optional
        First position forecast (default: ``window``).

    Returns
    -------
    vol : pd.Series
        Daily volatility forecast (decimal) for each day from ``start``.
    params : pd.DataFrame
        Estimated (omega, alpha, gamma, beta, nu) and convergence, indexed by the first day of each block.
    """
    r = 100 * np.asarray(returns, dtype=float)                       # in %
    start = window if start is None else start
    vol = np.full(len(r), np.nan)
    rows, prev = [], START
    for j in range(start, len(r), refit_every):
        p, _, ok = fit_gjr_t(r[j - window:j], start=prev)
        prev = p
        end = min(j + refit_every, len(r))
        seg = r[j - window:end]                                      # window + block
        s2 = gjr_variance(seg, *p[:4], s2_0=r[j - window:j].var())
        vol[j:end] = np.sqrt(s2[window:]) / 100                       # forecasts for the block days
        rows.append({"date": returns.index[j], "omega": p[0], "alpha": p[1], "gamma": p[2],
                     "beta": p[3], "nu": p[4], "converged": ok})
    params = pd.DataFrame(rows).set_index("date")
    return pd.Series(vol, index=returns.index, name="gjr_vol").iloc[start:], params


# =========================================================================== #
# 3. Simulation                                                               #
# =========================================================================== #
def simulate_gjr_paths(params, sigma_first, h=10, n_paths=20_000, seed=0):
    """Simulate h-day compounded index returns from GJR-GARCH(1,1)-t (Equations 13-14).

    Along each path the variance is updated with the simulated returns, so a large
    fall raises the volatility of the following days: the h-day distribution gets
    fatter tails and a negative skew that sqrt(h) scaling cannot produce.

    Parameters
    ----------
    params : sequence
        (omega, alpha, gamma, beta, nu) estimated on returns in %.
    sigma_first : float
        Volatility forecast for the first simulated day (daily, decimal).
    h : int
        Horizon in days.
    n_paths : int
        Number of paths.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray, shape (n_paths,)
        Compounded h-day returns (decimal).
    """
    om, al, ga, be, nu = params
    rng = np.random.default_rng(seed)
    s2 = np.full(n_paths, (100 * sigma_first) ** 2)                  # variance in %^2
    growth = np.ones(n_paths)
    for _ in range(h):
        z = rng.standard_t(nu, n_paths) * np.sqrt((nu - 2) / nu)     # t shocks with variance 1
        r = np.sqrt(s2) * z                                          # return in %
        growth *= 1 + r / 100
        s2 = om + (al + ga * (r < 0)) * r ** 2 + be * s2             # GJR update along the path
    return growth - 1

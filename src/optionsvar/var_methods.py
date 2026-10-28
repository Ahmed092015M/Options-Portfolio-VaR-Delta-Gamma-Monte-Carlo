"""
VaR methods for an options book on one day: delta-normal, delta-gamma and full revaluation.

Sections
--------
1. Delta-normal                 var_delta_normal                                    (Step 5.1, Equation 7)
2. Delta-gamma                  delta_gamma_moments, var_cornish_fisher,
                                var_delta_gamma_mc                                  (Step 5.1, Equations 8-11)
3. Full revaluation, VaR / ES   full_reval_pnl, var_es                              (Step 5.2, Equation 12)

Conventions: P&L in USD, VaR and ES are positive losses; sigma_d is the daily
volatility of index returns (risk-model volatility, e.g. EWMA or GJR).
"""
import numpy as np
from scipy.stats import norm

from .book import MATURITY, book_value

__all__ = ["Z99", "var_delta_normal", "delta_gamma_moments", "var_cornish_fisher",
           "var_delta_gamma_mc", "full_reval_pnl", "var_es"]

Z99 = norm.ppf(0.99)                                    # 2.3263


# =========================================================================== #
# 1. Delta-normal                                                             #
# =========================================================================== #
def var_delta_normal(delta, S, sigma_d, p=0.01):
    """Delta-normal VaR (Equation 7): z |delta| S sigma_d.

    The P&L is approximated by delta x dS with dS ~ N(0, (S sigma_d)^2): linear and normal.

    Parameters
    ----------
    delta : float
        Book delta in USD per index point.
    S : float
        Index level.
    sigma_d : float
        Daily volatility of index returns (decimal).
    p : float
        Tail probability (0.01 -> VaR 99%).

    Returns
    -------
    float
        VaR as a positive loss (USD).
    """
    return norm.ppf(1 - p) * abs(delta) * S * sigma_d


# =========================================================================== #
# 2. Delta-gamma                                                              #
# =========================================================================== #
def delta_gamma_moments(delta, gamma, theta, S, sigma_d):
    """Mean, standard deviation and skewness of the quadratic P&L (Equation 9).

    P&L = theta + delta dS + 0.5 gamma dS^2, with dS ~ N(0, s^2) and s = S sigma_d:
    mean = theta + 0.5 gamma s^2, variance = delta^2 s^2 + 0.5 gamma^2 s^4,
    third central moment = 3 delta^2 gamma s^4 + gamma^3 s^6.

    Returns
    -------
    (float, float, float)
        Mean, standard deviation, skewness.
    """
    s = S * sigma_d                                         # std of the index move in points
    mean = theta + 0.5 * gamma * s ** 2
    var = delta ** 2 * s ** 2 + 0.5 * gamma ** 2 * s ** 4
    skew = (3 * delta ** 2 * gamma * s ** 4 + gamma ** 3 * s ** 6) / var ** 1.5
    return mean, np.sqrt(var), skew


def var_cornish_fisher(delta, gamma, theta, S, sigma_d, p=0.01):
    """Delta-gamma VaR with the Cornish-Fisher skewness correction (Equation 10).

    z_CF = z + (z^2 - 1) skew / 6;  VaR = -(mean + z_CF sd).
    Only a first-order correction: it can be badly wrong (even negative) when the
    skewness is large, e.g. for a long straddle.
    """
    mean, sd, skew = delta_gamma_moments(delta, gamma, theta, S, sigma_d)
    z = norm.ppf(p)                                         # -2.326 at 1%
    z_cf = z + (z ** 2 - 1) * skew / 6
    return -(mean + z_cf * sd)


def var_delta_gamma_mc(delta, gamma, theta, S, sigma_d, p=0.01, n=1_000_000, seed=0):
    """Delta-gamma VaR from the exact quantile of the quadratic P&L, by simulation (Equation 11).

    Parameters
    ----------
    delta, gamma, theta : float
        Book Greeks (USD); theta over the VaR horizon.
    S, sigma_d : float
        Index level and volatility of the index return over the horizon.
    p : float
        Tail probability.
    n : int
        Number of normal draws.
    seed : int
        Random seed.

    Returns
    -------
    float
        VaR as a positive loss (USD).
    """
    dS = np.random.default_rng(seed).normal(0.0, S * sigma_d, n)
    pnl = theta + delta * dS + 0.5 * gamma * dS ** 2
    return -np.quantile(pnl, p)


# =========================================================================== #
# 3. Full revaluation, VaR / ES                                               #
# =========================================================================== #
def full_reval_pnl(legs, S, r, q, sigma, returns, dsigma=0.0, T=MATURITY, dt=1 / 252):
    """P&L of each sub-book by full revaluation on index-return scenarios (Equation 12).

    Every scenario reprices every option with Black-Scholes: no Taylor approximation.

    Parameters
    ----------
    legs : pd.DataFrame
        Output of ``book_legs`` (strikes fixed today).
    S, r, q, sigma : float
        Today's index, rate, dividend yield and implied volatility.
    returns : array
        Scenario index returns over the horizon (decimal); floored at -99%
        (an index cannot fall more than 100%, and FHS can rescale a large past
        return beyond that in a very volatile period).
    dsigma : float or array
        Scenario change of implied volatility (decimal), 0 = unchanged;
        the new volatility is floored at 0.01%.
    T : float
        Today's time to expiry (years).
    dt : float
        Horizon in years (one trading day by default).

    Returns
    -------
    dict of {sub-book or 'total': np.ndarray}
        Scenario P&L in USD.
    """
    returns = np.maximum(np.asarray(returns, dtype=float), -0.99)
    v0 = book_value(legs, S, T, r, q, sigma)
    v1 = book_value(legs, S * (1 + returns), T - dt, r, q, np.maximum(sigma + np.asarray(dsigma), 1e-4))
    return {k: v1[k] - v0[k] for k in v0}


def var_es(pnl, p_var=0.01, p_es=0.025):
    """VaR and ES as positive losses from scenario P&Ls (VaR 99%, ES 97.5% by default).

    Parameters
    ----------
    pnl : array
        Scenario P&Ls.
    p_var, p_es : float
        Tail probabilities of VaR and ES.

    Returns
    -------
    (float, float)
        VaR, ES (average loss beyond the p_es quantile).
    """
    x = np.asarray(pnl, dtype=float)
    var = -np.quantile(x, p_var)
    q_es = np.quantile(x, p_es)
    return var, -x[x <= q_es].mean()

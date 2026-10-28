"""
Black-Scholes pricing of European index options with a continuous dividend yield.

Sections
--------
1. Prices     bs_price     (Step 2.1, Equation 3)
2. Greeks     bs_greeks    (Step 2.2, Equation 5)

Every function is vectorised: each numeric input may be a scalar or a numpy array,
and arrays are broadcast together, so one call prices thousands of scenarios.

Units: prices in index points (multiply by 100 for USD per SPX contract);
sigma, r and q are decimals per year; T is in years.
"""
import numpy as np
from scipy.stats import norm

__all__ = ["bs_price", "bs_greeks"]


# =========================================================================== #
# 1. Prices                                                                   #
# =========================================================================== #
def bs_price(S, K, T, r, q, sigma, kind):
    """Black-Scholes price of a European call or put (Equation 3).

    d1 = [ln(S/K) + (r - q + sigma^2 / 2) T] / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)
    call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

    Parameters
    ----------
    S : float or array
        Index level.
    K : float or array
        Strike.
    T : float or array
        Time to expiry in years (> 0).
    r : float or array
        Continuously compounded risk-free rate (decimal).
    q : float or array
        Continuous dividend yield (decimal).
    sigma : float or array
        Volatility per year (decimal, > 0).
    kind : str
        "call" or "put".

    Returns
    -------
    float or np.ndarray
        Option price in index points.
    """
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    disc_S, disc_K = S * np.exp(-q * T), K * np.exp(-r * T)        # discounted index and strike
    if kind == "call":
        return disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    if kind == "put":
        return disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


# =========================================================================== #
# 2. Greeks                                                                   #
# =========================================================================== #
def bs_greeks(S, K, T, r, q, sigma, kind, dt=1 / 252):
    """Black-Scholes Greeks of a European call or put (Equation 5).

    delta = e^{-qT} N(d1) (call) or e^{-qT} (N(d1) - 1) (put)
    gamma = e^{-qT} n(d1) / (S sigma sqrt(T))           (same for calls and puts)
    vega  = S e^{-qT} n(d1) sqrt(T) / 100               (per volatility point)
    theta = price(T - dt) - price(T)                    (one-day repricing, not dV/dt)

    Theta is a repricing rather than the analytic derivative so that it equals exactly
    the time decay used by the delta-gamma P&L over the VaR horizon.

    Parameters
    ----------
    S, K, T, r, q, sigma : float or array
        As in ``bs_price``.
    kind : str
        "call" or "put".
    dt : float
        Time step for theta, in years (default: one trading day).

    Returns
    -------
    dict of float or np.ndarray
        'delta' (per index point), 'gamma' (per index point squared),
        'vega' (per 1 volatility point, i.e. sigma + 0.01),
        'theta' (price change over dt, everything else fixed).
    """
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    eq = np.exp(-q * T)
    delta = eq * norm.cdf(d1) if kind == "call" else eq * (norm.cdf(d1) - 1)
    gamma = eq * norm.pdf(d1) / (S * sigma * sqT)
    vega = S * eq * norm.pdf(d1) * sqT / 100                       # per volatility point
    theta = bs_price(S, K, T - dt, r, q, sigma, kind) - bs_price(S, K, T, r, q, sigma, kind)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

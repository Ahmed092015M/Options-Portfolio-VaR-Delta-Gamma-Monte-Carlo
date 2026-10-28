"""
Unit tests for the optionsvar package (run with `pytest -v` from the project folder).

Every test checks a function against an independent answer: a closed form, a
hand-worked example from the theory notes, a finite difference, a symmetry, or a
simulation with known properties. No test needs the internet.
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from optionsvar.backtest import acerbi_szekely_z2, christoffersen_test, kupiec_test
from optionsvar.book import (BOOK_SPEC, MATURITY, NOTIONAL, book_greeks, book_legs, book_value,
                             daily_book_pnl, realised_hday_pnl)
from optionsvar.data import align_market_data
from optionsvar.engine import block_hday_var, rolling_option_var, rolling_option_var_vega
from optionsvar.garch import fit_gjr_t, gjr_t_loglik, gjr_variance, rolling_gjr_vol, simulate_gjr_paths
from optionsvar.plots import (plot_approximations, plot_backtest, plot_hday_backtest, plot_market,
                              plot_var_through_crises)
from optionsvar.pricing import bs_greeks, bs_price
from optionsvar.var_methods import (delta_gamma_moments, full_reval_pnl, var_cornish_fisher,
                                    var_delta_gamma_mc, var_delta_normal, var_es)
from optionsvar.volatility import ewma_vol

# Theory-note market: S = K = 5,000, one month, r 4%, q 1.5%, sigma 20%
S0, K0, T0, R0, Q0, V0 = 5000.0, 5000.0, 21 / 252, 0.04, 0.015, 0.20
SD0 = V0 / np.sqrt(252)


@pytest.fixture(scope="module")
def market():
    """Synthetic market, 1,400 days: GJR-t index returns, VIX linked to the true volatility."""
    rng = np.random.default_rng(0)
    n = 1400
    s2, r = np.empty(n), np.empty(n)
    s2[0] = 1.0
    for t in range(n):
        if t:
            s2[t] = 0.02 + 0.19 * (r[t - 1] < 0) * r[t - 1] ** 2 + 0.89 * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * rng.standard_t(7) * np.sqrt(5 / 7)
    idx = pd.bdate_range("2000-01-03", periods=n)
    mkt = pd.DataFrame({"S": 1000 * np.cumprod(1 + r / 100),
                        "VIX": 100 * np.sqrt(252 * s2) / 100 * 1.2 * np.exp(0.05 * rng.standard_normal(n))},
                       index=idx)
    mkt["sigma"], mkt["rate_pct"] = mkt["VIX"] / 100, 3.0
    mkt["r"] = np.log(1.03)
    return mkt


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def test_align_market_data_calendar_and_rate_fill():
    d = pd.bdate_range("2024-01-01", periods=6)
    spx = pd.Series([100, 101, 102, 103, 104, 105.0], index=d)
    vix = pd.Series([20, 21, 22, 23, 24.0], index=d.delete(2))                  # VIX missing on day 3
    rate = pd.Series([5.0, 4.0], index=d[[0, 3]])                              # rate only on days 1 and 4
    m = align_market_data(spx, vix, rate)
    assert list(m.index) == list(d.delete(2))
    assert list(m["rate_pct"]) == [5.0, 5.0, 4.0, 4.0, 4.0]                    # forward-filled, no look-ahead
    assert np.allclose(m["sigma"], m["VIX"] / 100)
    assert np.isclose(m["r"].iloc[0], np.log(1.05))


# --------------------------------------------------------------------------- #
# Pricing                                                                     #
# --------------------------------------------------------------------------- #
def test_bs_theory_example():
    assert bs_price(S0, K0, T0, R0, Q0, V0, "call") == pytest.approx(120.16, abs=0.01)
    assert bs_price(S0, K0, T0, R0, Q0, V0, "put") == pytest.approx(109.76, abs=0.01)


def test_put_call_parity_random():
    rng = np.random.default_rng(1)
    n = 5000
    S, K = rng.uniform(1000, 8000, n), rng.uniform(1000, 8000, n)
    T, r, q, s = rng.uniform(0.01, 2, n), rng.uniform(0, 0.08, n), rng.uniform(0, 0.03, n), rng.uniform(0.05, 0.9, n)
    gap = bs_price(S, K, T, r, q, s, "call") - bs_price(S, K, T, r, q, s, "put") - (S * np.exp(-q * T) - K * np.exp(-r * T))
    assert np.abs(gap).max() < 1e-9


def test_bs_limits_and_bad_kind():
    assert bs_price(S0, 4800, 1e-8, R0, Q0, V0, "call") == pytest.approx(200, abs=1e-3)     # payoff at expiry
    fwd = S0 * np.exp(-Q0 * T0) - 4800 * np.exp(-R0 * T0)
    assert bs_price(S0, 4800, T0, R0, Q0, 1e-8, "call") == pytest.approx(fwd, abs=1e-6)     # no volatility
    with pytest.raises(ValueError):
        bs_price(S0, K0, T0, R0, Q0, V0, "straddle")


def test_greeks_vs_finite_differences():
    for kind in ("call", "put"):
        g = bs_greeks(S0, 5100, T0, R0, Q0, V0, kind)
        P = lambda S=S0, s=V0: bs_price(S, 5100, T0, R0, Q0, s, kind)
        assert g["delta"] == pytest.approx((P(S0 + 1) - P(S0 - 1)) / 2, abs=1e-5)
        assert g["gamma"] == pytest.approx(P(S0 + 1) - 2 * P() + P(S0 - 1), abs=1e-6)
        assert g["vega"] == pytest.approx((P(s=V0 + 1e-4) - P(s=V0 - 1e-4)) / 2e-4 / 100, abs=1e-5)
        assert g["theta"] == pytest.approx(bs_price(S0, 5100, T0 - 1 / 252, R0, Q0, V0, kind) - P())


def test_greeks_call_put_relations():
    c, p = bs_greeks(S0, K0, T0, R0, Q0, V0, "call"), bs_greeks(S0, K0, T0, R0, Q0, V0, "put")
    assert c["delta"] - p["delta"] == pytest.approx(np.exp(-Q0 * T0))
    assert c["gamma"] == pytest.approx(p["gamma"]) and c["vega"] == pytest.approx(p["vega"])


# --------------------------------------------------------------------------- #
# Book                                                                        #
# --------------------------------------------------------------------------- #
def test_book_legs_and_value():
    legs = book_legs(S0)
    assert list(legs["strike"]) == [5000, 5000, 4750, 5000, 5250]
    v = book_value(legs, S0, T0, R0, Q0, V0)
    assert v["long straddle"] == pytest.approx(229_920, abs=1)
    assert v["total"] == pytest.approx(v["long straddle"] + v["short puts"] + v["call spread"])


def test_book_greeks_theory_values_and_fd():
    legs = book_legs(S0)
    G = book_greeks(legs, S0, T0, R0, Q0, V0)
    assert G.loc["long straddle", "delta"] == pytest.approx(51.7, abs=0.05)
    assert G.loc["long straddle", "vega"] == pytest.approx(11_478, abs=1)
    up, mid, dn = (book_value(legs, S0 + d, T0, R0, Q0, V0)["total"] for d in (1, 0, -1))
    assert G.loc["total", "delta"] == pytest.approx((up - dn) / 2, abs=1e-3)
    assert G.loc["total", "gamma"] == pytest.approx(up - 2 * mid + dn, abs=1e-4)


def test_daily_pnl_hand_check_and_notional(market):
    pnl = daily_book_pnl(market)
    t = 500
    p, d = market.iloc[t - 1], market.index[t]
    lg = book_legs(p["S"])
    before = book_value(lg, p["S"], MATURITY, p["r"], 0.015, p["sigma"])
    m = market.loc[d]
    after = book_value(lg, m["S"], MATURITY - 1 / 252, m["r"], 0.015, m["sigma"])
    assert pnl.loc[d, "total"] == pytest.approx(after["total"] - before["total"])
    scaled = daily_book_pnl(market, notional=NOTIONAL)
    assert scaled.loc[d, "total"] == pytest.approx(pnl.loc[d, "total"] * 5000 / p["S"])


def test_hold_vol_and_h1(market):
    full = daily_book_pnl(market, notional=NOTIONAL)
    assert np.allclose(realised_hday_pnl(market, h=1, notional=NOTIONAL).values, full.values)
    flat = market.assign(sigma=0.2)
    assert np.allclose(daily_book_pnl(flat, hold_vol=True).values, daily_book_pnl(flat).values)
    assert len(realised_hday_pnl(market, h=10)) == len(market) - 10


# --------------------------------------------------------------------------- #
# Volatility and VaR methods                                                  #
# --------------------------------------------------------------------------- #
def test_ewma_recursion():
    x = pd.Series([0.01, -0.02, 0.03, 0.0])
    v = ewma_vol(x, lam=0.9, init_window=2)
    s2 = [np.mean([0.01 ** 2, 0.02 ** 2])]
    for r in x.values[:-1]:
        s2.append(0.9 * s2[-1] + 0.1 * r ** 2)
    assert np.allclose(v.values, np.sqrt(s2))


def test_delta_normal_formula():
    assert var_delta_normal(-530, S0, SD0) == pytest.approx(stats.norm.ppf(0.99) * 530 * S0 * SD0)


def test_delta_gamma_theory_example():
    G = book_greeks(book_legs(S0), S0, T0, R0, Q0, V0).loc["long straddle"]
    d, g = -G["delta"], -G["gamma"]                                              # short straddle
    assert var_cornish_fisher(d, g, 0, S0, SD0) == pytest.approx(42_268, rel=2e-3)
    assert var_delta_gamma_mc(d, g, 0, S0, SD0) == pytest.approx(38_950, rel=5e-3)


def test_delta_gamma_moments_vs_simulation():
    rng = np.random.default_rng(2)
    dS = rng.normal(0, S0 * SD0, 2_000_000)
    pl = 10 + 50 * dS - 0.5 * 3 * dS ** 2
    m, s, k = delta_gamma_moments(50, -3, 10, S0, SD0)
    assert m == pytest.approx(pl.mean(), rel=0.01)
    assert s == pytest.approx(pl.std(), rel=0.01)
    assert k == pytest.approx(stats.skew(pl), abs=0.03)


def test_zero_gamma_all_methods_agree():
    dn = var_delta_normal(530, S0, SD0)
    assert var_cornish_fisher(530, 0, 0, S0, SD0) == pytest.approx(dn)
    assert var_delta_gamma_mc(530, 0, 0, S0, SD0) == pytest.approx(dn, rel=0.005)


def test_full_reval_theory_example_and_forward():
    rng = np.random.default_rng(1)
    ret = rng.normal(0, SD0, 1_000_000)
    legs = book_legs(S0)
    st = full_reval_pnl(legs[legs["sub-book"] == "long straddle"], S0, R0, Q0, V0, ret)["total"]
    assert var_es(-st)[0] == pytest.approx(32_959, rel=2e-3)                    # short straddle
    fwd = pd.DataFrame([("f", "call", 1.0, 10, S0), ("f", "put", 1.0, -10, S0)],
                       columns=["sub-book", "kind", "moneyness", "contracts", "strike"])
    d = book_greeks(fwd, S0, T0, R0, Q0, V0).loc["total", "delta"]
    assert var_es(full_reval_pnl(fwd, S0, R0, Q0, V0, ret)["total"])[0] == pytest.approx(
        var_delta_normal(d, S0, SD0), rel=0.01)


def test_full_reval_floors():
    pl = full_reval_pnl(book_legs(S0), S0, R0, Q0, V0, np.array([-1.5, 0.0]), dsigma=np.array([-1.0, 0.0]))
    assert np.all(np.isfinite(pl["total"]))


def test_var_es_on_normal():
    x = np.random.default_rng(3).standard_normal(2_000_000)
    v, e = var_es(x)
    assert v == pytest.approx(stats.norm.ppf(0.99), abs=0.01)
    assert e == pytest.approx(stats.norm.pdf(stats.norm.ppf(0.975)) / 0.025, abs=0.01)


# --------------------------------------------------------------------------- #
# GJR-GARCH                                                                   #
# --------------------------------------------------------------------------- #
def test_gjr_variance_recursion():
    r = np.array([1.0, -2.0, 0.5])
    s2 = gjr_variance(r, 0.1, 0.05, 0.1, 0.8, s2_0=1.0)
    assert s2[1] == pytest.approx(0.1 + 0.05 * 1 + 0.8 * 1.0)
    assert s2[2] == pytest.approx(0.1 + 0.15 * 4 + 0.8 * s2[1])


def test_gjr_loglik_equals_scipy_t():
    r = np.random.default_rng(4).standard_normal(300)
    p = (0.05, 0.03, 0.1, 0.85, 6.0)
    s = np.sqrt(gjr_variance(r, *p[:4]))
    scale = s * np.sqrt((6 - 2) / 6)
    assert gjr_t_loglik(p, r) == pytest.approx(stats.t.logpdf(r / scale, 6).sum() - np.log(scale).sum())
    assert gjr_t_loglik((0.05, 0.03, 0.1, 0.85, 1.5), r) == -1e10


def test_fit_recovers_parameters():
    rng = np.random.default_rng(5)
    om, al, ga, be, nu = 0.03, 0.02, 0.15, 0.86, 7.0
    n, r, s2 = 6000, np.empty(6000), om / (1 - al - ga / 2 - be)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_t(nu) * np.sqrt((nu - 2) / nu)
        s2 = om + (al + ga * (r[t] < 0)) * r[t] ** 2 + be * s2
    p, _, ok = fit_gjr_t(r)
    assert ok
    assert p[2] == pytest.approx(ga, abs=0.06) and p[3] == pytest.approx(be, abs=0.04)
    assert p[4] == pytest.approx(nu, abs=2.5)


def test_rolling_gjr_no_look_ahead(market):
    ret = market["S"].pct_change().iloc[1:1130]
    a, par = rolling_gjr_vol(ret, window=1000, refit_every=50)
    b_ret = ret.copy()
    b_ret.iloc[1060] = -0.2
    b, _ = rolling_gjr_vol(b_ret, window=1000, refit_every=50)
    k = 1060 - 1000
    assert np.allclose(a.iloc[:k + 1], b.iloc[:k + 1]) and not np.isclose(a.iloc[k + 1], b.iloc[k + 1])
    assert len(par) == 3 and par["converged"].all()


def test_simulate_constant_variance():
    sd = 0.012
    g = simulate_gjr_paths(((100 * sd) ** 2, 0, 0, 0, 100.0), sd, h=10, n_paths=200_000)
    assert np.log1p(g).std() == pytest.approx(sd * np.sqrt(10), rel=0.01)


def test_simulate_gjr_negative_skew():
    g = simulate_gjr_paths((0.02, 0.0, 0.19, 0.89, 7.0), 0.012, h=10, n_paths=100_000)
    assert stats.skew(np.log1p(g)) < -0.3


# --------------------------------------------------------------------------- #
# Engines                                                                     #
# --------------------------------------------------------------------------- #
def test_engine_look_ahead_and_hand_check(market):
    ret = market["S"].pct_change().iloc[1:]
    ew = ewma_vol(ret)
    j0 = 1200
    base = rolling_option_var(market, ew, start=j0 - 2, end=j0 + 3, n_sim=5000)
    mf = market.copy()
    mf.iloc[j0 + 1:, mf.columns.get_loc("S")] *= 0.7                            # return j0 = -30%
    fake = rolling_option_var(mf, ewma_vol(mf["S"].pct_change().iloc[1:]), start=j0 - 2, end=j0 + 3, n_sim=5000)
    upto = ret.index[j0]
    assert np.allclose(base.loc[:upto].values, fake.loc[:upto].values)
    assert not np.allclose(base.iloc[-1].values, fake.iloc[-1].values)
    d = base.index[0]
    t = ret.index.get_loc(d)
    p = market.iloc[t]
    g = book_greeks(book_legs(p["S"]), p["S"], MATURITY, p["r"], 0.015, p["sigma"]).loc["short puts"]
    size = NOTIONAL / (1000 * p["S"])
    assert base.loc[d, ("short puts", "DN", "VaR")] == pytest.approx(size * var_delta_normal(g["delta"], p["S"], ew.iloc[t]))


def test_engine_mc_t_and_vega_zero_change(market):
    ret = market["S"].pct_change().iloc[1:]
    ew = ewma_vol(ret)
    nu = pd.Series(7.0, index=ret.index)
    a = rolling_option_var(market, ew, start=1100, end=1103, n_sim=4000, nu=nu)
    assert ("total", "MC-t", "ES") in a.columns
    assert (a.xs("ES", axis=1, level=2) >= a.xs("VaR97.5", axis=1, level=2) - 1e-9).all().all()
    m0 = market.assign(VIX=20.0)                                                 # no VIX change in any scenario
    v = rolling_option_var_vega(m0, ew, pd.Series(1.0, index=ret.index), start=1100, end=1103)
    assert np.allclose(v.xs("FHS-vega", axis=1, level=1).values, a.xs("FHS", axis=1, level=1).values)


def test_block_engine(market):
    ret = market["S"].pct_change().iloc[1:]
    ew = ewma_vol(ret)
    one = rolling_option_var(market, ew, start=1300, n_sim=2000)
    par = pd.DataFrame({"omega": [0.02], "alpha": [0.0], "gamma": [0.19], "beta": [0.89], "nu": [7.0]},
                       index=[ret.index[0]])
    blk = block_hday_var(market, ew, par, one, start=1300, n_sim=2000)
    assert len(blk) == len(range(1300, len(ret) - 9, 10))
    d = blk.index[0]
    assert blk.loc[d, ("total", "sqrt10 FHS", "VaR")] == pytest.approx(np.sqrt(10) * one.loc[d, ("total", "FHS", "VaR")])
    assert (blk.xs("ES", axis=1, level=2) >= blk.xs("VaR97.5", axis=1, level=2) - 1e-9).all().all()


# --------------------------------------------------------------------------- #
# Backtests                                                                   #
# --------------------------------------------------------------------------- #
def test_kupiec_and_christoffersen():
    n = 8000
    even = np.zeros(n, int); even[::100] = 1
    pairs = np.zeros(n, int); pairs[::200] = 1; pairs[1::200] = 1
    assert kupiec_test(even)[1] > 0.9 and christoffersen_test(even)[1] > 0.3
    assert kupiec_test(pairs)[1] > 0.5 and christoffersen_test(pairs)[1] < 0.001
    many = np.zeros(n, int); many[::33] = 1
    assert kupiec_test(many)[1] < 1e-6
    assert kupiec_test(np.zeros(n, int))[0] == 0                                 # zero breaches is valid


def test_z2_correct_and_too_low():
    x = np.random.default_rng(6).standard_normal(200_000)
    v, e = stats.norm.ppf(0.975), stats.norm.pdf(stats.norm.ppf(0.975)) / 0.025
    assert acerbi_szekely_z2(x, v, e) == pytest.approx(0, abs=0.03)
    assert acerbi_szekely_z2(2 * x, v, e) < -0.5


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def test_plots_return_figures(market):
    import matplotlib.pyplot as plt
    figs = [plot_market(market, market["VIX"])]
    bt = pd.DataFrame({"breaches": [80, 300, 2, 0], "Z2": [0.0, -5.0, 0.5, 0.9]},
                      index=pd.MultiIndex.from_product([["short puts", "total"], ["DN", "FHS"]]))
    figs.append(plot_backtest(bt, 72.7, -0.119))
    figs.append(plot_hday_backtest(bt.assign(**{"avg VaR 10d": 1e5}), 7.3))
    moves = np.linspace(-10, 10, 5)
    prof = pd.DataFrame({"full": moves, "delta": moves, "delta-gamma": moves}, index=moves)
    acc = pd.DataFrame({"DN error %": 1.0, "DG error %": 0.5},
                       index=pd.MultiIndex.from_product([[1, 10], [10, 20], ["short puts", "total"]],
                                                        names=["days", "vol %", "book"]))
    figs.append(plot_approximations({"1 day": prof, "10 days": prof}, acc))
    s = pd.Series(np.arange(len(market), dtype=float), index=market.index)
    figs.append(plot_var_through_crises(-s, {"A": s}, [(market.index[10], market.index[50], "x")], "t"))
    assert all(isinstance(f, matplotlib.figure.Figure) for f in figs)
    plt.close("all")

"""
Charts for the options VaR project.

Every function draws one chart and returns the Figure, so the notebook can save it
(``fig.savefig``) before showing it (``plt.show()``).

Sections
--------
1. Helpers                 CRISES, BOOK_COLORS, _clean, _shade
2. Market                  plot_market                     (Step 1.7)
3. Approximations          plot_approximations             (Step 6.2)
4. Backtests               plot_hday_backtest              (Step 8.6)
                           plot_backtest                   (Step 9.3)
                           plot_var_through_crises         (Step 9.4)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["CRISES", "BOOK_COLORS", "plot_market", "plot_approximations", "plot_hday_backtest",
           "plot_backtest", "plot_var_through_crises"]


# =========================================================================== #
# 1. Helpers                                                                  #
# =========================================================================== #
# Stress periods shaded on time charts: (start, end, label)
CRISES = [
    ("1998-08-01", "1998-10-31", "LTCM"),
    ("2000-03-01", "2002-10-31", "Dot-com"),
    ("2008-09-01", "2009-03-31", "GFC"),
    ("2020-02-20", "2020-04-30", "Covid"),
    ("2022-01-01", "2022-10-31", "Inflation"),
]

BOOK_COLORS = {"long straddle": "#2ca02c", "short puts": "#d62728", "call spread": "#1f77b4", "total": "black"}
METHOD_COLORS = {"full": "black", "delta": "#1f77b4", "delta-gamma": "#ff7f0e"}


def _clean(ax):
    """Remove the top and right frame lines."""
    ax.spines[["top", "right"]].set_visible(False)


def _shade(ax, shade, y_text):
    """Shade the stress periods and write their names at height y_text."""
    for start, end, text in shade or []:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.15, lw=0)
        ax.text(pd.Timestamp(start), y_text, text, fontsize=8, va="top")


def _grouped_bars(ax, tab, col, books, meths, floor=None):
    """Bars of tab.loc[(book, method), col], grouped by method, one colour per book."""
    x = np.arange(len(meths))
    w = 0.8 / len(books)
    for i, b in enumerate(books):
        vals = np.array([tab.loc[(b, m), col] for m in meths], dtype=float)
        if floor is not None:
            vals = np.maximum(vals, floor)
        ax.bar(x + (i - (len(books) - 1) / 2) * w, vals, w, label=b, color=BOOK_COLORS.get(b))
    ax.set_xticks(x, meths, fontsize=9)
    return x, w


# =========================================================================== #
# 2. Market                                                                   #
# =========================================================================== #
def plot_market(mkt, realised, shade=CRISES):
    """S&P 500 on a log scale (top); VIX and realised volatility (bottom).

    Parameters
    ----------
    mkt : pd.DataFrame
        Columns 'S' (index level) and 'VIX'.
    realised : pd.Series
        Realised volatility in the same units as VIX (% per year), aligned on mkt's dates.
    shade : list of (start, end, label), optional
        Periods shaded in grey.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})

    ax = axes[0]
    _shade(ax, shade, mkt["S"].max())
    ax.plot(mkt.index, mkt["S"], lw=1, color="black")
    ax.set_yscale("log")
    ax.set_ylabel("S&P 500 (log scale)")
    ax.set_title("S&P 500 index", fontsize=11)
    _clean(ax)

    ax = axes[1]
    _shade(ax, shade, mkt["VIX"].max())
    ax.plot(mkt.index, mkt["VIX"], lw=0.9, label="VIX (expected vol, next month)")
    ax.plot(realised.index, realised, lw=0.9, alpha=0.8, label="realised vol over the next 21 days")
    ax.set_ylabel("% per year")
    ax.set_title("Implied (VIX) vs realised volatility", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    _clean(ax)

    fig.tight_layout()
    return fig


# =========================================================================== #
# 3. Approximations                                                           #
# =========================================================================== #
def plot_approximations(profiles, accuracy, book="short puts"):
    """P&L profiles of one sub-book (full vs delta vs delta-gamma) and VaR errors vs volatility.

    Parameters
    ----------
    profiles : dict of {label: pd.DataFrame}
        One panel each (e.g. "1 day", "10 days"); index = index move in %,
        columns 'full', 'delta', 'delta-gamma' (P&L in USD).
    accuracy : pd.DataFrame
        Output of Step 6.1: index (days, vol %, book); columns 'DN error %', 'DG error %'.
    book : str
        Sub-book drawn in the profile panels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(profiles)
    fig, axes = plt.subplots(1, n + 1, figsize=(5.2 * (n + 1), 4.3))
    for ax, (label, prof) in zip(axes[:n], profiles.items()):
        for col in prof.columns:
            ax.plot(prof.index, prof[col] / 1e3, lw=1.6 if col == "full" else 1.2,
                    ls="-" if col == "full" else "--", color=METHOD_COLORS.get(col), label=col)
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("index move (%)")
        ax.set_ylabel("P&L (USD thousand)")
        ax.set_title(f"{book.capitalize()}: P&L over {label}", fontsize=11)
        ax.legend(fontsize=8)
        _clean(ax)

    ax = axes[-1]
    for days, ls in [(1, "-"), (10, "--")]:
        for b, mk in [("short puts", "o"), ("total", "s")]:
            sub = accuracy.xs((days, b), level=("days", "book"))
            ax.plot(sub.index, sub["DN error %"], ls=ls, marker=mk, color="#1f77b4", ms=4,
                    label=f"DN, {b}, {days}d")
            ax.plot(sub.index, sub["DG error %"], ls=ls, marker=mk, color="#ff7f0e", ms=4,
                    label=f"DG, {b}, {days}d")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("implied = risk volatility (% a year)")
    ax.set_ylabel("VaR error vs full revaluation (%)")
    ax.set_title("VaR 99% error vs full revaluation (solid 1 day, dashed 10 days)", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    _clean(ax)
    fig.tight_layout()
    return fig


# =========================================================================== #
# 4. Backtests                                                                #
# =========================================================================== #
def plot_hday_backtest(tab, expected):
    """10-day backtest: breaches (left) and average VaR (right) per method and book.

    Parameters
    ----------
    tab : pd.DataFrame
        Index = (book, method); columns 'breaches' and 'avg VaR 10d'.
    expected : float
        Expected number of breaches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    books = list(dict.fromkeys(tab.index.get_level_values(0)))
    meths = list(dict.fromkeys(tab.index.get_level_values(1)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    _grouped_bars(axes[0], tab, "breaches", books, meths)
    axes[0].axhline(expected, color="black", ls="--", lw=1, label=f"expected {expected:.1f}")
    axes[0].set_title("10-day VaR 99% breaches (non-overlapping blocks)", fontsize=11)
    t2 = tab.assign(k=tab["avg VaR 10d"] / 1e3)
    _grouped_bars(axes[1], t2, "k", books, meths)
    axes[1].set_ylabel("USD thousand")
    axes[1].set_title("Average 10-day VaR 99% (capital)", fontsize=11)
    for ax in axes:
        ax.legend(fontsize=8)
        _clean(ax)
    fig.tight_layout()
    return fig


def plot_backtest(bt, expected, crit, z_floor=-3.0):
    """99% VaR breaches (left, log scale) and ES Z2 (right, clipped) per method and book.

    Parameters
    ----------
    bt : pd.DataFrame
        Index = (book, method); columns 'breaches' and 'Z2'.
    expected : float
        Expected number of breaches.
    crit : float
        5% critical value of Z2.
    z_floor : float
        Z2 values below this are drawn at this level (and labelled).

    Returns
    -------
    matplotlib.figure.Figure
    """
    books = list(dict.fromkeys(bt.index.get_level_values(0)))
    meths = list(dict.fromkeys(bt.index.get_level_values(1)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    _grouped_bars(axes[0], bt, "breaches", books, meths, floor=0.8)        # 0.8: visible on a log scale
    x, w = _grouped_bars(axes[1], bt, "Z2", books, meths, floor=z_floor)
    for i, b in enumerate(books):                                         # label the clipped Z2 values
        for j, m in enumerate(meths):
            v = bt.loc[(b, m), "Z2"]
            if v < z_floor:
                axes[1].text(x[j] + (i - (len(books) - 1) / 2) * w, z_floor, f"{v:.0f}",
                             ha="center", va="top", fontsize=7)
    for ax in axes:
        ax.legend(fontsize=8)
        _clean(ax)
    axes[0].set_yscale("log")
    axes[0].axhline(expected, color="black", ls="--", lw=1)
    axes[0].set_title(f"VaR 99% breaches vs the full P&L (expected {expected:.1f}, log scale)", fontsize=11)
    axes[1].axhline(crit, color="black", ls="--", lw=1)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_title(f"ES 97.5% test Z2 (reject below {crit:.3f}; clipped at {z_floor:.0f})", fontsize=11)
    fig.tight_layout()
    return fig


def plot_var_through_crises(pnl, var_lines, periods, title):
    """Daily P&L (bars) against several -VaR forecasts (lines), one panel per period.

    Parameters
    ----------
    pnl : pd.Series
        Realised daily P&L.
    var_lines : dict of {label: pd.Series}
        VaR forecasts (positive losses), drawn as -VaR.
    periods : list of (start, end, label)
        Periods to show, one panel each.
    title : str
        Overall title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, len(periods), figsize=(7 * len(periods), 4.5), sharey=False)
    for ax, (a, b, lab) in zip(np.atleast_1d(axes), periods):
        p = pnl.loc[a:b]
        ax.bar(p.index, p.values / 1e3, width=1.0, color=np.where(p.values < 0, "#d62728", "#7f7f7f"), alpha=0.6)
        for label, v in var_lines.items():
            ax.plot(v.loc[a:b].index, -v.loc[a:b].values / 1e3, lw=1.4, label=f"-VaR {label}")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(lab, fontsize=11)
        ax.set_ylabel("USD thousand")
        ax.legend(fontsize=8, loc="lower left")
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        _clean(ax)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig

"""
The options book: positions, value, Greeks and realised P&L.

Sections
--------
1. Definition         MULTIPLIER, MATURITY, N_CONTRACTS, NOTIONAL, BOOK_SPEC, book_legs   (Step 3.1)
2. Value and Greeks   book_value, book_greeks                                             (Steps 3.1 - 3.2)
3. Realised P&L       daily_book_pnl, realised_hday_pnl                                   (Steps 3.3 - 3.4, 8.4)

The book is rebuilt every day at constant moneyness: on day t-1 we buy or sell
one-month options with strikes = moneyness x S_{t-1}, and on day t we revalue the
same options (same strikes, one day less to expiry). This "constant-maturity
rolling book" keeps the risk profile comparable across 36 years.

With ``notional`` set, the number of contracts is scaled every day so that each
leg always holds USD 5m of index exposure per 10 contracts
(contracts = 10 x 5,000,000 / (100 x 10 x S)); otherwise the book would grow 21x
with the index between 1990 and 2026.
"""
import pandas as pd

from .pricing import bs_greeks, bs_price

__all__ = ["MULTIPLIER", "MATURITY", "N_CONTRACTS", "NOTIONAL", "BOOK_SPEC",
           "book_legs", "book_value", "book_greeks", "daily_book_pnl", "realised_hday_pnl"]


# =========================================================================== #
# 1. Definition                                                               #
# =========================================================================== #
MULTIPLIER = 100                     # SPX options: USD 100 per index point
MATURITY = 21 / 252                  # one month, in years
N_CONTRACTS = 10                     # contracts per leg
NOTIONAL = 5_000_000                 # USD of index notional per leg (= 10 contracts at S = 5,000)

# (sub-book, kind, moneyness K/S, contracts: + bought / - sold)
BOOK_SPEC = [
    ("long straddle", "call", 1.00, +N_CONTRACTS),
    ("long straddle", "put",  1.00, +N_CONTRACTS),
    ("short puts",    "put",  0.95, -N_CONTRACTS),
    ("call spread",   "call", 1.00, +N_CONTRACTS),
    ("call spread",   "call", 1.05, -N_CONTRACTS),
]


def book_legs(S, spec=BOOK_SPEC):
    """Positions of the book for a given index level (strikes = moneyness x S).

    Parameters
    ----------
    S : float
        Index level on the day the book is built.
    spec : list of (sub-book, kind, moneyness, contracts)
        Book definition.

    Returns
    -------
    pd.DataFrame
        One row per leg: 'sub-book', 'kind', 'moneyness', 'contracts', 'strike'.
    """
    legs = pd.DataFrame(spec, columns=["sub-book", "kind", "moneyness", "contracts"])
    legs["strike"] = legs["moneyness"] * S
    return legs


# =========================================================================== #
# 2. Value and Greeks                                                         #
# =========================================================================== #
def book_value(legs, S, T, r, q, sigma):
    """Value in USD of each sub-book and of the whole book, on one or many market scenarios.

    Parameters
    ----------
    legs : pd.DataFrame
        Output of ``book_legs`` (strikes fixed).
    S, T, r, q, sigma : float or array
        Market inputs; arrays give one value per scenario (full revaluation).

    Returns
    -------
    dict of {sub-book or 'total': float or np.ndarray}
    """
    values = {}
    for _, leg in legs.iterrows():
        v = leg["contracts"] * MULTIPLIER * bs_price(S, leg["strike"], T, r, q, sigma, leg["kind"])
        values[leg["sub-book"]] = values.get(leg["sub-book"], 0.0) + v
    values["total"] = sum(values[k] for k in list(values))
    return values


def book_greeks(legs, S, T, r, q, sigma):
    """Greeks of each sub-book and of the whole book, in USD (Equation 6).

    Greeks add up across legs: book delta = sum of contracts x 100 x option delta, etc.

    Parameters
    ----------
    legs : pd.DataFrame
        Output of ``book_legs``.
    S, T, r, q, sigma : float
        Market inputs.

    Returns
    -------
    pd.DataFrame
        Index = sub-books + 'total'; columns 'delta' (USD per index point),
        'gamma' (change of delta per index point), 'vega' (USD per vol point),
        'theta' (USD per day).
    """
    rows = {}
    for _, leg in legs.iterrows():
        g = bs_greeks(S, leg["strike"], T, r, q, sigma, leg["kind"])
        scale = leg["contracts"] * MULTIPLIER
        acc = rows.setdefault(leg["sub-book"], {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0})
        for k in acc:
            acc[k] += scale * float(g[k])
    out = pd.DataFrame(rows).T
    out.loc["total"] = out.sum()
    return out


# =========================================================================== #
# 3. Realised P&L                                                             #
# =========================================================================== #
def _book_pnl(mkt, h, q, hold_vol, notional, spec):
    """Shared core of the 1-day and h-day realised P&L (vectorised over all days)."""
    S0, S1 = mkt["S"].values[:-h], mkt["S"].values[h:]
    s0, s1 = mkt["sigma"].values[:-h], mkt["sigma"].values[h:]
    r0, r1 = mkt["r"].values[:-h], mkt["r"].values[h:]
    if hold_vol:
        s1 = s0                                                   # index-only P&L: implied vol held
    size = 1.0 if notional is None else notional / (MULTIPLIER * N_CONTRACTS * S0)   # scale per day
    out = {}
    for book, kind, money, n in spec:
        K = money * S0                                            # strikes set on the first day
        v0 = bs_price(S0, K, MATURITY, r0, q, s0, kind)
        v1 = bs_price(S1, K, MATURITY - h / 252, r1, q, s1, kind)  # h days less to expiry
        out[book] = out.get(book, 0.0) + n * size * MULTIPLIER * (v1 - v0)
    df = pd.DataFrame(out, index=mkt.index[1:len(mkt) - h + 1])
    df["total"] = df.sum(axis=1)
    return df


def daily_book_pnl(mkt, q=0.015, hold_vol=False, notional=None, spec=BOOK_SPEC):
    """Realised clean daily P&L of each sub-book: built on day t-1, revalued on day t.

    Parameters
    ----------
    mkt : pd.DataFrame
        Columns 'S', 'sigma' (VIX / 100) and 'r' (continuous rate), index = date.
    q : float
        Continuous dividend yield.
    hold_vol : bool
        If True, keep sigma at its day t-1 value (index-only P&L, no vega effect).
    notional : float or None
        If given, each leg holds this USD index notional per N_CONTRACTS on every day
        (contracts = n x notional / (MULTIPLIER x N_CONTRACTS x S_{t-1})); if None, fixed contracts.
    spec : list
        Book definition (see BOOK_SPEC).

    Returns
    -------
    pd.DataFrame
        Index = date (from the second day); one column per sub-book and 'total' (USD).
    """
    return _book_pnl(mkt, 1, q, hold_vol, notional, spec)


def realised_hday_pnl(mkt, h=10, q=0.015, hold_vol=False, notional=None, spec=BOOK_SPEC):
    """Realised P&L of each sub-book over h days: built on day t-1, held unchanged, revalued on day t-1+h.

    With h = 1 this equals ``daily_book_pnl``.

    Parameters
    ----------
    mkt : pd.DataFrame
        Columns 'S', 'sigma', 'r'; index = date.
    h : int
        Holding period in trading days.
    q, hold_vol, notional, spec
        As in ``daily_book_pnl``.

    Returns
    -------
    pd.DataFrame
        Index = the first day of each h-day period (date of t); one column per sub-book and 'total'.
    """
    return _book_pnl(mkt, h, q, hold_vol, notional, spec)

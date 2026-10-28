"""
VaR and ES backtests.

Sections
--------
1. VaR coverage     kupiec_test, christoffersen_test     (Step 7.3)
2. ES               acerbi_szekely_z2                     (Step 7.3)

A "hit" (breach) is a day when the realised loss exceeds the VaR forecast.
"""
import numpy as np
from scipy import stats
from scipy.special import xlogy

__all__ = ["kupiec_test", "christoffersen_test", "acerbi_szekely_z2"]


def _lr_uc(h, p):
    """Kupiec likelihood ratio: binomial likelihood at p vs at the observed breach rate."""
    n, x = len(h), int(h.sum())
    return -2 * ((xlogy(n - x, 1 - p) + xlogy(x, p)) - (xlogy(n - x, 1 - x / n) + xlogy(x, x / n)))


# =========================================================================== #
# 1. VaR coverage                                                             #
# =========================================================================== #
def kupiec_test(hits, p=0.01):
    """Kupiec unconditional coverage test: is the breach rate equal to p?

    LR_uc ~ chi2(1) under the null. xlogy(0, 0) = 0 handles zero breaches.

    Parameters
    ----------
    hits : array of 0/1
        1 on breach days.
    p : float
        Expected breach probability.

    Returns
    -------
    (int, float)
        Number of breaches, p-value.
    """
    h = np.asarray(hits, dtype=int)
    return int(h.sum()), float(stats.chi2.sf(_lr_uc(h, p), 1))


def christoffersen_test(hits, p=0.01):
    """Christoffersen conditional coverage test: right breach rate AND no clustering.

    Breaches follow a first-order Markov chain; under the null the probability of a
    breach does not depend on whether yesterday was a breach. LR_cc = LR_uc + LR_ind ~ chi2(2).

    Returns
    -------
    (int, float)
        Number of consecutive breaches (n11), p-value.
    """
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum()); n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum()); n11 = int(((prev == 1) & (curr == 1)).sum())
    pi01, pi11 = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    lr_ind = -2 * ((xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi))
                   - (xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)))
    return n11, float(stats.chi2.sf(_lr_uc(h, p) + lr_ind, 2))


# =========================================================================== #
# 2. ES                                                                       #
# =========================================================================== #
def acerbi_szekely_z2(pnl, var, es, p=0.025):
    """Acerbi-Szekely Z2 test statistic for ES.

    Z2 = sum_t X_t 1[X_t < -VaR_t] / (T p ES_t) + 1, with X_t the P&L.
    Expected value 0 if ES is right; negative if losses beyond VaR are larger than
    ES predicts (ES too low); positive if ES is too conservative. The critical value
    is simulated (about -0.11 for 8,000 days at 5%).

    Parameters
    ----------
    pnl : array
        Realised P&L.
    var, es : array
        VaR and ES forecasts at level p (positive losses).
    p : float
        ES tail probability (2.5%).

    Returns
    -------
    float
    """
    x, v, e = (np.asarray(a, dtype=float) for a in (pnl, var, es))
    return float((x * (x < -v) / (p * e)).sum() / len(x) + 1)

"""
optionsvar – market risk of an S&P 500 options book.

Project 6 of the Financial Risk Forecasting series: a book of one-month SPX
options (long straddle, short puts, call spread) priced with Black-Scholes at
the VIX, and its 1-day and 10-day VaR / ES by delta-normal, delta-gamma,
Monte Carlo and filtered historical simulation, with EWMA or GJR-GARCH-t
volatility, implied-volatility (vega) risk, and backtests.

Modules
-------
data         S&P 500 (Yahoo), VIX and 3-month T-bill (FRED) download and alignment
pricing      Black-Scholes prices and Greeks with a continuous dividend yield
book         the options book: positions, value, Greeks, realised 1-day and h-day P&L
volatility   EWMA volatility forecast
var_methods  delta-normal, delta-gamma (Cornish-Fisher, exact quadratic), full revaluation
garch        GJR-GARCH(1,1)-t: estimation, rolling forecasts, path simulation
engine       out-of-sample VaR / ES engines (1-day, 10-day blocks, joint index + VIX)
backtest     Kupiec, Christoffersen and Acerbi-Szekely Z2 tests
plots        every chart of the project
"""

__version__ = "1.0.0"

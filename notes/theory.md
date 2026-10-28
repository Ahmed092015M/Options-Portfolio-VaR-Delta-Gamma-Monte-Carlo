# Project 6 – Options Portfolio VaR: Delta-Normal vs Delta-Gamma vs Monte Carlo – Theory Notes

*Methods follow Daníelsson, Financial Risk Forecasting, Chapters 6 (analytical VaR for options) and 7 (simulation methods), used as a guide only. This file goes in the repo as `notes/theory.md`.*

These notes use the same frameworks as Projects 1–5:

- **Part A** applies the problem-solving framework to the project as a whole.
- **Part B** applies the deep-learning framework to each technique: concept, why it was developed, the maths (with a definition for every equation and every variable), when it is valid, its limitations, and how to derive, implement and validate it. Each technique ends with a small real-life example.
- **Part C** is a one-page summary of all the equations.
- **Part D** is the development plan, step by step.

The worked examples use round, illustrative numbers (S&P 500 at 5,000, volatility 20%, rate 4%, dividend yield 1.5%, one-month options). Our own results will come from our data.

**Units and conventions.**

- Index levels in points; option prices in index points per option; positions in **contracts of 100 options** (the SPX multiplier), so a 1-point move in an option's price is USD 100 per contract.
- Volatilities in % per year (VIX = 20 means 20%); time in years (one trading day = 1/252).
- P&L, VaR and ES in **USD**; VaR and ES are reported as **positive losses**.
- As in every project, the value for day *t* uses information up to day *t − 1* only.

---

## Decisions for this project

| Topic | Decision |
|---|---|
| Data | **S&P 500 index** (Yahoo, `^GSPC`) and **VIX** (FRED, `VIXCLS`), daily from January 1990; **3-month Treasury bill** rate (FRED, `DGS3MO`) as the risk-free rate. Downloaded fresh in Step 1 |
| Pricing | **Black–Scholes** with a continuous dividend yield, implied volatility = VIX / 100 for every strike (flat smile; a limitation we measure) |
| Book | Three positions in one-month SPX options, rolled daily at constant moneyness: **long straddle** (ATM call + ATM put), **short puts** (5% out of the money), **call spread** (long ATM call, short 5% OTM call). Sizes fixed in Step 3 |
| Methods | 1. **Delta-normal** VaR; 2. **Delta-gamma** VaR (Cornish–Fisher and exact quantile of the quadratic); 3. **Monte Carlo** with full revaluation; extension: **GARCH** simulation of the index, and implied-volatility (vega) risk from VIX |
| Evaluation | Accuracy against full revaluation; out-of-sample VaR 99% / ES 97.5% backtests (Kupiec, Christoffersen, Acerbi–Szekely, as in Projects 3–5); 1-day and 10-day |
| Main message to test | Delta-normal **badly underestimates** the risk of non-linear positions, and can even get its sign wrong |

---

## Part A – The project's problem

### A1. What do I need to determine?
Projects 3–5 measured the risk of positions whose value is close to linear in the risk factors (equities, bonds with modest convexity). An option's value is **strongly non-linear** in the underlying and also depends on volatility and time. This project answers:

> **How should a risk team measure the VaR of an options book, and how wrong are linear (delta-normal) approximations for non-linear positions?**

More precisely:

1. How large the error of delta-normal and delta-gamma VaR is compared with full revaluation, for each kind of position (long gamma, short gamma, a spread with capped payoff).
2. Which method passes out-of-sample backtests over three decades (1998, 2008, 2020, 2022 included).
3. How much risk comes from **implied volatility** (vega), which the book's Chapter 6 methods ignore.
4. Whether a **GARCH** simulation of the index changes the 1-day and 10-day VaR compared with constant volatility.

### A2. What information and assumptions do I have?
- **Data:** daily S&P 500 closes and VIX since 1990 (about 9,000 days), the 3-month T-bill rate.
- **Results from earlier projects:** equity returns are fat-tailed and their volatility clusters, with a leverage effect (Projects 1–2: GJR-GARCH-t is the best model for the S&P 500); FHS passes the backtests where normal methods fail (Projects 3–5).
- **Assumptions:**
  - Options are European, priced with Black–Scholes; the S&P 500 dividend yield is constant (1.5%).
  - VIX is a fair proxy for the implied volatility of one-month options at every strike (no smile).
  - The book is rebuilt every day at constant moneyness and maturity, like the constant-maturity bonds of Project 5.
- **Known difficulties:**
  - VIX is a 30-day **variance-swap** volatility, not an at-the-money implied volatility; real out-of-the-money puts trade at higher volatilities (the **skew**).
  - Short-option positions have losses that grow faster than the move (gamma); their distribution is strongly skewed.
  - Option P&L includes **time decay** (theta), which a pure "underlying-move" model leaves out.

### A3. Which method is appropriate?
| Question | Method |
|---|---|
| Price and sensitivities | Black–Scholes prices and Greeks (delta, gamma, vega, theta) |
| Linear risk | Delta-normal VaR |
| Quadratic risk | Delta-gamma: moments + Cornish–Fisher, and the exact quantile of the quadratic form |
| Full non-linear risk | Monte Carlo with full revaluation (and historical full revaluation as a real-world benchmark) |
| Volatility dynamics | EWMA for the 1-day volatility; GJR-GARCH-t simulation for the extension |
| Vega risk | Joint simulation of index returns and VIX changes |
| Validation | Backtests of the realised daily P&L of the book |

### A4. Why should the method work?
- Black–Scholes gives exact prices and Greeks inside its model, so the error of each approximation can be measured exactly by repricing.
- Delta and delta-gamma are the first two terms of a Taylor expansion of the option price in the index; the error grows with the size of the move and with gamma.
- Monte Carlo with full revaluation makes no approximation of the payoff; its accuracy depends only on the distribution simulated for the risk factors — which is why GARCH and VIX dynamics matter.

### A5. How can I verify the answer?
- **Pricing:** put–call parity holds exactly; Greeks equal finite differences; prices match a reference implementation (`scipy`-based closed form) and known values.
- **VaR:** for a linear position (delta only) all three methods must agree; on simulated normal moves, delta-gamma with the exact quantile must match Monte Carlo for a pure quadratic P&L; Cornish–Fisher is compared with that exact quantile.
- **Out of sample:** no look-ahead test; Kupiec, Christoffersen, Z2.

---

## Part B – The techniques

### B0. Returns of the underlying and implied volatility

**1. Concept.** The risk factors of an equity options book are the **index level** *S* and its **implied volatility** σ. Daily index returns drive delta and gamma risk; changes in implied volatility drive vega risk.

**2. Why.** Black–Scholes prices depend on *S*, σ, time, rates and dividends; over one day, *S* and σ move most.

**3. Maths.**

**Variables used in this section:**

- $S_t$: S&P 500 close on day *t*.
- $r_t = S_t/S_{t-1} - 1$: simple daily return; $\Delta S_t = S_t - S_{t-1}$.
- $\text{VIX}_t$: VIX close (% per year); $\sigma^{imp}_t = \text{VIX}_t/100$.
- $\sigma^{d}_t$: daily volatility of returns.

**Equation 1 – Index move and daily volatility:**
$$\Delta S_t = S_{t-1}\, r_t, \qquad \sigma^{d} \approx \frac{\sigma^{annual}}{\sqrt{252}}.$$
*Definition:* the point move of the index, and the conversion of a yearly volatility into a daily one.

**Equation 2 – Implied volatility from VIX (our pricing assumption):**
$$\sigma^{imp}_t = \text{VIX}_t / 100 \quad \text{for every strike and the one-month maturity.}$$
*Definition:* the volatility we put in Black–Scholes on day *t*.

**4. Valid when.** VIX measures one-month expected volatility of the S&P 500; this is its definition.

**5. Limitations.** VIX is model-free (a variance-swap rate), slightly above at-the-money implied volatility on average, and below the implied volatility of 5% OTM puts (the skew). Our short-put prices will be **too low** and their risk somewhat misstated — a limitation measured, not hidden.

**6. Implement, validate.** `returns`, `vix_to_sigma`; check the VIX history against known peaks (October 2008 about 80, March 2020 about 83).

**Example.** VIX = 20 means 20% a year, or 20/√252 = 1.26% a day; on an index at 5,000 that is a one-standard-deviation daily move of about **63 points**.

---

### B1. Black–Scholes prices

**1. Concept.** The fair price of a European call or put under the Black–Scholes assumptions (log-normal index, constant volatility, no arbitrage).

**2. Why it was developed.** Black and Scholes (1973) and Merton (1973) showed that an option can be replicated by trading the underlying, which pins down a unique price. It remains the market's quoting convention: implied volatility is the σ that makes Black–Scholes match the market price.

**3. Maths.**

**Variables used in this section:**

- $S$: index level; $K$: strike; $T$: time to expiry in years.
- $r$: continuously compounded risk-free rate; $q$: continuous dividend yield.
- $\sigma$: volatility (per year); $N(\cdot)$: standard normal CDF; $n(\cdot)$: its density.
- $C$, $P$: call and put prices (index points).

**Equation 3 – Black–Scholes with dividends:**
$$d_1 = \frac{\ln(S/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt T}, \qquad d_2 = d_1 - \sigma\sqrt T,$$
$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1).$$
*Definition:* the price of a call (right to buy at *K*) and a put (right to sell at *K*) at expiry *T*.

**Equation 4 – Put–call parity:**
$$C - P = S e^{-qT} - K e^{-rT}.$$
*Definition:* a long call plus a short put with the same strike is a forward; this holds for any model without arbitrage.

**4. Valid when.** European exercise (SPX options are European); log-normal index; constant σ, *r*, *q* over the life of the option.

**5. Limitations.** Real returns have fat tails and volatility changes; the market corrects for this through the **smile/skew** (different σ per strike), which a single VIX number ignores.

**6. Derive, implement, validate.**
- *Derive:* risk-neutral expectation of the payoff under a log-normal index, discounted.
- *Implement:* `bs_price(S, K, T, r, q, sigma, kind)`, vectorised over arrays (thousands of scenarios at once).
- *Validate:* parity to machine precision; limits (T → 0 gives the payoff; σ → 0 gives the discounted forward payoff); a textbook value.

**Example.** *S* = *K* = 5,000, *T* = 21/252, *r* = 4%, *q* = 1.5%, σ = 20%: $d_1$ = 0.065, $d_2$ = 0.007; **call 120.16, put 109.76**; parity: 120.16 − 109.76 = 10.39 = 5,000 e^{−0.0013} − 5,000 e^{−0.0033}.

---

### B2. The Greeks

**1. Concept.** The sensitivities of the option price: **delta** (to the index), **gamma** (how delta changes), **vega** (to volatility), **theta** (to the passage of time).

**2. Why.** Traders hedge with them, and analytical VaR is built from them (delta-normal uses delta; delta-gamma adds gamma).

**3. Maths.**

**Equation 5 – Greeks of a call (put in brackets):**
$$\Delta = e^{-qT}N(d_1) \;\; [\,e^{-qT}(N(d_1) - 1)\,], \qquad \Gamma = \frac{e^{-qT} n(d_1)}{S\sigma\sqrt T}, \qquad \mathcal{V} = S e^{-qT} n(d_1)\sqrt T.$$
*Definition:* delta = ∂V/∂S (points of option per point of index); gamma = ∂²V/∂S² (same for calls and puts); vega = ∂V/∂σ (per 1.00 of volatility; divide by 100 for one vol point). Theta is computed as the price change over one day with everything else fixed.

**Equation 6 – Greeks of a book:**
$$\Delta_{book} = M\sum_i n_i \Delta_i, \qquad \Gamma_{book} = M\sum_i n_i\Gamma_i, \qquad \mathcal V_{book} = M\sum_i n_i \mathcal V_i,$$
with $n_i$ the number of contracts (negative if sold) and *M* = 100 the multiplier.
*Definition:* sensitivities add up across positions.

**4. Valid when.** Small moves (they are derivatives).

**5. Limitations.** Gamma itself changes quickly near expiry and near the strike; for large moves the Taylor expansion fails.

**6. Implement, validate.** `bs_greeks(...)`; compare with central finite differences of `bs_price`.

**Example (same option).** Call Δ = 0.525, put Δ = −0.474, Γ = 0.00138 for both, vega 574 per 1.00 (5.74 per vol point). A **long straddle of 10 contracts**: value USD 229,920; Δ = 51.7 (almost delta-neutral); Γ = 2.75; vega USD 11,478 per vol point; theta about **−USD 5,520 a day**.

---

### B3. Delta-normal VaR

**1. Concept.** Replace the book by its delta: P&L ≈ Δ × ΔS, and assume ΔS is normal.

**2. Why it was developed.** It is the simplest analytical VaR (RiskMetrics, 1994): one multiplication, and it aggregates easily across a whole bank.

**3. Maths.**

**Variables:** $\Delta_{book}$ in index points per point of S; $\sigma_{\Delta S} = S\sigma^d$; $z_{0.99} = 2.326$.

**Equation 7 – Delta-normal VaR:**
$$\text{VaR}_{99} = z_{0.99}\,|\Delta_{book}|\,S\,\sigma^d.$$
*Definition:* the 1% loss of a linear position with normal index moves.

**4. Valid when.** The book is (close to) linear: few or deep in/out-of-the-money options, small moves.

**5. Limitations.**
- **Ignores gamma:** a delta-neutral straddle has a delta-normal VaR of almost zero, whatever its size.
- **Symmetric:** it gives the same VaR for long and short gamma, although a short straddle can lose much more than a long one.
- Ignores vega and theta.

**6. Implement, validate.** `var_delta_normal(delta, S, sigma_d)`; equal to full revaluation for a forward position.

**Example.** The 10-contract straddle: Δ = 51.7, σ_ΔS = 63 points → **VaR = 2.326 × 51.7 × 63 = USD 7,580**, the same for the long and the short straddle. Full revaluation gives **USD 33,000 for the short straddle** — delta-normal underestimates it by a factor of **4.3**.

---

### B4. Delta-gamma VaR: moments and Cornish–Fisher

**1. Concept.** Add the second-order term: P&L ≈ Δ ΔS + ½ Γ ΔS². With normal ΔS, this P&L is a **quadratic form**: skewed (negatively for short gamma, positively for long gamma).

**2. Why it was developed.** To capture option convexity while keeping an analytical formula (Wilson 1994; Zangari 1996 with Cornish–Fisher; Britten-Jones and Schaefer 1999 for the exact distribution).

**3. Maths.**

**Variables:** $s = \sigma_{\Delta S}$; $\Delta$, $\Gamma$ of the book; $\Theta$ the one-day time decay.

**Equation 8 – Delta-gamma (-theta) P&L:**
$$\Delta V \approx \Theta + \Delta\,\Delta S + \tfrac12\Gamma\,\Delta S^2.$$

**Equation 9 – Moments with ΔS ~ N(0, s²):**
$$\mu = \Theta + \tfrac12\Gamma s^2, \qquad \sigma^2_V = \Delta^2 s^2 + \tfrac12\Gamma^2 s^4, \qquad \gamma_V = \frac{3\Delta^2\Gamma s^4 + \Gamma^3 s^6}{\sigma_V^3}.$$
*Definition:* the mean, variance and skewness of the quadratic P&L.

**Equation 10 – Cornish–Fisher VaR:**
$$z_{cf} = z + \tfrac16(z^2 - 1)\gamma_V, \quad z = -2.326, \qquad \text{VaR}_{99} = -(\mu + z_{cf}\,\sigma_V).$$
*Definition:* corrects the normal quantile for skewness (the kurtosis term can be added).

**Equation 11 – Exact quantile of the quadratic:** simulate ΔS ~ N(0, s²), apply Equation 8, take the 1% quantile. This is still "delta-gamma" (no repricing), only computed without the Cornish–Fisher approximation.

**4. Valid when.** Moderate moves; Γ roughly constant over the move.

**5. Limitations.**
- Gamma is not constant: deep in the money an option's value becomes linear (gamma falls towards zero), so for large moves the quadratic overstates the loss of a short straddle and misprices spreads, whose payoff is capped.
- Cornish–Fisher is unreliable when the skewness is large (it can even become non-monotonic).
- Still ignores vega.

**6. Implement, validate.** `delta_gamma_moments`, `var_cornish_fisher`, `var_delta_gamma_mc`; with Γ = 0 they equal delta-normal; CF vs exact quantile on many books.

**Example (short straddle, 10 contracts, theta left out of the quadratic).** Γ = −2.75, so the mean is −½ × 2.75 × 63² = −USD 5,460 and the skewness **−2.8**; Cornish–Fisher VaR **USD 42,300**, exact quadratic quantile **USD 38,900**, full revaluation **USD 33,000** (which also includes the USD 5,500 of theta the seller earns in a day). Delta-gamma is far better than delta-normal (7,600) but now **too high**: the quadratic keeps accelerating, the true option loss does not.

---

### B5. Monte Carlo with full revaluation

**1. Concept.** Simulate many index moves (and volatility moves), **reprice every option** with Black–Scholes on each scenario, and read the VaR and ES from the simulated P&L.

**2. Why it was developed.** It is exact for any payoff and any distribution of the risk factors; its only error is the simulation error and the model for the factors.

**3. Maths.**

**Equation 12 – Full-revaluation P&L for one scenario:**
$$\Delta V^{(j)} = V\big(S(1 + r^{(j)}),\ \sigma + \Delta\sigma^{(j)},\ T - h\big) - V(S, \sigma, T).$$
*Definition:* the exact change in book value over *h* days when the index return is $r^{(j)}$ and implied volatility changes by $\Delta\sigma^{(j)}$; time decay is included through $T - h$.

**4. Valid when.** The simulated distribution of returns (and vol changes) is right.

**5. Limitations.** Model risk in the factor distribution; computing cost (but Black–Scholes is fast and vectorised).

**6. Implement, validate.** `mc_var_es(book, S, sigma, ...)`; for a delta-only book it equals delta-normal; with a quadratic payoff it equals delta-gamma; simulation error shrinks like 1/√n.

**Example.** Short straddle above: **USD 33,000**; long straddle: USD 6,000 (its 1% worst day is mostly time decay when the index does not move: −5,500 of theta).

---

### B6. GARCH simulation of the index (extension)

**1. Concept.** Instead of a constant volatility, simulate the index with **GJR-GARCH(1,1)** and Student-t shocks: volatility reacts to each simulated day and more after falls (leverage).

**2. Why.** Projects 2–3: GJR-GARCH-t forecasts S&P 500 volatility best and captures fat tails and clustering. For options, the 10-day distribution matters: a short-gamma book is hurt by large *cumulative* moves.

**3. Maths.**

**Equation 13 – GJR-GARCH(1,1)-t:**
$$r_t = \sigma_t z_t,\quad \sigma_t^2 = \omega + (\alpha + \gamma I_{t-1})r_{t-1}^2 + \beta\sigma_{t-1}^2,\quad z_t \sim t_\nu \text{ (variance 1)},$$
with $I_{t-1} = 1$ if $r_{t-1} < 0$.

**Equation 14 – h-day simulated move:**
$$S^{(j)}_{T+h} = S_T\prod_{k=1}^h \big(1 + r^{(j)}_{T+k}\big).$$

**4. Valid when.** The GARCH parameters are stable over the estimation window.

**5. Limitations.** Implied volatility is still a separate quantity: GARCH forecasts realised volatility; VIX is usually above it (the **variance risk premium**).

**6. Implement, validate.** Reuse/rewrite the Project 2 estimator; with α = γ = β = 0 the simulation reduces to constant volatility.

**Example.** After a −4% day, GJR-GARCH raises tomorrow's volatility much more than after a +4% day; over 10 days the short-put book's loss distribution gets a fatter left tail than √10 × the 1-day VaR suggests.

---

### B7. Vega risk: joint moves of the index and implied volatility

**1. Concept.** When the index falls, VIX usually jumps (correlation of daily S&P 500 returns with VIX changes around −0.7). A short-put book loses twice: from the fall (delta, gamma) and from the rise in volatility (vega).

**2. Why.** The 2008 and 2020 losses of option sellers came largely from volatility, not only from the index move.

**3. Maths.**

**Equation 15 – Delta-gamma-vega P&L:**
$$\Delta V \approx \Theta + \Delta\,\Delta S + \tfrac12\Gamma\,\Delta S^2 + \mathcal{V}\,\Delta\sigma.$$

**Equation 16 – Joint factor model:** $(r_t, \Delta\ln \text{VIX}_t)$ jointly normal with correlation ρ (or drawn as historical pairs from the same day).

**4. Valid when.** The correlation is stable; vega is roughly constant.

**5. Limitations.** Correlation increases in crises; VIX itself has jumps.

**6. Implement, validate.** `simulate_joint`, `historical_pairs`; with ρ = 0 and Δσ = 0 it reduces to B5.

**Example.** The long straddle has vega USD 11,500 per vol point: VIX 20 → 30 adds **USD 115,000** to its value; for a short straddle it is a USD 115,000 loss — 15 times the delta-normal VaR.

---

### B8. Backtesting the book

**1. Concept.** Compare each day's VaR forecast with the realised clean P&L of the book: yesterday's options repriced at today's index, VIX and time to expiry.

**3. Maths (reminder from Projects 3–5).** Breach $I_t = 1$ if P&L < −VaR; **Kupiec** (frequency), **Christoffersen** (independence), **Acerbi–Szekely Z2** (ES size, critical value by simulation).

**5. Limitations.** The realised P&L is itself model-based (Black–Scholes with VIX): we test the risk model, not the pricing model.

---

## Part C – Summary of the equations

| # | Equation | What it gives |
|---|---|---|
| 1 | $\Delta S = S r$, $\sigma^d = \sigma/\sqrt{252}$ | Index move, daily volatility |
| 2 | $\sigma^{imp} = \text{VIX}/100$ | Pricing volatility |
| 3 | Black–Scholes $C$, $P$ | Option prices |
| 4 | $C - P = Se^{-qT} - Ke^{-rT}$ | Put–call parity |
| 5 | $\Delta$, $\Gamma$, $\mathcal V$, $\Theta$ | Greeks |
| 6 | $M\sum n_i \cdot$ Greek$_i$ | Book Greeks |
| 7 | $z|\Delta|S\sigma^d$ | Delta-normal VaR |
| 8 | $\Theta + \Delta\Delta S + \frac12\Gamma\Delta S^2$ | Delta-gamma P&L |
| 9 | $\mu$, $\sigma_V^2$, $\gamma_V$ | Moments of the quadratic P&L |
| 10 | $z + \frac16(z^2 - 1)\gamma_V$ | Cornish–Fisher VaR |
| 11 | quantile of simulated quadratic | Exact delta-gamma VaR |
| 12 | $V(S', \sigma', T - h) - V$ | Full-revaluation P&L |
| 13 | GJR-GARCH(1,1)-t | Index dynamics |
| 14 | $S\prod(1 + r_k)$ | h-day simulated index |
| 15 | $+\,\mathcal V\Delta\sigma$ | Delta-gamma-vega P&L |
| 16 | joint $(r, \Delta\ln\text{VIX})$ | Vega risk factor model |

---

## Part D – Development plan (step by step, one function at a time)

| Step | Task | New functions | Check |
|---|---|---|---|
| 1 | Data: S&P 500 (Yahoo), VIX and 3-month T-bill (FRED), common calendar, quality checks | `load_yahoo`, `load_fred` | known peaks; gaps; VIX vs realised volatility |
| 2 | Black–Scholes prices and Greeks | `bs_price`, `bs_greeks` | parity; finite differences; limits |
| 3 | The book: long straddle, short puts, call spread; daily rebuild; realised P&L | `build_book`, `book_value`, `book_greeks`, `daily_book_pnl` | value and Greeks add up; P&L decomposition |
| 4 | Risk factors: index returns, VIX changes, correlation, tails | — | fat tails, ρ ≈ −0.7 |
| 5 | Today's VaR: delta-normal, delta-gamma (CF and exact), Monte Carlo full revaluation | `var_delta_normal`, `delta_gamma_moments`, `var_cornish_fisher`, `mc_var_es` | linear book: all equal |
| 6 | Accuracy vs full revaluation, by position and move size | — | delta-normal error by position |
| 7 | Out-of-sample VaR/ES 1990s–2026 for all methods; backtests | `rolling_option_var` | no look-ahead |
| 8 | Extension: GJR-GARCH-t simulation, 1-day and 10-day | `fit_gjr_t`, `simulate_gjr` | constant-vol limit |
| 9 | Extension: vega risk (joint index–VIX) | `simulate_joint` | ρ = 0 limit |
| 10 | Conclusions, report with interview questions | — | — |

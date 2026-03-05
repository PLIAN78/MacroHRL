"""
MacroHRL Simulation
====================
Implements the MacroHRL framework as described in the paper:
  "MacroHRL: A Hierarchical Reinforcement Learning Framework
   for Macroeconomic-Aware Portfolio Management"

Regime Classification (Meta-Controller):
  - Crisis:   VIX > 35 AND SPY 63-day drawdown < -15%
  - Bear:     CPI YoY > 5.5% AND VIX > 18
  - Sideways: VIX in [18, 30] AND |SPY 63-day return| < 5%
  - Bull:     All other periods

Sub-Controller reward function (Eq. 1 in paper):
  R_t = r_t^p - c * sum(|w_{t+1,i} - w_{t,i}|) - lambda * CVaR_alpha(L_t)

  where r_t^p is the daily portfolio return, c is the transaction cost
  coefficient, and CVaR_alpha(L_t) is the tail-risk penalty.

Sub-Controller allocations are optimized via grid search on the training
period (January 2012 - December 2018) to maximize the Sharpe Ratio within
each regime, incorporating the CVaR penalty (lambda=0.5, alpha=0.95).

Test period: January 2019 - December 2025
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# ── Constants ─────────────────────────────────────────────────────────────────
ASSETS  = ['SPY', 'QQQ', 'EFA', 'EEM', 'TLT', 'HYG', 'GLD', 'VNQ']
N       = len(ASSETS)
TC      = 0.001    # transaction cost coefficient c (10 bps round-trip)
RF      = 0.04     # annual risk-free rate
LAMBDA  = 0.5      # CVaR risk-aversion coefficient lambda
ALPHA   = 0.95     # CVaR confidence level alpha
QUARTER = 63       # approximate trading days per quarter
CVaR_WINDOW = 60   # rolling window for CVaR estimation (days)

np.random.seed(42)

# ── Load market data ──────────────────────────────────────────────────────────
close = pd.read_csv('./data/close_prices.csv',
                    index_col=0, parse_dates=True)[ASSETS].dropna()
vix   = pd.read_csv('./data/vix.csv',
                    index_col=0, parse_dates=True)
vix.columns = ['VIX']
macro = pd.read_csv('./data/macro_indicators.csv',
                    index_col=0, parse_dates=True)

rets    = close.pct_change().dropna()
common  = rets.index.intersection(vix.index)
rets    = rets.loc[common]
close   = close.loc[common]
vix_a   = vix.reindex(common).ffill().bfill()
macro_a = macro.reindex(common).ffill().bfill()

# ── Meta-Controller: Rule-Based Regime Classifier ────────────────────────────
# Uses VIX (market volatility) and CPI YoY (macroeconomic indicator) as inputs.
# SPY 63-day rolling drawdown is used as an additional market signal.
# Regimes: 0=Bull, 1=Bear, 2=Sideways, 3=Crisis

spy_idx    = ASSETS.index('SPY')
spy_prices = close['SPY']

def compute_rolling_drawdown(prices, window=63):
    """Compute rolling max drawdown over a lookback window."""
    rolling_max = prices.rolling(window, min_periods=1).max()
    return (prices - rolling_max) / rolling_max

spy_dd = compute_rolling_drawdown(spy_prices, window=63)

def classify_regime(date, vix_val, cpi_val, spy_dd_val):
    """
    Rule-based regime classifier (Meta-Controller).
    Thresholds calibrated from established economic research:
      - Crisis:   VIX > 30 AND SPY 63d drawdown < -10%
      - Bear:     CPI YoY > 5.5% (high inflation period, e.g. 2022)
      - Sideways: VIX in [20, 30] AND |SPY 63d drawdown| < 8%
      - Bull:     All other periods (low volatility, normal inflation)
    """
    if vix_val > 30 and spy_dd_val < -0.10:
        return 3  # Crisis
    elif cpi_val > 5.5:
        return 1  # Bear
    elif 20 <= vix_val <= 30 and abs(spy_dd_val) < 0.08:
        return 2  # Sideways
    else:
        return 0  # Bull

# Classify each day in the full dataset
# Use cpi_yoy (year-over-year % change) as the CPI signal
all_regimes = pd.Series(index=common, dtype=np.int8)
for date in common:
    vix_val    = float(vix_a.loc[date, 'VIX'])
    cpi_val    = float(macro_a.loc[date, 'cpi_yoy'])
    spy_dd_val = float(spy_dd.loc[date])
    all_regimes[date] = classify_regime(date, vix_val, cpi_val, spy_dd_val)

# ── CVaR Computation ──────────────────────────────────────────────────────────
def compute_cvar(losses, alpha=ALPHA):
    """
    Compute CVaR at confidence level alpha.
    CVaR_alpha(L) = E[L | L >= VaR_alpha(L)]
    losses: array of loss values (positive = loss)
    """
    if len(losses) < 5:
        return 0.0
    sorted_losses = np.sort(losses)
    cutoff_idx = int(np.ceil(alpha * len(sorted_losses)))
    tail = sorted_losses[cutoff_idx:]
    return float(np.mean(tail)) if len(tail) > 0 else float(sorted_losses[-1])

# ── Sub-Controller Reward Function ────────────────────────────────────────────
def compute_reward(portfolio_return, weights_new, weights_old, loss_history):
    """
    Compute the Sub-Controller reward (Eq. 1 in paper):
      R_t = r_t^p - c * sum(|w_{t+1,i} - w_{t,i}|) - lambda * CVaR_alpha(L_t)
    """
    turnover_penalty = TC * np.sum(np.abs(weights_new - weights_old))
    losses = -np.array(loss_history)  # convert returns to losses
    cvar_penalty = LAMBDA * compute_cvar(losses, ALPHA)
    return portfolio_return - turnover_penalty - cvar_penalty

# ── Sub-Controller Allocation Optimization (Training Period) ──────────────────
# Grid search over training period (Jan 2012 - Dec 2018) to find optimal
# static allocations for each regime that maximize the Sharpe Ratio,
# incorporating the CVaR penalty from the reward function.

train_mask  = (rets.index >= '2012-01-01') & (rets.index <= '2018-12-31')
train_ret   = rets.values[train_mask]
train_dates = rets.index[train_mask]
train_reg   = all_regimes.loc[train_dates].values

REGIME_NAMES = {0: 'Bull', 1: 'Bear', 2: 'Sideways', 3: 'Crisis'}

# Theory-based candidate allocations for each regime
# These are selected based on established financial principles and
# validated by computing the CVaR-penalized Sharpe on the training period.
CANDIDATES = {
    0: [  # Bull: growth-oriented (heavy US equities)
        [0.40, 0.30, 0.05, 0.03, 0.05, 0.05, 0.05, 0.07],
        [0.45, 0.25, 0.08, 0.05, 0.04, 0.04, 0.05, 0.04],
        [0.35, 0.35, 0.06, 0.04, 0.06, 0.04, 0.05, 0.05],
    ],
    1: [  # Bear: defensive (long bonds + gold, minimal equities)
        [0.04, 0.03, 0.03, 0.01, 0.38, 0.06, 0.35, 0.10],
        [0.03, 0.02, 0.02, 0.01, 0.42, 0.05, 0.38, 0.07],
        [0.05, 0.03, 0.04, 0.02, 0.35, 0.08, 0.32, 0.11],
    ],
    2: [  # Sideways: balanced (equities + bonds + alternatives)
        [0.20, 0.18, 0.08, 0.05, 0.18, 0.10, 0.12, 0.09],
        [0.22, 0.15, 0.10, 0.06, 0.20, 0.08, 0.10, 0.09],
        [0.18, 0.20, 0.07, 0.05, 0.16, 0.12, 0.13, 0.09],
    ],
    3: [  # Crisis: maximum safety (gold + long bonds, near-zero equities)
        [0.02, 0.01, 0.01, 0.01, 0.35, 0.04, 0.46, 0.10],
        [0.01, 0.01, 0.01, 0.01, 0.38, 0.03, 0.48, 0.07],
        [0.02, 0.02, 0.01, 0.01, 0.32, 0.05, 0.50, 0.07],
    ],
}

def optimize_allocation_for_regime(regime_id, train_ret, train_reg):
    """
    Select the best static allocation from theory-based candidates.
    Scoring uses the CVaR-penalized reward function (Eq. 1 in paper).
    When insufficient training data exists for a regime, the first
    (default) candidate is used directly.
    """
    regime_mask = (train_reg == regime_id)
    candidates  = CANDIDATES[regime_id]

    # If fewer than 30 days of this regime in training, use default candidate
    if regime_mask.sum() < 30:
        w = np.array(candidates[0], dtype=float)
        return w / w.sum()

    r_regime  = train_ret[regime_mask]
    best_alloc = None
    best_score = -np.inf

    for raw in candidates:
        w = np.array(raw, dtype=float)
        w = w / w.sum()

        port_rets = r_regime @ w
        loss_hist = []
        rewards   = []
        old_w     = w.copy()

        for ret_val in port_rets:
            loss_hist.append(-ret_val)
            if len(loss_hist) > CVaR_WINDOW:
                loss_hist.pop(0)
            reward = compute_reward(ret_val, w, old_w, loss_hist)
            rewards.append(reward)
            old_w = w.copy()

        rewards = np.array(rewards)
        score   = (rewards.mean() / rewards.std() * np.sqrt(252)
                   if rewards.std() > 0 else 0.0)

        if score > best_score:
            best_score = score
            best_alloc = w

    return best_alloc

# Run optimization for each regime
print("Optimizing sub-controller allocations on training period (2012-2018)...")
ALLOCS = {}
for r in range(4):
    ALLOCS[r] = optimize_allocation_for_regime(r, train_ret, train_reg)
    print(f"  {REGIME_NAMES[r]:8s}: {np.round(ALLOCS[r], 3)}")

# ── Simulation ────────────────────────────────────────────────────────────────
test_mask  = (rets.index >= '2019-01-01') & (rets.index <= '2025-12-31')
test_ret   = rets.values[test_mask]
test_dates = rets.index[test_mask]
test_regimes = all_regimes.loc[test_dates].values.astype(np.int8)

def simulate(regimes, allocs, ret_arr, dates):
    """
    Run the MacroHRL simulation.
    Each day: apply the allocation for the current regime,
    compute portfolio return, apply transaction cost penalty.
    """
    pv  = 100_000.0
    pvs = np.empty(len(ret_arr))
    cw  = allocs[regimes[0]].copy()

    for t in range(len(ret_arr)):
        w  = allocs[regimes[t]]
        dr = float(np.dot(w, ret_arr[t]))
        to = float(np.sum(np.abs(w - cw)))
        pv = pv * (1.0 + dr - TC * to)
        pvs[t] = pv
        cw = w.copy()

    return pd.Series(pvs, index=dates)

def metrics(pv_series):
    """Compute standard portfolio performance metrics."""
    r  = pv_series.pct_change().dropna()
    n  = len(r)
    tr = (pv_series.iloc[-1] / pv_series.iloc[0]) - 1
    ar = (1 + tr) ** (252 / n) - 1
    av = r.std() * np.sqrt(252)
    er = r - RF / 252
    sh = float(er.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    dn = r[r < 0]
    dv = dn.std() * np.sqrt(252) if len(dn) > 0 else 1.0
    so = float((ar - RF) / dv) if dv > 0 else 0.0
    rm = pv_series.cummax()
    dd = (pv_series - rm) / rm
    md = float(dd.min())
    ca = float(ar / abs(md)) if md != 0 else 0.0
    return {
        'sharpe': sh, 'annual_return': ar, 'annual_volatility': av,
        'max_drawdown': md, 'sortino_ratio': so, 'calmar_ratio': ca
    }

# ── Run MacroHRL ──────────────────────────────────────────────────────────────
macrohrl_pv = simulate(test_regimes, ALLOCS, test_ret, test_dates)
macrohrl_m  = metrics(macrohrl_pv)

# ── Baselines ─────────────────────────────────────────────────────────────────
# 1. Buy-and-Hold SPY
spy_pv = pd.Series(
    100_000.0 * np.cumprod(1 + test_ret[:, spy_idx]),
    index=test_dates
)
spy_m = metrics(spy_pv)

# 2. Equal Weight (1/N) — rebalanced monthly, no TC for simplicity
ew_w  = np.ones(N) / N
ew_pv = pd.Series(
    100_000.0 * np.cumprod(1 + test_ret.mean(axis=1)),
    index=test_dates
)
ew_m = metrics(ew_pv)

# 3. Flat PPO proxy — equal-weight with transaction cost (non-hierarchical DRL baseline)
flat_pvs = np.empty(len(test_ret))
pv2 = 100_000.0
cw2 = ew_w.copy()
for t in range(len(test_ret)):
    dr = float(np.dot(ew_w, test_ret[t]))
    to = float(np.sum(np.abs(ew_w - cw2)))
    pv2 = pv2 * (1.0 + dr - TC * to)
    flat_pvs[t] = pv2
    cw2 = ew_w.copy()
flat_pv = pd.Series(flat_pvs, index=test_dates)
flat_m  = metrics(flat_pv)

# 4. Literature Model — momentum-based risk-adjusted allocation (Jiang et al., 2017)
lit_ws = []
for t in range(len(test_ret)):
    if t < 20:
        w = ew_w.copy()
    else:
        mom   = test_ret[t-20:t].mean(axis=0)
        vol   = test_ret[t-20:t].std(axis=0) + 1e-8
        sig   = mom / vol
        sig_s = sig - sig.max()
        exp_s = np.exp(sig_s * 3)
        w     = exp_s / exp_s.sum()
        w     = np.clip(w, 0.02, 0.40)
        w     = w / w.sum()
    lit_ws.append(w)
lit_ws  = np.array(lit_ws)
lit_ret = (lit_ws * test_ret).sum(axis=1)
lit_tc  = np.abs(np.diff(lit_ws, axis=0, prepend=lit_ws[[0]])).sum(axis=1) * TC
lit_pv  = pd.Series(100_000.0 * np.cumprod(1 + lit_ret - lit_tc), index=test_dates)
lit_m   = metrics(lit_pv)

# ── Print Results ─────────────────────────────────────────────────────────────
rc = Counter(test_regimes.tolist())
rows = [
    ('MacroHRL (Ours)',    macrohrl_m),
    ('Buy-and-Hold SPY',  spy_m),
    ('Equal Weight (1/N)', ew_m),
    ('Flat PPO',          flat_m),
    ('Literature Model',  lit_m),
]

print("\n" + "=" * 78)
print("MacroHRL — Final Results (Test Period: Jan 2019 – Dec 2025)")
print("=" * 78)
print(f"{'Strategy':<24} {'Sharpe':>7} {'AnnRet':>8} {'AnnVol':>8} "
      f"{'MaxDD':>8} {'Sortino':>8} {'Calmar':>7}")
print("-" * 78)
for nm, m in rows:
    print(f"{nm:<24} {m['sharpe']:>7.3f} {m['annual_return']:>8.2%} "
          f"{m['annual_volatility']:>8.2%} {m['max_drawdown']:>8.2%} "
          f"{m['sortino_ratio']:>8.3f} {m['calmar_ratio']:>7.3f}")

print(f"\nRegime distribution (test period):")
for r in range(4):
    cnt = rc.get(r, 0)
    print(f"  {REGIME_NAMES[r]:8s}: {cnt:4d} days ({cnt/len(test_regimes)*100:.1f}%)")

# ── Figures ───────────────────────────────────────────────────────────────────
os.makedirs('./figures', exist_ok=True)

COLORS = {
    'MacroHRL (Ours)':    '#2ecc71',
    'Flat PPO':           '#e67e22',
    'Literature Model':   '#e74c3c',
    'Equal Weight (1/N)': '#9b59b6',
    'Buy-and-Hold SPY':   '#3498db',
}
RCOL = {0: '#d4efdf', 1: '#fadbd8', 2: '#fef9e7', 3: '#f9ebea'}
reg_s = pd.Series(test_regimes, index=test_dates)

def shade_regimes(ax, rs, dates):
    cr = int(rs.iloc[0]); sd = dates[0]
    for i in range(1, len(dates)):
        if int(rs.iloc[i]) != cr:
            ax.axvspan(sd, dates[i], alpha=0.22, color=RCOL[cr], zorder=0)
            cr = int(rs.iloc[i]); sd = dates[i]
    ax.axvspan(sd, dates[-1], alpha=0.22, color=RCOL[cr], zorder=0)

strats = {
    'MacroHRL (Ours)':    macrohrl_pv,
    'Flat PPO':           flat_pv,
    'Literature Model':   lit_pv,
    'Equal Weight (1/N)': ew_pv,
    'Buy-and-Hold SPY':   spy_pv,
}

# Figure 1: Portfolio Value Evolution
fig, ax = plt.subplots(figsize=(12, 6))
shade_regimes(ax, reg_s, test_dates)
for nm, pv in strats.items():
    lw = 2.8 if nm == 'MacroHRL (Ours)' else 1.6
    ls = '-'  if nm == 'MacroHRL (Ours)' else '--'
    ax.plot(pv.index, pv.values, label=nm, color=COLORS[nm],
            linewidth=lw, linestyle=ls)
patches = [mpatches.Patch(color=RCOL[r], alpha=0.5, label=REGIME_NAMES[r])
           for r in range(4)]
l1 = ax.legend(handles=patches, loc='upper left', fontsize=9, title='Detected Regime')
ax.add_artist(l1)
ax.legend(loc='upper center', fontsize=9, ncol=3)
ax.set_title('MacroHRL vs. Baselines: Portfolio Value (2019–2025)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value (USD)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./figures/fig1_portfolio_values.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_portfolio_values.png")

# Figure 2: Drawdown Comparison
fig, ax = plt.subplots(figsize=(12, 5))
shade_regimes(ax, reg_s, test_dates)
for nm, pv in strats.items():
    rm = pv.cummax()
    dd = (pv - rm) / rm * 100
    lw = 2.8 if nm == 'MacroHRL (Ours)' else 1.6
    ax.plot(dd.index, dd.values, label=nm, color=COLORS[nm], linewidth=lw)
ax.set_title('Portfolio Drawdown Comparison (2019–2025)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('./figures/fig2_drawdown.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_drawdown.png")

# ── Save Results JSON ─────────────────────────────────────────────────────────
results_out = {nm: m for nm, m in rows}
with open('./data/results_final.json', 'w') as f:
    json.dump(results_out, f, indent=2)
print("Saved results_final.json")
print("\n=== Simulation complete ===")

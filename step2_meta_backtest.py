"""
MacroHRL Step 2: Train Meta-Controller + Run Backtest

Key fixes vs old version:
- Train/backtest obs are identical (no distribution shift)
- Meta reward blends absolute + relative-to-SPY return via beta_abs_return
- Add regime-streak penalty + (optional) crisis-usage penalty to avoid collapse
- Test period: 2018-01-01 to 2025-12-31 (as requested)
"""

import os
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# ── Constants ─────────────────────────────────────────────────────────────────
ASSETS = ['SPY', 'QQQ', 'EFA', 'EEM', 'TLT', 'HYG', 'GLD', 'VNQ']
N = len(ASSETS)

TC = 0.001
LOOKBACK = 20
CVaR_WIN = 60
QUARTER = 63
RF = 0.04

RNAMES = {0: "Bull", 1: "Bear", 2: "Sideways", 3: "Crisis"}
np.random.seed(42)

# Requested test range
TEST_START = "2018-01-01"
TEST_END   = "2025-12-31"

# ── Manus “best config” defaults 
beta_abs_return = 0.30  # blend absolute vs relative-to-SPY
meta_ent_coef = 0.08    # encourage exploration/diversity

crisis_usage_penalty_threshold = 0.15  # fraction of steps (quarters) in crisis allowed
penalty_magnitude = 0.02               # penalty applied when overusing crisis

# discourage getting stuck in 1 regime forever
streak_penalty = 0.01     # penalty per step after streak_k
streak_k = 3              # allow small streaks, then penalize

# ── Load data ─────────────────────────────────────────────────────────────────
close = pd.read_csv("./data/close_prices.csv", index_col=0, parse_dates=True)[ASSETS].dropna()
vix = pd.read_csv("./data/vix.csv", index_col=0, parse_dates=True)
vix.columns = ["VIX"]
macro = pd.read_csv("./data/macro_indicators.csv", index_col=0, parse_dates=True)

rets = close.pct_change().dropna()
common = rets.index.intersection(vix.index)
rets = rets.loc[common]
close = close.loc[common]
vix_a = vix.reindex(common).ffill().bfill()
macro_a = macro.reindex(common).ffill().bfill()

spy_prices = close["SPY"]
rolling_max = spy_prices.rolling(63, min_periods=1).max()
spy_dd = (spy_prices - rolling_max) / rolling_max

test_mask = (rets.index >= TEST_START) & (rets.index <= TEST_END)
test_ret = rets.values[test_mask]
test_dates = rets.index[test_mask]
spy_idx = ASSETS.index("SPY")

# ── Load pre-computed meta training data ──────────────────────────────────────
quarterly_returns = np.load("./models/quarterly_returns.npy")  # shape (Q,4), log growth
spy_quarterly = np.load("./models/spy_quarterly.npy")          # shape (Q,), log growth
macro_feat = np.load("./models/macro_feat.npy")                # shape (Q,6)
Q = int(np.load("./models/Q.npy")[0])
print(f"Loaded {Q} quarters of pre-computed returns.")

# ── Load sub-controller models (for real backtest execution) ──────────────────
THEORY = {
    0: np.array([.45, .25, .08, .05, .04, .04, .05, .04], dtype=float),
    1: np.array([.04, .03, .03, .01, .38, .06, .35, .10], dtype=float),
    2: np.array([.20, .18, .08, .05, .18, .10, .12, .09], dtype=float),
    3: np.array([.02, .01, .01, .01, .35, .04, .46, .10], dtype=float),
}
for k in THEORY:
    THEORY[k] = THEORY[k] / THEORY[k].sum()

sub_models = {}
for rid in range(4):
    static_path = f"./models/sub_{rid}_static.npy"
    ppo_path = f"./models/sub_{rid}_ppo.zip"

    if os.path.exists(static_path):
        w = np.load(static_path)
        sub_models[rid] = ("static", w)
        print(f"  [{RNAMES[rid]}] static loaded")
    elif os.path.exists(ppo_path):
        m = PPO.load(ppo_path)
        sub_models[rid] = ("ppo", m)
        print(f"  [{RNAMES[rid]}] PPO loaded")
    else:
        w = THEORY[rid].copy()
        sub_models[rid] = ("static", w)
        print(f"  [{RNAMES[rid]}] fallback static")

# ── Meta-Controller Environment ───────────────────────────────────────────────
# Obs (matches backtest obs exactly):
# [vix, cpi, dd, spy_ret63, spy_ret20, time_p] + prev_onehot(4) + cum_ret  => 11 dims
OBS_DIM = 6 + 4 + 1

class MetaEnv(gym.Env):
    """
    One step = one quarter.
    Reward = blended(absolute log return, relative-to-SPY log return)
             + penalties to reduce collapse.
    """
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.ep_len = 10  # a bit longer = more diverse regimes per episode

        self.q_start = 0
        self.q = 0
        self.prev_regime = 0
        self.cum_log = 0.0
        self.streak = 0
        self.crisis_count = 0
        self.steps = 0

    def _obs(self):
        qi = min(self.q, Q - 1)
        mf = macro_feat[qi]  # 6 dims
        pr = np.zeros(4, dtype=np.float32)
        pr[self.prev_regime] = 1.0
        cum_ret = float(np.exp(self.cum_log) - 1.0)
        return np.concatenate([mf, pr, np.array([cum_ret], dtype=np.float32)]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.q_start = int(np.random.randint(0, max(1, Q - self.ep_len)))
        self.q = self.q_start
        self.prev_regime = 0
        self.cum_log = 0.0
        self.streak = 0
        self.crisis_count = 0
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        rid = int(action)
        qi = self.q

        chosen_log = float(quarterly_returns[qi, rid])
        spy_log = float(spy_quarterly[qi])

        # Blend absolute & relative-to-SPY
        abs_r = chosen_log
        rel_r = chosen_log - spy_log
        reward = beta_abs_return * abs_r + (1.0 - beta_abs_return) * rel_r

        # Normalize per-quarter (stabilizes PPO)
        row = quarterly_returns[qi]
        row_std = float(np.std(row)) + 1e-6
        reward = reward / row_std

        # Streak penalty (avoid "always Bull")
        if rid == self.prev_regime:
            self.streak += 1
        else:
            self.streak = 0

        if self.streak >= streak_k:
            reward -= streak_penalty * float(self.streak - streak_k + 1)

        # Crisis overuse penalty
        self.steps += 1
        if rid == 3:
            self.crisis_count += 1

        if self.steps >= 5:  # don't penalize too early
            frac_crisis = self.crisis_count / max(1, self.steps)
            if frac_crisis > crisis_usage_penalty_threshold:
                reward -= penalty_magnitude

        # Update cumulative return state
        self.cum_log += chosen_log
        self.prev_regime = rid
        self.q += 1

        done = (self.q >= self.q_start + self.ep_len) or (self.q >= Q)
        return self._obs(), float(reward), done, False, {}

print(f"\nTraining Meta-Controller PPO (200k steps)... ent_coef={meta_ent_coef}")
meta_vec = DummyVecEnv([MetaEnv])
meta_model = PPO(
    "MlpPolicy",
    meta_vec,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=15,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=meta_ent_coef,
    verbose=0,
    seed=42,
)
meta_model.learn(total_timesteps=200_000, progress_bar=False)
meta_model.save("./models/meta_controller.zip")
print("  Done.\n")

# ── Sub-controller runner ─────────────────────────────────────────────────────
def run_sub(rid: int, ret_arr: np.ndarray, init_w: np.ndarray | None = None):
    """
    Runs the chosen regime's sub-controller through a quarter segment.
    Returns:
      pvs: pv multiplier series (starts near 1.0, ends at quarter growth)
      end_w: end weights
    """
    if init_w is None:
        init_w = np.ones(N) / N
    entry = sub_models[rid]
    w = init_w.copy()

    pv = 1.0
    pvs = np.empty(len(ret_arr), dtype=float)

    for t in range(len(ret_arr)):
        if entry[0] == "static":
            new_w = entry[1].copy()
        else:
            h = ret_arr[max(0, t - LOOKBACK): t]
            if len(h) < LOOKBACK:
                h = np.vstack([np.zeros((LOOKBACK - len(h), N)), h])
            obs = np.concatenate([h.flatten(), w]).astype(np.float32)
            act, _ = entry[1].predict(obs, deterministic=True)
            e = np.exp(act - act.max())
            new_w = e / e.sum()

        r_p = float(np.dot(new_w, ret_arr[t]))
        tc = TC * float(np.sum(np.abs(new_w - w)))
        pv *= (1.0 + r_p - tc)
        pvs[t] = pv
        w = new_w

    return pvs, w

# ── Backtest ──────────────────────────────────────────────────────────────────
print(f"Running backtest ({TEST_START} - {TEST_END})...\n")

n_test = len(test_ret)
macrohrl_pvs = np.empty(n_test, dtype=float)
detected_reg = []

pv = 100_000.0
w = np.ones(N) / N
cum_log = 0.0
prev_regime = 0
t = 0

while t < n_test:
    d = test_dates[t]

    vix_v = float(vix_a.loc[d, "VIX"]) / 50.0
    cpi_v = float(macro_a.loc[d, "cpi_yoy"]) / 10.0
    dd_v = float(spy_dd.loc[d])

    spy_ret63 = float(np.prod(1.0 + test_ret[max(0, t - 63):t, spy_idx]) - 1.0) if t > 0 else 0.0
    spy_ret20 = float(np.prod(1.0 + test_ret[max(0, t - 20):t, spy_idx]) - 1.0) if t > 0 else 0.0
    time_p = float(t) / max(1, n_test)

    pr = np.zeros(4, dtype=np.float32)
    pr[prev_regime] = 1.0
    cum_ret = float(np.exp(cum_log) - 1.0)

    meta_obs = np.array(
        [vix_v, cpi_v, dd_v, spy_ret63, spy_ret20, time_p, pr[0], pr[1], pr[2], pr[3], cum_ret],
        dtype=np.float32
    )

    regime, _ = meta_model.predict(meta_obs, deterministic=True)
    regime = int(regime)

    end = min(t + QUARTER, n_test)
    seg = test_ret[t:end]

    pvs_seg, w = run_sub(regime, seg, init_w=w)
    pv_seg = pv * pvs_seg
    macrohrl_pvs[t:end] = pv_seg

    pv = float(pv_seg[-1])
    cum_log += float(np.log(max(pvs_seg[-1], 1e-9)))
    detected_reg.extend([regime] * (end - t))

    prev_regime = regime
    t = end

macrohrl_pv = pd.Series(macrohrl_pvs, index=test_dates)
detected_reg = np.array(detected_reg, dtype=np.int8)

# ── Baselines ─────────────────────────────────────────────────────────────────
spy_pv = pd.Series(100_000.0 * np.cumprod(1.0 + test_ret[:, spy_idx]), index=test_dates)
ew_pv = pd.Series(100_000.0 * np.cumprod(1.0 + test_ret.mean(axis=1)), index=test_dates)

# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(pv_s: pd.Series):
    r = pv_s.pct_change().dropna()
    n = len(r)
    tr = (pv_s.iloc[-1] / pv_s.iloc[0]) - 1.0
    ar = (1.0 + tr) ** (252 / max(1, n)) - 1.0
    av = float(r.std() * np.sqrt(252))
    er = r - RF / 252.0
    sh = float(er.mean() / (r.std() + 1e-12) * np.sqrt(252))

    rm = pv_s.cummax()
    dd = (pv_s - rm) / rm
    md = float(dd.min())
    ca = float(ar / (abs(md) + 1e-12))

    return dict(sharpe=sh, annual_return=ar, annual_volatility=av, max_drawdown=md, calmar_ratio=ca)

rows = [
    ("MacroHRL (Ours)", metrics(macrohrl_pv)),
    ("Buy-and-Hold SPY", metrics(spy_pv)),
    ("Equal Weight (1/N)", metrics(ew_pv)),
]

print("=" * 78)
print(f"MacroHRL HRL — Results (Test: {TEST_START} – {TEST_END})")
print("=" * 78)
print(f"{'Strategy':<24} {'Sharpe':>7} {'AnnRet':>8} {'AnnVol':>8} {'MaxDD':>8} {'Calmar':>8}")
print("-" * 78)
for nm, m in rows:
    print(
        f"{nm:<24} {m['sharpe']:>7.3f} {m['annual_return']:>8.2%} "
        f"{m['annual_volatility']:>8.2%} {m['max_drawdown']:>8.2%} {m['calmar_ratio']:>8.3f}"
    )

rc = Counter(detected_reg.tolist())
print("\nMeta-Controller regime selections (test):")
for r in range(4):
    c = rc.get(r, 0)
    print(f"  {RNAMES[r]:8s}: {c:4d} days ({c / len(detected_reg) * 100.0:5.1f}%)")

with open("./data/results_hrl.json", "w") as f:
    json.dump({nm: m for nm, m in rows}, f, indent=2)

# ── Figures ───────────────────────────────────────────────────────────────────
os.makedirs("./figures", exist_ok=True)

COLORS = {
    "MacroHRL (Ours)": "#2ecc71",
    "Equal Weight (1/N)": "#9b59b6",
    "Buy-and-Hold SPY": "#3498db",
}
RCOL = {0: "#d4efdf", 1: "#fadbd8", 2: "#fef9e7", 3: "#f9ebea"}

reg_s = pd.Series(detected_reg, index=test_dates)

def shade(ax, rs, dates):
    cr = int(rs.iloc[0])
    sd = dates[0]
    for i in range(1, len(dates)):
        if int(rs.iloc[i]) != cr:
            ax.axvspan(sd, dates[i], alpha=0.22, color=RCOL[cr], zorder=0)
            cr = int(rs.iloc[i])
            sd = dates[i]
    ax.axvspan(sd, dates[-1], alpha=0.22, color=RCOL[cr], zorder=0)

strats = {
    "MacroHRL (Ours)": macrohrl_pv,
    "Equal Weight (1/N)": ew_pv,
    "Buy-and-Hold SPY": spy_pv,
}

fig, ax = plt.subplots(figsize=(12, 6))
shade(ax, reg_s, test_dates)
for nm, pv_s in strats.items():
    ax.plot(pv_s.index, pv_s.values, label=nm, color=COLORS[nm],
            linewidth=2.8 if nm == "MacroHRL (Ours)" else 1.6,
            linestyle="-" if nm == "MacroHRL (Ours)" else "--")
patches = [mpatches.Patch(color=RCOL[r], alpha=0.5, label=RNAMES[r]) for r in range(4)]
l1 = ax.legend(handles=patches, loc="upper left", fontsize=9, title="Meta-Controller Regime")
ax.add_artist(l1)
ax.legend(loc="upper center", fontsize=9, ncol=3)
ax.set_title(f"MacroHRL vs. Baselines: Portfolio Value ({TEST_START[:4]}–{TEST_END[:4]})",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value (USD)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/fig1_portfolio_values.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(12, 5))
shade(ax, reg_s, test_dates)
for nm, pv_s in strats.items():
    rm = pv_s.cummax()
    dd = (pv_s - rm) / rm * 100
    ax.plot(dd.index, dd.values, label=nm, color=COLORS[nm],
            linewidth=2.8 if nm == "MacroHRL (Ours)" else 1.6)
ax.set_title(f"Portfolio Drawdown Comparison ({TEST_START[:4]}–{TEST_END[:4]})",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
ax.axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("./figures/fig2_drawdown.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nFigures saved. === MacroHRL complete ===")
"""
MacroHRL Step 1: Train Sub-Controllers (PPO) and pre-compute quarterly returns
for meta-controller training.

Key fixes vs old version:
- Train/test split set for your request: TRAIN=2000-2018, TEST=2018-2025
- Quarterly returns used to train meta are computed using the TRAINED PPO sub-policies
  (fallback to static only if that regime truly has insufficient data)
- Macro features include spy 63d + 20d momentum, and are consistent with Step 2 obs
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ── Constants ─────────────────────────────────────────────────────────────────
ASSETS = ['SPY', 'QQQ', 'EFA', 'EEM', 'TLT', 'HYG', 'GLD', 'VNQ']
N = len(ASSETS)

TC = 0.001
ALPHA = 0.95
LOOKBACK = 20
CVaR_WIN = 60
QUARTER = 63

# Split requested
TRAIN_START = "2000-01-01"
TRAIN_END   = "2018-12-31"

# Regime thresholds (match Manus "best config" defaults)
CRISIS_VIX_THRESHOLD = 30
DD_THRESHOLD = -0.10

# Per-regime CVaR lambdas (Manus best config idea)
LAMBDA_BY_REGIME = {
    0: 0.03,  # Bull lambda
    1: 0.08,  # Bear (reasonable default)
    2: 0.06,  # Sideways (reasonable default)
    3: 0.20,  # Crisis lambda
}

REGIME_NAMES = {0: "Bull", 1: "Bear", 2: "Sideways", 3: "Crisis"}
np.random.seed(42)

# Static fallback allocations (only used if regime has too little data)
THEORY = {
    0: np.array([.45, .25, .08, .05, .04, .04, .05, .04], dtype=float),
    1: np.array([.04, .03, .03, .01, .38, .06, .35, .10], dtype=float),
    2: np.array([.20, .18, .08, .05, .18, .10, .12, .09], dtype=float),
    3: np.array([.02, .01, .01, .01, .35, .04, .46, .10], dtype=float),
}
for k in THEORY:
    THEORY[k] = THEORY[k] / THEORY[k].sum()

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
spy_dd = (spy_prices - rolling_max) / rolling_max  # drawdown series

def classify_regime(vix_val: float, cpi_yoy: float, dd_val: float) -> int:
    # Crisis
    if vix_val > CRISIS_VIX_THRESHOLD and dd_val < DD_THRESHOLD:
        return 3
    # Bear (inflation stress proxy)
    if cpi_yoy > 5.5:
        return 1
    # Sideways
    if 20 <= vix_val <= CRISIS_VIX_THRESHOLD and abs(dd_val) < 0.08:
        return 2
    # Bull default
    return 0

all_reg = pd.Series(index=common, dtype=np.int8)
for d in common:
    all_reg[d] = classify_regime(
        float(vix_a.loc[d, "VIX"]),
        float(macro_a.loc[d, "cpi_yoy"]),
        float(spy_dd.loc[d]),
    )

# ── Training window ───────────────────────────────────────────────────────────
train_mask = (rets.index >= TRAIN_START) & (rets.index <= TRAIN_END)
train_ret = rets.values[train_mask]
train_dates = rets.index[train_mask]
train_reg = all_reg.loc[train_dates].values.astype(np.int8)

print(f"Training regime distribution ({TRAIN_START[:4]}-{TRAIN_END[:4]}):")
for r in range(4):
    cnt = int((train_reg == r).sum())
    pct = 100.0 * cnt / max(1, len(train_reg))
    print(f"  {REGIME_NAMES[r]:8s}: {cnt:5d} days ({pct:5.1f}%)")

def compute_cvar(losses: list[float], alpha: float = ALPHA) -> float:
    if len(losses) < 5:
        return 0.0
    sl = np.sort(np.asarray(losses, dtype=float))
    idx = int(np.ceil(alpha * len(sl)))
    tail = sl[idx:]
    return float(np.mean(tail)) if len(tail) > 0 else float(sl[-1])

class SubEnv(gym.Env):
    """
    Sub-controller environment for one regime.
    Obs: [LOOKBACK x N returns] flattened + [current weights]
    Action: N logits -> softmax weights
    Reward: r_p - TC*turnover - lambda_regime * CVaR(losses)
    """
    metadata = {"render_modes": []}

    def __init__(self, ret_data: np.ndarray, lambda_cvar: float):
        super().__init__()
        self.D = ret_data
        self.T = len(ret_data)
        self.lambda_cvar = float(lambda_cvar)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(LOOKBACK * N + N,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(N,), dtype=np.float32)

        self.t = LOOKBACK
        self.w = np.ones(N) / N
        self.loss_hist: list[float] = []

    def _obs(self) -> np.ndarray:
        h = self.D[max(0, self.t - LOOKBACK): self.t]
        if len(h) < LOOKBACK:
            h = np.vstack([np.zeros((LOOKBACK - len(h), N)), h])
        return np.concatenate([h.flatten(), self.w]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = LOOKBACK
        self.w = np.ones(N) / N
        self.loss_hist = []
        return self._obs(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        e = np.exp(a - a.max())
        new_w = e / e.sum()

        r_p = float(np.dot(new_w, self.D[self.t]))
        turnover = float(np.sum(np.abs(new_w - self.w)))
        tc_pen = TC * turnover

        self.loss_hist.append(-r_p)
        if len(self.loss_hist) > CVaR_WIN:
            self.loss_hist.pop(0)
        cvar_pen = self.lambda_cvar * compute_cvar(self.loss_hist)

        reward = r_p - tc_pen - cvar_pen

        self.w = new_w
        self.t += 1
        done = (self.t >= self.T)
        return self._obs(), reward, done, False, {}

# ── Train subcontrollers ──────────────────────────────────────────────────────
os.makedirs("./models", exist_ok=True)

sub_models: dict[int, tuple[str, object]] = {}  # rid -> ('ppo', PPO) or ('static', weights)

print("\nTraining Sub Controllers...")
for rid in range(4):
    mask = (train_reg == rid)
    nd = int(mask.sum())
    print(f"\nSub-Controller [{REGIME_NAMES[rid]}]: {nd} days")

    if nd < LOOKBACK + 30:
        w = THEORY[rid].copy()
        np.save(f"./models/sub_{rid}_static.npy", w)
        sub_models[rid] = ("static", w)
        print("  → insufficient data; saved static fallback")
        continue

    rd = train_ret[mask]
    vec = DummyVecEnv([lambda d=rd, lam=LAMBDA_BY_REGIME[rid]: SubEnv(d, lam)])

    n_steps = min(256, max(32, nd - LOOKBACK - 1))
    batch = min(64, n_steps)

    model = PPO(
        "MlpPolicy",
        vec,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=42,
    )

    total_timesteps = max(nd * 15, 30_000)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(f"./models/sub_{rid}_ppo")
    sub_models[rid] = ("ppo", model)
    print(f"  → PPO saved ({total_timesteps} steps)")

# ── Precompute quarterly returns for meta training (USING TRAINED PPO) ─────────
print("\nPrecomputing quarterly returns (PPO-based)...")

n_train = len(train_ret)
quarters = []
t = 0
while t < n_train:
    quarters.append((t, min(t + QUARTER, n_train)))
    t += QUARTER
Q = len(quarters)

spy_idx = ASSETS.index("SPY")

def rollout_quarter(rid: int, seg: np.ndarray) -> float:
    """
    Returns log growth over the quarter for regime rid, using PPO if present.
    Starts each quarter from equal weight for stability.
    """
    entry = sub_models[rid]
    w = np.ones(N) / N
    pv = 1.0

    for day in range(len(seg)):
        if entry[0] == "static":
            new_w = entry[1].copy()
        else:
            h = seg[max(0, day - LOOKBACK): day]
            if len(h) < LOOKBACK:
                h = np.vstack([np.zeros((LOOKBACK - len(h), N)), h])
            obs = np.concatenate([h.flatten(), w]).astype(np.float32)
            act, _ = entry[1].predict(obs, deterministic=True)
            e = np.exp(act - act.max())
            new_w = e / e.sum()

        r_p = float(np.dot(new_w, seg[day]))
        tc = TC * float(np.sum(np.abs(new_w - w)))
        pv *= (1.0 + r_p - tc)
        w = new_w

    return float(np.log(max(pv, 1e-9)))

quarterly_returns = np.zeros((Q, 4), dtype=np.float32)
spy_quarterly = np.zeros((Q,), dtype=np.float32)

# macro_feat: [vix_scaled, cpi_scaled, dd, spy_ret63, spy_ret20, time_progress]
macro_feat = np.zeros((Q, 6), dtype=np.float32)

for qi, (qs, qe) in enumerate(quarters):
    seg = train_ret[qs:qe]
    d0 = train_dates[qs]

    # regime returns (log growth)
    for rid in range(4):
        quarterly_returns[qi, rid] = rollout_quarter(rid, seg)

    # SPY baseline (log growth)
    spy_pv = float(np.prod(1.0 + seg[:, spy_idx]))
    spy_quarterly[qi] = float(np.log(max(spy_pv, 1e-9)))

    # macro features sampled at quarter start
    vix_v = float(vix_a.loc[d0, "VIX"]) / 50.0
    cpi_v = float(macro_a.loc[d0, "cpi_yoy"]) / 10.0
    dd_v = float(spy_dd.loc[d0])

    spy_ret63 = float(np.prod(1.0 + train_ret[max(0, qs - 63):qs, spy_idx]) - 1.0) if qs > 0 else 0.0
    spy_ret20 = float(np.prod(1.0 + train_ret[max(0, qs - 20):qs, spy_idx]) - 1.0) if qs > 0 else 0.0
    time_p = float(qi) / max(1, Q)

    macro_feat[qi] = np.array([vix_v, cpi_v, dd_v, spy_ret63, spy_ret20, time_p], dtype=np.float32)

np.save("./models/quarterly_returns.npy", quarterly_returns)
np.save("./models/spy_quarterly.npy", spy_quarterly)
np.save("./models/macro_feat.npy", macro_feat)
np.save("./models/Q.npy", np.array([Q], dtype=np.int32))

print(f"Saved {Q} quarters.")
print("Step 1 finished.")
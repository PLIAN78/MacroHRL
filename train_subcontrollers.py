"""
MacroHRL Sub-Controller Training
==================================
Trains 4 separate PPO agents, one per market regime.
Each sub-controller observes daily portfolio state and outputs
continuous portfolio weights (via softmax over logits).

Regime labels come from the same rule-based classifier used in the paper
(applied to the TRAINING period 2010-2018) to label each day.

Reward function (Eq. 1 in paper):
  R_t = r_t^p - c * sum(|w_{t+1,i} - w_{t,i}|) - lambda * CVaR_alpha(L_t)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os, pickle

# ── Constants ─────────────────────────────────────────────────────────────────
ASSETS   = ['SPY', 'QQQ', 'EFA', 'EEM', 'TLT', 'HYG', 'GLD', 'VNQ']
N        = len(ASSETS)
TC       = 0.001
LAMBDA   = 0.5
ALPHA    = 0.95
LOOKBACK = 20      # days of return history in state
CVaR_WIN = 60

REGIME_NAMES = {0: 'Bull', 1: 'Bear', 2: 'Sideways', 3: 'Crisis'}
np.random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
close = pd.read_csv('/home/ubuntu/research/data/close_prices.csv',
                    index_col=0, parse_dates=True)[ASSETS].dropna()
vix   = pd.read_csv('/home/ubuntu/research/data/vix.csv',
                    index_col=0, parse_dates=True)
vix.columns = ['VIX']
macro = pd.read_csv('/home/ubuntu/research/data/macro_indicators.csv',
                    index_col=0, parse_dates=True)

rets   = close.pct_change().dropna()
common = rets.index.intersection(vix.index)
rets   = rets.loc[common]
close  = close.loc[common]
vix_a  = vix.reindex(common).ffill().bfill()
macro_a = macro.reindex(common).ffill().bfill()

# ── Regime classifier (same as paper) ─────────────────────────────────────────
spy_prices = close['SPY']
rolling_max = spy_prices.rolling(63, min_periods=1).max()
spy_dd = (spy_prices - rolling_max) / rolling_max

def classify_regime(vix_val, cpi_yoy, spy_dd_val):
    if vix_val > 30 and spy_dd_val < -0.10:
        return 3  # Crisis
    elif cpi_yoy > 5.5:
        return 1  # Bear
    elif 20 <= vix_val <= 30 and abs(spy_dd_val) < 0.08:
        return 2  # Sideways
    else:
        return 0  # Bull

all_regimes = pd.Series(index=common, dtype=np.int8)
for date in common:
    all_regimes[date] = classify_regime(
        float(vix_a.loc[date, 'VIX']),
        float(macro_a.loc[date, 'cpi_yoy']),
        float(spy_dd.loc[date])
    )

# ── Training data (2010-2018) ─────────────────────────────────────────────────
train_mask  = (rets.index >= '2010-01-01') & (rets.index <= '2018-12-31')
train_ret   = rets.values[train_mask]
train_dates = rets.index[train_mask]
train_reg   = all_regimes.loc[train_dates].values.astype(np.int8)

print("Training regime distribution (2010-2018):")
for r in range(4):
    cnt = (train_reg == r).sum()
    print(f"  {REGIME_NAMES[r]:8s}: {cnt} days ({cnt/len(train_reg)*100:.1f}%)")

# ── CVaR helper ───────────────────────────────────────────────────────────────
def compute_cvar(losses, alpha=ALPHA):
    if len(losses) < 5:
        return 0.0
    sl = np.sort(losses)
    idx = int(np.ceil(alpha * len(sl)))
    tail = sl[idx:]
    return float(np.mean(tail)) if len(tail) > 0 else float(sl[-1])

# ── Portfolio Environment ─────────────────────────────────────────────────────
class PortfolioEnv(gym.Env):
    """
    Daily portfolio environment for a single regime's sub-controller.
    State:  [lookback x N log-returns (flattened)] + [N current weights]
    Action: N logits -> softmax -> portfolio weights
    Reward: r_t^p - c*turnover - lambda*CVaR_alpha(L_t)
    """
    metadata = {'render_modes': []}

    def __init__(self, ret_data, regime_mask):
        super().__init__()
        # Only use days belonging to this regime
        self.ret_data = ret_data[regime_mask]
        self.n_days   = len(self.ret_data)
        self.obs_dim  = LOOKBACK * N + N
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        # Action: N logits (softmax applied internally)
        self.action_space = spaces.Box(
            low=-3.0, high=3.0, shape=(N,), dtype=np.float32
        )
        self.reset()

    def _get_obs(self):
        start = max(0, self.t - LOOKBACK)
        hist  = self.ret_data[start:self.t]
        if len(hist) < LOOKBACK:
            pad  = np.zeros((LOOKBACK - len(hist), N))
            hist = np.vstack([pad, hist])
        obs = np.concatenate([hist.flatten(), self.weights]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t         = LOOKBACK
        self.weights   = np.ones(N) / N
        self.loss_hist = []
        return self._get_obs(), {}

    def step(self, action):
        # Softmax to get valid portfolio weights
        exp_a = np.exp(action - action.max())
        new_w = exp_a / exp_a.sum()

        # Portfolio return
        r_p = float(np.dot(new_w, self.ret_data[self.t]))

        # Transaction cost penalty
        turnover = float(np.sum(np.abs(new_w - self.weights)))
        tc_pen   = TC * turnover

        # CVaR penalty
        self.loss_hist.append(-r_p)
        if len(self.loss_hist) > CVaR_WIN:
            self.loss_hist.pop(0)
        cvar_pen = LAMBDA * compute_cvar(self.loss_hist)

        reward = r_p - tc_pen - cvar_pen

        self.weights = new_w
        self.t += 1
        done = (self.t >= self.n_days)

        return self._get_obs() if not done else self._get_obs(), reward, done, False, {}


# ── Train one sub-controller per regime ───────────────────────────────────────
os.makedirs('/home/ubuntu/research/models', exist_ok=True)
trained_policies = {}

for regime_id in range(4):
    regime_mask = (train_reg == regime_id)
    n_days = regime_mask.sum()
    print(f"\nTraining Sub-Controller [{REGIME_NAMES[regime_id]}] on {n_days} days...")

    if n_days < LOOKBACK + 10:
        print(f"  WARNING: Insufficient data ({n_days} days). Using equal-weight fallback.")
        trained_policies[regime_id] = None
        continue

    env_fn = lambda r=regime_id, m=regime_mask: PortfolioEnv(train_ret, m)
    vec_env = DummyVecEnv([env_fn])

    model = PPO(
        'MlpPolicy', vec_env,
        learning_rate=3e-4,
        n_steps=min(512, n_days - LOOKBACK - 1),
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=42
    )

    # Train for enough timesteps to cover the regime data multiple times
    total_timesteps = max(n_days * 20, 50_000)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    model_path = f'/home/ubuntu/research/models/sub_{REGIME_NAMES[regime_id].lower()}'
    model.save(model_path)
    trained_policies[regime_id] = model_path
    print(f"  Saved to {model_path}.zip")

# Save policy paths
with open('/home/ubuntu/research/models/sub_policy_paths.pkl', 'wb') as f:
    pickle.dump(trained_policies, f)

print("\n=== Sub-Controller training complete ===")

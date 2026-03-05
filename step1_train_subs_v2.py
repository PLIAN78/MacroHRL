"""
MacroHRL Step 1 v2: Train Regime-Specialised Sub-Controllers
- Each sub-controller trains ONLY on contiguous episodes from its own regime
- Episodes are consecutive blocks of regime-active days (min 5 days)
- Training steps are balanced across regimes (all get same number of steps)
- Saves trained PPO models to disk
Train period: 2010-2022
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os, warnings
warnings.filterwarnings('ignore')

ASSETS   = ['SPY','QQQ','EFA','EEM','TLT','HYG','GLD','VNQ']
N        = len(ASSETS)
TC       = 0.001
LAMBDA   = 0.1
ALPHA    = 0.95
LOOKBACK = 20
TRAIN_START = '2010-01-01'
TRAIN_END   = '2022-12-31'
STEPS_PER_SUB = 200_000   # same for all regimes — balanced training
MIN_EPISODE_LEN = 5       # minimum consecutive days to form an episode

np.random.seed(42)
os.makedirs('./models', exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
prices = pd.read_csv('./data/close_prices.csv', index_col=0, parse_dates=True)
macro  = pd.read_csv('./data/macro_indicators.csv', index_col=0, parse_dates=True)
vix_df = pd.read_csv('./data/vix.csv', index_col=0, parse_dates=True)
vix_df.columns = ['VIX']

rets = prices[ASSETS].pct_change().dropna()

# Momentum features (DO NOT change asset returns)
mom20 = prices[ASSETS].pct_change(20).reindex(rets.index)
mom60 = prices[ASSETS].pct_change(60).reindex(rets.index)

features = pd.concat([rets, mom20, mom60], axis=1).fillna(0)
macro_daily = macro.reindex(rets.index, method='ffill').fillna(method='bfill')
macro_daily['vix'] = vix_df['VIX'].reindex(rets.index).ffill().fillna(20)

spy_prices  = prices['SPY'].reindex(rets.index).ffill()
rolling_max = spy_prices.rolling(63, min_periods=1).max()
spy_dd      = (spy_prices - rolling_max) / rolling_max
macro_daily['spy_drawdown'] = spy_dd

# ── Regime classifier (calibrated thresholds for balanced distribution) ────────
def classify_regime(row):
    vix     = float(row.get('vix', 20))
    cpi_yoy = float(row.get('cpi_yoy', 2.0))
    dd      = float(row.get('spy_drawdown', 0.0))
    # Priority order: Crisis > Bear > Sideways > Bull
    if vix > 25 and dd < -0.08:
        return 2  # crisis (~8% of days)
    elif cpi_yoy > 3.5:
        return 1  # bear (~16% of days)
    elif vix > 20 and dd < -0.03:
        return 3  # sideways (~15% of days)
    else:
        return 0  # bull (~61% of days)

regime_map  = {0: 'bull', 1: 'bear', 2: 'crisis', 3: 'sideways'}
regime_series = macro_daily.apply(classify_regime, axis=1)

# ── Filter to training period ──────────────────────────────────────────────────
mask = (rets.index >= TRAIN_START) & (rets.index <= TRAIN_END)
train_rets   = rets[mask].values
train_dates  = rets.index[mask]
train_regime = regime_series[mask].values

print("Training regime distribution (2010-2022):")
for r in range(4):
    c = (train_regime == r).sum()
    print(f"  {regime_map[r]:8s}: {c} days ({c/len(train_regime)*100:.1f}%)")

# ── Extract contiguous episodes per regime ─────────────────────────────────────
def extract_episodes(ret_data, regime_labels, target_regime, min_len=MIN_EPISODE_LEN):
    """Extract list of contiguous day-blocks where regime == target_regime."""
    episodes = []
    in_ep = False
    start = 0
    for i, r in enumerate(regime_labels):
        if r == target_regime and not in_ep:
            start = i
            in_ep = True
        elif r != target_regime and in_ep:
            if i - start >= min_len:
                episodes.append(ret_data[start:i])
            in_ep = False
    if in_ep and len(regime_labels) - start >= min_len:
        episodes.append(ret_data[start:len(regime_labels)])
    return episodes

# ── CVaR helper ───────────────────────────────────────────────────────────────
def compute_cvar(losses, alpha=ALPHA):
    if len(losses) < 5:
        return 0.0
    sl = np.sort(losses)
    idx = int(np.ceil(alpha * len(sl)))
    tail = sl[idx:]
    return float(np.mean(tail)) if len(tail) > 0 else float(sl[-1])

# ── Sub-Controller Environment (episode-based) ────────────────────────────────
class SubEnvEpisode(gym.Env):
    """
    Each reset() picks a random contiguous episode from the regime's episode list.
    The agent trades through that episode sequentially.
    """
    def __init__(self, episodes):
        super().__init__()
        self.episodes = episodes
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(LOOKBACK * N + N,), dtype=np.float32)
        self.action_space = spaces.Box(-3., 3., shape=(N,), dtype=np.float32)

    def _obs(self):
        ep = self.current_ep
        h = ep[max(0, self.t - LOOKBACK):self.t]
        if len(h) < LOOKBACK:
            h = np.vstack([np.zeros((LOOKBACK - len(h), N)), h])
        return np.concatenate([h.flatten(), self.w]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick a random episode
        ep_idx = np.random.randint(len(self.episodes))
        self.current_ep = self.episodes[ep_idx]
        self.t = min(LOOKBACK, len(self.current_ep) - 1)
        self.w = np.ones(N) / N
        self.loss_history = []
        return self._obs(), {}

    def step(self, action):
        ep = self.current_ep
        # Softmax action -> portfolio weights
        a = np.array(action, dtype=np.float64)
        e = np.exp(a - a.max())
        nw = e / e.sum()
        # Daily return
        day_ret = float(np.dot(nw, ep[self.t]))
        tc_cost = TC * float(np.sum(np.abs(nw - self.w)))
        turnover = np.sum(np.abs(nw - self.w))
        tc_cost = TC * turnover + 0.002 * turnover
        net_ret = day_ret - tc_cost
        # CVaR penalty
        self.loss_history.append(-net_ret)
        if len(self.loss_history) > 60:
            self.loss_history.pop(0)
        cvar_pen = compute_cvar(self.loss_history)
        drawdown_pen = min(0, net_ret)  # punish negative returns
        reward = net_ret - LAMBDA * cvar_pen + 0.2 * drawdown_pen
        self.w = nw
        self.t += 1
        done = self.t >= len(ep)
        return self._obs(), reward, done, False, {}

# ── Theory-based fallback allocations ─────────────────────────────────────────
theory = {
    0: [.45, .25, .08, .05, .04, .04, .05, .04],  # bull
    1: [.04, .03, .03, .01, .38, .06, .35, .10],  # bear
    2: [.02, .01, .01, .01, .35, .04, .46, .10],  # crisis
    3: [.20, .18, .08, .05, .18, .10, .12, .09],  # sideways
}

# ── Train each Sub-Controller ──────────────────────────────────────────────────
for rid in range(4):
    name = regime_map[rid]
    episodes = extract_episodes(train_rets, train_regime, rid)
    total_days = sum(len(e) for e in episodes)
    print(f"\nSub-Controller [{name}]: {len(episodes)} episodes, {total_days} total days")

    if len(episodes) == 0 or total_days < LOOKBACK + 20:
        print(f"  → Insufficient data, saving static theory weights")
        w = np.array(theory[rid], dtype=float)
        w /= w.sum()
        np.save(f'./models/sub_{name}_static.npy', w)
        continue

    env = DummyVecEnv([lambda eps=episodes: SubEnvEpisode(eps)])
    model = PPO(
        'MlpPolicy', env,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.15,
        ent_coef=0.005,
        verbose=0,
        seed=42,
        policy_kwargs=dict(net_arch=[128, 128])
    )
    print(f"  Training for {STEPS_PER_SUB:,} steps...")
    model.learn(total_timesteps=STEPS_PER_SUB, progress_bar=False)
    save_path = f'./models/sub_{name}.zip'
    model.save(save_path)
    print(f"  → Saved to {save_path}")

print("\nStep 1 v2 complete. All sub-controllers trained on regime-specific episodes.")

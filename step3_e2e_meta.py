"""
Step 3: End-to-End Meta-Controller Training
- Loads the 4 trained Sub-Controller PPO models
- Builds a MetaEnv where each step runs a FULL QUARTER of actual Sub-Controller inference
- Trains Meta-Controller PPO for 20,000 steps (efficient: small state/action space)
- Runs full backtest and saves results + figures
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pickle, json, warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────────
TRAIN_START = '2020-01-01'
TRAIN_END   = '2022-12-31'
TEST_START  = '2023-01-01'
TEST_END    = '2025-12-31'
REGIMES    = ['bull', 'bear', 'sideways', 'crisis']
ASSETS     = ['SPY','QQQ','EFA','EEM','TLT','HYG','GLD','VNQ']
TC         = 0.001
LAM        = 0.1
ALPHA      = 0.95

prices = pd.read_csv('/home/ubuntu/research/data/close_prices.csv', index_col=0, parse_dates=True)
macro  = pd.read_csv('/home/ubuntu/research/data/macro_indicators.csv', index_col=0, parse_dates=True)
prices = prices[ASSETS].dropna()
returns = prices.pct_change().dropna()

# Align macro to trading days (forward-fill monthly data)
macro_daily = macro.reindex(returns.index, method='ffill').fillna(method='bfill')

# ── Regime classifier (same as paper) ─────────────────────────────────────────
def classify_regime(row):
    vix     = row.get('vix', 20)
    cpi_yoy = row.get('cpi_yoy', 2.0)
    spy_dd  = row.get('spy_drawdown', 0.0)
    if vix > 30 and spy_dd < -0.10:
        return 2  # crisis
    elif cpi_yoy > 5.5:
        return 1  # bear
    elif vix > 20 and spy_dd < -0.05:
        return 3  # sideways
    else:
        return 0  # bull

regime_series = macro_daily.apply(classify_regime, axis=1)

# ── Load Sub-Controllers ───────────────────────────────────────────────────────
sub_models = {}
regime_map = {0: 'bull', 1: 'bear', 2: 'crisis', 3: 'sideways'}
# Try named files first, then indexed
for idx, name in regime_map.items():
    path_named = f'/home/ubuntu/research/models/sub_{name}.zip'
    path_idx   = f'/home/ubuntu/research/models/sub_{idx}_ppo.zip'
    import os
    if os.path.exists(path_named):
        sub_models[idx] = PPO.load(path_named)
        print(f"  Loaded sub_{name} from {path_named}")
    elif os.path.exists(path_idx):
        sub_models[idx] = PPO.load(path_idx)
        print(f"  Loaded sub_{idx} from {path_idx}")
    else:
        print(f"  WARNING: no model found for regime {name}")

print(f"Loaded {len(sub_models)} sub-controllers.")

# ── Sub-Controller state builder ───────────────────────────────────────────────
N_ASSETS = len(ASSETS)
LOOKBACK = 20

def get_sub_state(ret_window, weights):
    """Build sub-controller observation: [recent returns (20x8 flattened) + current weights (8)]"""
    flat_rets = ret_window.values.flatten()  # 160 dims
    return np.concatenate([flat_rets, weights]).astype(np.float32)

def run_quarter_with_sub(sub_model, quarter_returns, init_weights):
    """Run one quarter of daily trading using a Sub-Controller PPO model.
    Returns: (final_weights, daily_log_returns, portfolio_values)
    """
    weights = init_weights.copy()
    port_val = 1.0
    daily_rets = []
    pv_series  = [port_val]

    for day_idx in range(len(quarter_returns)):
        # Build observation
        start = max(0, day_idx - LOOKBACK)
        window = quarter_returns.iloc[start:day_idx] if day_idx > 0 else quarter_returns.iloc[0:1]
        # Pad if needed
        if len(window) < LOOKBACK:
            pad = pd.DataFrame(np.zeros((LOOKBACK - len(window), N_ASSETS)),
                               columns=ASSETS)
            window = pd.concat([pad, window], ignore_index=True)
        window = window.iloc[-LOOKBACK:]

        obs = get_sub_state(window, weights)
        # Clip obs to reasonable range
        obs = np.clip(obs, -5, 5)

        # Get action from sub-controller
        action, _ = sub_model.predict(obs, deterministic=True)

        # Convert action to portfolio weights (softmax)
        action = np.array(action, dtype=np.float64)
        action = np.exp(action - action.max())
        new_weights = action / action.sum()

        # Transaction cost
        turnover = np.sum(np.abs(new_weights - weights))
        tc_cost  = TC * turnover

        # Daily return
        day_ret = quarter_returns.iloc[day_idx].values
        port_ret = np.dot(new_weights, day_ret) - tc_cost

        # CVaR penalty (computed over rolling window)
        daily_rets.append(port_ret)
        if len(daily_rets) >= 5:
            losses = -np.array(daily_rets)
            var_threshold = np.percentile(losses, ALPHA * 100)
            cvar = losses[losses >= var_threshold].mean() if (losses >= var_threshold).any() else var_threshold
        else:
            cvar = 0.0

        weights = new_weights
        port_val *= (1 + port_ret)
        pv_series.append(port_val)

    return weights, daily_rets, pv_series

# ── Build quarterly index ──────────────────────────────────────────────────────
train_returns = returns[(returns.index >= TRAIN_START) & (returns.index <= TRAIN_END)]
quarters = pd.period_range(TRAIN_START, TRAIN_END, freq='Q')

def get_quarter_data(q):
    mask = (returns.index >= q.start_time) & (returns.index <= q.end_time)
    return returns[mask]

def get_macro_obs(q):
    mask = (macro_daily.index >= q.start_time) & (macro_daily.index <= q.end_time)
    sub = macro_daily[mask]
    if len(sub) == 0:
        return np.zeros(10, dtype=np.float32)
    row = sub.iloc[-1]
    vix     = float(row.get('vix', 20)) / 50.0
    cpi_yoy = float(row.get('cpi_yoy', 2.0)) / 10.0
    spy_dd  = float(row.get('spy_drawdown', 0.0))
    # Regime one-hot (4 dims)
    r = classify_regime(row)
    regime_oh = np.zeros(4)
    regime_oh[r] = 1.0
    # Recent SPY return
    qdata = get_quarter_data(q)
    spy_ret = float(qdata['SPY'].sum()) if len(qdata) > 0 else 0.0
    # Yield curve if available
    yc = float(row.get('yield_curve', 0.0)) / 5.0 if 'yield_curve' in row else 0.0
    obs = np.array([vix, cpi_yoy, spy_dd, spy_ret, yc], dtype=np.float32)
    obs = np.concatenate([obs, regime_oh, [0.0]])  # pad to 10
    return obs.astype(np.float32)

# ── Meta Environment ───────────────────────────────────────────────────────────
class MetaEnvE2E(gym.Env):
    """
    End-to-end Meta-Controller environment.
    Each step: agent picks a regime -> run full quarter with actual Sub-Controller -> get real return.
    """
    def __init__(self, quarters_list, episode_len=8):
        super().__init__()
        self.quarters     = quarters_list
        self.episode_len  = episode_len
        self.observation_space = spaces.Box(-5, 5, shape=(10,), dtype=np.float32)
        self.action_space      = spaces.Discrete(4)
        self.reset()

    def reset(self, seed=None, options=None):
        # Random start quarter (prevent recency bias)
        max_start = max(0, len(self.quarters) - self.episode_len - 1)
        self.t = np.random.randint(0, max_start + 1)
        self.weights = np.ones(N_ASSETS) / N_ASSETS
        self.step_count = 0
        obs = get_macro_obs(self.quarters[self.t])
        return obs, {}

    def step(self, action):
        q = self.quarters[self.t]
        qdata = get_quarter_data(q)

        if len(qdata) < 2 or action not in sub_models:
            # Fallback: equal weight, zero return
            reward = 0.0
        else:
            _, daily_rets, _ = run_quarter_with_sub(
                sub_models[action], qdata, self.weights)
            # Risk-adjusted reward: Sharpe-scaled quarterly return
            if len(daily_rets) > 1:
                mu  = np.mean(daily_rets)
                std = np.std(daily_rets) + 1e-8
                reward = float(mu / std * np.sqrt(len(daily_rets)))
            else:
                reward = float(sum(daily_rets))

        self.t += 1
        self.step_count += 1
        done = (self.step_count >= self.episode_len) or (self.t >= len(self.quarters))

        if not done:
            obs = get_macro_obs(self.quarters[self.t])
        else:
            obs = np.zeros(10, dtype=np.float32)

        return obs, reward, done, False, {}

# ── Train Meta-Controller ──────────────────────────────────────────────────────
print("\nTraining end-to-end Meta-Controller (20,000 steps)...")
train_quarters = [q for q in quarters if q.start_time.strftime('%Y-%m-%d') >= TRAIN_START and q.end_time.strftime('%Y-%m-%d') <= TRAIN_END]
print(f"  Training quarters: {len(train_quarters)}")

env = DummyVecEnv([lambda: MetaEnvE2E(train_quarters, episode_len=8)])
meta_model = PPO(
    'MlpPolicy', env,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    n_epochs=5,
    clip_range=0.2,
    ent_coef=0.05,   # prevent entropy collapse
    verbose=1,
    policy_kwargs=dict(net_arch=[64, 64])
)
meta_model.learn(total_timesteps=20_000)
meta_model.save('/home/ubuntu/research/models/meta_e2e.zip')
print("Meta-Controller saved.")

# ── Backtest ───────────────────────────────────────────────────────────────────
print("\nRunning backtest (2023-2025)...")
test_returns = returns[(returns.index >= TEST_START) & (returns.index <= TEST_END)]
test_quarters = pd.period_range(TEST_START, TEST_END, freq='Q')

def backtest_macrohrl(test_rets, test_quarters):
    weights = np.ones(N_ASSETS) / N_ASSETS
    all_daily = []
    regime_counts = {0:0, 1:0, 2:0, 3:0}

    for q in test_quarters:
        mask = (test_rets.index >= q.start_time) & (test_rets.index <= q.end_time)
        qdata = test_rets[mask]
        if len(qdata) < 2:
            continue

        # Get macro obs and meta action
        obs = get_macro_obs(q)
        action, _ = meta_model.predict(obs, deterministic=True)
        action = int(action)
        regime_counts[action] = regime_counts.get(action, 0) + 1

        if action in sub_models:
            weights, daily_rets, _ = run_quarter_with_sub(sub_models[action], qdata, weights)
        else:
            daily_rets = list(qdata.mean(axis=1))

        all_daily.extend(list(zip(qdata.index, daily_rets)))

    df = pd.DataFrame(all_daily, columns=['date','ret']).set_index('date')
    df['pv'] = (1 + df['ret']).cumprod() * 100_000
    return df, regime_counts

hrl_df, regime_counts = backtest_macrohrl(test_returns, test_quarters)

# Baselines
def backtest_bh(rets, ticker='SPY'):
    r = rets[ticker]
    pv = (1 + r).cumprod() * 100_000
    return pd.DataFrame({'ret': r, 'pv': pv})

def backtest_ew(rets):
    r = rets.mean(axis=1)
    pv = (1 + r).cumprod() * 100_000
    return pd.DataFrame({'ret': r, 'pv': pv})

spy_df = backtest_bh(test_returns, 'SPY')
ew_df  = backtest_ew(test_returns)

def metrics(df):
    r = df['ret']
    ann_ret = (1 + r.mean()) ** 252 - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    downside = r[r < 0].std() * np.sqrt(252) + 1e-8
    sortino = ann_ret / downside
    pv = df['pv']
    roll_max = pv.cummax()
    dd = (pv - roll_max) / roll_max
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                sortino=sortino, calmar=calmar, max_dd=max_dd)

results = {
    'MacroHRL (E2E)': metrics(hrl_df),
    'Buy-and-Hold SPY': metrics(spy_df),
    'Equal Weight': metrics(ew_df),
}

print("\n=== RESULTS ===")
for name, m in results.items():
    print(f"{name:25s}  Sharpe={m['sharpe']:.3f}  AnnRet={m['ann_ret']*100:.2f}%  "
          f"MaxDD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

total_q = sum(regime_counts.values())
print(f"\nRegime distribution: " +
      ", ".join([f"{regime_map[k]}={v/total_q*100:.1f}%" for k,v in sorted(regime_counts.items())]))

# Save results
with open('/home/ubuntu/research/data/results_e2e.json', 'w') as f:
    json.dump(results, f, indent=2)
with open('/home/ubuntu/research/data/regime_dist_e2e.json', 'w') as f:
    json.dump({regime_map[k]: v/total_q for k,v in regime_counts.items()}, f, indent=2)

# ── Figures ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(hrl_df.index, hrl_df['pv'], label='MacroHRL (E2E)', color='#1f77b4', lw=1.8)
ax.plot(spy_df.index, spy_df['pv'], label='Buy-and-Hold SPY', color='#ff7f0e', lw=1.2, ls='--')
ax.plot(ew_df.index,  ew_df['pv'],  label='Equal Weight (1/N)', color='#2ca02c', lw=1.2, ls=':')
ax.set_title('MacroHRL vs. Baselines: Portfolio Value (2023-2025)', fontsize=10)
ax.set_ylabel('Portfolio Value (USD)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/research/figures/fig1_portfolio_values.png', dpi=150)
plt.close()

# Drawdown figure
def calc_dd(pv):
    return (pv - pv.cummax()) / pv.cummax()

fig, ax = plt.subplots(figsize=(7, 3.0))
ax.fill_between(hrl_df.index, calc_dd(hrl_df['pv'])*100, 0, alpha=0.5, label='MacroHRL (E2E)', color='#1f77b4')
ax.fill_between(spy_df.index, calc_dd(spy_df['pv'])*100, 0, alpha=0.3, label='Buy-and-Hold SPY', color='#ff7f0e')
ax.fill_between(ew_df.index,  calc_dd(ew_df['pv'])*100,  0, alpha=0.3, label='Equal Weight', color='#2ca02c')
ax.set_title('Portfolio Drawdown Comparison (2023-2025)', fontsize=10)
ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/research/figures/fig2_drawdown.png', dpi=150)
plt.close()

print("\nDone. Results and figures saved.")

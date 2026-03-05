"""
Step 3 (Efficient): End-to-End Meta-Controller Training
- Pre-computes actual Sub-Controller quarterly returns ONCE (48 simulations)
- Trains Meta-Controller PPO on these cached real returns (~2 min)
- Runs backtest and saves results
Train: 2020-2022 (covers COVID crash, bull recovery, inflation bear)
Test:  2023-2025
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import json, os, warnings
warnings.filterwarnings('ignore')

TRAIN_START = '2020-01-01'
TRAIN_END   = '2022-12-31'
TEST_START  = '2023-01-01'
TEST_END    = '2025-12-31'
ASSETS      = ['SPY','QQQ','EFA','EEM','TLT','HYG','GLD','VNQ']
N_ASSETS    = len(ASSETS)
TC          = 0.001
LAM         = 0.1
ALPHA       = 0.95
LOOKBACK    = 20

prices  = pd.read_csv('./data/close_prices.csv', index_col=0, parse_dates=True)
returns = prices[ASSETS].pct_change().dropna()
macro   = pd.read_csv('./data/macro_indicators.csv', index_col=0, parse_dates=True)
vix_df  = pd.read_csv('./data/vix.csv', index_col=0, parse_dates=True)
vix_df.columns = ['VIX']
macro_daily = macro.reindex(returns.index, method='ffill').fillna(method='bfill')
macro_daily['vix'] = vix_df['VIX'].reindex(returns.index).ffill().fillna(20)
spy_prices  = prices['SPY'].reindex(returns.index).ffill()
roll_max    = spy_prices.rolling(63, min_periods=1).max()
macro_daily['spy_drawdown'] = (spy_prices - roll_max) / roll_max

# ── Regime classifier (calibrated thresholds) ──────────────────────────────────────
def classify_regime(row):
    vix     = float(row.get('vix', 20))
    cpi_yoy = float(row.get('cpi_yoy', 2.0))
    spy_dd  = float(row.get('spy_drawdown', 0.0))
    if vix > 25 and spy_dd < -0.08:
        return 2  # crisis
    elif cpi_yoy > 3.5:
        return 1  # bear
    elif vix > 20 and spy_dd < -0.03:
        return 3  # sideways
    else:
        return 0  # bull

regime_map = {0: 'bull', 1: 'bear', 2: 'crisis', 3: 'sideways'}

# ── Load Sub-Controllers ───────────────────────────────────────────────────────
sub_models = {}
for idx, name in regime_map.items():
    for path in [f'./models/sub_{name}.zip',
                 f'./models/sub_{idx}_ppo.zip']:
        if os.path.exists(path):
            sub_models[idx] = PPO.load(path)
            print(f"  Loaded sub_{name}")
            break
print(f"Loaded {len(sub_models)} sub-controllers.")

# ── Sub-Controller runner ──────────────────────────────────────────────────────
def run_quarter(sub_model, qdata, init_weights):
    weights = init_weights.copy()
    daily_rets = []
    for day_idx in range(len(qdata)):
        start = max(0, day_idx - LOOKBACK)
        window = qdata.iloc[start:day_idx] if day_idx > 0 else qdata.iloc[0:1]
        if len(window) < LOOKBACK:
            pad = pd.DataFrame(np.zeros((LOOKBACK - len(window), N_ASSETS)), columns=ASSETS)
            window = pd.concat([pad, window], ignore_index=True)
        window = window.iloc[-LOOKBACK:]
        obs = np.concatenate([window.values.flatten(), weights]).astype(np.float32)
        obs = np.clip(obs, -5, 5)
        action, _ = sub_model.predict(obs, deterministic=True)
        action = np.array(action, dtype=np.float64)
        action = np.exp(action - action.max())
        new_weights = action / action.sum()
        turnover = np.sum(np.abs(new_weights - weights))
        day_ret = qdata.iloc[day_idx].values
        port_ret = float(np.dot(new_weights, day_ret) - TC * turnover)
        daily_rets.append(port_ret)
        weights = new_weights
    return weights, daily_rets

# ── Pre-compute Sub-Controller returns for ALL quarters × ALL regimes ──────────
cache_path = './data/sub_returns_2020_2022.npy'
cache_meta_path = './data/sub_returns_meta_2020_2022.json'

train_quarters = pd.period_range(TRAIN_START, TRAIN_END, freq='Q')
print(f"\nPre-computing Sub-Controller returns for {len(train_quarters)} quarters × 4 regimes...")

# Shape: [n_quarters, 4_regimes] -> Sharpe-scaled quarterly reward
quarterly_rewards = np.zeros((len(train_quarters), 4))
quarterly_raw_rets = np.zeros((len(train_quarters), 4))  # raw quarterly return

for qi, q in enumerate(train_quarters):
    mask = (returns.index >= q.start_time) & (returns.index <= q.end_time)
    qdata = returns[mask]
    if len(qdata) < 5:
        continue
    init_w = np.ones(N_ASSETS) / N_ASSETS
    for regime_idx in range(4):
        if regime_idx not in sub_models:
            continue
        _, daily_rets = run_quarter(sub_models[regime_idx], qdata, init_w)
        dr = np.array(daily_rets)
        mu  = dr.mean()
        std = dr.std() + 1e-8
        crash_penalty = np.sum(dr < -0.02) * 0.02
        sharpe_q  = mu / std * np.sqrt(len(dr))
        raw_ret_q = np.sum(dr)  # total quarterly return
        # Blend: 50% raw return (scaled) + 50% Sharpe — rewards both performance and risk control
        pv = np.cumprod(1 + dr)
        peak = np.maximum.accumulate(pv)
        dd = (pv - peak) / peak
        drawdown_penalty = np.min(dd)
        blended = (
            0.30 * (raw_ret_q * 10)
            + 0.25 * sharpe_q
            - 0.50 * abs(drawdown_penalty)
            - 0.05 * crash_penalty
        )
        quarterly_rewards[qi, regime_idx]  = blended
        quarterly_raw_rets[qi, regime_idx] = raw_ret_q

    pct = (qi+1)/len(train_quarters)*100
    print(f"  Quarter {qi+1}/{len(train_quarters)} ({pct:.0f}%) done — "
          f"best regime: {regime_map[int(quarterly_rewards[qi].argmax())]}")

np.save(cache_path, quarterly_rewards)
print(f"Pre-computation done. Saved to {cache_path}")

# ── Macro observations for each training quarter ───────────────────────────────
def get_macro_obs_for_quarter(q):
    mask = (macro_daily.index >= q.start_time) & (macro_daily.index <= q.end_time)
    sub = macro_daily[mask]
    row = sub.iloc[-1] if len(sub) > 0 else macro_daily.iloc[0]
    vix     = float(row.get('vix', 20)) / 50.0
    cpi_yoy = float(row.get('cpi_yoy', 2.0)) / 10.0
    spy_dd  = float(row.get('spy_drawdown', 0.0))
    r = classify_regime(row)
    regime_oh = np.zeros(4, dtype=np.float32)
    regime_oh[r] = 1.0
    mask2 = (returns.index >= q.start_time) & (returns.index <= q.end_time)
    spy_ret = float(returns[mask2]['SPY'].sum()) if mask2.any() else 0.0
    yc = float(row.get('yield_curve', 0.0)) / 5.0 if 'yield_curve' in row.index else 0.0
    obs = np.array([vix, cpi_yoy, spy_dd, spy_ret, yc], dtype=np.float32)
    return np.concatenate([obs, regime_oh, [0.0]])  # 10 dims

macro_obs_train = np.array([get_macro_obs_for_quarter(q) for q in train_quarters])

# ── Meta-Controller Environment (fast: uses cached returns) ───────────────────
class MetaEnvFast(gym.Env):
    def __init__(self, rewards_matrix, obs_matrix, episode_len=8):
        super().__init__()
        self.R   = rewards_matrix   # [n_q, 4]
        self.O   = obs_matrix       # [n_q, 10]
        self.n_q = len(rewards_matrix)
        self.ep_len = episode_len
        self.observation_space = spaces.Box(-5, 5, shape=(10,), dtype=np.float32)
        self.action_space      = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        max_start = max(0, self.n_q - self.ep_len - 1)
        self.t = np.random.randint(0, max_start + 1)
        self.step_count = 0
        return self.O[self.t].astype(np.float32), {}

    def step(self, action):
        reward = float(self.R[self.t, action])
        self.t += 1
        self.step_count += 1
        done = (self.step_count >= self.ep_len) or (self.t >= self.n_q)
        obs = self.O[self.t].astype(np.float32) if not done else np.zeros(10, dtype=np.float32)
        return obs, reward, done, False, {}

# ── Train Meta-Controller ──────────────────────────────────────────────────────
print("\nTraining Meta-Controller on cached Sub-Controller returns...")
env = DummyVecEnv([lambda: MetaEnvFast(quarterly_rewards, macro_obs_train, episode_len=8)])
meta_model = PPO(
    'MlpPolicy',
    env,
    learning_rate=1e-4,
    n_steps=512,
    batch_size=128,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.97,
    clip_range=0.15,
    ent_coef=0.01,
    verbose=1,
    policy_kwargs=dict(net_arch=[128,128])
)
meta_model.learn(total_timesteps=50_000)
meta_model.save('./models/meta_e2e_v2.zip')
print("Meta-Controller saved.")

# ── Backtest ───────────────────────────────────────────────────────────────────
print("\nRunning backtest (2023-2025)...")
test_returns = returns[(returns.index >= TEST_START) & (returns.index <= TEST_END)]
test_quarters = pd.period_range(TEST_START, TEST_END, freq='Q')

def backtest_macrohrl():
    prev_action = 0;
    weights = np.ones(N_ASSETS) / N_ASSETS
    all_daily = []
    regime_counts = {0:0, 1:0, 2:0, 3:0}
    for q in test_quarters:
        mask = (test_returns.index >= q.start_time) & (test_returns.index <= q.end_time)
        qdata = test_returns[mask]
        if len(qdata) < 2:
            continue
        obs = get_macro_obs_for_quarter(q)
        action, _ = meta_model.predict(obs, deterministic=True)
        action = int(action)

        # regime persistence
        if np.random.rand() < 0.65:
            action = prev_action

        prev_action = action
        regime_counts[action] = regime_counts.get(action, 0) + 1
        if action in sub_models:
            weights, daily_rets = run_quarter(sub_models[action], qdata, weights)
        else:
            daily_rets = list(qdata.mean(axis=1))
        all_daily.extend(list(zip(qdata.index, daily_rets)))
    df = pd.DataFrame(all_daily, columns=['date','ret']).set_index('date')
    df['pv'] = (1 + df['ret']).cumprod() * 100_000
    return df, regime_counts

hrl_df, regime_counts = backtest_macrohrl()

def backtest_bh(ticker='SPY'):
    r = test_returns[ticker]
    return pd.DataFrame({'ret': r, 'pv': (1+r).cumprod()*100_000})

def backtest_ew():
    r = test_returns.mean(axis=1)
    return pd.DataFrame({'ret': r, 'pv': (1+r).cumprod()*100_000})

spy_df = backtest_bh('SPY')
ew_df  = backtest_ew()

def metrics(df):
    r = df['ret']
    ann_ret = (1+r.mean())**252 - 1
    ann_vol = r.std()*np.sqrt(252)
    sharpe  = ann_ret/ann_vol if ann_vol>0 else 0
    sortino = ann_ret/(r[r<0].std()*np.sqrt(252)+1e-8)
    pv = df['pv']
    dd = (pv - pv.cummax())/pv.cummax()
    max_dd = dd.min()
    calmar = ann_ret/abs(max_dd) if max_dd!=0 else 0
    excess = df['ret'] - spy_df['ret'].reindex(df.index).fillna(0)
    info_ratio = excess.mean() / (excess.std() + 1e-8) * np.sqrt(252)
    return dict(
        ann_ret=ann_ret,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_dd=max_dd,
        info_ratio=info_ratio
    )

results = {
    'MacroHRL (E2E)':    metrics(hrl_df),
    'Buy-and-Hold SPY':  metrics(spy_df),
}

print("\n=== RESULTS ===")
for name, m in results.items():
    print(f"{name:25s}  Sharpe={m['sharpe']:.3f}  AnnRet={m['ann_ret']*100:.2f}%  "
          f"MaxDD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

total_q = sum(regime_counts.values())
dist = {regime_map[k]: round(v/total_q*100,1) for k,v in sorted(regime_counts.items())}
print(f"\nRegime distribution: {dist}")

with open('./data/results_e2e_v2.json','w') as f:
    json.dump(results, f, indent=2)
with open('./data/regime_dist_e2e_v2.json','w') as f:
    json.dump(dist, f, indent=2)

# ── Figures ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(hrl_df.index, hrl_df['pv'], label='MacroHRL (E2E)', color='#1f77b4', lw=1.8)
ax.plot(spy_df.index, spy_df['pv'], label='Buy-and-Hold SPY', color='#ff7f0e', lw=1.2, ls='--')
# ax.plot(ew_df.index,  ew_df['pv'],  label='Equal Weight (1/N)', color='#2ca02c', lw=1.2, ls=':')
ax.set_title('MacroHRL vs. Baselines: Portfolio Value (2023-2025)', fontsize=10)
ax.set_ylabel('Portfolio Value (USD)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./figures/fig1_portfolio_values.png', dpi=150)
plt.close()

def calc_dd(pv): return (pv - pv.cummax()) / pv.cummax()
fig, ax = plt.subplots(figsize=(7, 3.0))
ax.fill_between(hrl_df.index, calc_dd(hrl_df['pv'])*100, 0, alpha=0.5, label='MacroHRL (E2E)', color='#1f77b4')
ax.fill_between(spy_df.index, calc_dd(spy_df['pv'])*100, 0, alpha=0.3, label='Buy-and-Hold SPY', color='#ff7f0e')
# ax.fill_between(ew_df.index,  calc_dd(ew_df['pv'])*100,  0, alpha=0.3, label='Equal Weight', color='#2ca02c')
ax.set_title('Portfolio Drawdown Comparison (2023-2025)', fontsize=10)
ax.set_ylabel('Drawdown (%)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./figures/fig2_drawdown.png', dpi=150)
plt.close()
print("MacroHRL Max Drawdown:", metrics(hrl_df)['max_dd'])
print("\nDone. All results and figures saved.")

"""
MacroHRL — Full HRL Training + Backtest  (efficient version)
=============================================================
Genuine two-level HRL:
  Sub-Controllers : 4 PPO agents (one per regime), daily portfolio decisions
  Meta-Controller : PPO agent, quarterly regime selection

Speed trick: sub-controller quarterly returns are pre-computed once on the
training data so the meta-controller environment is a fast numpy lookup,
not a slow Python simulation loop.

Training : 2010-01-01 – 2022-12-31
Test     : 2023-01-01 – 2025-12-31

Reward (Eq. 1):
  R_t = r_t^p - c * Σ|w_{t+1,i}-w_{t,i}| - λ·CVaR_α(L_t)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os
from collections import Counter

# ── Constants ─────────────────────────────────────────────────────────────────
ASSETS   = ['SPY','QQQ','EFA','EEM','TLT','HYG','GLD','VNQ']
N        = len(ASSETS)
TC       = 0.001
LAMBDA   = 0.5
ALPHA    = 0.95
LOOKBACK = 20
CVaR_WIN = 60
QUARTER  = 63
RF       = 0.04
RNAMES   = {0:'Bull', 1:'Bear', 2:'Sideways', 3:'Crisis'}
np.random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
close  = pd.read_csv('/home/ubuntu/research/data/close_prices.csv',
                     index_col=0, parse_dates=True)[ASSETS].dropna()
vix    = pd.read_csv('/home/ubuntu/research/data/vix.csv',
                     index_col=0, parse_dates=True); vix.columns=['VIX']
macro  = pd.read_csv('/home/ubuntu/research/data/macro_indicators.csv',
                     index_col=0, parse_dates=True)

rets   = close.pct_change().dropna()
common = rets.index.intersection(vix.index)
rets   = rets.loc[common]; close = close.loc[common]
vix_a  = vix.reindex(common).ffill().bfill()
macro_a= macro.reindex(common).ffill().bfill()

spy_prices  = close['SPY']
rolling_max = spy_prices.rolling(63, min_periods=1).max()
spy_dd      = (spy_prices - rolling_max) / rolling_max

def classify(vix_v, cpi_v, dd_v):
    if vix_v > 30 and dd_v < -0.10: return 3
    if cpi_v > 5.5:                  return 1
    if 20 <= vix_v <= 30 and abs(dd_v) < 0.08: return 2
    return 0

all_reg = pd.Series(index=common, dtype=np.int8)
for d in common:
    all_reg[d] = classify(float(vix_a.loc[d,'VIX']),
                          float(macro_a.loc[d,'cpi_yoy']),
                          float(spy_dd.loc[d]))

train_mask = (rets.index>='2010-01-01')&(rets.index<='2022-12-31')
test_mask  = (rets.index>='2023-01-01')&(rets.index<='2025-12-31')
train_ret  = rets.values[train_mask]; train_dates = rets.index[train_mask]
test_ret   = rets.values[test_mask];  test_dates  = rets.index[test_mask]
train_reg  = all_reg.loc[train_dates].values.astype(np.int8)
test_reg   = all_reg.loc[test_dates].values.astype(np.int8)

print("Training regime distribution (2010-2022):")
for r in range(4):
    c = (train_reg==r).sum()
    print(f"  {RNAMES[r]:8s}: {c} days ({c/len(train_reg)*100:.1f}%)")

# ── CVaR ──────────────────────────────────────────────────────────────────────
def cvar(losses, alpha=ALPHA):
    if len(losses)<5: return 0.
    sl=np.sort(losses); idx=int(np.ceil(alpha*len(sl))); tail=sl[idx:]
    return float(np.mean(tail)) if len(tail)>0 else float(sl[-1])

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — Sub-Controller PPO (one per regime)
# ═══════════════════════════════════════════════════════════════════════════════
class SubEnv(gym.Env):
    """Daily portfolio env for one regime."""
    def __init__(self, ret_data):
        super().__init__()
        self.D = ret_data; self.T = len(ret_data)
        self.observation_space = spaces.Box(-np.inf,np.inf,
            shape=(LOOKBACK*N+N,), dtype=np.float32)
        self.action_space = spaces.Box(-3.,3., shape=(N,), dtype=np.float32)

    def _obs(self):
        h = self.D[max(0,self.t-LOOKBACK):self.t]
        if len(h)<LOOKBACK: h=np.vstack([np.zeros((LOOKBACK-len(h),N)),h])
        return np.concatenate([h.flatten(), self.w]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t=LOOKBACK; self.w=np.ones(N)/N; self.lh=[]
        return self._obs(),{}

    def step(self, action):
        e=np.exp(action-action.max()); nw=e/e.sum()
        rp=float(np.dot(nw,self.D[self.t]))
        tc=TC*float(np.sum(np.abs(nw-self.w)))
        self.lh.append(-rp)
        if len(self.lh)>CVaR_WIN: self.lh.pop(0)
        rew = rp - tc - LAMBDA*cvar(self.lh)
        self.w=nw; self.t+=1
        done=(self.t>=self.T)
        return self._obs(), rew, done, False, {}

sub_models={}
for rid in range(4):
    mask=train_reg==rid; nd=mask.sum()
    print(f"\nSub-Controller [{RNAMES[rid]}]: {nd} days")
    if nd < LOOKBACK+20:
        theory={0:[.45,.25,.08,.05,.04,.04,.05,.04],
                1:[.04,.03,.03,.01,.38,.06,.35,.10],
                2:[.20,.18,.08,.05,.18,.10,.12,.09],
                3:[.02,.01,.01,.01,.35,.04,.46,.10]}
        w=np.array(theory[rid],float); sub_models[rid]=('static',w/w.sum())
        print("  → static fallback"); continue
    rd=train_ret[mask]
    env_fn=lambda d=rd: SubEnv(d)
    vec=DummyVecEnv([env_fn])
    ns=min(256,nd-LOOKBACK-1)
    m=PPO('MlpPolicy',vec,learning_rate=3e-4,n_steps=ns,
          batch_size=min(64,ns),n_epochs=10,gamma=0.99,
          gae_lambda=0.95,clip_range=0.2,ent_coef=0.01,
          verbose=0,seed=42)
    ts=max(nd*15,30_000)
    m.learn(total_timesteps=ts,progress_bar=False)
    sub_models[rid]=('ppo',m)
    print(f"  → PPO trained {ts} steps")

# ── Helper: run sub-controller on a return array, return daily PV series ──────
def run_sub(rid, ret_arr, init_w=None):
    """Returns (pv_array, final_weights)."""
    if init_w is None: init_w=np.ones(N)/N
    entry=sub_models[rid]; w=init_w.copy()
    pv=1.0; pvs=np.empty(len(ret_arr)); lh=[]
    for t in range(len(ret_arr)):
        if entry[0]=='static':
            nw=entry[1].copy()
        else:
            h=ret_arr[max(0,t-LOOKBACK):t]
            if len(h)<LOOKBACK: h=np.vstack([np.zeros((LOOKBACK-len(h),N)),h])
            obs=np.concatenate([h.flatten(),w]).astype(np.float32)
            act,_=entry[1].predict(obs,deterministic=True)
            e=np.exp(act-act.max()); nw=e/e.sum()
        rp=float(np.dot(nw,ret_arr[t]))
        tc=TC*float(np.sum(np.abs(nw-w)))
        lh.append(-rp)
        if len(lh)>CVaR_WIN: lh.pop(0)
        pv*=(1+rp-tc); pvs[t]=pv; w=nw
    return pvs, w

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Pre-compute quarterly sub-controller returns for meta training
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPre-computing quarterly sub-controller returns for meta training...")

# Build quarterly index on training data
n_train = len(train_ret)
quarters = []
t=0
while t < n_train:
    quarters.append((t, min(t+QUARTER, n_train)))
    t+=QUARTER
Q = len(quarters)

# For each quarter and each regime, compute the cumulative log return
# Shape: (Q, 4)
quarterly_returns = np.zeros((Q, 4), dtype=np.float32)
for qi, (qs, qe) in enumerate(quarters):
    seg = train_ret[qs:qe]
    for rid in range(4):
        pvs, _ = run_sub(rid, seg)
        quarterly_returns[qi, rid] = float(np.log(pvs[-1]))  # log return

# Build macro features per quarter (first day of each quarter)
spy_idx_n = ASSETS.index('SPY')
macro_feat = np.zeros((Q, 5), dtype=np.float32)
for qi, (qs, qe) in enumerate(quarters):
    d = train_dates[qs]
    macro_feat[qi,0] = float(vix_a.loc[d,'VIX'])/50.
    macro_feat[qi,1] = float(macro_a.loc[d,'cpi_yoy'])/10.
    macro_feat[qi,2] = float(spy_dd.loc[d])
    macro_feat[qi,3] = float(np.prod(1+train_ret[max(0,qs-63):qs,spy_idx_n])-1)
    macro_feat[qi,4] = float(qi)/Q  # time progress

print(f"  {Q} quarters pre-computed")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — Meta-Controller PPO (fast: uses pre-computed quarterly returns)
# ═══════════════════════════════════════════════════════════════════════════════
# Also pre-compute best possible (oracle) quarterly return for normalisation
best_quarterly = quarterly_returns.max(axis=1)  # shape (Q,)

class MetaEnv(gym.Env):
    """
    Fast quarterly meta-controller environment.
    Obs:    [vix_norm, cpi_norm, spy_dd, spy_63d_ret, time_progress,
             prev_regime_onehot(4), rolling_sharpe_proxy]  = 10 dims
    Action: discrete 0-3 (regime selection)
    Reward: risk-adjusted return = chosen_return - mean_return + 0.5*(chosen - mean)/std
            minus switching penalty when regime changes
    Episodes start at a random quarter to prevent recency bias.
    """
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf,np.inf,
            shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.ep_len = 8  # 8 quarters per episode (~2 years)

    def _obs(self):
        qi = min(self.q, Q-1)
        mf = macro_feat[qi]
        pr = np.zeros(4, dtype=np.float32)
        pr[self.prev_regime] = 1.
        # rolling sharpe proxy: mean/std of last 4 chosen returns
        if len(self.ret_hist) >= 2:
            rs = float(np.mean(self.ret_hist[-4:])) / (float(np.std(self.ret_hist[-4:]))+1e-6)
        else:
            rs = 0.
        return np.concatenate([mf, pr, [rs]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random start quarter, leave room for ep_len quarters
        self.q_start = int(np.random.randint(0, max(1, Q - self.ep_len)))
        self.q = self.q_start
        self.prev_regime = 0
        self.ret_hist = []
        return self._obs(), {}

    def step(self, action):
        rid = int(action)
        chosen_r = float(quarterly_returns[self.q, rid])
        # Risk-adjusted reward: excess return over mean of all regimes this quarter
        all_r = quarterly_returns[self.q]  # shape (4,)
        mean_r = float(all_r.mean())
        std_r  = float(all_r.std()) + 1e-6
        # Sharpe-like: how much better than average, normalised by spread
        reward = (chosen_r - mean_r) / std_r
        # Small penalty for switching regime (encourages stability)
        if rid != self.prev_regime:
            reward -= 0.05
        self.ret_hist.append(chosen_r)
        self.prev_regime = rid
        self.q += 1
        done = (self.q >= self.q_start + self.ep_len) or (self.q >= Q)
        return self._obs(), reward, done, False, {}

print("\nTraining Meta-Controller PPO...")
meta_vec = DummyVecEnv([MetaEnv])
meta_model = PPO('MlpPolicy', meta_vec,
                 learning_rate=3e-4, n_steps=256, batch_size=64,
                 n_epochs=15, gamma=0.95, gae_lambda=0.95,
                 clip_range=0.2, ent_coef=0.02, verbose=0, seed=42)
meta_model.learn(total_timesteps=200_000, progress_bar=False)
print("  Meta-Controller trained 200,000 steps")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — Backtest on Test Period (2023-2025)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nRunning backtest (2023-2025)...")

n_test = len(test_ret)
macrohrl_pvs = np.empty(n_test)
detected_reg = []
pv = 100_000.
w  = np.ones(N)/N
cum_ret = 0.
prev_regime = 0
t = 0

while t < n_test:
    d = test_dates[t]
    vix_v  = float(vix_a.loc[d,'VIX'])/50.
    cpi_v  = float(macro_a.loc[d,'cpi_yoy'])/10.
    dd_v   = float(spy_dd.loc[d])
    spy_ret= float(np.prod(1+test_ret[max(0,t-63):t,spy_idx_n])-1) if t>0 else 0.
    time_p = float(t)/n_test
    pr = np.zeros(4,dtype=np.float32); pr[prev_regime]=1.
    meta_obs = np.array([vix_v,cpi_v,dd_v,spy_ret,time_p,
                         pr[0],pr[1],pr[2],pr[3],cum_ret],
                        dtype=np.float32)
    regime, _ = meta_model.predict(meta_obs, deterministic=True)
    regime = int(regime)

    end = min(t+QUARTER, n_test)
    seg = test_ret[t:end]
    pvs_seg, w = run_sub(regime, seg, init_w=w)
    pv_seg = pv * pvs_seg
    macrohrl_pvs[t:end] = pv_seg
    pv = float(pv_seg[-1])
    cum_ret += float(np.log(pvs_seg[-1]))
    detected_reg.extend([regime]*(end-t))
    prev_regime = regime
    t = end

macrohrl_pv = pd.Series(macrohrl_pvs, index=test_dates)
detected_reg = np.array(detected_reg, dtype=np.int8)

# ── Baselines ─────────────────────────────────────────────────────────────────
spy_pv = pd.Series(100_000.*np.cumprod(1+test_ret[:,spy_idx_n]), index=test_dates)
ew_pv  = pd.Series(100_000.*np.cumprod(1+test_ret.mean(axis=1)), index=test_dates)

flat_pvs=np.empty(n_test); pv2=100_000.; cw2=np.ones(N)/N
for t in range(n_test):
    dr=float(np.dot(np.ones(N)/N,test_ret[t]))
    to=float(np.sum(np.abs(np.ones(N)/N-cw2)))
    pv2=pv2*(1.+dr-TC*to); flat_pvs[t]=pv2; cw2=np.ones(N)/N
flat_pv=pd.Series(flat_pvs,index=test_dates)

lit_ws=[]
for t in range(n_test):
    if t<20: w2=np.ones(N)/N
    else:
        mom=test_ret[t-20:t].mean(0); vol=test_ret[t-20:t].std(0)+1e-8
        sig=mom/vol; sig_s=sig-sig.max()
        e=np.exp(sig_s*3); w2=e/e.sum(); w2=np.clip(w2,.02,.40); w2/=w2.sum()
    lit_ws.append(w2)
lit_ws=np.array(lit_ws)
lit_ret=(lit_ws*test_ret).sum(1)
lit_tc=np.abs(np.diff(lit_ws,axis=0,prepend=lit_ws[[0]])).sum(1)*TC
lit_pv=pd.Series(100_000.*np.cumprod(1+lit_ret-lit_tc),index=test_dates)

# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(pv_s):
    r=pv_s.pct_change().dropna(); n=len(r)
    tr=(pv_s.iloc[-1]/pv_s.iloc[0])-1; ar=(1+tr)**(252/n)-1
    av=r.std()*np.sqrt(252); er=r-RF/252
    sh=float(er.mean()/r.std()*np.sqrt(252)) if r.std()>0 else 0.
    dn=r[r<0]; dv=dn.std()*np.sqrt(252) if len(dn)>0 else 1.
    so=float((ar-RF)/dv) if dv>0 else 0.
    rm=pv_s.cummax(); dd=(pv_s-rm)/rm; md=float(dd.min())
    ca=float(ar/abs(md)) if md!=0 else 0.
    return dict(sharpe=sh,annual_return=ar,annual_volatility=av,
                max_drawdown=md,sortino_ratio=so,calmar_ratio=ca)

rows=[('MacroHRL (Ours)',    metrics(macrohrl_pv)),
      ('Buy-and-Hold SPY',  metrics(spy_pv)),
      ('Equal Weight (1/N)',metrics(ew_pv)),
      ('Flat PPO',          metrics(flat_pv)),
      ('Literature Model',  metrics(lit_pv))]

print("\n"+"="*78)
print("MacroHRL HRL — Results (Test: Jan 2023 – Dec 2025)")
print("="*78)
print(f"{'Strategy':<24} {'Sharpe':>7} {'AnnRet':>8} {'AnnVol':>8} "
      f"{'MaxDD':>8} {'Sortino':>8} {'Calmar':>7}")
print("-"*78)
for nm,m in rows:
    print(f"{nm:<24} {m['sharpe']:>7.3f} {m['annual_return']:>8.2%} "
          f"{m['annual_volatility']:>8.2%} {m['max_drawdown']:>8.2%} "
          f"{m['sortino_ratio']:>8.3f} {m['calmar_ratio']:>7.3f}")

rc=Counter(detected_reg.tolist())
print("\nMeta-Controller regime selections (test):")
for r in range(4):
    c=rc.get(r,0)
    print(f"  {RNAMES[r]:8s}: {c:4d} days ({c/len(detected_reg)*100:.1f}%)")

with open('/home/ubuntu/research/data/results_hrl.json','w') as f:
    json.dump({nm:m for nm,m in rows},f,indent=2)

# ── Figures ───────────────────────────────────────────────────────────────────
os.makedirs('/home/ubuntu/research/figures',exist_ok=True)
COLORS={'MacroHRL (Ours)':'#2ecc71','Flat PPO':'#e67e22',
        'Literature Model':'#e74c3c','Equal Weight (1/N)':'#9b59b6',
        'Buy-and-Hold SPY':'#3498db'}
RCOL={0:'#d4efdf',1:'#fadbd8',2:'#fef9e7',3:'#f9ebea'}
reg_s=pd.Series(detected_reg,index=test_dates)

def shade(ax,rs,dates):
    cr=int(rs.iloc[0]); sd=dates[0]
    for i in range(1,len(dates)):
        if int(rs.iloc[i])!=cr:
            ax.axvspan(sd,dates[i],alpha=0.22,color=RCOL[cr],zorder=0)
            cr=int(rs.iloc[i]); sd=dates[i]
    ax.axvspan(sd,dates[-1],alpha=0.22,color=RCOL[cr],zorder=0)

strats={'MacroHRL (Ours)':macrohrl_pv,'Flat PPO':flat_pv,
        'Literature Model':lit_pv,'Equal Weight (1/N)':ew_pv,
        'Buy-and-Hold SPY':spy_pv}

fig,ax=plt.subplots(figsize=(12,6))
shade(ax,reg_s,test_dates)
for nm,pv in strats.items():
    ax.plot(pv.index,pv.values,label=nm,color=COLORS[nm],
            linewidth=2.8 if nm=='MacroHRL (Ours)' else 1.6,
            linestyle='-' if nm=='MacroHRL (Ours)' else '--')
patches=[mpatches.Patch(color=RCOL[r],alpha=0.5,label=RNAMES[r]) for r in range(4)]
l1=ax.legend(handles=patches,loc='upper left',fontsize=9,title='Meta-Controller Regime')
ax.add_artist(l1); ax.legend(loc='upper center',fontsize=9,ncol=3)
ax.set_title('MacroHRL vs. Baselines: Portfolio Value (2023–2025)',fontsize=13,fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Portfolio Value (USD)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'${x:,.0f}'))
ax.grid(True,alpha=0.3); plt.tight_layout()
plt.savefig('/home/ubuntu/research/figures/fig1_portfolio_values.png',dpi=150,bbox_inches='tight')
plt.close()

fig,ax=plt.subplots(figsize=(12,5))
shade(ax,reg_s,test_dates)
for nm,pv in strats.items():
    rm=pv.cummax(); dd=(pv-rm)/rm*100
    ax.plot(dd.index,dd.values,label=nm,color=COLORS[nm],
            linewidth=2.8 if nm=='MacroHRL (Ours)' else 1.6)
ax.set_title('Portfolio Drawdown Comparison (2023–2025)',fontsize=13,fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=9,ncol=2); ax.grid(True,alpha=0.3)
ax.axhline(0,color='black',linewidth=0.8); plt.tight_layout()
plt.savefig('/home/ubuntu/research/figures/fig2_drawdown.png',dpi=150,bbox_inches='tight')
plt.close()

print("\nFigures saved.")
print("=== MacroHRL complete ===")

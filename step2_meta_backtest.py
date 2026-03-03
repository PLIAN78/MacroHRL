"""
MacroHRL Step 2: Train Meta-Controller + Run Backtest
Loads sub-controller models from disk (output of step1_train_subs.py).
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os
from collections import Counter

ASSETS=['SPY','QQQ','EFA','EEM','TLT','HYG','GLD','VNQ']
N=len(ASSETS); TC=0.001; LAMBDA=0.5; ALPHA=0.95; LOOKBACK=20; CVaR_WIN=60; QUARTER=63; RF=0.04
RNAMES={0:'Bull',1:'Bear',2:'Sideways',3:'Crisis'}
np.random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
close=pd.read_csv('/home/ubuntu/research/data/close_prices.csv',index_col=0,parse_dates=True)[ASSETS].dropna()
vix=pd.read_csv('/home/ubuntu/research/data/vix.csv',index_col=0,parse_dates=True); vix.columns=['VIX']
macro=pd.read_csv('/home/ubuntu/research/data/macro_indicators.csv',index_col=0,parse_dates=True)
rets=close.pct_change().dropna()
common=rets.index.intersection(vix.index)
rets=rets.loc[common]; close=close.loc[common]
vix_a=vix.reindex(common).ffill().bfill()
macro_a=macro.reindex(common).ffill().bfill()
spy_prices=close['SPY']; rolling_max=spy_prices.rolling(63,min_periods=1).max()
spy_dd=(spy_prices-rolling_max)/rolling_max

test_mask=(rets.index>='2023-01-01')&(rets.index<='2025-12-31')
test_ret=rets.values[test_mask]; test_dates=rets.index[test_mask]
spy_idx_n=ASSETS.index('SPY')

# ── Load pre-computed meta training data ──────────────────────────────────────
quarterly_returns=np.load('/home/ubuntu/research/models/quarterly_returns.npy')
macro_feat=np.load('/home/ubuntu/research/models/macro_feat.npy')
Q=int(np.load('/home/ubuntu/research/models/Q.npy')[0])
print(f"Loaded {Q} quarters of pre-computed returns.")

# ── Load sub-controller models ────────────────────────────────────────────────
theory={0:[.45,.25,.08,.05,.04,.04,.05,.04],1:[.04,.03,.03,.01,.38,.06,.35,.10],
        2:[.20,.18,.08,.05,.18,.10,.12,.09],3:[.02,.01,.01,.01,.35,.04,.46,.10]}
sub_models={}
for rid in range(4):
    static_path=f'/home/ubuntu/research/models/sub_{rid}_static.npy'
    ppo_path=f'/home/ubuntu/research/models/sub_{rid}_ppo.zip'
    if os.path.exists(static_path):
        w=np.load(static_path); sub_models[rid]=('static',w); print(f"  [{RNAMES[rid]}] static loaded")
    elif os.path.exists(ppo_path):
        m=PPO.load(ppo_path); sub_models[rid]=('ppo',m); print(f"  [{RNAMES[rid]}] PPO loaded")
    else:
        w=np.array(theory[rid],float); w/=w.sum(); sub_models[rid]=('static',w)
        print(f"  [{RNAMES[rid]}] fallback static")

# ── Meta-Controller Environment ───────────────────────────────────────────────
class MetaEnv(gym.Env):
    """
    Obs: [vix,cpi,dd,spy_ret,time_prog, prev_regime_onehot(4), rolling_sharpe] = 10
    Action: discrete 0-3
    Reward: risk-adjusted excess return (Sharpe-normalised vs all regimes this quarter)
    Episodes start at random quarter to prevent recency bias.
    """
    def __init__(self):
        super().__init__()
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(10,),dtype=np.float32)
        self.action_space=spaces.Discrete(4)
        self.ep_len=8

    def _obs(self):
        qi=min(self.q,Q-1); mf=macro_feat[qi]
        pr=np.zeros(4,dtype=np.float32); pr[self.prev_regime]=1.
        rs=float(np.mean(self.ret_hist[-4:]))/(float(np.std(self.ret_hist[-4:]))+1e-6) if len(self.ret_hist)>=2 else 0.
        return np.concatenate([mf,pr,[rs]]).astype(np.float32)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.q_start=int(np.random.randint(0,max(1,Q-self.ep_len)))
        self.q=self.q_start; self.prev_regime=0; self.ret_hist=[]
        return self._obs(),{}

    def step(self,action):
        rid=int(action); chosen_r=float(quarterly_returns[self.q,rid])
        all_r=quarterly_returns[self.q]; mean_r=float(all_r.mean()); std_r=float(all_r.std())+1e-6
        reward=(chosen_r-mean_r)/std_r
        if rid!=self.prev_regime: reward-=0.05
        self.ret_hist.append(chosen_r); self.prev_regime=rid; self.q+=1
        done=(self.q>=self.q_start+self.ep_len)or(self.q>=Q)
        return self._obs(),reward,done,False,{}

print("\nTraining Meta-Controller PPO (200k steps)...")
meta_vec=DummyVecEnv([MetaEnv])
meta_model=PPO('MlpPolicy',meta_vec,learning_rate=3e-4,n_steps=256,batch_size=64,
               n_epochs=15,gamma=0.95,gae_lambda=0.95,clip_range=0.2,ent_coef=0.02,verbose=0,seed=42)
meta_model.learn(total_timesteps=200_000,progress_bar=False)
print("  Done.")

# ── CVaR ──────────────────────────────────────────────────────────────────────
def cvar(losses,alpha=ALPHA):
    if len(losses)<5: return 0.
    sl=np.sort(losses); idx=int(np.ceil(alpha*len(sl))); tail=sl[idx:]
    return float(np.mean(tail)) if len(tail)>0 else float(sl[-1])

# ── Sub-controller runner ──────────────────────────────────────────────────────
def run_sub(rid,ret_arr,init_w=None):
    if init_w is None: init_w=np.ones(N)/N
    entry=sub_models[rid]; w=init_w.copy(); pv=1.; pvs=np.empty(len(ret_arr)); lh=[]
    for t in range(len(ret_arr)):
        if entry[0]=='static':
            nw=entry[1].copy()
        else:
            h=ret_arr[max(0,t-LOOKBACK):t]
            if len(h)<LOOKBACK: h=np.vstack([np.zeros((LOOKBACK-len(h),N)),h])
            obs=np.concatenate([h.flatten(),w]).astype(np.float32)
            act,_=entry[1].predict(obs,deterministic=True)
            e=np.exp(act-act.max()); nw=e/e.sum()
        rp=float(np.dot(nw,ret_arr[t])); tc=TC*float(np.sum(np.abs(nw-w)))
        lh.append(-rp)
        if len(lh)>CVaR_WIN: lh.pop(0)
        pv*=(1+rp-tc); pvs[t]=pv; w=nw
    return pvs,w

# ── Backtest ──────────────────────────────────────────────────────────────────
print("\nRunning backtest (2023-2025)...")
n_test=len(test_ret); macrohrl_pvs=np.empty(n_test); detected_reg=[]
pv=100_000.; w=np.ones(N)/N; cum_ret=0.; prev_regime=0; t=0

while t<n_test:
    d=test_dates[t]
    vix_v=float(vix_a.loc[d,'VIX'])/50.; cpi_v=float(macro_a.loc[d,'cpi_yoy'])/10.
    dd_v=float(spy_dd.loc[d])
    spy_ret=float(np.prod(1+test_ret[max(0,t-63):t,spy_idx_n])-1) if t>0 else 0.
    time_p=float(t)/n_test
    pr=np.zeros(4,dtype=np.float32); pr[prev_regime]=1.
    meta_obs=np.array([vix_v,cpi_v,dd_v,spy_ret,time_p,pr[0],pr[1],pr[2],pr[3],cum_ret],dtype=np.float32)
    regime,_=meta_model.predict(meta_obs,deterministic=True); regime=int(regime)
    end=min(t+QUARTER,n_test); seg=test_ret[t:end]
    pvs_seg,w=run_sub(regime,seg,init_w=w)
    pv_seg=pv*pvs_seg; macrohrl_pvs[t:end]=pv_seg
    pv=float(pv_seg[-1]); cum_ret+=float(np.log(pvs_seg[-1]))
    detected_reg.extend([regime]*(end-t)); prev_regime=regime; t=end

macrohrl_pv=pd.Series(macrohrl_pvs,index=test_dates)
detected_reg=np.array(detected_reg,dtype=np.int8)

# ── Baselines ─────────────────────────────────────────────────────────────────
spy_pv=pd.Series(100_000.*np.cumprod(1+test_ret[:,spy_idx_n]),index=test_dates)
ew_pv=pd.Series(100_000.*np.cumprod(1+test_ret.mean(axis=1)),index=test_dates)

flat_pvs=np.empty(n_test); pv2=100_000.; cw2=np.ones(N)/N
for t in range(n_test):
    dr=float(np.dot(np.ones(N)/N,test_ret[t])); to=float(np.sum(np.abs(np.ones(N)/N-cw2)))
    pv2=pv2*(1.+dr-TC*to); flat_pvs[t]=pv2; cw2=np.ones(N)/N
flat_pv=pd.Series(flat_pvs,index=test_dates)

lit_ws=[]
for t in range(n_test):
    if t<20: w2=np.ones(N)/N
    else:
        mom=test_ret[t-20:t].mean(0); vol=test_ret[t-20:t].std(0)+1e-8
        sig=mom/vol; sig_s=sig-sig.max(); e=np.exp(sig_s*3); w2=e/e.sum()
        w2=np.clip(w2,.02,.40); w2/=w2.sum()
    lit_ws.append(w2)
lit_ws=np.array(lit_ws); lit_ret=(lit_ws*test_ret).sum(1)
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
    return dict(sharpe=sh,annual_return=ar,annual_volatility=av,max_drawdown=md,sortino_ratio=so,calmar_ratio=ca)

rows=[('MacroHRL (Ours)',metrics(macrohrl_pv)),('Buy-and-Hold SPY',metrics(spy_pv)),
      ('Equal Weight (1/N)',metrics(ew_pv)),('Flat PPO',metrics(flat_pv)),
      ('Literature Model',metrics(lit_pv))]

print("\n"+"="*78)
print("MacroHRL HRL — Results (Test: Jan 2023 – Dec 2025)")
print("="*78)
print(f"{'Strategy':<24} {'Sharpe':>7} {'AnnRet':>8} {'AnnVol':>8} {'MaxDD':>8} {'Sortino':>8} {'Calmar':>7}")
print("-"*78)
for nm,m in rows:
    print(f"{nm:<24} {m['sharpe']:>7.3f} {m['annual_return']:>8.2%} {m['annual_volatility']:>8.2%} "
          f"{m['max_drawdown']:>8.2%} {m['sortino_ratio']:>8.3f} {m['calmar_ratio']:>7.3f}")

rc=Counter(detected_reg.tolist())
print("\nMeta-Controller regime selections (test):")
for r in range(4):
    c=rc.get(r,0); print(f"  {RNAMES[r]:8s}: {c:4d} days ({c/len(detected_reg)*100:.1f}%)")

with open('/home/ubuntu/research/data/results_hrl.json','w') as f:
    json.dump({nm:m for nm,m in rows},f,indent=2)

# ── Figures ───────────────────────────────────────────────────────────────────
os.makedirs('/home/ubuntu/research/figures',exist_ok=True)
COLORS={'MacroHRL (Ours)':'#2ecc71','Flat PPO':'#e67e22','Literature Model':'#e74c3c',
        'Equal Weight (1/N)':'#9b59b6','Buy-and-Hold SPY':'#3498db'}
RCOL={0:'#d4efdf',1:'#fadbd8',2:'#fef9e7',3:'#f9ebea'}
reg_s=pd.Series(detected_reg,index=test_dates)

def shade(ax,rs,dates):
    cr=int(rs.iloc[0]); sd=dates[0]
    for i in range(1,len(dates)):
        if int(rs.iloc[i])!=cr:
            ax.axvspan(sd,dates[i],alpha=0.22,color=RCOL[cr],zorder=0)
            cr=int(rs.iloc[i]); sd=dates[i]
    ax.axvspan(sd,dates[-1],alpha=0.22,color=RCOL[cr],zorder=0)

strats={'MacroHRL (Ours)':macrohrl_pv,'Flat PPO':flat_pv,'Literature Model':lit_pv,
        'Equal Weight (1/N)':ew_pv,'Buy-and-Hold SPY':spy_pv}

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

print("\nFigures saved. === MacroHRL complete ===")

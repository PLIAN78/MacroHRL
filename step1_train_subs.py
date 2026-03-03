"""
MacroHRL Step 1: Train Sub-Controllers and pre-compute quarterly returns.
Saves models and quarterly_returns.npy to disk so Step 2 can run independently.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

ASSETS=['SPY','QQQ','EFA','EEM','TLT','HYG','GLD','VNQ']
N=len(ASSETS); TC=0.001; LAMBDA=0.5; ALPHA=0.95; LOOKBACK=20; CVaR_WIN=60; QUARTER=63
RNAMES={0:'Bull',1:'Bear',2:'Sideways',3:'Crisis'}
np.random.seed(42)

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

def classify(vix_v,cpi_v,dd_v):
    if vix_v>30 and dd_v<-0.10: return 3
    if cpi_v>5.5: return 1
    if 20<=vix_v<=30 and abs(dd_v)<0.08: return 2
    return 0

all_reg=pd.Series(index=common,dtype=np.int8)
for d in common:
    all_reg[d]=classify(float(vix_a.loc[d,'VIX']),float(macro_a.loc[d,'cpi_yoy']),float(spy_dd.loc[d]))

train_mask=(rets.index>='2010-01-01')&(rets.index<='2022-12-31')
train_ret=rets.values[train_mask]; train_dates=rets.index[train_mask]
train_reg=all_reg.loc[train_dates].values.astype(np.int8)

print("Training regime distribution (2010-2022):")
for r in range(4):
    c=(train_reg==r).sum(); print(f"  {RNAMES[r]:8s}: {c} ({c/len(train_reg)*100:.1f}%)")

def cvar(losses,alpha=ALPHA):
    if len(losses)<5: return 0.
    sl=np.sort(losses); idx=int(np.ceil(alpha*len(sl))); tail=sl[idx:]
    return float(np.mean(tail)) if len(tail)>0 else float(sl[-1])

class SubEnv(gym.Env):
    def __init__(self,ret_data):
        super().__init__()
        self.D=ret_data; self.T=len(ret_data)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(LOOKBACK*N+N,),dtype=np.float32)
        self.action_space=spaces.Box(-3.,3.,shape=(N,),dtype=np.float32)
    def _obs(self):
        h=self.D[max(0,self.t-LOOKBACK):self.t]
        if len(h)<LOOKBACK: h=np.vstack([np.zeros((LOOKBACK-len(h),N)),h])
        return np.concatenate([h.flatten(),self.w]).astype(np.float32)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed); self.t=LOOKBACK; self.w=np.ones(N)/N; self.lh=[]
        return self._obs(),{}
    def step(self,action):
        e=np.exp(action-action.max()); nw=e/e.sum()
        rp=float(np.dot(nw,self.D[self.t])); tc=TC*float(np.sum(np.abs(nw-self.w)))
        self.lh.append(-rp)
        if len(self.lh)>CVaR_WIN: self.lh.pop(0)
        rew=rp-tc-LAMBDA*cvar(self.lh); self.w=nw; self.t+=1
        return self._obs(),rew,(self.t>=self.T),False,{}

os.makedirs('/home/ubuntu/research/models',exist_ok=True)
theory={0:[.45,.25,.08,.05,.04,.04,.05,.04],1:[.04,.03,.03,.01,.38,.06,.35,.10],
        2:[.20,.18,.08,.05,.18,.10,.12,.09],3:[.02,.01,.01,.01,.35,.04,.46,.10]}

for rid in range(4):
    mask=train_reg==rid; nd=mask.sum()
    print(f"\nSub-Controller [{RNAMES[rid]}]: {nd} days")
    if nd<LOOKBACK+20:
        w=np.array(theory[rid],float); w/=w.sum()
        np.save(f'/home/ubuntu/research/models/sub_{rid}_static.npy',w)
        print("  → static"); continue
    rd=train_ret[mask]
    vec=DummyVecEnv([lambda d=rd: SubEnv(d)])
    ns=min(256,nd-LOOKBACK-1)
    m=PPO('MlpPolicy',vec,learning_rate=3e-4,n_steps=ns,batch_size=min(64,ns),
          n_epochs=10,gamma=0.99,gae_lambda=0.95,clip_range=0.2,ent_coef=0.01,verbose=0,seed=42)
    ts=max(nd*15,30_000)
    m.learn(total_timesteps=ts,progress_bar=False)
    m.save(f'/home/ubuntu/research/models/sub_{rid}_ppo')
    print(f"  → PPO saved ({ts} steps)")

# Pre-compute quarterly returns using vectorised numpy (no PPO inference — use static weights for speed)
# For the meta-controller training we use the STATIC theory allocations to pre-compute quarterly returns
# The actual backtest will use the trained PPO sub-controllers
print("\nPre-computing quarterly returns (static allocs for meta training)...")
n_train=len(train_ret); quarters=[]; t=0
while t<n_train: quarters.append((t,min(t+QUARTER,n_train))); t+=QUARTER
Q=len(quarters)

static_w=np.array([theory[r] for r in range(4)],dtype=float)
for r in range(4): static_w[r]/=static_w[r].sum()

quarterly_returns=np.zeros((Q,4),dtype=np.float32)
for qi,(qs,qe) in enumerate(quarters):
    seg=train_ret[qs:qe]
    for rid in range(4):
        w=static_w[rid]
        daily_r=(seg*w).sum(axis=1)
        # include TC at entry only
        tc_entry=TC*float(np.sum(np.abs(w-np.ones(N)/N)))
        pv=np.prod(1+daily_r)*(1-tc_entry)
        quarterly_returns[qi,rid]=float(np.log(max(pv,1e-6)))

np.save('/home/ubuntu/research/models/quarterly_returns.npy',quarterly_returns)

# Save macro features per quarter
spy_idx_n=ASSETS.index('SPY')
macro_feat=np.zeros((Q,5),dtype=np.float32)
for qi,(qs,qe) in enumerate(quarters):
    d=train_dates[qs]
    macro_feat[qi,0]=float(vix_a.loc[d,'VIX'])/50.
    macro_feat[qi,1]=float(macro_a.loc[d,'cpi_yoy'])/10.
    macro_feat[qi,2]=float(spy_dd.loc[d])
    macro_feat[qi,3]=float(np.prod(1+train_ret[max(0,qs-63):qs,spy_idx_n])-1)
    macro_feat[qi,4]=float(qi)/Q

np.save('/home/ubuntu/research/models/macro_feat.npy',macro_feat)
np.save('/home/ubuntu/research/models/Q.npy',np.array([Q]))
print(f"  Saved {Q} quarters. Step 1 complete.")

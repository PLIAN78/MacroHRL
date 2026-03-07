# MacroHRL  
## A Hierarchical Reinforcement Learning Framework for Risk-Aware Portfolio Management with Drawdown Minimization

**Authors:**  
Neelesh Nayak, Peter Lian, Tony Xia  
University of Waterloo  

---

# Overview

**MacroHRL** is a hierarchical reinforcement learning framework designed for **risk-aware portfolio management**. Unlike traditional portfolio optimization methods that focus solely on maximizing returns, MacroHRL prioritizes:

- 📉 Drawdown minimization  
- 🛡️ Capital preservation  
- 🔄 Regime-adaptive allocation  
- ⚠️ Tail-risk suppression  

The system uses a **two-level hierarchical architecture**:

- A **Meta-Controller (PPO agent)** selects macroeconomic regimes quarterly  
- Specialized **Sub-Controllers (PPO agents)** manage daily portfolio allocation  

This design enables the system to adapt to changing macroeconomic conditions while maintaining strong risk control.

---

# System Architecture

Below is the MacroHRL framework pipeline:

![MacroHRL Architecture](figures/fig7_architecture.png)

The system consists of:

**Macro Inputs**
- CPI (inflation)
- VIX (volatility index)
- Yield Curve

**Processing**
- Rule-based macro regime classifier

**Hierarchical RL Framework**
- **Meta-Controller:** selects active regime policy quarterly  
- **Sub-Controllers:** execute daily allocation decisions  

**Output**
- Portfolio weight vector  

---

# Key Contributions

## 1. Hierarchical Reinforcement Learning for Finance

Portfolio management is formulated as a **Hierarchical Markov Decision Process (HMDP)**:

- Long-term macro decision making  
- Short-term tactical asset allocation  

The Meta-Controller handles **strategic regime decisions**, while Sub-Controllers handle **daily portfolio execution**.

---

## 2. Regime Specialization

Markets are classified into four macro regimes:

- 🟢 **Bull**
- 🔴 **Bear**
- ⚠️ **Crisis**
- ➖ **Sideways**

Each regime is assigned a **dedicated PPO trading agent trained on historical data specific to that environment**, allowing the system to learn regime-specific behaviors.

---

## 3. Risk-Aware Reward Function

Sub-Controllers optimize a **CVaR-penalized reward function**:

\[
R_k = r_k^p - c \sum |w_{k,i} - w_{k-1,i}| - \lambda \cdot CVaR_\alpha(L_k)
\]

Where:

- \( r_k^p \) = portfolio return  
- \( c \) = transaction cost  
- \( \lambda \) = risk aversion coefficient  
- \( CVaR \) = conditional value-at-risk of recent losses  

This reward explicitly penalizes **tail risk**, encouraging strategies that **avoid catastrophic drawdowns**.

---

# Dataset

## Assets (Daily Data)

Eight major ETFs are used for portfolio construction:

- SPY — US equities  
- QQQ — NASDAQ equities  
- EFA — Developed markets  
- EEM — Emerging markets  
- TLT — Long-term treasury bonds  
- HYG — High-yield corporate bonds  
- GLD — Gold  
- VNQ — Real estate  

---

## Macroeconomic Indicators

Macroeconomic signals used for regime detection:

- **VIX** — market volatility  
- **CPI** — inflation  
- **Yield Curve** — economic expectations  

Data is sourced from **FRED and Yahoo Finance**.

---

## Time Period

| Phase | Years |
|------|------|
| Training | 2010 – 2022 |
| Testing (Out-of-Sample) | 2023 – 2025 |

---

# Results

## Portfolio Performance

MacroHRL produces a significantly smoother equity curve compared to a traditional benchmark.

![Portfolio Value Comparison](figures/fig1_portfolio_values.png)

---

## Drawdown Comparison

MacroHRL demonstrates strong downside protection.

![Drawdown Comparison](figures/fig2_drawdown.png)

---

# Performance Metrics (Out-of-Sample 2023–2025)

| Strategy | Sharpe | Annual Return | Max Drawdown | Calmar |
|-----------|--------|---------------|--------------|--------|
| **MacroHRL (Selected)** | **1.753** | **28.07%** | **-9.90%** | **2.835** |
| Buy & Hold SPY | 1.616 | 24.80% | -18.76% | 1.322 |

---

# Key Insight

MacroHRL achieves approximately:

**~47% reduction in maximum drawdown compared to SPY**

while maintaining strong risk-adjusted returns.

This demonstrates that hierarchical RL can effectively balance:

- return generation  
- volatility control  
- tail-risk suppression  

---

# Why MacroHRL Matters

Traditional strategies struggle during:

- market regime transitions  
- black swan events  
- correlation breakdowns  

MacroHRL addresses these challenges by:

- separating **macro strategy** from **tactical execution**
- learning **specialized regime policies**
- explicitly optimizing **downside risk**

This makes the framework suitable for **risk-sensitive institutional portfolio management**.

---

# Hyperparameters (Selected from Sweep)

| Parameter | Value |
|-----------|------|
| VIX Threshold (Crisis) | 30 |
| Drawdown Threshold (Crisis) | -10% |
| Bull Risk-Aversion λ | 0.05 |
| Crisis Risk-Aversion λ | 0.30 |
| Meta-Controller Entropy | 0.02 |
| Transaction Cost | 0.001 |

These values were selected through a large hyperparameter sweep optimizing **annualized return under a max drawdown constraint (<10%)**.

---

# Future Work

Potential future improvements include:

- additional macroeconomic signals  
- multi-agent portfolio coordination  
- LLM-assisted macro reasoning  
- real-time deployment pipelines  
- integration with alternative datasets  

---

# Citation

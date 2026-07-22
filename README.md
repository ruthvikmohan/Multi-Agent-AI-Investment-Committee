# Multi-Agent AI Investment Committee Dashboard
### 📊 Production-Grade Financial Engineering Orchestration Framework

An institutional-grade multi-agent autonomous framework built using **CrewAI**, **Streamlit**, and **Groq Cloud Infrastructure (Llama 3.3 70B)**. The system couples statistical econometric execution models (**GARCH**, **Historical VaR**) with unstructured real-time fundamental narrative data parsing engines to produce verified asset management briefs.

---

## 🔬 Core Financial Engineering Mathematics

Unlike generic generative AI implementations, this framework executes empirical econometric calculations within its local raw data layer before contextualizing prompts for the agent layer.

### 1. Volatility Dynamics via GARCH(1,1)
To capture time-varying volatility clusters and avoid the systemic errors of basic rolling variance metrics, the quantitative engine fits a Generalized Autoregressive Conditional Heteroskedasticity (GARCH(1,1)) framework via Maximum Likelihood Estimation (MLE):

```math
\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
```

Where:
```math

* `\sigma_t^2`: Conditional variance forecast for period \(t\)
* `\omega`: Baseline long-term variance constant (\(\omega > 0\))
* `\alpha`: ARCH parameter measuring reaction to recent localized return shocks (\(\alpha \geq 0\))
* `\beta`: GARCH parameter measuring historical volatility persistence (\(\beta \geq 0\))
* *Constraint:* Stationarity is enforced by ensuring \(\alpha + \beta < 1\).

```

The calculated conditional variance is annualized over standard market profiles:

```math
\sigma_{\text{annual}} = \sqrt{\sigma_t^2} \times \sqrt{252} \times 100
```

### 2. Risk Exposure Containment: 95% Historical Value-at-Risk (VaR)
To establish structural portfolio allocation ceilings, the system calculates the 1-Day Historical Value-at-Risk at a \(c = 0.95\) confidence threshold. Given a vector of daily log returns 
```math
\(R_t = \ln(P_t / P_{t-1})\), the metric isolates the fifth percentile boundary:
```

```math
\text{VaR}_{95\%} = - \text{Percentile}\left(R_t, 5\right) \times 100
```

This establishes that there is a statistical probability of exactly 5% that the asset's downside loss will exceed this calculated percentage over a 24-hour horizon.

### 3. Momentum Oscillators: MACD & Relative Strength Index (RSI)
*   **MACD Line:** Measures trend convergence using Exponential Moving Averages:
    
    ```math
    \text{MACD} = \text{EMA}_{12}(P) - \text{EMA}_{26}(P)
    ```
    
    ```math
    \text{Signal Line} = \text{EMA}_{9}(\text{MACD})
    ```
    
*   **RSI (14-Day):** Quantifies velocity imbalances across directional closes:
    
    ```math
    \text{RSI} = 100 - \left[ \frac{100}{1 + \frac{\sum \text{Gains}}{\sum \text{Losses}}} \right]
    ```

---

## 🤖 System Architecture & Agent Design Pattern

The framework utilizes a sequential multi-agent debate loop to neutralize individual bias and reach an institutional consensus.

```text
       [Raw Market Data API Layer: yFinance]
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
  [Quantitative Math]          [Text Narrative Scraping]
  ├─ RSI / MACD Line           └─ Earnings News Stream
  └─ GARCH Variance / VaR
         │                             │
         ▼                             ▼
┌────────────────────────────────────────────────────────┐
│                   CREWAI AGENT LAYER                   │
├────────────────────────────────────────────────────────┤
│ 👤 Agent 1: Technical Quant Analyst                    │
│    └─ Input: RSI, MACD, GARCH Volatility               │
│                                                        │
│ 👤 Agent 2: Fundamental Text Analyst                   │
│    └─ Input: Live Scraping News Transcripts            │
│                                                        │
│ 👤 Agent 3: Investment Risk Manager (Chair)            │
│    └─ Input: Agent 1 + Agent 2 Reports + 95% VaR       │
└────────────────────────────────────────────────────────┘
                        │
                        ▼
       [Unified Streamlit Dashboard WebUI]
```

1. **Technical Quant Analyst:** Evaluates technical indicators and momentum shifts. It identifies overbought conditions and changes in directional trends.
2. **Fundamental Text Analyst:** Processes unstructured corporate text feeds to identify operational risks, competitive changes, and general sentiment indicators.
3. **Risk Manager (Committee Chair):** Reconciles technical and fundamental reports. It uses the empirical 95% VaR value to set asset exposure ceilings and issues a final directive (BUY/SELL/HOLD).

---

## 🛠️ Production Tech Stack & Free Cloud Orchestration

*   **Orchestration Engine:** `CrewAI` (Sequential Process Execution Paradigm)
*   **Compute Models:** `Groq Cloud Infrastructure API` running `llama-3.3-70b-versatile`
*   **Econometric Math Engine:** `arch` (MLE optimization models), `numpy`, `pandas`
*   **Data Scraper:** `yfinance` (Zero-cost API wrapper)
*   **Frontend UI:** `Streamlit Web Server`

---

## 🚀 Execution & Local Deployment

### 1. Clone & Install Dependencies
```bash
git clone https://github.com
cd YOUR_REPO_NAME
pip install streamlit crewai yfinance pandas numpy groq langchain-openai pyngrok litellm arch
```

### 2. Launch the Streamlit Interface Natively
```bash
streamlit run app.py
```

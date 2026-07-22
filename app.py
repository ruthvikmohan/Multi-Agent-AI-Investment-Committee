import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from crewai import Agent, Task, Crew, Process, LLM
from arch import arch_model

# Framework patch for Groq prompt-caching bug
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

st.set_page_config(page_title="AI Investment Committee", layout="wide")
st.title("🤖 Multi-Agent AI Investment Committee Dashboard")
st.caption("Production Architecture: CrewAI, Groq Llama 3.3, GARCH(1,1), Historical VaR, and Live Fundamental Parsing")

with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    ticker = st.selectbox("Select Stock Ticker:", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SAP", "SIE.DE"])
    run_button = st.button("Run Committee Analysis", type="primary")

def fetch_advanced_financial_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period="1y")
    if df.empty:
        return None
        
    # --- MATH ENGINE 1: Technical Indicators (RSI & MACD) ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # --- MATH ENGINE 2: Log Returns & Value-at-Risk (VaR) ---
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    clean_returns = df['Log_Returns'].dropna()
    
    # 95% 1-Day Historical Value-at-Risk percentage
    var_95 = -np.percentile(clean_returns, 5) * 100

    # --- MATH ENGINE 3: GARCH(1,1) Volatility Modeling ---
    scaled_returns = clean_returns * 100
    try:
        model_garch = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='Normal')
        res_garch = model_garch.fit(disp='off')
        
        forecast_df = res_garch.forecast(horizon=1).variance
        latest_variance = forecast_df.iloc[-1].values
        
        # FIX: Flatten array and pull item directly to prevent type conversion errors
        garch_annualized_vol = float(np.sqrt(latest_variance).flatten()[0]) * np.sqrt(252)
    except Exception as e:
        garch_annualized_vol = float(clean_returns.std() * np.sqrt(252) * 100)

    # --- TEXT ENGINE: Live Market & Fundamental News Scraping ---
    raw_news = stock.news
    news_summary = ""
    if raw_news:
        for idx, item in enumerate(raw_news[:3]):
            content = item.get('content', {})
            title = content.get('title', 'No Title')
            summary = content.get('summary', '')
            news_summary += f"[{idx+1}] {title}: {summary}\n"
    else:
        news_summary = "No recent fundamental news transcripts or releases found."

    latest = df.iloc[-1]
    return {
        "current_price": round(latest['Close'], 2),
        "rsi_14": round(latest['RSI_14'], 2) if not pd.isna(latest['RSI_14']) else "N/A",
        "macd": round(latest['MACD'], 3) if not pd.isna(latest['MACD']) else "N/A",
        "macd_signal": round(latest['MACD_Signal'], 3) if not pd.isna(latest['MACD_Signal']) else "N/A",
        "var_95": round(var_95, 2),
        "garch_volatility": round(garch_annualized_vol, 2),
        "news_feed": news_summary
    }

if run_button:
    if not groq_api_key:
        st.error("Please provide your Groq API key in the sidebar to proceed.")
    else:
        with st.spinner(f"Extracting math metrics, parsing filings, and convening advanced investment committee..."):
            metrics = fetch_advanced_financial_data(ticker)
            if metrics is None:
                st.error("Failed to retrieve data for the chosen ticker.")
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${metrics['current_price']}")
                col2.metric("GARCH (1,1) Volatility", f"{metrics['garch_volatility']}%")
                col3.metric("14-Day RSI / MACD", f"{metrics['rsi_14']} / {metrics['macd']}")
                col4.metric("95% Historical VaR", f"{metrics['var_95']}%")

                os.environ["GROQ_API_KEY"] = groq_api_key
                llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=groq_api_key)

                quant_analyst = Agent(
                    role="Technical Quant Analyst",
                    goal=f"Analyze raw technical price structures, MACD trend lines, and GARCH statistical volatility variables for {ticker}.",
                    backstory="You are a quantitative specialist. You interpret momentum oscillators and conditional heteroskedasticity data frames to identify market regime shifts.",
                    llm=llm,
                    verbose=True
                )
                
                fundamental_analyst = Agent(
                    role="Fundamental Text Analyst",
                    goal=f"Scan and digest public corporate text streams, news feeds, and narrative data regarding {ticker} to establish sentiment context.",
                    backstory="You are an equity analyst specializing in reading corporate releases, evaluating competitive moats, and finding systemic risks within text transcripts.",
                    llm=llm,
                    verbose=True
                )
                
                risk_manager = Agent(
                    role="Risk Manager",
                    goal=f"Evaluate Value-at-Risk limits and portfolio exposures for {ticker} to calculate safe asset sizing restrictions.",
                    backstory="You are a conservative risk controller. You translate empirical statistical losses like VaR down into operational capital allocation safety limits.",
                    llm=llm,
                    verbose=True
                )

                task1 = Task(
                    description=(
                        f"Evaluate these data strings for {ticker}: Price={metrics['current_price']}, "
                        f"RSI={metrics['rsi_14']}, MACD={metrics['macd']} (Signal Line={metrics['macd_signal']}), "
                        f"and GARCH Annualized Volatility={metrics['garch_volatility']}%. "
                        "Determine if technical momentum is strengthening or decaying based on crossovers and variance levels."
                    ), 
                    expected_output="A quantitative market momentum assessment paragraph and an absolute momentum score (1-10).", 
                    agent=quant_analyst
                )
                
                task2 = Task(
                    description=f"Read and evaluate the following text metrics extracted from recent market releases for {ticker}:\n{metrics['news_feed']}\nIdentify core operational highlights, corporate sentiment indices, and potential business challenges mentioned.", 
                    expected_output="A corporate fundamental briefing analyzing business health and a sentiment bias indicator (Bullish/Bearish/Neutral).", 
                    agent=fundamental_analyst
                )
                
                task3 = Task(
                    description=(
                        f"Review the momentum report from the Quant and the corporate narrative from the Fundamental analyst. "
                        f"Factor in that {ticker} possesses a 95% 1-day Historical Value-at-Risk (VaR) of {metrics['var_95']}%, "
                        "meaning there is a 5% historical probability of losing that specific percentage of asset value in a single day. "
                        "Synthesize these three domains into an executive corporate mandate."
                    ), 
                    expected_output="A comprehensive Markdown memo containing: 1. Action Verdict (BUY/SELL/HOLD), 2. Mathematical Rationale (referencing GARCH and VaR explicitly), 3. Allocation and Capital Protection Blueprint.", 
                    agent=risk_manager
                )

                crew = Crew(
                    agents=[quant_analyst, fundamental_analyst, risk_manager], 
                    tasks=[task1, task2, task3], 
                    process=Process.sequential
                )
                
                result = crew.kickoff()
                st.success("Analysis Complete!")
                st.markdown(result.raw)

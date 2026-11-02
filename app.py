"""
Main Application - Enriching Stock Market Data using the OpenAI API
Financial Data Science & Generative AI Intelligence Terminal

Run using:
    streamlit run app.py
"""

import os
from datetime import datetime
import pandas as pd
import streamlit as st

# Application Configuration & Constants
import config
from modules.data_collection import (
    fetch_historical_data,
    fetch_stock_news,
    fetch_multi_stocks,
    validate_ticker_symbol,
)
from modules.data_preprocessing import clean_and_validate_stock_data
from modules.stock_analysis import calculate_technical_indicators
from modules.openai_analysis import (
    generate_modular_ai_insights,
    generate_full_ai_report,
    analyze_news_sentiment,
    get_openai_client,
)
from modules.visualization import (
    plot_price_and_volume,
    plot_rsi_indicator,
    plot_macd_indicator,
    plot_stock_comparison,
    plot_historical_volatility,
)
from modules.report_generator import generate_pdf_report

# ------------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & CSS INJECTION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Market Enrichment | FinTech Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load and inject custom CSS
def load_custom_css():
    css_path = config.ASSETS_DIR / "custom.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()

# ------------------------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# ------------------------------------------------------------------------------
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = config.DEFAULT_INDIAN_TICKER
if "current_period" not in st.session_state:
    st.session_state.current_period = config.DEFAULT_TIMEFRAME
if "stock_df" not in st.session_state:
    st.session_state.stock_df = None
if "stock_info" not in st.session_state:
    st.session_state.stock_info = None
if "tech_summary" not in st.session_state:
    st.session_state.tech_summary = None
if "data_audit" not in st.session_state:
    st.session_state.data_audit = None
if "news_articles" not in st.session_state:
    st.session_state.news_articles = []
if "ai_insights" not in st.session_state:
    st.session_state.ai_insights = None
if "full_report" not in st.session_state:
    st.session_state.full_report = None
if "ai_cache" not in st.session_state:
    st.session_state.ai_cache = {}


# Helper currency symbol mapper
def get_currency_symbol(currency_code: str) -> str:
    mapping = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
    return mapping.get(currency_code.upper(), currency_code + " ")


# ------------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & NAVIGATION
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 16px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect width="32" height="32" rx="8" fill="#131B2E"/>
                    <rect x="0.5" y="0.5" width="31" height="31" rx="7.5" stroke="#3B82F6" stroke-opacity="0.3"/>
                    <path d="M6 22L12 15L17 19L26 8" stroke="#00E676" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M20 8H26V14" stroke="#00E676" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <div>
                    <h3 style="margin:0; font-size:17px; font-weight:800; color:#F8FAFC; letter-spacing:0.5px;">QUANTIX <span style="color:#38BDF8;">AI</span></h3>
                    <small style="color:#64748B; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:1px;">Market Terminal</small>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Market Domain Selector
    market_choice = st.radio(
        "Select Market Universe",
        options=["Indian Stocks (NSE)", "US Stocks", "Custom Search"],
        index=0,
        horizontal=False,
    )

    # Ticker Selection Logic
    if market_choice == "Indian Stocks (NSE)":
        predefined_options = list(config.INDIAN_STOCKS.keys())
        selected_key = st.selectbox(
            "Select Indian Bluechip",
            options=predefined_options,
            format_func=lambda x: f"{x} — {config.INDIAN_STOCKS[x]}",
            index=0,
        )
        target_ticker = selected_key
    elif market_choice == "US Stocks":
        predefined_options = list(config.US_STOCKS.keys())
        selected_key = st.selectbox(
            "Select US Tech / Bluechip",
            options=predefined_options,
            format_func=lambda x: f"{x} — {config.US_STOCKS[x]}",
            index=0,
        )
        target_ticker = selected_key
    else:
        custom_input = st.text_input(
            "Enter Ticker Symbol",
            value="NVDA",
            help="For Indian stocks append .NS (e.g. INFY.NS). For US stocks use bare ticker (e.g. MSFT).",
        )
        target_ticker = custom_input.strip().upper()

    # Timeframe selection
    selected_period = st.selectbox(
        "Historical Timeframe",
        options=list(config.TIMEFRAME_OPTIONS.keys()),
        format_func=lambda x: f"{config.TIMEFRAME_OPTIONS[x]} ({x})",
        index=2,  # 6mo default
    )

    st.markdown("---")

    # OpenAI API Configuration
    st.markdown("#### 🤖 OpenAI Intelligence Settings")
    
    # Check if key is available in environment
    env_has_key = bool(config.OPENAI_API_KEY)
    
    user_api_key = st.text_input(
        "OpenAI API Key",
        value=config.OPENAI_API_KEY if env_has_key else "",
        type="password",
        placeholder="sk-proj-...",
        help="Reads from .env automatically if set. You can also paste your OpenAI key here for demo testing.",
    )

    effective_api_key = user_api_key.strip() if user_api_key.strip() else None

    if effective_api_key:
        st.caption("🟢 OpenAI API Key configured")
    else:
        st.caption("🟡 No API Key entered — Running in Deterministic Academic Model")

    selected_model = st.selectbox(
        "OpenAI Model Engine",
        options=config.SUPPORTED_OPENAI_MODELS,
        index=0,
    )

    st.markdown("---")

    # Primary Action Button
    analyze_clicked = st.button("🚀 Analyze Stock", use_container_width=True, type="primary")

    # Secondary Refresh
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.session_state.stock_df = None
        analyze_clicked = True

    st.markdown(
        """
        <div style="font-size:10px; color:#64748B; margin-top:20px; line-height:1.4;">
            <b>System:</b> Financial Data Science Intelligence<br>
            <b>Stack:</b> Python 3.13, Streamlit, yfinance, Plotly, OpenAI SDK
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# 4. DATA PIPELINE EXECUTION ENGINE
# ------------------------------------------------------------------------------
def execute_pipeline(ticker: str, period: str):
    """Executes the full end-to-end data collection, preprocessing, and analysis pipeline."""
    with st.spinner(f"Acquiring real-time market data for {ticker}..."):
        raw_df, raw_info, err = fetch_historical_data(ticker, period=period)

        if err or raw_df.empty:
            st.error(f"❌ {err or 'Failed to retrieve stock data.'}")
            return False

        # Preprocessing Step
        clean_df, audit = clean_and_validate_stock_data(raw_df)

        # Technical Indicators Calculation
        enriched_df, tech_summary = calculate_technical_indicators(clean_df)

        # News Retrieval
        news = fetch_stock_news(ticker, max_items=5)

        # Commit to Session State
        st.session_state.current_ticker = ticker
        st.session_state.current_period = period
        st.session_state.stock_df = enriched_df
        st.session_state.stock_info = raw_info
        st.session_state.tech_summary = tech_summary
        st.session_state.data_audit = audit
        st.session_state.news_articles = news
        st.session_state.ai_insights = None  # Reset AI insights for new stock
        st.session_state.full_report = None
        return True


# Trigger pipeline if button clicked or if state is empty
if analyze_clicked or st.session_state.stock_df is None:
    try:
        clean_sym = validate_ticker_symbol(target_ticker)
        success = execute_pipeline(clean_sym, selected_period)
    except Exception as e:
        st.error(f"Invalid Ticker Input: {e}")
        st.stop()


# ------------------------------------------------------------------------------
# 5. MAIN DASHBOARD RENDER
# ------------------------------------------------------------------------------
df = st.session_state.stock_df
info = st.session_state.stock_info
tech = st.session_state.tech_summary
audit = st.session_state.data_audit
news = st.session_state.news_articles

if df is None or info is None or tech is None:
    st.info("👈 Please select a stock ticker from the sidebar and click **Analyze Stock** to begin.")
    st.stop()

curr_sym = get_currency_symbol(info.get("currency", "USD"))

# 1. Sleek Terminal Brand Header & Logo
st.markdown(
    """
    <div class="top-nav-banner">
        <div class="terminal-brand">
            <div class="brand-logo-icon">
                <svg width="30" height="30" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect width="32" height="32" rx="8" fill="#131B2E"/>
                    <rect x="0.5" y="0.5" width="31" height="31" rx="7.5" stroke="#3B82F6" stroke-opacity="0.4"/>
                    <path d="M6 22L12 15L17 19L26 8" stroke="#00E676" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M20 8H26V14" stroke="#00E676" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                    <circle cx="12" cy="15" r="1.5" fill="#38BDF8"/>
                    <circle cx="17" cy="19" r="1.5" fill="#38BDF8"/>
                    <circle cx="26" cy="8" r="2.2" fill="#00E676"/>
                </svg>
            </div>
            <div class="brand-text-container">
                <div class="brand-title">QUANTIX <span class="brand-accent">INTELLIGENCE</span></div>
                <div class="brand-tagline">INSTITUTIONAL QUANTITATIVE MARKET ANALYTICS</div>
            </div>
        </div>
        <div class="terminal-meta">
            <div class="market-status-badge">
                <span class="status-dot"></span>
                <span class="status-label">LIVE GLOBAL FEED</span>
            </div>
            <div class="terminal-pill">NSE • BSE • NASDAQ • NYSE</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Header Hero Banner
st.markdown(
    f"""
    <div class="hero-header">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
            <div>
                <h1 class="hero-title">{info.get('company_name', info['symbol'])} ({info['symbol']})</h1>
                <p class="hero-subtitle">
                    <b>Exchange:</b> {info.get('exchange', 'N/A')} &nbsp;|&nbsp;
                    <b>Sector:</b> {info.get('sector', 'N/A')} &nbsp;|&nbsp;
                    <b>Industry:</b> {info.get('industry', 'N/A')} &nbsp;|&nbsp;
                    <b>Currency:</b> {info.get('currency', 'USD')}
                </p>
            </div>
            <div style="text-align:right;">
                <div style="font-size:11px; color:#94A3B8;">Last Updated Data Date</div>
                <div style="font-size:14px; font-weight:600; color:#38BDF8;">{audit.get('date_end', 'Today')}</div>
                <div style="font-size:10px; color:#64748B;">Source: Yahoo Finance Official API Feed</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 6. STOCK OVERVIEW KPI METRIC CARDS
# ------------------------------------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)

daily_pct = tech.get("daily_pct_change", 0.0)
delta_class = "delta-positive" if daily_pct >= 0 else "delta-negative"
arrow = "▲" if daily_pct >= 0 else "▼"

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Latest Price</div>
            <div class="metric-value">{curr_sym}{tech.get('latest_close', 0.0):,.2f}</div>
            <div class="metric-delta {delta_class}">
                {arrow} {tech.get('daily_pct_change', 0.0):+.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Period Return</div>
            <div class="metric-value">{tech.get('total_period_return', 0.0):+.2f}%</div>
            <div style="font-size:11px; color:#94A3B8;">Open: {curr_sym}{tech.get('period_open', 0.0):,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">52-Week High</div>
            <div class="metric-value">{curr_sym}{info.get('fifty_two_week_high', 0.0):,.2f}</div>
            <div style="font-size:11px; color:#94A3B8;">Period High: {curr_sym}{tech.get('period_high', 0.0):,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">52-Week Low</div>
            <div class="metric-value">{curr_sym}{info.get('fifty_two_week_low', 0.0):,.2f}</div>
            <div style="font-size:11px; color:#94A3B8;">Period Low: {curr_sym}{tech.get('period_low', 0.0):,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    vol_ratio = tech.get("volume_ratio", 1.0)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Trading Volume</div>
            <div class="metric-value">{tech.get('latest_volume', 0):,.0f}</div>
            <div style="font-size:11px; color:{'#00E676' if vol_ratio > 1.2 else '#94A3B8'};">
                {vol_ratio:.2f}x of 20-Day SMA
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c6:
    trend_sent = tech.get("trend_sentiment", "Neutral")
    badge_style = "badge-bullish" if "Bullish" in trend_sent else "badge-bearish" if "Bearish" in trend_sent else "badge-neutral"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Technical Stance</div>
            <div style="margin-top:6px;">
                <span class="badge {badge_style}">{trend_sent}</span>
            </div>
            <div style="font-size:11px; color:#94A3B8; margin-top:8px;">Vol: {tech.get('annualized_volatility_pct', 0.0):.1f}% Ann.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 7. TABBED ANALYTICS WORKSPACE
# ------------------------------------------------------------------------------
tab_chart, tab_indicators, tab_ai, tab_news, tab_compare, tab_report = st.tabs([
    "📈 Price & Charts",
    "⚡ Technical Indicators",
    "🤖 AI-Powered Intelligence",
    "📰 Financial News Enrichment",
    "⚖️ Stock Comparison",
    "📑 Full AI Report & Data Export",
])

# ==============================================================================
# TAB 1: PRICE & CHARTS
# ==============================================================================
with tab_chart:
    st.markdown("### Interactive Market Dynamics & Moving Average Overlays")
    
    # Chart controls
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns(5)
    with ctrl_col1:
        c_type = st.selectbox("Chart Style", ["Candlestick", "Line Chart"], index=0)
    with ctrl_col2:
        t_sma20 = st.checkbox("Show SMA 20", value=True)
    with ctrl_col3:
        t_sma50 = st.checkbox("Show SMA 50", value=True)
    with ctrl_col4:
        t_sma200 = st.checkbox("Show SMA 200", value=True)
    with ctrl_col5:
        t_ema20 = st.checkbox("Show EMA 20", value=False)

    fig_price = plot_price_and_volume(
        df=df,
        ticker=info["symbol"],
        company_name=info["company_name"],
        chart_type=c_type,
        show_sma_20=t_sma20,
        show_sma_50=t_sma50,
        show_sma_200=t_sma200,
        show_ema_20=t_ema20,
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Secondary Volatility Chart
    with st.expander("📊 View Rolling 21-Day Annualized Volatility Curve", expanded=False):
        fig_vol = plot_historical_volatility(df, info["symbol"])
        st.plotly_chart(fig_vol, use_container_width=True)
        st.caption(
            "Mathematical formulation: Annualized Volatility is calculated as the standard deviation "
            "of daily log returns multiplied by $\\sqrt{252}$ trading days."
        )


# ==============================================================================
# TAB 2: TECHNICAL INDICATORS
# ==============================================================================
with tab_indicators:
    st.markdown("### Quantitative Indicator Suite & Algorithmic Signals")

    ind_col1, ind_col2 = st.columns(2)

    with ind_col1:
        st.plotly_chart(plot_rsi_indicator(df, info["symbol"]), use_container_width=True)
        st.markdown(
            f"""
            <div style="background:rgba(15,23,42,0.6); padding:12px; border-radius:8px; border:1px solid #1E293B;">
                <b>RSI (14-Day Wilder's Smoothing):</b> <span style="font-size:16px; color:#38BDF8;">{tech.get('rsi_14', 50):.2f}</span><br>
                <small style="color:#94A3B8;">Status: {tech.get('rsi_signal')}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ind_col2:
        st.plotly_chart(plot_macd_indicator(df, info["symbol"]), use_container_width=True)
        st.markdown(
            f"""
            <div style="background:rgba(15,23,42,0.6); padding:12px; border-radius:8px; border:1px solid #1E293B;">
                <b>MACD Line:</b> {tech.get('macd_line', 0):.4f} &nbsp;|&nbsp; 
                <b>Signal:</b> {tech.get('macd_signal_line', 0):.4f} &nbsp;|&nbsp; 
                <b>Hist:</b> {tech.get('macd_histogram', 0):.4f}<br>
                <small style="color:#94A3B8;">Momentum: {tech.get('macd_signal')}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Indicator Reference & Mathematical Foundation")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(
            """
            **Moving Averages (SMA & EMA)**
            - **SMA 20:** Short-term trend threshold
            - **SMA 50:** Medium-term trend benchmark
            - **SMA 200:** Institutional macro boundary
            - **Golden Cross:** SMA 50 crossing above SMA 200
            - **Death Cross:** SMA 50 crossing below SMA 200
            """
        )
    with m_col2:
        st.markdown(
            """
            **Relative Strength Index (RSI 14)**
            - Formulated by J. Welles Wilder Jr. (1978)
            - $RS = \\frac{\\text{Smoothed Gain}}{\\text{Smoothed Loss}}$
            - $RSI = 100 - \\frac{100}{1 + RS}$
            - Overbought threshold: $> 70$
            - Oversold threshold: $< 30$
            """
        )
    with m_col3:
        st.markdown(
            """
            **MACD (12, 26, 9)**
            - Developed by Gerald Appel
            - Fast EMA (12) minus Slow EMA (26)
            - 9-period EMA signal line
            - Histogram reflects momentum acceleration
            - Zero-line crossovers denote momentum shifts
            """
        )


# ==============================================================================
# TAB 3: AI-POWERED MARKET INTELLIGENCE
# ==============================================================================
with tab_ai:
    st.markdown("### 🤖 AI-Powered Market Intelligence")
    st.caption("Transforms structured quantitative data and indicators into actionable analytical narratives.")

    # Cache key for current query
    cache_key = f"{info['symbol']}_{selected_period}_{selected_model}"

    # Generate insights button or auto-load if cached
    ai_btn = st.button("✨ Generate / Refresh AI Analysis", type="primary")

    if ai_btn or (cache_key in st.session_state.ai_cache):
        if cache_key not in st.session_state.ai_cache or ai_btn:
            with st.spinner("Synthesizing multi-factor AI insights via OpenAI API..."):
                insights = generate_modular_ai_insights(
                    info=info,
                    tech_summary=tech,
                    news=news,
                    api_key=effective_api_key,
                    model=selected_model,
                )
                st.session_state.ai_cache[cache_key] = insights
                st.session_state.ai_insights = insights
        else:
            st.session_state.ai_insights = st.session_state.ai_cache[cache_key]

    insights = st.session_state.ai_insights

    if insights:
        if "api_error" in insights:
            st.warning(insights["api_error"])

        ai_tabs = st.tabs([
            "📋 Stock Summary",
            "📈 Trend Analysis",
            "⚡ Technical Explanation",
            "⚠️ Risk Analysis",
            "💡 Strategic Insights",
            "🎓 Beginner's Guide",
        ])

        with ai_tabs[0]:
            st.markdown(
                f"""
                <div class="ai-panel">
                    <span class="ai-pill">A. Executive Stock Summary</span>
                    <div style="font-size:14px; line-height:1.7; color:#E2E8F0;">
                        {insights.get('stock_summary', 'No summary available.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with ai_tabs[1]:
            st.markdown(
                f"""
                <div class="ai-panel">
                    <span class="ai-pill">B. Momentum & Trend Dynamics</span>
                    <div style="font-size:14px; line-height:1.7; color:#E2E8F0;">
                        {insights.get('trend_analysis', 'No trend analysis available.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with ai_tabs[2]:
            st.markdown(
                f"""
                <div class="ai-panel">
                    <span class="ai-pill">C. Technical Indicators Breakdown</span>
                    <div style="font-size:14px; line-height:1.7; color:#E2E8F0;">
                        {insights.get('technical_analysis', 'No technical analysis available.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with ai_tabs[3]:
            st.markdown(
                f"""
                <div class="ai-panel">
                    <span class="ai-pill">D. Quantitative Risk & Volatility</span>
                    <div style="font-size:14px; line-height:1.7; color:#E2E8F0;">
                        {insights.get('risk_analysis', 'No risk analysis available.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with ai_tabs[4]:
            st.markdown(
                f"""
                <div class="ai-panel">
                    <span class="ai-pill">E. Actionable Market Insights</span>
                    <div style="font-size:14px; line-height:1.7; color:#E2E8F0;">
                        {insights.get('market_insights', 'No market insights available.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with ai_tabs[5]:
            st.markdown(
                f"""
                <div class="ai-panel">
                    <span class="ai-pill">F. Beginner-Friendly Concept Translation</span>
                    <div style="font-size:14px; line-height:1.7; color:#E2E8F0;">
                        {insights.get('beginner_friendly', 'No beginner explanation available.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Click **✨ Generate / Refresh AI Analysis** above to query the OpenAI API and enrich the stock's indicators.")


# ==============================================================================
# TAB 4: FINANCIAL NEWS ENRICHMENT
# ==============================================================================
with tab_news:
    st.markdown("### Real-Time Financial News & Sentiment Enrichment")
    st.caption("Collects verified company news feeds and applies NLP sentiment and relevance scoring.")

    if not news:
        st.info(f"No recent news articles detected for symbol `{info['symbol']}` via financial wire feeds.")
    else:
        enrich_news_btn = st.button("🧠 Enrich News with AI Sentiment")
        
        enriched_news = news
        if enrich_news_btn:
            with st.spinner("Analyzing news sentiment and company relevance with OpenAI..."):
                enriched_news = analyze_news_sentiment(
                    articles=news,
                    ticker=info["symbol"],
                    company_name=info["company_name"],
                    api_key=effective_api_key,
                    model=selected_model,
                )
                st.session_state.news_articles = enriched_news

        for item in enriched_news:
            sentiment = item.get("sentiment", "Neutral")
            sent_color = "#00E676" if "Positive" in sentiment else "#FF5252" if "Negative" in sentiment else "#ECC94B"

            st.markdown(
                f"""
                <div class="news-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-size:11px; color:#94A3B8;">{item.get('source', 'Financial Wire')} • {item.get('publish_date', 'Recent')}</span>
                        <span style="font-size:10px; font-weight:700; color:{sent_color}; border:1px solid {sent_color}; padding:2px 8px; border-radius:12px;">
                            {sentiment}
                        </span>
                    </div>
                    <a href="{item.get('url', '#')}" target="_blank" class="news-title">{item.get('headline')}</a>
                    <p style="font-size:12px; color:#CBD5E1; margin:6px 0;">{item.get('description')}</p>
                    {f"<div style='font-size:11px; color:#A78BFA; background:rgba(79,70,229,0.15); padding:6px 10px; border-radius:6px; margin-top:6px;'><b>AI Relevance:</b> {item.get('relevance')}</div>" if 'relevance' in item else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# TAB 5: STOCK COMPARISON
# ==============================================================================
with tab_compare:
    st.markdown("### Multi-Asset Comparative Benchmarking")
    st.caption("Normalized historical return comparison: $R_t = \\frac{P_t - P_0}{P_0} \\times 100\\%$")

    comp_col1, comp_col2 = st.columns([1, 2])
    
    with comp_col1:
        # Determine comparison partner candidate
        default_pair = "TCS.NS" if info["symbol"] == "RELIANCE.NS" else "MSFT" if info["symbol"] == "AAPL" else "TCS.NS"
        second_ticker = st.text_input("Enter Comparison Ticker", value=default_pair).strip().upper()
        run_compare = st.button("Compare Assets", type="secondary")

    if second_ticker:
        with st.spinner(f"Fetching comparative data for {second_ticker}..."):
            multi_dict = fetch_multi_stocks([info["symbol"], second_ticker], period=selected_period)
            
            if len(multi_dict) >= 2:
                labels = {
                    info["symbol"]: f"{info['company_name']} ({info['symbol']})",
                    second_ticker: second_ticker,
                }
                fig_comp = plot_stock_comparison(multi_dict, labels)
                st.plotly_chart(fig_comp, use_container_width=True)

                # Summary return comparison table
                t1_df = multi_dict[info["symbol"]]
                t2_df = multi_dict[second_ticker]

                t1_ret = ((t1_df["Close"].iloc[-1] - t1_df["Close"].iloc[0]) / t1_df["Close"].iloc[0]) * 100
                t2_ret = ((t2_df["Close"].iloc[-1] - t2_df["Close"].iloc[0]) / t2_df["Close"].iloc[0]) * 100

                cmp_c1, cmp_c2 = st.columns(2)
                with cmp_c1:
                    st.metric(label=f"{info['symbol']} Period Return", value=f"{t1_ret:+.2f}%")
                with cmp_c2:
                    st.metric(label=f"{second_ticker} Period Return", value=f"{t2_ret:+.2f}%")
            else:
                st.warning(f"Could not load historical series for ticker '{second_ticker}'. Please verify the symbol.")


# ==============================================================================
# TAB 6: FULL AI REPORT & DATA EXPORT
# ==============================================================================
with tab_report:
    st.markdown("### Formal Academic Research Report & Data Export")

    rep_col1, rep_col2 = st.columns([2, 1])

    with rep_col1:
        gen_rep_btn = st.button("📄 Generate Full AI Equity Research Report", type="primary")

    if gen_rep_btn or (st.session_state.full_report is not None):
        if gen_rep_btn or (st.session_state.full_report is None):
            with st.spinner("Compiling formal structured academic research report..."):
                rep_content = generate_full_ai_report(
                    info=info,
                    tech_summary=tech,
                    news=news,
                    api_key=effective_api_key,
                    model=selected_model,
                )
                st.session_state.full_report = rep_content
        else:
            rep_content = st.session_state.full_report

        # Display Report in Markdown
        st.markdown("---")
        st.markdown(rep_content)

        # Download Actions
        st.markdown("---")
        st.markdown("#### 📥 Download Research Deliverables")
        
        down_col1, down_col2 = st.columns(2)
        
        # Markdown / Text Download
        with down_col1:
            st.download_button(
                label="📥 Download Report as Markdown (.md)",
                data=rep_content,
                file_name=f"{info['symbol']}_AI_Equity_Research_Report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # PDF Download
        with down_col2:
            try:
                pdf_bytes = generate_pdf_report(
                    report_markdown=rep_content,
                    ticker=info["symbol"],
                    company_name=info["company_name"],
                )
                st.download_button(
                    label="📕 Download Formal PDF Report (.pdf)",
                    data=pdf_bytes,
                    file_name=f"{info['symbol']}_AI_Equity_Research_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as pe:
                st.error(f"PDF compilation error: {pe}")

    st.markdown("---")
    st.markdown("### Historical Stock Market Data Table")

    # Export clean DataFrame with formatted columns
    display_df = df.copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    table_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Daily_Pct_Change",
        "SMA_20",
        "SMA_50",
        "RSI_14",
    ]
    available_cols = [c for c in table_cols if c in display_df.columns]

    st.dataframe(
        display_df[available_cols].style.format(
            {
                "Open": "{:.2f}",
                "High": "{:.2f}",
                "Low": "{:.2f}",
                "Close": "{:.2f}",
                "Adj Close": "{:.2f}",
                "Volume": "{:,.0f}",
                "Daily_Pct_Change": "{:+.2f}%",
                "SMA_20": "{:.2f}",
                "SMA_50": "{:.2f}",
                "RSI_14": "{:.2f}",
            }
        ),
        use_container_width=True,
        height=320,
    )

    # CSV Download Button
    csv_bytes = display_df.to_csv(index=True).encode("utf-8")
    st.download_button(
        label="📊 Download Historical Data as CSV (.csv)",
        data=csv_bytes,
        file_name=f"{info['symbol']}_{selected_period}_market_data.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------------------------
# 8. MANDATORY ACADEMIC & FINANCIAL DISCLAIMER FOOTER
# ------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="disclaimer-box">
        <b>ACADEMIC RESEARCH & FINANCIAL DISCLAIMER:</b> {config.FINANCIAL_DISCLAIMER}
    </div>
    """,
    unsafe_allow_html=True,
)

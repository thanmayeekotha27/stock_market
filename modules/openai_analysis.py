"""
OpenAI Analysis Module
Enriches quantitative financial indicators and company metrics with Generative AI insights
using the official OpenAI Python SDK. Designed for academic and analytical excellence.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

# System instructions enforcing financial rigor, hallucination prevention, and disclaimer rules
SYSTEM_PROMPT = """
You are an expert Senior Financial Data Scientist and Quantitative Analyst AI.
Your purpose is to enrich real-world stock market metrics and quantitative indicators with objective, educational insights.

STRICT OPERATING PRINCIPLES:
1. GROUNDED IN TRUTH: Base your analysis EXCLUSIVELY on the actual numerical data, indicators, and news provided in the user prompt. NEVER invent, hallucinate, or alter stock prices, moving averages, or technical values.
2. NO BUY/SELL ADVICE: Do NOT provide unsupported buy, sell, or hold financial recommendations. Frame observations in terms of technical momentum, risk factors, and historical context.
3. NO FUTURE GUARANTEES: Financial markets are stochastic. Never guarantee future price movements.
4. DISTINGUISH FACTS FROM INTERPRETATIONS: Clearly separate historical factual data (e.g. "RSI is currently 68.4") from interpretive commentary (e.g. "This indicates strong upward momentum nearing overbought territory").
5. ACCESSIBLE EXPLANATION: Explain complex quantitative concepts (e.g., Wilder's RSI, MACD divergence, exponential smoothing) clearly and intuitively.
6. MISSING DATA TRANSPARENCY: If certain indicators or news items are not provided or flagged as unavailable, state that explicitly.
7. COMPLIANCE: Always uphold a professional, objective financial analysis standard.
"""


def get_openai_client(api_key: Optional[str] = None) -> Optional[OpenAI]:
    """
    Initializes and returns an OpenAI client safely.
    Checks user-provided key, then environment variable OPENAI_API_KEY.
    """
    key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key == "your_openai_api_key_here":
        return None
    try:
        return OpenAI(api_key=key, timeout=30.0)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return None


def format_stock_context_payload(
    info: Dict[str, Any],
    tech_summary: Dict[str, Any],
    news: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Formats structured quantitative metrics and company background into a clean,
    token-efficient context string for OpenAI prompts.
    """
    currency = info.get("currency", "USD")
    sym = info.get("symbol", "N/A")
    name = info.get("company_name", sym)

    news_text = "No recent company-specific news articles available."
    if news and len(news) > 0:
        news_items = []
        for i, item in enumerate(news[:4], 1):
            news_items.append(
                f"{i}. [{item.get('source', 'News')} | {item.get('publish_date', 'Recent')}] "
                f"{item.get('headline', 'N/A')}\n   Snippet: {item.get('description', '')[:160]}"
            )
        news_text = "\n".join(news_items)

    sma_200_str = f"{tech_summary.get('sma_200', 'N/A'):.2f}" if tech_summary.get("has_sma_200") else "Unavailable (<200 trading days)"

    payload = f"""
=== STOCK OVERVIEW & FUNDAMENTALS ===
Company: {name} ({sym})
Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')} | Country: {info.get('country', 'N/A')}
Market Cap: {info.get('market_cap', 0):,} {currency}
Trailing/Forward P/E: {info.get('pe_ratio', 'N/A')} | Beta: {info.get('beta', 'N/A')}
52-Week High: {info.get('fifty_two_week_high', 0.0):.2f} {currency} | 52-Week Low: {info.get('fifty_two_week_low', 0.0):.2f} {currency}

=== QUANTITATIVE MARKET DATA & HISTORICAL PERFORMANCE ===
Latest Close Price: {tech_summary.get('latest_close', 0.0):.2f} {currency}
Latest Daily Change: {tech_summary.get('daily_pct_change', 0.0):+.2f}%
Period Range: Open = {tech_summary.get('period_open', 0.0):.2f}, High = {tech_summary.get('period_high', 0.0):.2f}, Low = {tech_summary.get('period_low', 0.0):.2f}
Total Period Return: {tech_summary.get('total_period_return', 0.0):+.2f}%
Annualized Volatility: {tech_summary.get('annualized_volatility_pct', 0.0):.2f}%
Total Analyzed Trading Days: {tech_summary.get('total_trading_days', 0)}

=== TECHNICAL INDICATORS ===
1. Simple Moving Averages:
   - SMA 20: {tech_summary.get('sma_20', 0.0):.2f} {currency}
   - SMA 50: {tech_summary.get('sma_50', 0.0):.2f} {currency}
   - SMA 200: {sma_200_str}
   - Moving Average Alignment: {tech_summary.get('moving_average_alignment', 'N/A')}
   - Long-Term Trend Signal: {tech_summary.get('cross_event', 'N/A')}
2. Exponential Moving Average:
   - EMA 20: {tech_summary.get('ema_20', 0.0):.2f} {currency}
3. Relative Strength Index (RSI 14):
   - Current RSI: {tech_summary.get('rsi_14', 50.0):.2f}
   - RSI Interpretation: {tech_summary.get('rsi_signal', 'Neutral')}
4. MACD (12, 26, 9):
   - MACD Line: {tech_summary.get('macd_line', 0.0):.4f}
   - Signal Line: {tech_summary.get('macd_signal_line', 0.0):.4f}
   - Histogram: {tech_summary.get('macd_histogram', 0.0):.4f}
   - MACD Momentum: {tech_summary.get('macd_signal', 'Neutral')}
5. Trading Volume Trends:
   - Latest Volume: {tech_summary.get('latest_volume', 0):,.0f}
   - 20-Day Average Volume: {tech_summary.get('volume_sma_20', 0):,.0f}
   - Volume Ratio: {tech_summary.get('volume_ratio', 1.0):.2f}x
   - Volume Assessment: {tech_summary.get('volume_signal', 'Normal')}

=== RECENT COMPANY NEWS & DEVELOPMENTS ===
{news_text}
"""
    return payload.strip()


def generate_modular_ai_insights(
    info: Dict[str, Any],
    tech_summary: Dict[str, Any],
    news: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> Dict[str, str]:
    """
    Generates all 6 modular AI insight dimensions:
      A. Stock Summary
      B. Trend Analysis
      C. Technical Analysis
      D. Risk Analysis
      E. Market Insights
      F. Beginner-Friendly Explanation
      
    Returns a dictionary of section names to generated markdown analysis.
    """
    client = get_openai_client(api_key)
    context = format_stock_context_payload(info, tech_summary, news)

    if not client:
        return _generate_offline_academic_insights(info, tech_summary, news)

    prompt = f"""
Analyze the following real-world quantitative stock market data:

{context}

Generate an in-depth financial evaluation strictly formatted as a JSON object with EXACTLY the following 6 keys:
{{
  "stock_summary": "Concise 2-3 paragraph explanation of the stock's recent performance, key price milestones, and valuation context.",
  "trend_analysis": "Detailed explanation of whether indicators suggest bullish, bearish, or sideways momentum, evaluating price action vs moving averages (SMA 20, 50, 200) and MACD.",
  "technical_analysis": "Comprehensive breakdown of the moving averages, RSI 14, MACD convergence/divergence, and volume dynamics in clear language.",
  "risk_analysis": "Objective identification of volatility ({tech_summary.get('annualized_volatility_pct', 0.0):.2f}% annualized), price drawdowns from high, volume anomalies, and systemic/macro risk considerations.",
  "market_insights": "Strategic analytical observations connecting technical momentum, trading volume signals, and broader sector context.",
  "beginner_friendly": "A warm, intuitive explanation of the stock's current state designed for a novice student or beginner investor, using analogies where helpful without jargon."
}}

Respond ONLY with valid JSON.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2500,
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return data

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        offline = _generate_offline_academic_insights(info, tech_summary, news)
        offline["api_error"] = f"OpenAI API Notice: {str(e)}. Displaying calculated statistical insights."
        return offline


def generate_full_ai_report(
    info: Dict[str, Any],
    tech_summary: Dict[str, Any],
    news: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generates a structured comprehensive academic research report:
      1. Company Overview
      2. Performance Summary
      3. Technical Indicators Analysis
      4. Market Trend & Momentum
      5. Risk Observations
      6. Important Insights
      7. Academic Conclusion
      8. Financial Disclaimer
    """
    client = get_openai_client(api_key)
    context = format_stock_context_payload(info, tech_summary, news)

    if not client:
        return _generate_offline_full_report(info, tech_summary, news)

    prompt = f"""
You are preparing a formal Institutional Financial Research Report.

Review the following verified stock data:
{context}

Generate a comprehensive, beautifully structured research report in Markdown format.
Include these exact section headings:

# Executive Equity Research Report: {info.get('company_name')} ({info.get('symbol')})
*Generated via Antigravity Financial Intelligence & OpenAI API Integration*

## 1. Company Overview
(Provide company profile, business model, sector/industry positioning, market cap, and valuation context.)

## 2. Performance Summary
(Analyze the period return, recent price trajectory, 52-week high/low relative position, and daily changes.)

## 3. Technical Indicators Analysis
(Examine SMA 20/50/200 alignment, EMA 20, RSI 14 oscillator status, MACD momentum, and volume validation.)

## 4. Market Trend & Momentum
(State clearly whether current momentum is Bullish, Bearish, or Consolidating, citing specific indicator values.)

## 5. Quantitative Risk Observations
(Analyze annualized volatility ({tech_summary.get('annualized_volatility_pct', 0.0):.2f}%), price swings, and risk factors.)

## 6. Strategic Market Insights & News Relevance
(Synthesize findings, volume trends, and news items if available.)

## 7. Conclusion & Research Takeaways
(Summarize the data-driven findings objectively without speculative buy/sell commands.)

## 8. Regulatory & Compliance Disclaimer
(Mandatory disclaimer stating that this report is generated for educational and analytical purposes and does not constitute financial advice.)
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        return response.choices[0].message.content or "No report content generated."
    except Exception as e:
        logger.error(f"OpenAI Full Report generation error: {e}")
        offline_rep = _generate_offline_full_report(info, tech_summary, news)
        return f"> **Notice:** OpenAI API call encountered an error: `{str(e)}`. Showing rule-based quantitative report below.\n\n" + offline_rep


def analyze_news_sentiment(
    articles: List[Dict[str, Any]],
    ticker: str,
    company_name: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """
    Enriches each news article with:
      - Short summary
      - Sentiment assessment (Positive, Negative, Neutral)
      - Relevance to company fundamentals vs market noise
    """
    if not articles:
        return []

    client = get_openai_client(api_key)
    if not client:
        # Fallback offline rule-based classification
        enriched = []
        for art in articles:
            enriched.append({
                **art,
                "ai_summary": art.get("description", "Headline news regarding company operations."),
                "sentiment": "Neutral / Informative",
                "relevance": "General market coverage for " + company_name,
            })
        return enriched

    articles_payload = json.dumps([{
        "id": i,
        "headline": a.get("headline"),
        "source": a.get("source"),
        "snippet": a.get("description", "")[:180]
    } for i, a in enumerate(articles)], indent=2)

    prompt = f"""
Analyze these recent news articles for {company_name} ({ticker}):

{articles_payload}

For each article, provide:
1. "ai_summary": A 1-2 sentence executive summary of the event.
2. "sentiment": One of ["Positive", "Negative", "Neutral", "Mixed"].
3. "relevance": A brief sentence explaining potential impact or relevance to {ticker}.

Return a JSON array of objects with keys: "id", "ai_summary", "sentiment", "relevance".
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1500,
        )
        parsed = json.loads(response.choices[0].message.content)
        items_map = {item["id"]: item for item in parsed.get("articles", parsed.get("items", [])) if "id" in item}
        
        # Merge back
        results = []
        for i, art in enumerate(articles):
            enrichment = items_map.get(i, {})
            results.append({
                **art,
                "ai_summary": enrichment.get("ai_summary", art.get("description", "")),
                "sentiment": enrichment.get("sentiment", "Neutral"),
                "relevance": enrichment.get("relevance", "Company market activity."),
            })
        return results

    except Exception as e:
        logger.warning(f"News sentiment analysis error: {e}")
        return [{
            **a,
            "ai_summary": a.get("description", ""),
            "sentiment": "Neutral",
            "relevance": "Company coverage",
        } for a in articles]


# ==============================================================================
# OFFLINE / FALLBACK RULE-BASED ANALYTICAL ENRICHMENT
# (Ensures zero-crash usability and academic rigor even when API keys are absent)
# ==============================================================================

def _generate_offline_academic_insights(
    info: Dict[str, Any],
    tech_summary: Dict[str, Any],
    news: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, str]:
    """Generates structured data science insights computed deterministically from metrics."""
    curr = info.get("currency", "USD")
    sym = info.get("symbol", "N/A")
    name = info.get("company_name", sym)
    close = tech_summary.get("latest_close", 0.0)
    ret = tech_summary.get("total_period_return", 0.0)
    rsi = tech_summary.get("rsi_14", 50.0)
    vol = tech_summary.get("annualized_volatility_pct", 0.0)
    trend = tech_summary.get("trend_sentiment", "Neutral")
    vol_ratio = tech_summary.get("volume_ratio", 1.0)

    return {
        "stock_summary": (
            f"**{name} ({sym})** is currently trading at **{close:.2f} {curr}**, reflecting a **{ret:+.2f}%** "
            f"total return over the analyzed timeframe. The 52-week range spans from a low of "
            f"**{info.get('fifty_two_week_low', 0):.2f}** to a high of **{info.get('fifty_two_week_high', 0):.2f} {curr}**. "
            f"With an annualized historical volatility of **{vol:.2f}%**, price action shows characteristic equity market fluctuations."
        ),
        "trend_analysis": (
            f"Momentum indicators currently reflect a **{trend}** stance. The stock's Moving Average structure indicates: "
            f"**{tech_summary.get('moving_average_alignment')}**. The 20-day Simple Moving Average is at **{tech_summary.get('sma_20', 0):.2f}**, "
            f"while the 50-day SMA sits at **{tech_summary.get('sma_50', 0):.2f}**. Long-term structure is signaled by: "
            f"*{tech_summary.get('cross_event')}*."
        ),
        "technical_analysis": (
            f"• **RSI (14-period Wilder's Smoothing):** Currently registered at **{rsi:.2f}**, placing the asset in "
            f"*{tech_summary.get('rsi_signal')}*.\n"
            f"• **MACD (12, 26, 9):** MACD Line is at **{tech_summary.get('macd_line', 0):.4f}** vs Signal Line at **{tech_summary.get('macd_signal_line', 0):.4f}** "
            f"(Histogram: **{tech_summary.get('macd_histogram', 0):.4f}**), indicating: *{tech_summary.get('macd_signal')}*.\n"
            f"• **Volume Dynamics:** Recent session volume is **{vol_ratio:.2f}x** of its 20-day SMA, indicating *{tech_summary.get('volume_signal')}*."
        ),
        "risk_analysis": (
            f"The asset exhibits an annualized volatility profile of **{vol:.2f}%**, measured using daily logarithmic returns. "
            f"The period maximum was **{tech_summary.get('period_high', 0):.2f}** and the minimum was **{tech_summary.get('period_low', 0):.2f}**, "
            f"representing a maximum drawdown spread of **{abs(tech_summary.get('period_high', 0) - tech_summary.get('period_low', 0)):.2f} {curr}**. "
            f"Traders should monitor potential support/resistance at the 50-day moving average."
        ),
        "market_insights": (
            f"1. **Volume-Price Confirmation:** Trading volume at {vol_ratio:.2f}x average indicates {'heightened trader interest' if vol_ratio > 1.2 else 'steady baseline trading'}.\n"
            f"2. **Mean-Reversion Potential:** With RSI at {rsi:.2f}, the asset {'faces potential pullback pressure' if rsi > 70 else 'is in a stable technical balance zone' if rsi >= 40 else 'may experience short-covering rallies'}.\n"
            f"3. **Macro Sensitivity:** As a leading enterprise in the **{info.get('sector', 'broad')}** sector, price behavior is closely tied to industry benchmark indices."
        ),
        "beginner_friendly": (
            f"Imagine the stock price as a car on a highway: the 20-day moving average is like its recent speed limit, "
            f"and the 50-day is its average journey pace. Right now, {name} is trading at **{close:.2f} {curr}**, which is "
            f"{'ahead of' if close > tech_summary.get('sma_20', 0) else 'behind'} its short-term average.\n\n"
            f"The **RSI speedometer** reads **{rsi:.1f} out of 100**. Scores between 40 and 60 represent normal cruising speed. "
            f"Scores above 70 indicate high-speed overheating (overbought), while scores under 30 show the engine is dragging (oversold). "
            f"Overall, this stock is currently showing **{trend.lower()}** behavior."
        ),
    }


def _generate_offline_full_report(
    info: Dict[str, Any],
    tech_summary: Dict[str, Any],
    news: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Generates a complete markdown research report using quantitative deterministic formulas."""
    insights = _generate_offline_academic_insights(info, tech_summary, news)
    sym = info.get("symbol", "N/A")
    name = info.get("company_name", sym)
    curr = info.get("currency", "USD")

    return f"""# Executive Equity Research Report: {name} ({sym})
*Generated via Antigravity Quantitative Data Engine*

---

## 1. Company Overview
**{name}** operates within the **{info.get('sector', 'N/A')}** sector, specializing in **{info.get('industry', 'N/A')}**.
- **Exchange:** {info.get('exchange', 'N/A')}
- **Market Capitalization:** {info.get('market_cap', 0):,} {curr}
- **P/E Ratio:** {info.get('pe_ratio', 'N/A')} | **Beta:** {info.get('beta', 'N/A')}
- **Business Profile:** {info.get('business_summary', 'N/A')[:400]}...

## 2. Performance Summary
{insights['stock_summary']}

## 3. Technical Indicators Analysis
{insights['technical_analysis']}

## 4. Market Trend & Momentum
{insights['trend_analysis']}

## 5. Quantitative Risk Observations
{insights['risk_analysis']}

## 6. Strategic Market Insights & News Relevance
{insights['market_insights']}

## 7. Conclusion & Research Takeaways
Quantitative modeling of {sym} shows a primary **{tech_summary.get('trend_sentiment')}** posture with an RSI oscillator reading of **{tech_summary.get('rsi_14', 0):.2f}** and annualized volatility of **{tech_summary.get('annualized_volatility_pct', 0):.2f}%**. The convergence of price relative to key moving averages provides valuable technical insight for market observation.

---
## 8. Regulatory & Compliance Disclaimer
**DISCLAIMER:** This report was generated by the Financial Intelligence Engine for analytical and educational demonstration only ("Enriching Stock Market Data using the OpenAI API"). All calculations are for informational purposes and do not constitute financial, investment, or legal advice.
"""

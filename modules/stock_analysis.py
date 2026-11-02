"""
Stock Analysis Module
Computes financial technical indicators, statistical volatility,
and algorithmic momentum signals with academic rigor.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI) using J. Welles Wilder's smoothing method.
    
    Formula:
        RS = Smoothed_Avg_Gain / Smoothed_Avg_Loss
        RSI = 100 - (100 / (1 + RS))
    """
    if len(series) < window + 1:
        return pd.Series(50.0, index=series.index)

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing uses alpha = 1 / window
    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Formula:
        MACD Line = EMA(Close, 12) - EMA(Close, 26)
        Signal Line = EMA(MACD Line, 9)
        Histogram = MACD Line - Signal Line
    """
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_technical_indicators(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enriches preprocessed stock DataFrame with 10 quantitative financial indicators
    and produces an analytical summary.
    
    Args:
        df: Preprocessed DataFrame containing 'Close', 'High', 'Low', 'Volume'
        
    Returns:
        Tuple of (enriched_df, indicators_summary_dict)
    """
    if df.empty or "Close" not in df.columns:
        return df, {}

    enriched = df.copy()
    close = enriched["Close"]
    n_records = len(enriched)

    # 1. Simple Moving Averages
    enriched["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
    enriched["SMA_50"] = close.rolling(window=50, min_periods=1).mean()
    
    # SMA 200 requires sufficient historical observations
    if n_records >= 200:
        enriched["SMA_200"] = close.rolling(window=200, min_periods=200).mean()
    else:
        enriched["SMA_200"] = np.nan

    # 2. Exponential Moving Average (EMA 20)
    enriched["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    # 3. Relative Strength Index (RSI 14)
    enriched["RSI_14"] = compute_rsi(close, window=14)

    # 4. MACD (12, 26, 9)
    macd_line, signal_line, hist = compute_macd(close, 12, 26, 9)
    enriched["MACD_Line"] = macd_line
    enriched["MACD_Signal"] = signal_line
    enriched["MACD_Hist"] = hist

    # 5. Historical Volatility (Annualized standard deviation of daily log returns)
    # sigma_ann = sigma_daily * sqrt(252) * 100
    if "Log_Return" in enriched.columns:
        daily_log_ret = enriched["Log_Return"]
    else:
        daily_log_ret = np.log(close / close.shift(1)).fillna(0)

    # 21-day rolling volatility
    rolling_vol = daily_log_ret.rolling(window=21, min_periods=5).std() * np.sqrt(252) * 100.0
    enriched["Historical_Vol_21d"] = rolling_vol.fillna(0.0)

    # Overall period annualized volatility
    period_volatility = float(daily_log_ret.std() * np.sqrt(252) * 100.0) if len(daily_log_ret) > 1 else 0.0

    # 6. Trading Volume Trends
    vol = enriched["Volume"]
    enriched["Volume_SMA_20"] = vol.rolling(window=20, min_periods=1).mean()
    latest_vol = float(vol.iloc[-1])
    latest_vol_sma = float(enriched["Volume_SMA_20"].iloc[-1])
    vol_ratio = (latest_vol / latest_vol_sma) if latest_vol_sma > 0 else 1.0

    # 7. Extremum (Max and Min prices in period)
    period_high = float(enriched["High"].max())
    period_low = float(enriched["Low"].min())
    period_open = float(enriched["Open"].iloc[0])
    latest_close = float(close.iloc[-1])
    total_period_return = ((latest_close - period_open) / period_open) * 100.0

    # Latest indicator values
    latest_sma_20 = float(enriched["SMA_20"].iloc[-1])
    latest_sma_50 = float(enriched["SMA_50"].iloc[-1])
    has_sma_200 = not np.isnan(enriched["SMA_200"].iloc[-1])
    latest_sma_200 = float(enriched["SMA_200"].iloc[-1]) if has_sma_200 else None
    latest_ema_20 = float(enriched["EMA_20"].iloc[-1])
    latest_rsi = float(enriched["RSI_14"].iloc[-1])
    latest_macd = float(enriched["MACD_Line"].iloc[-1])
    latest_macd_signal = float(enriched["MACD_Signal"].iloc[-1])
    latest_macd_hist = float(enriched["MACD_Hist"].iloc[-1])
    latest_daily_pct = float(enriched["Daily_Pct_Change"].iloc[-1]) if "Daily_Pct_Change" in enriched.columns else 0.0

    # Algorithmic Interpretations
    # A. RSI Assessment
    if latest_rsi >= 70:
        rsi_signal = "Overbought (>70) — Caution for potential mean reversion or profit-taking"
        rsi_sentiment = "Bearish"
    elif latest_rsi <= 30:
        rsi_signal = "Oversold (<30) — Potential selling exhaustion / technical bounce zone"
        rsi_sentiment = "Bullish"
    elif latest_rsi >= 55:
        rsi_signal = "Bullish Momentum (55-70) — Positive price accumulation"
        rsi_sentiment = "Bullish"
    elif latest_rsi <= 45:
        rsi_signal = "Bearish Drift (30-45) — Sustained selling pressure"
        rsi_sentiment = "Bearish"
    else:
        rsi_signal = "Neutral Zone (45-55) — Consolidation / equilibrium"
        rsi_sentiment = "Neutral"

    # B. MACD Assessment
    if latest_macd > latest_macd_signal and latest_macd_hist > 0:
        macd_signal = "Bullish Crossover — Positive upward momentum accelerating"
        macd_sentiment = "Bullish"
    elif latest_macd < latest_macd_signal and latest_macd_hist < 0:
        macd_signal = "Bearish Crossover — Downward divergence or distribution"
        macd_sentiment = "Bearish"
    else:
        macd_signal = "Neutral Convergence — Minimal momentum divergence"
        macd_sentiment = "Neutral"

    # C. Moving Average Alignment
    ma_alignment = "Mixed"
    if latest_close > latest_sma_20 > latest_sma_50:
        ma_alignment = "Strong Bullish Stack (Price > SMA20 > SMA50)"
        trend_sentiment = "Bullish"
    elif latest_close < latest_sma_20 < latest_sma_50:
        ma_alignment = "Strong Bearish Stack (Price < SMA20 < SMA50)"
        trend_sentiment = "Bearish"
    elif latest_close > latest_sma_20:
        ma_alignment = "Short-term Bullish above 20-day SMA"
        trend_sentiment = "Mild Bullish"
    else:
        ma_alignment = "Short-term Bearish below 20-day SMA"
        trend_sentiment = "Mild Bearish"

    # D. Golden / Death Cross detection (if 50 and 200 exist)
    cross_event = "Insufficient data for 200-day SMA evaluation (requires >=200 trading days)"
    if has_sma_200:
        if latest_sma_50 > latest_sma_200:
            cross_event = "Golden Cross Territory (50 SMA above 200 SMA) — Long-term macro uptrend"
        else:
            cross_event = "Death Cross Territory (50 SMA below 200 SMA) — Long-term macro downtrend"

    # E. Volume Trend
    if vol_ratio > 1.5:
        vol_signal = f"Significant Volume Spike ({vol_ratio:.2f}x of 20-day average) — High institutional participation"
    elif vol_ratio < 0.7:
        vol_signal = f"Below-Average Volume ({vol_ratio:.2f}x of 20-day average) — Low conviction consolidation"
    else:
        vol_signal = f"Normal Trading Volume ({vol_ratio:.2f}x of 20-day average)"

    # Comprehensive summary package for UI and AI prompts
    summary = {
        "latest_close": latest_close,
        "daily_pct_change": latest_daily_pct,
        "period_open": period_open,
        "period_high": period_high,
        "period_low": period_low,
        "total_period_return": total_period_return,
        "annualized_volatility_pct": period_volatility,
        "sma_20": latest_sma_20,
        "sma_50": latest_sma_50,
        "sma_200": latest_sma_200,
        "has_sma_200": has_sma_200,
        "ema_20": latest_ema_20,
        "rsi_14": latest_rsi,
        "rsi_signal": rsi_signal,
        "rsi_sentiment": rsi_sentiment,
        "macd_line": latest_macd,
        "macd_signal_line": latest_macd_signal,
        "macd_histogram": latest_macd_hist,
        "macd_signal": macd_signal,
        "macd_sentiment": macd_sentiment,
        "moving_average_alignment": ma_alignment,
        "trend_sentiment": trend_sentiment,
        "cross_event": cross_event,
        "latest_volume": latest_vol,
        "volume_sma_20": latest_vol_sma,
        "volume_ratio": vol_ratio,
        "volume_signal": vol_signal,
        "total_trading_days": n_records,
    }

    return enriched, summary

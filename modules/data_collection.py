"""
Data Collection Module
Fetches real-world historical stock data, company metadata, and news using yfinance.
Designed with enterprise data engineering standards and robust error handling.
"""

from datetime import datetime
import logging
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_ticker_symbol(ticker: str) -> str:
    """Sanitize and format stock ticker symbol."""
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Stock ticker symbol must be a non-empty string.")
    return ticker.strip().upper()


def get_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Fetch comprehensive company profile and current valuation metrics.
    
    Args:
        ticker: Stock symbol (e.g. 'RELIANCE.NS', 'AAPL')
        
    Returns:
        Dictionary containing company fundamentals and market statistics.
    """
    clean_ticker = validate_ticker_symbol(ticker)
    ticker_obj = yf.Ticker(clean_ticker)
    
    try:
        raw_info = ticker_obj.info or {}
    except Exception as e:
        logger.warning(f"Could not retrieve full info for {clean_ticker}: {e}")
        raw_info = {}

    # Extract price info safely with fallbacks
    curr_price = (
        raw_info.get("currentPrice")
        or raw_info.get("regularMarketPrice")
        or raw_info.get("previousClose")
        or 0.0
    )
    prev_close = raw_info.get("previousClose") or curr_price

    price_change = curr_price - prev_close if curr_price and prev_close else 0.0
    price_change_pct = (price_change / prev_close * 100) if prev_close else 0.0

    info_summary = {
        "symbol": clean_ticker,
        "company_name": raw_info.get("shortName") or raw_info.get("longName") or clean_ticker,
        "sector": raw_info.get("sector", "N/A"),
        "industry": raw_info.get("industry", "N/A"),
        "country": raw_info.get("country", "N/A"),
        "currency": raw_info.get("currency", "USD"),
        "exchange": raw_info.get("exchange", "N/A"),
        "current_price": float(curr_price) if curr_price else 0.0,
        "previous_close": float(prev_close) if prev_close else 0.0,
        "price_change": float(price_change),
        "price_change_pct": float(price_change_pct),
        "open": float(raw_info.get("open") or raw_info.get("regularMarketOpen") or 0.0),
        "day_high": float(raw_info.get("dayHigh") or raw_info.get("regularMarketDayHigh") or 0.0),
        "day_low": float(raw_info.get("dayLow") or raw_info.get("regularMarketDayLow") or 0.0),
        "fifty_two_week_high": float(raw_info.get("fiftyTwoWeekHigh") or 0.0),
        "fifty_two_week_low": float(raw_info.get("fiftyTwoWeekLow") or 0.0),
        "market_cap": raw_info.get("marketCap", 0),
        "pe_ratio": raw_info.get("trailingPE") or raw_info.get("forwardPE") or None,
        "dividend_yield": raw_info.get("dividendYield") or 0.0,
        "beta": raw_info.get("beta") or None,
        "business_summary": raw_info.get("longBusinessSummary", "No company summary available."),
        "website": raw_info.get("website", ""),
    }
    return info_summary


def fetch_historical_data(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d"
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[str]]:
    """
    Fetch OHLCV historical stock data from Yahoo Finance.
    
    Args:
        ticker: Stock symbol
        period: Time horizon ('1mo', '3mo', '6mo', '1y', '5y')
        interval: Granularity ('1d')
        
    Returns:
        Tuple of (DataFrame, info_dict, error_message)
    """
    try:
        clean_ticker = validate_ticker_symbol(ticker)
        ticker_obj = yf.Ticker(clean_ticker)
        
        # Download historical data including Adj Close
        df = ticker_obj.history(period=period, interval=interval, auto_adjust=False)
        
        if df is None or df.empty:
            return pd.DataFrame(), {}, f"No historical market data found for ticker '{clean_ticker}'. Please verify the symbol."
        
        # Verify essential columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame(), {}, f"Incomplete dataset for '{clean_ticker}': missing column '{col}'."
                
        # If Adj Close is missing, fill from Close
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        # Fetch metadata
        info = get_stock_info(clean_ticker)
        
        # Update current price from latest close if raw info had 0
        if info["current_price"] == 0.0 and not df.empty:
            latest_close = float(df["Close"].iloc[-1])
            info["current_price"] = latest_close
            if len(df) > 1:
                prev_c = float(df["Close"].iloc[-2])
                info["previous_close"] = prev_c
                info["price_change"] = latest_close - prev_c
                info["price_change_pct"] = (info["price_change"] / prev_c) * 100

        # Ensure 52w high/low fallbacks if missing
        if info["fifty_two_week_high"] == 0.0 and not df.empty:
            info["fifty_two_week_high"] = float(df["High"].max())
        if info["fifty_two_week_low"] == 0.0 and not df.empty:
            info["fifty_two_week_low"] = float(df["Low"].min())

        return df, info, None

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame(), {}, f"Data acquisition error: {str(e)}"


def fetch_stock_news(ticker: str, max_items: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve real-world company news using Yahoo Finance search and ticker feed.
    
    Args:
        ticker: Stock symbol
        max_items: Maximum articles to retrieve
        
    Returns:
        List of standardized news article dictionaries.
    """
    clean_ticker = validate_ticker_symbol(ticker)
    articles: List[Dict[str, Any]] = []

    try:
        # 1. Try yf.Search first (provides rich titles and links)
        try:
            search_obj = yf.Search(clean_ticker, news_count=max_items)
            if hasattr(search_obj, "news") and search_obj.news:
                for item in search_obj.news[:max_items]:
                    pub_time = item.get("providerPublishTime")
                    if pub_time:
                        pub_str = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M")
                    else:
                        pub_str = "Recent"

                    articles.append({
                        "headline": item.get("title", "Untitled News"),
                        "source": item.get("publisher", "Financial News"),
                        "publish_date": pub_str,
                        "url": item.get("link", "#"),
                        "description": item.get("summary") or item.get("description") or "Click link to read full article.",
                    })
        except Exception as se:
            logger.debug(f"yf.Search news lookup error for {clean_ticker}: {se}")

        # 2. Fallback to ticker.news if search returned nothing
        if not articles:
            ticker_obj = yf.Ticker(clean_ticker)
            raw_news = ticker_obj.news or []
            for item in raw_news[:max_items]:
                pub_time = item.get("providerPublishTime")
                pub_str = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M") if pub_time else "Recent"
                articles.append({
                    "headline": item.get("title", "Market Update"),
                    "source": item.get("publisher", "Market News"),
                    "publish_date": pub_str,
                    "url": item.get("link", "#"),
                    "description": item.get("summary") or "Real-time stock update.",
                })

    except Exception as e:
        logger.warning(f"Error retrieving news for {clean_ticker}: {e}")

    return articles


def fetch_multi_stocks(tickers: List[str], period: str = "6mo") -> Dict[str, pd.DataFrame]:
    """
    Fetch historical close prices for multiple stocks for comparison.
    
    Args:
        tickers: List of ticker symbols
        period: Time period
        
    Returns:
        Dictionary mapping ticker -> DataFrame of historical data
    """
    result = {}
    for ticker in tickers:
        df, _, err = fetch_historical_data(ticker, period=period)
        if err is None and not df.empty:
            result[ticker] = df
    return result

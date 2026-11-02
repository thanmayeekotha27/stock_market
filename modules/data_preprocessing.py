"""
Data Preprocessing Module
Implements statistical data cleaning, type sanitization, date normalization,
and basic return/rolling feature engineering following financial engineering principles.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def clean_and_validate_stock_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans, validates, and normalizes raw OHLCV stock market data.
    
    Processing Steps:
        1. Duplicate timestamp deduplication
        2. Timezone localization stripping for uniform indexing
        3. Sorting chronologically
        4. Numeric casting & validation (Open, High, Low, Close, Volume)
        5. Forward-fill and backward-fill missing price quotes
        6. Filtering anomalous records (e.g. High < Low or Close <= 0)
        7. Calculating daily returns and percentage changes
        8. Rolling volume features
        
    Args:
        df: Raw DataFrame from data collection
        
    Returns:
        Tuple of (clean_df, audit_metrics_dict)
    """
    if df.empty:
        return df, {"status": "Empty dataset", "rows": 0}

    clean_df = df.copy()

    # Step 1: Ensure DateTime Index is timezone-naive and sorted
    if isinstance(clean_df.index, pd.DatetimeIndex):
        if clean_df.index.tz is not None:
            clean_df.index = clean_df.index.tz_localize(None)
    else:
        clean_df.index = pd.to_datetime(clean_df.index).tz_localize(None)

    clean_df = clean_df.sort_index()

    # Step 2: Remove duplicate timestamps
    initial_rows = len(clean_df)
    clean_df = clean_df[~clean_df.index.duplicated(keep="last")]
    duplicates_removed = initial_rows - len(clean_df)

    # Step 3: Enforce strict numeric data types
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if "Adj Close" not in clean_df.columns and "Close" in clean_df.columns:
        clean_df["Adj Close"] = clean_df["Close"]

    present_numeric = [col for col in numeric_cols if col in clean_df.columns]
    for col in present_numeric:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Step 4: Handle Missing Values
    null_counts_before = int(clean_df[present_numeric].isnull().sum().sum())
    clean_df[present_numeric] = clean_df[present_numeric].ffill().bfill()
    clean_df = clean_df.dropna(subset=["Open", "High", "Low", "Close"])

    # Step 5: Data Sanity Constraints (Financial Validation)
    # Ensure High is at least max(Open, Close) and Low is at most min(Open, Close)
    clean_df["High"] = np.maximum(clean_df["High"], np.maximum(clean_df["Open"], clean_df["Close"]))
    clean_df["Low"] = np.minimum(clean_df["Low"], np.minimum(clean_df["Open"], clean_df["Close"]))
    clean_df["Volume"] = clean_df["Volume"].clip(lower=0)
    
    # Filter out impossible negative or zero prices
    clean_df = clean_df[clean_df["Close"] > 0]

    # Step 6: Return Calculations
    # Daily Return: (Close_t - Close_{t-1}) / Close_{t-1}
    clean_df["Daily_Return"] = clean_df["Close"].pct_change()
    clean_df["Daily_Pct_Change"] = clean_df["Daily_Return"] * 100.0

    # Log Returns: ln(Close_t / Close_{t-1}) for continuous compounding & volatility modeling
    clean_df["Log_Return"] = np.log(clean_df["Close"] / clean_df["Close"].shift(1))

    # Cumulative Returns: (1 + R).cumprod() - 1
    clean_df["Cumulative_Return"] = (1 + clean_df["Daily_Return"].fillna(0)).cumprod() - 1

    # Step 7: Rolling Averages for price smoothing (7-day and 20-day)
    clean_df["Rolling_Mean_7"] = clean_df["Close"].rolling(window=7, min_periods=1).mean()
    clean_df["Rolling_Std_7"] = clean_df["Close"].rolling(window=7, min_periods=1).std().fillna(0)

    # Step 8: Volume Analysis
    clean_df["Volume_SMA_20"] = clean_df["Volume"].rolling(window=20, min_periods=1).mean()
    clean_df["Volume_Ratio"] = np.where(
        clean_df["Volume_SMA_20"] > 0,
        clean_df["Volume"] / clean_df["Volume_SMA_20"],
        1.0
    )
    clean_df["Volume_Pct_Change"] = clean_df["Volume"].pct_change() * 100.0

    # Fill any remaining NaNs in initial rows
    clean_df["Daily_Return"] = clean_df["Daily_Return"].fillna(0.0)
    clean_df["Daily_Pct_Change"] = clean_df["Daily_Pct_Change"].fillna(0.0)
    clean_df["Log_Return"] = clean_df["Log_Return"].fillna(0.0)
    clean_df["Volume_Pct_Change"] = clean_df["Volume_Pct_Change"].fillna(0.0)

    # Compile data audit metrics
    audit_report = {
        "status": "Validated",
        "initial_records": initial_rows,
        "final_records": len(clean_df),
        "duplicates_dropped": duplicates_removed,
        "missing_imputed": null_counts_before,
        "date_start": clean_df.index[0].strftime("%Y-%m-%d") if not clean_df.empty else "N/A",
        "date_end": clean_df.index[-1].strftime("%Y-%m-%d") if not clean_df.empty else "N/A",
        "data_completeness_pct": 100.0 if not clean_df.empty else 0.0,
    }

    return clean_df, audit_report

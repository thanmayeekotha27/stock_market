"""
End-to-End Test Suite
Tests data acquisition, preprocessing, technical indicator computation,
OpenAI payload formatting, and PDF report compilation.
"""

import sys
import unittest
import pandas as pd
import numpy as np

import config
from modules.data_collection import (
    validate_ticker_symbol,
    get_stock_info,
    fetch_historical_data,
    fetch_stock_news,
    fetch_multi_stocks,
)
from modules.data_preprocessing import clean_and_validate_stock_data
from modules.stock_analysis import (
    compute_rsi,
    compute_macd,
    calculate_technical_indicators,
)
from modules.openai_analysis import (
    format_stock_context_payload,
    _generate_offline_academic_insights,
    _generate_offline_full_report,
)
from modules.report_generator import generate_pdf_report


class TestStockEnrichmentPipeline(unittest.TestCase):

    def test_01_ticker_validation(self):
        self.assertEqual(validate_ticker_symbol("  aapl  "), "AAPL")
        self.assertEqual(validate_ticker_symbol("reliance.ns"), "RELIANCE.NS")
        with self.assertRaises(ValueError):
            validate_ticker_symbol("")

    def test_02_synthetic_data_preprocessing(self):
        # Create synthetic OHLCV data
        dates = pd.date_range("2026-01-01", periods=60, freq="B")
        prices = 100 + np.cumsum(np.random.randn(60) * 1.5)
        raw_df = pd.DataFrame(
            {
                "Open": prices - 0.5,
                "High": prices + 1.5,
                "Low": prices - 1.5,
                "Close": prices,
                "Adj Close": prices,
                "Volume": np.random.randint(100000, 500000, size=60),
            },
            index=dates,
        )
        # Inject intentional duplicate and NaN
        raw_df.iloc[5, 0] = np.nan

        clean_df, audit = clean_and_validate_stock_data(raw_df)
        self.assertFalse(clean_df.empty)
        self.assertEqual(audit["status"], "Validated")
        self.assertIn("Daily_Return", clean_df.columns)
        self.assertIn("Daily_Pct_Change", clean_df.columns)
        self.assertIn("Log_Return", clean_df.columns)
        self.assertIn("Cumulative_Return", clean_df.columns)

    def test_03_technical_indicators(self):
        dates = pd.date_range("2026-01-01", periods=100, freq="B")
        prices = 150 + np.cumsum(np.random.randn(100) * 2.0)
        df = pd.DataFrame(
            {
                "Open": prices - 1.0,
                "High": prices + 2.0,
                "Low": prices - 2.0,
                "Close": prices,
                "Volume": 200000 + np.random.randint(0, 100000, size=100),
            },
            index=dates,
        )
        clean_df, _ = clean_and_validate_stock_data(df)
        enriched_df, summary = calculate_technical_indicators(clean_df)

        self.assertIn("SMA_20", enriched_df.columns)
        self.assertIn("SMA_50", enriched_df.columns)
        self.assertIn("EMA_20", enriched_df.columns)
        self.assertIn("RSI_14", enriched_df.columns)
        self.assertIn("MACD_Line", enriched_df.columns)
        self.assertIn("MACD_Signal", enriched_df.columns)
        self.assertIn("MACD_Hist", enriched_df.columns)

        self.assertGreater(summary["latest_close"], 0)
        self.assertTrue(0 <= summary["rsi_14"] <= 100)
        self.assertIn(summary["trend_sentiment"], ["Bullish", "Mild Bullish", "Bearish", "Mild Bearish", "Neutral"])

    def test_04_ai_payload_and_offline_generation(self):
        mock_info = {
            "symbol": "RELIANCE.NS",
            "company_name": "Reliance Industries Ltd",
            "sector": "Energy",
            "industry": "Oil & Gas Refining",
            "currency": "INR",
            "fifty_two_week_high": 3000.0,
            "fifty_two_week_low": 2200.0,
            "market_cap": 18000000000000,
        }
        mock_tech = {
            "latest_close": 2850.0,
            "daily_pct_change": 1.25,
            "period_open": 2700.0,
            "period_high": 2900.0,
            "period_low": 2650.0,
            "total_period_return": 5.56,
            "annualized_volatility_pct": 18.5,
            "sma_20": 2810.0,
            "sma_50": 2780.0,
            "sma_200": 2650.0,
            "has_sma_200": True,
            "ema_20": 2820.0,
            "rsi_14": 62.4,
            "rsi_signal": "Bullish Momentum",
            "macd_line": 15.2,
            "macd_signal_line": 12.0,
            "macd_histogram": 3.2,
            "macd_signal": "Bullish Crossover",
            "moving_average_alignment": "Bullish Stack",
            "trend_sentiment": "Bullish",
            "cross_event": "Golden Cross Territory",
            "latest_volume": 4500000,
            "volume_sma_20": 4000000,
            "volume_ratio": 1.125,
            "volume_signal": "Normal Volume",
            "total_trading_days": 125,
        }
        mock_news = [
            {
                "headline": "Reliance expands green energy commitments",
                "source": "Financial Express",
                "publish_date": "2026-11-01",
                "url": "https://example.com",
                "description": "Reliance announces capital investment in new gigafactories.",
            }
        ]

        payload = format_stock_context_payload(mock_info, mock_tech, mock_news)
        self.assertIn("RELIANCE.NS", payload)
        self.assertIn("2850.00", payload)
        self.assertIn("RSI", payload)

        insights = _generate_offline_academic_insights(mock_info, mock_tech, mock_news)
        self.assertIn("stock_summary", insights)
        self.assertIn("trend_analysis", insights)
        self.assertIn("technical_analysis", insights)
        self.assertIn("risk_analysis", insights)
        self.assertIn("market_insights", insights)
        self.assertIn("beginner_friendly", insights)

        report = _generate_offline_full_report(mock_info, mock_tech, mock_news)
        self.assertIn("Executive Equity Research Report", report)
        self.assertIn("DISCLAIMER", report)

    def test_05_pdf_report_compilation(self):
        sample_md = """# Executive Equity Research Report: Apple Inc. (AAPL)
## 1. Company Overview
Apple Inc. designs, manufactures, and markets smartphones, personal computers, and services.
## 2. Performance Summary
The stock has appreciated by 12.4% over the past 6 months.
- RSI: 58.2
- MACD: Bullish
## 8. Academic & Regulatory Disclaimer
DISCLAIMER: Developed for quantitative financial analysis demonstration. Not financial advice.
"""
        pdf_bytes = generate_pdf_report(sample_md, "AAPL", "Apple Inc.")
        self.assertTrue(len(pdf_bytes) > 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()

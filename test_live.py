"""
Live Integration Test
Tests real network fetching from Yahoo Finance, pipeline processing, and PDF export.
"""

import sys
import config
from modules.data_collection import fetch_historical_data, fetch_stock_news
from modules.data_preprocessing import clean_and_validate_stock_data
from modules.stock_analysis import calculate_technical_indicators
from modules.openai_analysis import _generate_offline_academic_insights, _generate_offline_full_report
from modules.report_generator import generate_pdf_report

def main():
    print("Testing live fetch for RELIANCE.NS...")
    df, info, err = fetch_historical_data("RELIANCE.NS", period="3mo")
    if err or df.empty:
        print(f"Error fetching: {err}")
        sys.exit(1)

    print(f"Success! Fetched {len(df)} rows for {info.get('company_name')}. Current price: {info.get('current_price')}")

    clean_df, audit = clean_and_validate_stock_data(df)
    enriched_df, tech = calculate_technical_indicators(clean_df)
    print(f"Calculated: SMA20={tech['sma_20']:.2f}, RSI={tech['rsi_14']:.2f}, MACD={tech['macd_line']:.4f}")

    news = fetch_stock_news("RELIANCE.NS", max_items=3)
    print(f"News retrieved: {len(news)} items")

    print("Testing US stock AAPL...")
    df_us, info_us, err_us = fetch_historical_data("AAPL", period="1mo")
    if err_us or df_us.empty:
        print(f"Error fetching AAPL: {err_us}")
        sys.exit(1)
    print(f"Success! Fetched {len(df_us)} rows for {info_us.get('company_name')}. Price: {info_us.get('current_price')}")

    print("Generating sample PDF report...")
    report_md = _generate_offline_full_report(info, tech, news)
    pdf_bytes = generate_pdf_report(report_md, "RELIANCE.NS", info["company_name"])
    print(f"PDF successfully compiled: {len(pdf_bytes):,} bytes")

    print("\n>>> ALL SYSTEM INTEGRATION TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    main()

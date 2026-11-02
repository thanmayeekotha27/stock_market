"""
Configuration module for the Stock Market Enrichment System.
Manages application constants, ticker universes, OpenAI settings, and environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Project Metadata
PROJECT_TITLE = "Enriching Stock Market Data using the OpenAI API"
PROJECT_SUBTITLE = "Financial Data Science & Generative AI Intelligence Engine"
PROJECT_VERSION = "2.0.0"
AUTHOR_LABEL = "Financial Data Science & AI"

# Stock Ticker Universes
INDIAN_STOCKS = {
    "RELIANCE.NS": "Reliance Industries Ltd",
    "TCS.NS": "Tata Consultancy Services Ltd",
    "INFY.NS": "Infosys Ltd",
    "HDFCBANK.NS": "HDFC Bank Ltd",
    "ICICIBANK.NS": "ICICI Bank Ltd",
    "SBIN.NS": "State Bank of India",
    "TATAMOTORS.NS": "Tata Motors Ltd",
    "ITC.NS": "ITC Limited",
}

US_STOCKS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. (Google)",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corporation",
}

TIMEFRAME_OPTIONS = {
    "1mo": "1 Month",
    "3mo": "3 Months",
    "6mo": "6 Months",
    "1y": "1 Year",
    "5y": "5 Years",
}

DEFAULT_TIMEFRAME = "6mo"
DEFAULT_INDIAN_TICKER = "RELIANCE.NS"
DEFAULT_US_TICKER = "AAPL"

# OpenAI Configuration
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SUPPORTED_OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-3.5-turbo",
]

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Technical Indicator Settings
SMA_FAST = 20
SMA_MEDIUM = 50
SMA_SLOW = 200
EMA_FAST = 20
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ANNUALIZED_TRADING_DAYS = 252

# Mandatory Financial Disclaimer
FINANCIAL_DISCLAIMER = (
    "DISCLAIMER: This application is an educational and financial data science analytical tool. "
    "All technical indicators, metrics, and AI-generated analyses are strictly for informational and analytical demonstration purposes. "
    "They DO NOT constitute financial, investment, legal, or tax advice. Stock market trading "
    "involves substantial risk of capital loss. Historical performance does not guarantee future results."
)

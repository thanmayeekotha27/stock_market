"""
Visualization Module
Generates high-performance, dark-themed interactive financial charts
using Plotly for stock prices, technical indicators, and multi-asset comparisons.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Fintech Palette Constants
COLOR_BG = "#0B0E14"
COLOR_PAPER = "#11151F"
COLOR_GRID = "#1E2433"
COLOR_TEXT = "#E2E8F0"
COLOR_MUTED = "#718096"
COLOR_BULLISH = "#00E676"  # Vibrant Emerald Green
COLOR_BEARISH = "#FF5252"  # Vibrant Crimson Red
COLOR_SMA20 = "#00D2FF"    # Cyan
COLOR_SMA50 = "#FFA000"    # Amber Gold
COLOR_SMA200 = "#E040FB"   # Neon Magenta
COLOR_EMA20 = "#B388FF"    # Light Violet
COLOR_RSI = "#8C9EFF"      # Indigo
COLOR_MACD = "#00E5FF"     # Neon Cyan
COLOR_SIGNAL = "#FFAB40"   # Tangerine


def apply_fintech_layout(
    fig: go.Figure,
    title: str = "",
    height: int = 500,
    show_legend: bool = True
) -> go.Figure:
    """Applies high-end dark fintech terminal styling to Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.96,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": COLOR_TEXT, "family": "Inter, sans-serif"},
        },
        template="plotly_dark",
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_BG,
        font={"color": COLOR_TEXT, "family": "Inter, sans-serif", "size": 11},
        height=height,
        margin={"l": 50, "r": 40, "t": 60, "b": 40},
        showlegend=show_legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 11, "color": COLOR_TEXT},
            "bgcolor": "rgba(17, 21, 31, 0.7)",
        },
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#1A202C",
            "font_size": 12,
            "font_family": "Inter, sans-serif",
            "font_color": "#FFFFFF",
        },
    )
    fig.update_xaxes(
        gridcolor=COLOR_GRID,
        gridwidth=1,
        zerolinecolor=COLOR_GRID,
        showline=True,
        linecolor=COLOR_GRID,
    )
    fig.update_yaxes(
        gridcolor=COLOR_GRID,
        gridwidth=1,
        zerolinecolor=COLOR_GRID,
        showline=True,
        linecolor=COLOR_GRID,
    )
    return fig


def plot_price_and_volume(
    df: pd.DataFrame,
    ticker: str,
    company_name: str,
    chart_type: str = "Candlestick",
    show_sma_20: bool = True,
    show_sma_50: bool = True,
    show_sma_200: bool = True,
    show_ema_20: bool = False,
) -> go.Figure:
    """
    Creates an interactive dual-panel chart:
    Panel 1: Candlestick/Line Price action with Moving Average overlays
    Panel 2: Color-coded Trading Volume bars
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{company_name} ({ticker}) — Price Action", "Trading Volume"),
    )

    # 1. Main Price Plot
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC Price",
                increasing_line_color=COLOR_BULLISH,
                decreasing_line_color=COLOR_BEARISH,
                increasing_fillcolor="rgba(0, 230, 118, 0.3)",
                decreasing_fillcolor="rgba(255, 82, 82, 0.3)",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                name="Close Price",
                mode="lines",
                line=dict(color="#38BDF8", width=2.2),
            ),
            row=1,
            col=1,
        )

    # Moving Average Overlays
    if show_sma_20 and "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_20"],
                name="SMA 20",
                mode="lines",
                line=dict(color=COLOR_SMA20, width=1.5),
            ),
            row=1,
            col=1,
        )

    if show_sma_50 and "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_50"],
                name="SMA 50",
                mode="lines",
                line=dict(color=COLOR_SMA50, width=1.6),
            ),
            row=1,
            col=1,
        )

    if show_sma_200 and "SMA_200" in df.columns and not df["SMA_200"].dropna().empty:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_200"],
                name="SMA 200",
                mode="lines",
                line=dict(color=COLOR_SMA200, width=1.8),
            ),
            row=1,
            col=1,
        )

    if show_ema_20 and "EMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_20"],
                name="EMA 20",
                mode="lines",
                line=dict(color=COLOR_EMA20, width=1.5, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # 2. Volume Panel (Color-coded: Bullish vs Bearish days)
    colors_vol = np.where(df["Close"] >= df["Open"], "rgba(0, 230, 118, 0.65)", "rgba(255, 82, 82, 0.65)")
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=colors_vol,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    if "Volume_SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Volume_SMA_20"],
                name="Volume 20 SMA",
                mode="lines",
                line=dict(color="#FFD700", width=1.2),
            ),
            row=2,
            col=1,
        )

    # Range Selector and slider
    fig.update_xaxes(
        row=1,
        col=1,
        rangeslider_visible=False,
    )
    fig.update_xaxes(
        row=2,
        col=1,
        rangeslider_visible=True,
        rangeslider_thickness=0.06,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
            bgcolor="#1A202C",
            activecolor="#2B6CB0",
            font=dict(color=COLOR_TEXT, size=10),
        ),
    )

    fig = apply_fintech_layout(fig, title=f"Price & Volume Dynamics | {ticker}", height=620)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def plot_rsi_indicator(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Plots RSI 14 with Overbought (70) and Oversold (30) threshold bands."""
    fig = go.Figure()

    if "RSI_14" not in df.columns:
        return fig

    # Reference bands
    fig.add_hrect(
        y0=70,
        y1=100,
        fillcolor="rgba(255, 82, 82, 0.12)",
        line_width=0,
        annotation_text="Overbought Zone (>70)",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color=COLOR_BEARISH,
    )
    fig.add_hrect(
        y0=0,
        y1=30,
        fillcolor="rgba(0, 230, 118, 0.12)",
        line_width=0,
        annotation_text="Oversold Zone (<30)",
        annotation_position="bottom left",
        annotation_font_size=10,
        annotation_font_color=COLOR_BULLISH,
    )

    # 70, 50, 30 lines
    fig.add_hline(y=70, line=dict(color=COLOR_BEARISH, width=1, dash="dash"))
    fig.add_hline(y=50, line=dict(color=COLOR_MUTED, width=1, dash="dot"))
    fig.add_hline(y=30, line=dict(color=COLOR_BULLISH, width=1, dash="dash"))

    # RSI Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI_14"],
            name="RSI 14",
            mode="lines",
            line=dict(color=COLOR_RSI, width=2),
        )
    )

    fig = apply_fintech_layout(fig, title=f"Relative Strength Index (RSI 14) | {ticker}", height=320)
    fig.update_yaxes(range=[10, 90], title_text="RSI Value")
    return fig


def plot_macd_indicator(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Plots MACD Line, Signal Line, and Momentum Histogram."""
    fig = go.Figure()

    if "MACD_Line" not in df.columns:
        return fig

    # Histogram
    hist_colors = np.where(df["MACD_Hist"] >= 0, COLOR_BULLISH, COLOR_BEARISH)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["MACD_Hist"],
            name="Histogram",
            marker_color=hist_colors,
            opacity=0.7,
        )
    )

    # MACD Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD_Line"],
            name="MACD (12, 26)",
            mode="lines",
            line=dict(color=COLOR_MACD, width=1.8),
        )
    )

    # Signal Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD_Signal"],
            name="Signal (9)",
            mode="lines",
            line=dict(color=COLOR_SIGNAL, width=1.5, dash="dash"),
        )
    )

    fig.add_hline(y=0, line=dict(color=COLOR_MUTED, width=1))

    fig = apply_fintech_layout(fig, title=f"MACD Convergence / Divergence | {ticker}", height=320)
    fig.update_yaxes(title_text="MACD Value")
    return fig


def plot_stock_comparison(
    dfs: Dict[str, pd.DataFrame],
    labels: Dict[str, str]
) -> go.Figure:
    """
    Plots normalized percentage returns for comparative stock benchmarking.
    Formula: Normalized_Return_t = ((Close_t - Close_0) / Close_0) * 100
    """
    fig = go.Figure()
    palette = ["#00E5FF", "#FF9100", "#00E676", "#E040FB", "#FF5252"]

    for i, (ticker, df) in enumerate(dfs.items()):
        if df.empty or "Close" not in df.columns:
            continue
        first_close = df["Close"].iloc[0]
        if first_close <= 0:
            continue
        norm_return = ((df["Close"] - first_close) / first_close) * 100.0
        label_text = labels.get(ticker, ticker)
        color = palette[i % len(palette)]

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=norm_return,
                name=f"{label_text} ({norm_return.iloc[-1]:+.2f}%)",
                mode="lines",
                line=dict(color=color, width=2.2),
            )
        )

    fig.add_hline(y=0, line=dict(color=COLOR_MUTED, width=1, dash="dash"))
    fig = apply_fintech_layout(
        fig,
        title="Comparative Historical Performance (Normalized Cumulative Returns %)",
        height=450
    )
    fig.update_yaxes(title_text="Cumulative Return (%)")
    return fig


def plot_historical_volatility(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Plots 21-day rolling annualized volatility curve."""
    fig = go.Figure()
    if "Historical_Vol_21d" not in df.columns:
        return fig

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Historical_Vol_21d"],
            name="21-Day Annualized Volatility",
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(140, 158, 255, 0.12)",
            line=dict(color="#A78BFA", width=1.8),
        )
    )

    fig = apply_fintech_layout(fig, title=f"Rolling Annualized Volatility (21-Day) | {ticker}", height=300)
    fig.update_yaxes(title_text="Volatility (%)")
    return fig

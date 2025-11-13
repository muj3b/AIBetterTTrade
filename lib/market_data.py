import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import requests


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    symbol: str
    interval: str
    latest_close: float
    change_24h: float
    change_4h: float
    momentum_1h: float
    rsi: float
    sma_fast: float
    sma_slow: float
    atr_pct: float
    volume_24h: float
    volatility_24h: float


@dataclass(frozen=True, slots=True)
class TechnicalSignal:
    signal: Literal["Bullish", "Bearish", "Neutral"]
    confidence: float
    rationale: str


def fetch_klines(symbol: str, interval: str = "15m", limit: int = 200) -> pd.DataFrame:
    """
    Fetch recent kline/candlestick data from Binance.

    Binance is used here for reliability and speed even if execution happens elsewhere.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if not data:
        raise ValueError(f"No kline data returned for {symbol}")

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def build_snapshot(symbol: str, interval: str, klines: pd.DataFrame) -> MarketSnapshot:
    """Compute summary statistics and indicators from candlestick data."""
    closes = klines["close"]
    highs = klines["high"]
    lows = klines["low"]
    volumes = klines["volume"]

    change_24h = _period_pct_change(closes, bars=int((24 * 60) / _interval_to_minutes(interval)))
    change_4h = _period_pct_change(closes, bars=int((4 * 60) / _interval_to_minutes(interval)))
    momentum_1h = _period_pct_change(closes, bars=max(1, int(60 / _interval_to_minutes(interval))))
    rsi = _calculate_rsi(closes, period=14)
    sma_fast = _safe_value(closes.rolling(window=20).mean().iloc[-1], closes.iloc[-1])
    sma_slow = _safe_value(closes.rolling(window=60).mean().iloc[-1], closes.iloc[-1])
    atr_value = _calculate_atr(highs, lows, closes, period=14)
    atr_pct = (atr_value / closes.iloc[-1] * 100) if closes.iloc[-1] else 0.0
    volatility_24h = closes.pct_change().rolling(window=int((24 * 60) / _interval_to_minutes(interval))).std().iloc[-1]
    volatility_24h = _safe_value(volatility_24h, 0.0)
    volume_24h = volumes.iloc[-int((24 * 60) / _interval_to_minutes(interval)):].sum()

    snapshot = MarketSnapshot(
        symbol=symbol,
        interval=interval,
        latest_close=closes.iloc[-1],
        change_24h=change_24h,
        change_4h=change_4h,
        momentum_1h=momentum_1h,
        rsi=rsi,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        atr_pct=atr_pct,
        volume_24h=volume_24h,
        volatility_24h=volatility_24h,
    )
    return snapshot


def derive_signal(snapshot: MarketSnapshot) -> TechnicalSignal:
    """
    Build a lightweight deterministic signal from the indicators.

    The goal is not to be perfect but to provide a sanity check for the LLM.
    """
    score = 0.0
    rationale_parts: list[str] = []

    trend_bias = 0.25 if snapshot.latest_close > snapshot.sma_slow else -0.25
    score += trend_bias
    rationale_parts.append(
        f"Price {'above' if trend_bias > 0 else 'below'} long SMA ({snapshot.latest_close:.0f} vs {snapshot.sma_slow:.0f})"
    )

    ma_cross_bias = 0.2 if snapshot.sma_fast > snapshot.sma_slow else -0.2
    score += ma_cross_bias
    rationale_parts.append("SMA20" + (">" if ma_cross_bias > 0 else "<") + "SMA60")

    if snapshot.rsi >= 55:
        rsi_bias = min(0.2, (snapshot.rsi - 55) / 100)
        score += rsi_bias
        rationale_parts.append(f"RSI strong ({snapshot.rsi:.1f})")
    elif snapshot.rsi <= 45:
        rsi_bias = -min(0.2, (45 - snapshot.rsi) / 100)
        score += rsi_bias
        rationale_parts.append(f"RSI weak ({snapshot.rsi:.1f})")

    if snapshot.momentum_1h >= 0.002:
        score += 0.15
        rationale_parts.append(f"1h momentum +{snapshot.momentum_1h * 100:.2f}%")
    elif snapshot.momentum_1h <= -0.002:
        score -= 0.15
        rationale_parts.append(f"1h momentum {snapshot.momentum_1h * 100:.2f}%")

    if snapshot.change_24h >= 0.005:
        score += 0.1
    elif snapshot.change_24h <= -0.005:
        score -= 0.1

    signal: Literal["Bullish", "Bearish", "Neutral"]
    if score > 0.15:
        signal = "Bullish"
    elif score < -0.15:
        signal = "Bearish"
    else:
        signal = "Neutral"

    confidence = max(0.45, min(0.95, 0.55 + abs(score)))
    rationale = "; ".join(rationale_parts)

    return TechnicalSignal(signal=signal, confidence=confidence, rationale=rationale)


def format_market_context(snapshot: MarketSnapshot, signal: TechnicalSignal) -> str:
    """Compact textual context fed to the LLM."""
    base_asset = snapshot.symbol.replace("USDT", "")
    context = (
        f"Symbol: {snapshot.symbol} | Interval: {snapshot.interval}\n"
        f"Last close: {snapshot.latest_close:.2f} USDT | 24h change: {snapshot.change_24h * 100:.2f}% | "
        f"4h change: {snapshot.change_4h * 100:.2f}% | 1h momentum: {snapshot.momentum_1h * 100:.2f}%\n"
        f"SMA20/SMA60: {snapshot.sma_fast:.2f}/{snapshot.sma_slow:.2f} | RSI-14: {snapshot.rsi:.1f} | "
        f"ATR%: {snapshot.atr_pct:.2f}% | 24h volatility σ: {snapshot.volatility_24h * 100:.2f}%\n"
        f"24h volume: {snapshot.volume_24h:.2f} {base_asset}\n"
        f"Deterministic guardrail: {signal.signal} (confidence {signal.confidence:.0%}) — {signal.rationale}"
    )
    return context


def _period_pct_change(series: pd.Series, bars: int) -> float:
    if len(series) <= bars or bars <= 0:
        return 0.0
    latest = series.iloc[-1]
    base = series.iloc[-bars - 1]
    if base == 0:
        return 0.0
    return (latest - base) / base


def _interval_to_minutes(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    multipliers = {"m": 1, "h": 60, "d": 60 * 24}
    if unit not in multipliers:
        raise ValueError(f"Unsupported interval {interval}")
    return value * multipliers[unit]


def _calculate_rsi(series: pd.Series, period: int) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    value = rsi.iloc[-1]
    return _safe_value(value, 50.0)


def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    value = atr.iloc[-1]
    return _safe_value(value, 0.0)


def _safe_value(value: float, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except Exception:
        pass
    if isinstance(value, float) and math.isnan(value):
        return fallback
    return float(value)

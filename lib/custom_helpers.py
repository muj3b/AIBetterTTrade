import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal, Optional


def get_timestamp() -> str:
    """Get current UTC timestamp in simple format: YYYY-MM-DD HH:MM:SS"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def configure_logger(run_name: str) -> None:
    """Configure logging to file in logs/ directory."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"{run_name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def calculate_stop_loss_price(entry_price: float, side: str, sl_percent: float) -> float:
    """
    Calculate stop-loss price based on entry price and percentage.

    Args:
        entry_price: Entry price of the position
        side: Position side - "BUY" (long) or "SELL" (short)
        sl_percent: Stop-loss percentage (e.g., 2.0 for 2%)

    Returns:
        Stop-loss trigger price
    """
    if side == "BUY":
        return entry_price * (1 - sl_percent / 100)
    else:
        return entry_price * (1 + sl_percent / 100)


def calculate_position_size(exchange, symbol: str, position_size: str | float | int) -> float:
    """
    Calculate position size in base currency.

    Supports two modes:
    1. Percentage string: "10%" → 10% of available capital
    2. Float/int: 100 → Fixed cost of 100 USDT

    Args:
        exchange: Exchange client (BitunixFutures or ForwardTester)
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        position_size: Position size specification (percentage string or fixed cost in USDT)

    Returns:
        Position size in base currency (e.g., BTC quantity)
    """
    capital = exchange.get_account_balance("USDT")
    current_price = exchange.get_current_price(symbol)

    if isinstance(position_size, str):
        if position_size.endswith("%"):
            try:
                percentage = float(position_size.rstrip("%"))
                if not 0 < percentage <= 100:
                    raise ValueError(f"Percentage must be between 0 and 100, got {percentage}%")
                fraction = percentage / 100
                capital_to_use = capital * fraction
            except ValueError as e:
                raise ValueError(f"Invalid percentage format '{position_size}': {e}")
        else:
            raise ValueError(f"String position_size must end with '%', got '{position_size}'")

    elif isinstance(position_size, (int, float)):
        if position_size <= 0:
            raise ValueError(f"Position size must be positive, got {position_size}")
        if position_size > capital:
            raise ValueError(f"Fixed amount {position_size} USDT exceeds available capital {capital:.2f} USDT")
        capital_to_use = position_size
    else:
        raise TypeError(f"position_size must be str, int, or float, got {type(position_size)}")

    qty = capital_to_use / current_price

    return qty


def open_position(
        exchange,
        symbol: str,
        direction: str,
        position_size: str | float | int,
        stop_loss_percent: float | None = None,
        **kwargs
) -> dict[str, str]:
    """
    Open a market position (buy or sell).

    This helper function handles:
    - Position sizing (percentage or fixed amount)
    - Stop-loss calculation (percentage-based)
    - Calling the exchange's place_order() method with computed values
    - Attaching position-level stop-loss (if supported by exchange)

    Works with both BitunixFutures and ForwardTester exchange clients.

    Args:
        exchange: Exchange client (BitunixFutures or ForwardTester)
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        direction: "buy" or "sell"
        position_size: Position size ("10%" or fixed amount like 100)
        stop_loss_percent: Optional stop-loss percentage (e.g., 2.0 for 2%)
        **kwargs: Additional parameters passed to exchange.place_order()

    Returns:
        Order response from exchange
    """
    side = direction.upper()

    qty = calculate_position_size(exchange, symbol, position_size)
    logging.info(f"Position size: {qty:.6f} {symbol.replace('USDT', '')}")

    order_response = exchange.place_order(
        symbol=symbol,
        qty=qty,
        side=side,
        trade_side="OPEN",
        order_type="MARKET",
        **kwargs
    )

    if stop_loss_percent is not None and hasattr(exchange, 'place_position_tpsl'):
        try:
            position = exchange.get_pending_positions(symbol=symbol)
            if position:
                entry_price = float(position.avgOpenPrice)
                sl_price = calculate_stop_loss_price(entry_price, side, stop_loss_percent)
                logging.info(f"Stop-loss: {sl_price:.2f} USDT ({stop_loss_percent}% from entry {entry_price:.2f})")

                exchange.place_position_tpsl(
                    symbol=symbol,
                    position_id=position.positionId,
                    sl_price=sl_price
                )
                logging.info("Position stop-loss attached successfully")
            else:
                logging.warning("Could not attach stop-loss: position not found")
        except Exception as e:
            logging.warning(f"Failed to attach position stop-loss: {e}")
    elif stop_loss_percent is not None:
        logging.info("Stop-loss not supported for this exchange (forward testing mode)")

    return order_response


def normalize_position_side(raw_side: Optional[str]) -> Optional[str]:
    """Map exchange-specific side labels to 'buy'/'sell' used internally."""
    if not raw_side:
        return None
    side = raw_side.strip().upper()
    mapping = {
        "BUY": "buy",
        "BID": "buy",
        "LONG": "buy",
        "SELL": "sell",
        "ASK": "sell",
        "SHORT": "sell",
    }
    for key, value in mapping.items():
        if side.startswith(key):
            return value
    logging.warning(f"Unrecognized position side '{raw_side}', treating as None")
    return None


def combine_signals(
        ai_signal: str,
        technical_signal: Optional[str],
        technical_confidence: Optional[float]
) -> Literal["Bullish", "Bearish", "Neutral"]:
    """
    Blend AI and deterministic signals.

    The AI remains the primary driver, but high-confidence technical checks
    can break ties or neuter conflicting bets.
    """
    normalized_ai = _normalize_signal(ai_signal)
    normalized_tech = _normalize_signal(technical_signal) if technical_signal else None

    if not normalized_tech or technical_confidence is None:
        return normalized_ai

    if normalized_ai == normalized_tech:
        return normalized_ai

    if normalized_ai == "Neutral" and technical_confidence >= 0.55:
        logging.info(
            "AI neutral but guardrail %.0f%% confident in %s, following guardrail",
            technical_confidence * 100,
            normalized_tech
        )
        return normalized_tech

    if normalized_tech == "Neutral":
        return normalized_ai

    if technical_confidence >= 0.75:
        logging.info(
            "Signal disagreement (AI %s vs technical %s %.0f%%) -> flattening to Neutral",
            normalized_ai,
            normalized_tech,
            technical_confidence * 100
        )
        return "Neutral"

    return normalized_ai


def _normalize_signal(signal: Optional[str]) -> Literal["Bullish", "Bearish", "Neutral"]:
    if not signal:
        return "Neutral"
    cleaned = signal.strip().capitalize()
    if cleaned not in {"Bullish", "Bearish", "Neutral"}:
        logging.warning("Unexpected signal '%s', defaulting to Neutral", signal)
        return "Neutral"
    return cleaned

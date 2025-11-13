import os
import logging
from dotenv import load_dotenv

from lib import ai, custom_helpers, market_data, ForwardTester, BitunixFutures, BitunixError

load_dotenv()

# ===================== CONFIGURATION =====================
RUN_NAME = "run_btc_template_prompt"
CRYPTO = "Bitcoin"
SYMBOL = "BTCUSDT"
LEVERAGE = 1
MARGIN_MODE = "ISOLATION"

# Position Size Configuration
# POSITION_SIZE = "10%"  # Use 10% of capital per trade
POSITION_SIZE = 20  # Use 20 USDT per trade

# Stop Loss Configuration (LIVE TRADING ONLY - not supported in forward testing yet)
STOP_LOSS_PERCENT = 10  # 10% stop-loss from entry price
# STOP_LOSS_PERCENT = None  # Disabled

# Forward Testing Configuration
FORWARD_TESTING_CONFIG = None
# FORWARD_TESTING_CONFIG = {
#     "run_name": RUN_NAME,
#     "initial_capital": 10000,
#     "fees": 0.0006,  # 0.06% taker fee
# }

MARKET_DATA_INTERVAL = "15m"


def _scale_position_size_spec(position_size: str | int | float, scale: float) -> str | float:
    """Scale the configured position size specification by a factor."""
    if scale == 1.0:
        return position_size
    if isinstance(position_size, str) and position_size.endswith("%"):
        try:
            numeric = float(position_size.rstrip("%"))
            adjusted = max(0.1, numeric * scale)
            return f"{adjusted:.2f}%"
        except ValueError:
            logging.warning("Invalid percentage format '%s', skipping scaling", position_size)
            return position_size
    if isinstance(position_size, (int, float)):
        adjusted = max(1e-8, float(position_size) * scale)
        return adjusted
    return position_size


PROMPT_TEMPLATE = """
You are a cryptocurrency market analyst AI.

You are helping a systematic trader that executes {crypto} futures trades once per day.

Use both the structured market context below (which contains live indicators) and your wider knowledge of macro/crypto flows to recommend an outlook for the next 24 hours (Bullish, Bearish, Neutral).

Explain only the highest-signal factors in 2 short paragraphs (~120 words total). Avoid repeating the provided stats verbatim—interpret them.

Market context:
{market_context}

Always return your answer by calling the supplied function with the outlook and your reasoning.
""".strip()

# ===================== PREP =====================
LLM_API_KEY = os.environ.get("LLM_API_KEY")
EXCHANGE_API_KEY = os.environ.get("EXCHANGE_API_KEY")
EXCHANGE_API_SECRET = os.environ.get("EXCHANGE_API_SECRET")

if not LLM_API_KEY:
    raise RuntimeError("Missing LLM_API_KEY environment variable")

if FORWARD_TESTING_CONFIG is None and (not EXCHANGE_API_KEY or not EXCHANGE_API_SECRET):
    raise RuntimeError("Live trading requires both EXCHANGE_API_KEY and EXCHANGE_API_SECRET")

# ===================== MAIN EXECUTION =====================
custom_helpers.configure_logger(RUN_NAME)
logging.info("=== Run Started ===")

market_context = "Real-time market context unavailable; fall back to general knowledge."
technical_guardrail = None
position_size_spec = POSITION_SIZE

try:
    klines = market_data.fetch_klines(symbol=SYMBOL, interval=MARKET_DATA_INTERVAL, limit=200)
    snapshot = market_data.build_snapshot(symbol=SYMBOL, interval=MARKET_DATA_INTERVAL, klines=klines)
    technical_guardrail = market_data.derive_signal(snapshot)
    market_context = market_data.format_market_context(snapshot, technical_guardrail)
    volatility_scale = 1.0
    if snapshot.atr_pct >= 4:
        volatility_scale = 0.5
    elif snapshot.atr_pct >= 3:
        volatility_scale = 0.75
    elif snapshot.atr_pct <= 1:
        volatility_scale = 1.1
    position_size_spec = _scale_position_size_spec(POSITION_SIZE, volatility_scale)
    if volatility_scale != 1.0:
        logging.info(
            "ATR %.2f%% -> scaling position size (x%.2f) to %s",
            snapshot.atr_pct,
            volatility_scale,
            position_size_spec,
        )
    logging.info(
        "Market context ready. Guardrail signal %s (%.0f%% confidence)",
        technical_guardrail.signal,
        technical_guardrail.confidence * 100,
    )
except Exception as e:
    logging.warning(f"Failed to build market context: {e}")

# Build AI prompt with latest context
prompt = PROMPT_TEMPLATE.format(crypto=CRYPTO, market_context=market_context)
logging.info(f"Market context snapshot:\n{market_context}")

# Initialize exchange client (real or forward testing)
if FORWARD_TESTING_CONFIG is not None:
    exchange = ForwardTester(FORWARD_TESTING_CONFIG)
    logging.info("Forward testing mode enabled")
else:
    exchange = BitunixFutures(EXCHANGE_API_KEY, EXCHANGE_API_SECRET)
    logging.info("Live trading mode enabled")

#  Call AI to get interpretation
try:
    outlook = ai.send_request(prompt, CRYPTO, LLM_API_KEY)
    interpretation = outlook.interpretation
    logging.info(f"AI Interpretation: {interpretation}")
except (ai.AIResponseError, Exception) as e:
    logging.warning(f"AI request failed, defaulting to Neutral: {e}")
    interpretation = "Neutral"
    outlook = None

final_signal = custom_helpers.combine_signals(
    interpretation,
    technical_guardrail.signal if technical_guardrail else None,
    technical_guardrail.confidence if technical_guardrail else None
)

if final_signal != interpretation:
    logging.info(f"Signal adjusted from {interpretation} to {final_signal} after guardrail check")
interpretation = final_signal

if outlook:
    ai.save_response(outlook, RUN_NAME)

# Call exchange to get current position status
try:
    position = exchange.get_pending_positions(symbol=SYMBOL)
    current_position = custom_helpers.normalize_position_side(position.side) if position else None
    logging.info(f"Current Position: {current_position}")
    logging.info(f"Available Capital: {exchange.get_account_balance('USDT')} USDT")

    # Execute trading actions
    exchange.set_margin_mode(SYMBOL, MARGIN_MODE)
    exchange.set_leverage(SYMBOL, LEVERAGE)

    match (interpretation, current_position):
        # Bullish cases
        case ("Bullish", None):
            logging.info("Bullish signal: Opening long position")
            custom_helpers.open_position(exchange, SYMBOL, direction="buy",
                                        position_size=position_size_spec, stop_loss_percent=STOP_LOSS_PERCENT)

        case ("Bullish", "sell"):
            logging.info("Bullish signal: Closing short, opening long")
            exchange.flash_close_position(position.positionId)
            custom_helpers.open_position(exchange, SYMBOL, direction="buy",
                                        position_size=position_size_spec, stop_loss_percent=STOP_LOSS_PERCENT)

        case ("Bullish", "buy"):
            logging.info("Bullish signal: Already in long position, holding")

        # Bearish cases
        case ("Bearish", None):
            logging.info("Bearish signal: Opening short position")
            custom_helpers.open_position(exchange, SYMBOL, direction="sell",
                                        position_size=position_size_spec, stop_loss_percent=STOP_LOSS_PERCENT)

        case ("Bearish", "buy"):
            logging.info("Bearish signal: Closing long, opening short")
            exchange.flash_close_position(position.positionId)
            custom_helpers.open_position(exchange, SYMBOL, direction="sell",
                                        position_size=position_size_spec, stop_loss_percent=STOP_LOSS_PERCENT)

        case ("Bearish", "sell"):
            logging.info("Bearish signal: Already in short position, holding")

        # Neutral cases
        case ("Neutral", "buy" | "sell"):
            logging.info(f"Neutral signal: Closing {current_position} position")
            exchange.flash_close_position(position.positionId)
        case ("Neutral", None):
            logging.info("Neutral signal: No position open, doing nothing")

    logging.info("=== Run Completed ===")

except (BitunixError, Exception) as e:
    logging.warning(f"Exchange operation failed, stopping execution: {e}")

    # SAFETY: Flash close any open position on error
    try:
        position = exchange.get_pending_positions(symbol=SYMBOL)
        if position:
            logging.warning("Emergency flash close triggered due to error")
            exchange.flash_close_position(position.positionId)
            logging.info("Emergency flash close completed")
    except Exception as close_error:
        logging.error(f"Failed to flash close position: {close_error}")

    logging.info("=== Run Failed ===")

from kalshi import KalshiRestClient
import datetime as dt
from zoneinfo import ZoneInfo

import polars as pl
from tqdm import tqdm

client = KalshiRestClient()

# Config
ET = ZoneInfo("America/New_York")
SERIES_TICKER = "KXBTC15M"
END_DT = dt.datetime(2026, 5, 27, 10, 30, tzinfo=ET)
START_DT = END_DT - dt.timedelta(days=1)
STATUS = "settled"
PERIOD_INTERVAL = 1  # minutes per candle
OUT_PATH = "data/candles.parquet"


def parse_iso(s: str) -> int:
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


# Fetch the markets that settled within the date range
markets = client.get_markets(
    series_ticker=SERIES_TICKER,
    status=STATUS,
    min_close_ts=int(START_DT.timestamp()),
    max_close_ts=int(END_DT.timestamp()),
)

# Index each market's open/close window and result by ticker
market_meta = {
    m["ticker"]: {
        "open_ts": parse_iso(m["open_time"]),
        "close_ts": parse_iso(m["close_time"]),
        "result": m.get("result"),
    }
    for m in markets
}

# Fetch candles one market at a time, flattening each into a row
rows: list[dict] = []
for ticker, meta in tqdm(market_meta.items(), desc="fetching candles"):
    result = client.get_market_candlesticks(
        market_tickers=ticker,
        start_ts=meta["open_ts"],
        end_ts=meta["close_ts"],
        period_interval=PERIOD_INTERVAL,
    )
    for entry in result:
        for c in entry["candlesticks"]:
            price = c.get("price") or {}
            bid = c.get("yes_bid") or {}
            ask = c.get("yes_ask") or {}
            rows.append(
                {
                    "ticker": ticker,
                    "ts": c["end_period_ts"],
                    "market_open_ts": meta["open_ts"],
                    "market_close_ts": meta["close_ts"],
                    "result": meta["result"],
                    "price_open": price.get("open_dollars"),
                    "price_high": price.get("high_dollars"),
                    "price_low": price.get("low_dollars"),
                    "price_close": price.get("close_dollars"),
                    "price_mean": price.get("mean_dollars"),
                    "price_previous": price.get("previous_dollars"),
                    "yes_bid_open": bid.get("open_dollars"),
                    "yes_bid_high": bid.get("high_dollars"),
                    "yes_bid_low": bid.get("low_dollars"),
                    "yes_bid_close": bid.get("close_dollars"),
                    "yes_ask_open": ask.get("open_dollars"),
                    "yes_ask_high": ask.get("high_dollars"),
                    "yes_ask_low": ask.get("low_dollars"),
                    "yes_ask_close": ask.get("close_dollars"),
                    "volume": c.get("volume_fp"),
                    "open_interest": c.get("open_interest_fp"),
                }
            )

candles_df = pl.DataFrame(rows)

# Coerce price/size columns to floats
float_cols = [
    c for c in candles_df.columns if c.startswith(("price_", "yes_bid_", "yes_ask_"))
] + [
    "volume",
    "open_interest",
]
candles_df = candles_df.with_columns(pl.col(float_cols).cast(pl.Float64, strict=False))

# Convert epoch seconds to UTC datetimes
for col in ("ts", "market_open_ts", "market_close_ts"):
    candles_df = candles_df.with_columns(
        pl.from_epoch(pl.col(col), time_unit="s").dt.replace_time_zone("UTC").alias(col)
    )

# Sort and write to parquet
candles_df = candles_df.sort(["ticker", "ts"])
candles_df.write_parquet(OUT_PATH)

from kalshi import KalshiRestClient, KalshiWSClient
from kalshi.messages import LifecycleEvent, MetadataUpdated, Settled, parse
import asyncio
import datetime as dt
import json
import os

from tqdm import tqdm

rest = KalshiRestClient()

# Config
SERIES_TICKER = "KXBTC15M"
OUT_DIR = "data/ws_raw"


def append_raw(handles: dict, ticker: str, data: dict) -> None:
    """Append one raw message (stamped with receive time) to the ticker's JSONL file."""
    handle = handles.get(ticker)
    if handle is None:
        handle = open(os.path.join(OUT_DIR, f"{ticker}.jsonl"), "a")
        handles[ticker] = handle
    handle.write(json.dumps({"recv_ts": dt.datetime.now(dt.UTC).timestamp(), "data": data}) + "\n")


async def collect() -> None:
    # Load the currently-open market(s) to start collecting from
    markets = rest.get_markets(SERIES_TICKER, status="open")
    tickers = [m["ticker"] for m in markets]
    print(f"subscribing to {len(tickers):,} open {SERIES_TICKER} markets -> {OUT_DIR}/")

    ws = KalshiWSClient()
    handles: dict = {}
    counts: dict = {}
    active = set(tickers)
    progress = tqdm(desc="messages", unit="msg")
    try:
        async for data in ws.connect(tickers):
            event = parse(data)

            # Kalshi streams lifecycle for every series — drop anything outside ours
            if isinstance(event, LifecycleEvent) and not event.market_ticker.startswith(f"{SERIES_TICKER}-"):
                continue

            # A new market just opened — subscribe and start collecting it
            if isinstance(event, MetadataUpdated) and event.market_ticker not in active:
                active.add(event.market_ticker)
                await ws.add_market(event.market_ticker)
                progress.write(f"+ {event.market_ticker}")

            # Persist the raw message under its ticker
            ticker = data.get("msg", {}).get("market_ticker", "_system")
            append_raw(handles, ticker, data)
            counts[ticker] = counts.get(ticker, 0) + 1
            progress.update(1)

            # Market settled — unsubscribe, close its file, and stop tracking it
            if isinstance(event, Settled) and event.market_ticker in active:
                active.discard(event.market_ticker)
                await ws.remove_market(event.market_ticker)
                handle = handles.pop(event.market_ticker, None)
                if handle is not None:
                    handle.close()
                progress.write(f"- {event.market_ticker}")
    finally:
        progress.close()
        for handle in handles.values():
            handle.close()
        print(f"wrote {sum(counts.values()):,} messages across {len(counts):,} files -> {OUT_DIR}/")


# Ensure the output directory exists, then stream until interrupted
os.makedirs(OUT_DIR, exist_ok=True)
try:
    asyncio.run(collect())
except KeyboardInterrupt:
    print("stopped")

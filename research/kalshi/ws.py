import json
import os
from typing import AsyncIterator
import websockets
from ._auth import load_private_key, build_auth_headers


class KalshiWSClient:

    def __init__(
        self,
        api_key_id: str | None = None,
        private_key_path: str | None = None,
        base_url: str | None = None,
    ) -> None:
        api_key_id = api_key_id or os.getenv("KALSHI_API_KEY_ID")
        private_key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH")
        base_url = base_url or os.getenv("KALSHI_WS_URL")

        missing = [
            name for name, value in (
                ("KALSHI_API_KEY_ID", api_key_id),
                ("KALSHI_PRIVATE_KEY_PATH", private_key_path),
                ("KALSHI_WS_URL", base_url),
            ) if not value
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required Kalshi environment variables: {', '.join(missing)}"
            )

        self._api_key_id = api_key_id
        self._private_key = load_private_key(private_key_path)
        self._base_url = base_url
        self._ws: websockets.ClientConnection | None = None
        self._next_id = 0
        self._orderbook_sid: int | None = None

    async def connect(self, market_tickers: list[str]) -> AsyncIterator[dict]:
        """Connect, send two subscribes (ticker-filtered orderbook+lifecycle and
        unfiltered user_orders), and yield each parsed message."""
        ws_headers = build_auth_headers(
            self._api_key_id, self._private_key, "GET", "/trade-api/ws/v2"
        )

        async with websockets.connect(self._base_url, additional_headers=ws_headers) as websocket:
            self._ws = websocket

            # Ticker-filtered channels — dynamically updated via add/remove_market.
            self._next_id += 1
            await websocket.send(json.dumps({
                "id": self._next_id,
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta", "market_lifecycle_v2"],
                    "market_tickers": market_tickers,
                },
            }))

            # user_orders unfiltered so fills on later-added tickers still arrive.
            self._next_id += 1
            await websocket.send(json.dumps({
                "id": self._next_id,
                "cmd": "subscribe",
                "params": {"channels": ["user_orders"]},
            }))

            try:
                async for message in websocket:
                    data = json.loads(message)
                    if data.get("type") == "subscribed":
                        msg = data.get("msg", {})
                        if msg.get("channel") == "orderbook_delta":
                            self._orderbook_sid = msg.get("sid")
                    yield data
            finally:
                self._ws = None
                self._orderbook_sid = None

    async def add_market(self, ticker: str) -> None:
        """Add a market to the orderbook subscription via update_subscription."""
        await self._update_subscription("add_markets", ticker)

    async def remove_market(self, ticker: str) -> None:
        """Remove a market from the orderbook subscription via update_subscription."""
        await self._update_subscription("delete_markets", ticker)

    async def _update_subscription(self, action: str, ticker: str) -> None:
        if self._ws is None or self._orderbook_sid is None:
            return
        self._next_id += 1
        await self._ws.send(json.dumps({
            "id": self._next_id,
            "cmd": "update_subscription",
            "params": {
                "sids": [self._orderbook_sid],
                "action": action,
                "market_tickers": [ticker],
            },
        }))
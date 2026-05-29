import os
import uuid
from urllib.parse import urlparse

import requests

from ._auth import build_auth_headers, load_private_key
from dotenv import load_dotenv

load_dotenv(override=True)

class KalshiRestClient:
    def __init__(
        self,
        api_key_id: str | None = None,
        private_key_path: str | None = None,
        base_url: str | None = None,
    ) -> None:
        api_key_id = api_key_id or os.getenv("KALSHI_API_KEY_ID")
        private_key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH")
        base_url = base_url or os.getenv("KALSHI_API_BASE")

        missing = [
            name for name, value in (
                ("KALSHI_API_KEY_ID", api_key_id),
                ("KALSHI_PRIVATE_KEY_PATH", private_key_path),
                ("KALSHI_API_BASE", base_url),
            ) if not value
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required Kalshi environment variables: {', '.join(missing)}"
            )

        self._api_key_id = api_key_id
        self._private_key = load_private_key(private_key_path)
        self._base_url = base_url
        self._session = requests.Session()


    def _get(self, path: str, params: dict | None = None) -> requests.Response:
        """Make an authenticated GET request to the Kalshi API."""
        # Signing requires the full URL path from root (e.g. /trade-api/v2/portfolio/balance)
        sign_path = urlparse(self._base_url + path).path
        headers = build_auth_headers(self._api_key_id, self._private_key, "GET", sign_path)
        r = self._session.get(self._base_url + path, headers=headers, params=params)
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} {r.reason} for {r.url}: {r.text}", response=r
            )
        return r

    def _post(self, path: str, body: dict) -> requests.Response:
        """Make an authenticated POST request to the Kalshi API."""
        sign_path = urlparse(self._base_url + path).path
        headers = build_auth_headers(self._api_key_id, self._private_key, "POST", sign_path)
        r = self._session.post(self._base_url + path, headers=headers, json=body)
        r.raise_for_status()
        return r

    def _delete(self, path: str) -> requests.Response:
        """Make an authenticated DELETE request to the Kalshi API."""
        sign_path = urlparse(self._base_url + path).path
        headers = build_auth_headers(self._api_key_id, self._private_key, "DELETE", sign_path)
        r = self._session.delete(self._base_url + path, headers=headers)
        r.raise_for_status()
        return r

    def get_balance(self) -> dict:
        response = self._get("/portfolio/balance")
        return response.json()

    def get_markets(
        self,
        series_ticker: str,
        status: str = "open",
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        params: dict = {
            "series_ticker": series_ticker,
            "status": status,
            "limit": limit,
        }
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        if max_close_ts is not None:
            params["max_close_ts"] = max_close_ts

        all_markets: list[dict] = []
        while True:
            response = self._get("/markets", params=params)
            body = response.json()
            all_markets.extend(body.get("markets", []))
            cursor = body.get("cursor")
            if not cursor:
                return all_markets
            params["cursor"] = cursor

    def place_order(
        self,
        *,
        ticker: str,
        side: str,                              # "yes" | "no"
        action: str,                            # "buy" | "sell"
        count: int,
        price_cents: int,                       # integer 1..99 — the limit price for `side`
        order_type: str = "limit",              # "limit" | "market"
        time_in_force: str | None = None,       # omit = GTC; otherwise Kalshi's enum
        client_order_id: str | None = None,
    ) -> dict:
        """Submit an order. Kalshi quotes prices as integer cents (1..99) and
        splits the price field by side: a yes-side limit uses `yes_price`,
        a no-side limit uses `no_price`. `time_in_force` is omitted for GTC.
        """
        body: dict = {
            "action": action,
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "count": count,
            "side": side,
            "ticker": ticker,
            "type": order_type,
        }
        if order_type == "limit":
            body["yes_price" if side == "yes" else "no_price"] = price_cents
        if time_in_force is not None:
            body["time_in_force"] = time_in_force
        response = self._post("/portfolio/orders", body=body)
        return response.json()

    def get_orders(self, ticker: str | None = None, status: str | None = None) -> list[dict]:
        params: dict = {}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        response = self._get("/portfolio/orders", params=params or None)
        return response.json().get("orders", [])

    def cancel_order(self, order_id: str) -> dict:
        response = self._delete(f"/portfolio/orders/{order_id}")
        return response.json()

    def get_market_candlesticks(
        self,
        *,
        market_tickers: list[str] | str,
        start_ts: int,
        end_ts: int,
        period_interval: int,
        include_latest_before_start: bool | None = None,
    ) -> list[dict]:
        tickers = (
            market_tickers if isinstance(market_tickers, str) else ",".join(market_tickers)
        )
        params: dict = {
            "market_tickers": tickers,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        if include_latest_before_start is not None:
            params["include_latest_before_start"] = str(include_latest_before_start).lower()
        response = self._get("/markets/candlesticks", params=params)
        return response.json().get("markets", [])


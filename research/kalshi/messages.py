from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Subscribed:
    channel: str
    sid: int


@dataclass(frozen=True, slots=True)
class OrderbookSnapshot:
    market_ticker: str
    yes_levels: list[tuple[str, str]]
    no_levels: list[tuple[str, str]]


@dataclass(frozen=True, slots=True)
class OrderbookDelta:
    market_ticker: str
    side: str  # "yes" | "no"
    price: str  # dollars, e.g. "0.3800"
    delta_fp: str  # signed size, e.g. "-12.00"


@dataclass(frozen=True, slots=True)
class MetadataUpdated:
    market_ticker: str
    floor_strike: float | None


@dataclass(frozen=True, slots=True)
class Determined:
    market_ticker: str
    determination_ts: int
    result: str  # "yes" | "no"
    settlement_value: str  # "1.0000" | "0.0000"


@dataclass(frozen=True, slots=True)
class Settled:
    market_ticker: str
    settled_ts: int


@dataclass(frozen=True, slots=True)
class UnknownLifecycle:
    market_ticker: str
    event_type: str | None
    raw: dict


@dataclass(frozen=True, slots=True)
class UserOrder:
    order_id: str
    client_order_id: str
    ticker: str
    status: str               # "resting" | "canceled" | "executed"
    side: str                 # "yes" | "no"
    yes_price: str            # limit price in YES terms (dollars)
    initial_count: str
    fill_count: str
    remaining_count: str
    taker_fill_cost: str      # actual dollars paid for taker fills
    taker_fees: str
    last_updated_ts_ms: int


@dataclass(frozen=True, slots=True)
class Ok:
    raw: dict


@dataclass(frozen=True, slots=True)
class EventLifecycle:
    raw: dict


@dataclass(frozen=True, slots=True)
class Error:
    raw: dict


@dataclass(frozen=True, slots=True)
class Unhandled:
    raw: dict


LifecycleEvent = (
    MetadataUpdated | Determined | Settled | UnknownLifecycle
)
WSEvent = (
    Subscribed
    | OrderbookSnapshot
    | OrderbookDelta
    | LifecycleEvent
    | UserOrder
    | Ok
    | EventLifecycle
    | Error
    | Unhandled
)


def parse(data: dict) -> WSEvent:
    msg = data.get("msg", {})
    match data.get("type"):
        case "subscribed":
            return Subscribed(channel=msg.get("channel", ""), sid=msg.get("sid", 0))
        case "orderbook_snapshot":
            return OrderbookSnapshot(
                market_ticker=msg["market_ticker"],
                yes_levels=msg.get("yes_dollars_fp", []),
                no_levels=msg.get("no_dollars_fp", []),
            )
        case "orderbook_delta":
            return OrderbookDelta(
                market_ticker=msg["market_ticker"],
                side=msg["side"],
                price=msg["price_dollars"],
                delta_fp=msg["delta_fp"],
            )
        case "market_lifecycle_v2":
            return _parse_lifecycle(msg)
        case "user_order":
            return UserOrder(
                order_id=msg.get("order_id", ""),
                client_order_id=msg.get("client_order_id", ""),
                ticker=msg.get("ticker", ""),
                status=msg.get("status", ""),
                side=msg.get("side", ""),
                yes_price=msg.get("yes_price_dollars", ""),
                initial_count=msg.get("initial_count_fp", "0"),
                fill_count=msg.get("fill_count_fp", "0"),
                remaining_count=msg.get("remaining_count_fp", "0"),
                taker_fill_cost=msg.get("taker_fill_cost_dollars", "0"),
                taker_fees=msg.get("taker_fees_dollars", "0"),
                last_updated_ts_ms=msg.get("last_updated_ts_ms", 0),
            )
        case "event_lifecycle":
            return EventLifecycle(raw=data)
        case "ok":
            return Ok(raw=data)
        case "error":
            return Error(raw=data)
        case _:
            return Unhandled(raw=data)


def _parse_lifecycle(msg: dict) -> LifecycleEvent:
    ticker = msg.get("market_ticker", "")
    match msg.get("event_type"):
        case "metadata_updated":
            return MetadataUpdated(
                market_ticker=ticker,
                floor_strike=msg.get("floor_strike"),
            )
        case "determined":
            return Determined(
                market_ticker=ticker,
                determination_ts=msg.get("determination_ts", 0),
                result=msg.get("result", ""),
                settlement_value=msg.get("settlement_value", ""),
            )
        case "settled":
            return Settled(
                market_ticker=ticker,
                settled_ts=msg.get("settled_ts", 0),
            )
        case event_type:
            return UnknownLifecycle(
                market_ticker=ticker,
                event_type=event_type,
                raw=msg,
            )

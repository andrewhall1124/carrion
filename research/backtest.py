import polars as pl
import math
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

# Config
DATA_PATH = "data/candles.parquet"
OUT_PATH = "results/backtest.png"
BUY_PRICE = 0.99             # buy yes the first time the ask reaches 99c
WIN_PROB = 0.995             # assumed conversion rate for sizing
BANKROLL = 200.0
KELLY_FRACTION = 0.25        # quarter Kelly
FEE_MULTIPLIER = 0.07        # Kalshi fee model


def kelly_criterion(edge: float, odds: float) -> float:
    return edge / odds


def kalshi_fee(price: float, n_contracts: int) -> float:
    raw_fee = FEE_MULTIPLIER * n_contracts * price * (1 - price)
    return math.ceil(raw_fee * 100) / 100


candles_df = pl.read_parquet(DATA_PATH)
series_ticker = candles_df["ticker"][0].split("-")[0]

# Find the first candle per market where the yes ask reaches 99c — one entry per market
entries_df = (
    candles_df
    .filter(pl.col("yes_ask_close").ge(BUY_PRICE), pl.col("yes_ask_close").lt(1.0))
    .sort("ts")
    .group_by("ticker")
    .first()
    .select("ticker", "ts", "result")
    .sort("ts")
)

# Fixed fractional-Kelly stake: f* depends only on the (constant) buy price and win prob
edge = (WIN_PROB - BUY_PRICE) / BUY_PRICE
odds = (1 - BUY_PRICE) / BUY_PRICE
bet_fraction = KELLY_FRACTION * kelly_criterion(edge, odds)

# Walk markets in chronological order, compounding the bankroll trade by trade
bankroll = BANKROLL
rows: list[dict] = [{"ts": entries_df["ts"][0], "trade": 0, "bankroll": bankroll, "pnl": 0.0, "fee": 0.0, "won": None}]
for trade, (ts, result) in enumerate(tqdm(entries_df.select("ts", "result").iter_rows(), total=entries_df.height, desc="backtesting"), start=1):
    n_contracts = int((bet_fraction * bankroll) // BUY_PRICE)
    if n_contracts <= 0:
        continue
    fee = kalshi_fee(BUY_PRICE, n_contracts)
    won = result == "yes"
    raw_pnl = n_contracts * (1 - BUY_PRICE) if won else -n_contracts * BUY_PRICE
    pnl = raw_pnl - fee
    bankroll += pnl
    rows.append({"ts": ts, "trade": trade, "bankroll": bankroll, "pnl": pnl, "fee": fee, "won": won})

results_df = pl.DataFrame(rows)

# Summary metrics
n_trades = results_df.height - 1
n_losses = results_df.filter(pl.col("won") == False).height
win_rate = 1 - n_losses / n_trades
total_fees = results_df["fee"].sum()
total_return = bankroll / BANKROLL - 1
running_max = results_df["bankroll"].cum_max()
max_drawdown = (results_df["bankroll"] / running_max - 1).min()

# Equity curve over time, with a starting-bankroll reference and loss markers
ts = results_df["ts"].to_numpy()
equity = results_df["bankroll"].to_numpy()
losses = results_df.filter(pl.col("won") == False)

fig, ax = plt.subplots(figsize=(11, 6))
ax.axhline(BANKROLL, color="black", linestyle="--", linewidth=1, label="Starting bankroll")
ax.plot(ts, equity, color="C0", linewidth=1.5, label="Bankroll")
ax.scatter(losses["ts"], losses["bankroll"], color="tab:red", s=40, zorder=5, label=f"Losing trade ({n_losses})")
ax.set_xlabel("Date")
ax.set_ylabel("Bankroll (\\$)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"{series_ticker} backtest — buy yes at 99¢, ¼-Kelly sizing\n"
    f"\\${bankroll:,.0f} final ({total_return:+.0%}) · {n_trades:,} trades · "
    f"{win_rate:.2%} win rate · max drawdown {max_drawdown:.0%}",
    fontsize=13,
)
fig.text(
    0.5,
    0.01,
    f"start \\${BANKROLL:,.0f} · {KELLY_FRACTION:g}× Kelly ({bet_fraction:.0%} of bankroll/trade) "
    f"· assumed win prob {WIN_PROB:.2%} · Kalshi fee × {FEE_MULTIPLIER} · \\${total_fees:,.2f} fees paid",
    ha="center",
    color="gray",
    fontsize=8,
)

fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")

print(f"final ${bankroll:,.2f} ({total_return:+.1%}) over {n_trades:,} trades, {n_losses} losses -> {OUT_PATH}")

import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# Config
DATA_PATH = "data/candles.parquet"
COLUMN = "yes_ask_close"
OUT_PATH = "results/calibration_over_time.png"
EXTREMES = [0.01, 0.99]  # longshot and favorite buckets to track over time
MIN_COUNT = 50  # drop time-to-expiry buckets too thin to estimate
Z = 1.96  # 95% confidence

candles_df = pl.read_parquet(DATA_PATH)
series_ticker = candles_df["ticker"][0].split("-")[0]
n_markets = candles_df["ticker"].n_unique()
start, end = candles_df["ts"].min(), candles_df["ts"].max()

# Bucket the extreme prices and tally outcomes per minute of time to expiry
tte_df = (
    candles_df
    .with_columns(
        pl.col("result").replace({"yes": 1, "no": 0}).cast(pl.Int32),
        pl.col("market_close_ts").sub("ts").dt.total_minutes().alias("tte"),
        pl.col(COLUMN).truncate(decimals=2).replace({0.0: 0.01}),
    )
    .filter(pl.col(COLUMN).is_in(EXTREMES))
    .group_by(COLUMN, "tte")
    .agg(
        pl.len().alias("count"),
        pl.col("result").mean().alias("observed"),
    )
    .filter(pl.col("count") >= MIN_COUNT)
    .sort(COLUMN, "tte")
)
max_count = tte_df["count"].max()

# Edge to the yes buyer (observed - price), in cents, with a 95% Wilson CI per minute
fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(0, color="black", linewidth=1, zorder=1)
for price_level, color in zip(EXTREMES, ["tab:red", "tab:green"]):
    bucket = tte_df.filter(pl.col(COLUMN) == price_level)
    tte = bucket["tte"].to_numpy()
    n = bucket["count"].to_numpy()
    obs = bucket["observed"].to_numpy()

    denom = 1 + Z**2 / n
    center = (obs + Z**2 / (2 * n)) / denom
    margin = (Z / denom) * np.sqrt(obs * (1 - obs) / n + Z**2 / (4 * n**2))
    edge = (obs - price_level) * 100
    ci_lo = (center - margin - price_level) * 100
    ci_hi = (center + margin - price_level) * 100

    ax.vlines(tte, ci_lo, ci_hi, color=color, alpha=0.5, zorder=2)
    ax.plot(tte, edge, color=color, linewidth=1, alpha=0.6, zorder=3)
    ax.scatter(
        tte,
        edge,
        s=12 + 280 * n / max_count,
        color=color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
        label=f"{int(round(price_level * 100))}¢ contracts (size ∝ count)",
    )

ax.invert_xaxis()  # time flows left → right toward settlement
ax.set_ylim(-3, 3)
ax.set_xlabel("Time to expiry (minutes)")
ax.set_ylabel("Edge to yes buyer — observed − price (¢)")
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.3)

# Two-line title plus a run-parameter caption
fig.suptitle(
    f"{series_ticker} extreme-contract edge by time to expiry\n"
    f"Mispricing persists across the market's life — it does not decay into settlement",
    fontsize=13,
)
fig.text(
    0.5,
    0.01,
    f"column={COLUMN} · {n_markets:,} markets · {start:%Y-%m-%d} → {end:%Y-%m-%d} "
    f"· buckets with ≥{MIN_COUNT} obs · 95% Wilson CI",
    ha="center",
    color="gray",
    fontsize=8,
)

fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")

print(f"wrote extreme-contract edge over time -> {OUT_PATH}")

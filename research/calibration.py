import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# Config
DATA_PATH = "data/candles.parquet"
COLUMN = "yes_ask_close"
OUT_PATH = "results/calibration.png"
OUT_PATH_EXTREMES = "results/calibration_extremes.png"
EXTREMES = [0.01, 0.02, 0.98, 0.99]  # longshot and favorite buckets to spotlight
Z = 1.96  # 95% confidence

candles_df = pl.read_parquet(DATA_PATH)
series_ticker = candles_df["ticker"][0].split("-")[0]
n_markets = candles_df["ticker"].n_unique()
start, end = candles_df["ts"].min(), candles_df["ts"].max()

# Bucket the predicted price into cents and tally outcomes per bucket
cal_df = (
    candles_df
    .with_columns(
        pl.col("result").replace({"yes": 1, "no": 0}).cast(pl.Int32),
        pl.col(COLUMN).truncate(decimals=2).replace({0.0: 0.01}),
    )
    .filter(pl.col(COLUMN).ne(1))
    .group_by(COLUMN)
    .agg(
        pl.len().alias("count"),
        pl.col("result").mean().alias("observed"),
    )
    .sort(COLUMN)
)

price = cal_df[COLUMN].to_numpy()
count = cal_df["count"].to_numpy()
observed = cal_df["observed"].to_numpy()

# Wilson score interval for each bucket's conversion proportion
denom = 1 + Z**2 / count
center = (observed + Z**2 / (2 * count)) / denom
margin = (Z / denom) * np.sqrt(observed * (1 - observed) / count + Z**2 / (4 * count**2))
ci_lo, ci_hi = center - margin, center + margin

# Reliability diagram of observed conversion vs. predicted price
fig, ax_cal = plt.subplots(figsize=(9, 9))

# Perfect-calibration diagonal and the 95% band around observed conversion
ax_cal.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.5, zorder=1, label="Perfect calibration")
ax_cal.fill_between(price, ci_lo, ci_hi, color="C0", alpha=0.2, zorder=2, label="95% CI")
ax_cal.plot(price, observed, color="C0", linewidth=1, alpha=0.5, zorder=3)
ax_cal.scatter(
    price,
    observed,
    s=12 + 280 * count / count.max(),
    color="C0",
    alpha=0.85,
    edgecolor="white",
    linewidth=0.5,
    zorder=4,
    label="Observed (size ∝ count)",
)
ax_cal.set_xlim(0, 1)
ax_cal.set_ylim(0, 1)
ax_cal.set_aspect("equal")
ax_cal.set_xlabel("Predicted price — yes ask close ($)")
ax_cal.set_ylabel("Empirical conversion rate")
ax_cal.legend(loc="upper left", framealpha=0.9)
ax_cal.grid(True, alpha=0.3)

# Title plus a run-parameter caption
fig.suptitle(
    f"{series_ticker} calibration — yes ask close vs. realized outcome\n"
    f"Well-calibrated overall — observed conversion tracks the price across the full range",
    fontsize=13,
)
fig.text(
    0.5,
    0.04,
    f"column={COLUMN} · {n_markets:,} markets · {np.sum(count):,} candle-minutes "
    f"· {start:%Y-%m-%d} → {end:%Y-%m-%d} · excludes $1.00",
    ha="center",
    color="gray",
    fontsize=8,
)

fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")

print(f"wrote calibration over {np.sum(count):,} candle-minutes -> {OUT_PATH}")

# Zoom in on the extreme buckets, where a sub-cent gap is invisible on the [0,1] plot
extremes_df = cal_df.filter(pl.col(COLUMN).is_in(EXTREMES)).sort(COLUMN)
ext_price = extremes_df[COLUMN].to_numpy()
ext_count = extremes_df["count"].to_numpy()
ext_obs = extremes_df["observed"].to_numpy()

# Wilson score interval for each extreme proportion
e_denom = 1 + Z**2 / ext_count
e_center = (ext_obs + Z**2 / (2 * ext_count)) / e_denom
e_margin = (Z / e_denom) * np.sqrt(ext_obs * (1 - ext_obs) / ext_count + Z**2 / (4 * ext_count**2))
e_lo, e_hi = e_center - e_margin, e_center + e_margin

# Edge to the yes buyer (observed conversion - price paid), in cents
edge = (ext_obs - ext_price) * 100
yerr = np.clip(np.vstack([(ext_obs - e_lo) * 100, (e_hi - ext_obs) * 100]), 0, None)
labels = [f"{int(round(p * 100))}¢" for p in ext_price]
colors = ["tab:red" if e < 0 else "tab:green" for e in edge]

# Bar chart of the per-bucket edge with 95% Wilson error bars
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.axhline(0, color="black", linewidth=1, zorder=1)
ax2.bar(labels, edge, yerr=yerr, color=colors, alpha=0.85, capsize=8, width=0.6, zorder=2)
for x, e, o, n, lo_len, hi_len in zip(labels, edge, ext_obs, ext_count, yerr[0], yerr[1]):
    # Anchor each label past the end of the whisker so it never overlaps the CI
    ax2.annotate(
        f"{e:+.2f}¢\nconverts {o * 100:.2f}%\n(n={n:,})",
        (x, e + hi_len if e > 0 else e - lo_len),
        textcoords="offset points",
        xytext=(0, 10 if e > 0 else -38),
        ha="center",
        fontsize=9,
    )
ax2.set_ylim(-2.5, 2.5)
ax2.set_xlabel("Contract price")
ax2.set_ylabel("Edge to yes buyer — observed − price (¢)")
ax2.grid(True, axis="y", alpha=0.3)

fig2.suptitle(
    f"{series_ticker} extreme-contract miscalibration\n"
    f"Longshots (1–2¢) overpriced · favorites (98–99¢) underpriced",
    fontsize=13,
)
fig2.text(
    0.5,
    0.02,
    f"column={COLUMN} · {n_markets:,} markets · {start:%Y-%m-%d} → {end:%Y-%m-%d} · 95% Wilson CI",
    ha="center",
    color="gray",
    fontsize=8,
)

fig2.savefig(OUT_PATH_EXTREMES, dpi=300, bbox_inches="tight")

print(f"wrote extreme-contract miscalibration -> {OUT_PATH_EXTREMES}")

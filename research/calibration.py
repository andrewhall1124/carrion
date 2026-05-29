import polars as pl
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

df = pl.read_parquet('data/candles.parquet')
column = 'yes_ask_close'

df = (
    df
    .with_columns(
        pl.col('market_close_ts').sub('ts').dt.total_minutes().alias('time_to_expiry'),
        pl.col('result').replace({'yes': '1', 'no': '0'}).cast(pl.Int32),
        pl.col(column).alias('original'),
        # pl.col(column).truncate(decimals=2).replace({0: 0.01})
    )
    .filter(
        # pl.col('time_to_expiry').eq(10),
        pl.col(column).ne(1),
        pl.col(column).ge(0.99)
    )
    .group_by(column)
    .agg(
        pl.len().alias('count'),
        pl.col('result').mean()
    )
    .with_columns(
        pl.col(column).sub('result').alias('delta')
    )
    .sort(column)
)

# x = [0.01 * i for i in range(1, 100)]
x = [0.99, 0.9999]


plt.figure(figsize=(10, 6))

sns.lineplot(df, x='yes_ask_close', y='result', label="Observed Calibration") # , hue='time_to_expiry'

plt.plot(x, x, label="Perfect Calibration", color='red', linestyle='--')

plt.title('Calibration')

plt.xlabel("Yes Ask Close")
plt.ylabel("Converted (%)")

plt.legend()

plt.tight_layout()

plt.show()
plt.clf()

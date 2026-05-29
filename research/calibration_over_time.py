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
    )
    .filter(
        pl.col(column).ge(.99),
        pl.col(column).ne(1)
    )
    .group_by('time_to_expiry')
    .agg(
        pl.col('result').mean(),
        pl.len().alias('count'),
    )
    .with_columns(
        pl.col('result').sub(0.99).alias('edge')
    )
    .sort('time_to_expiry')
)

print(df)
print(df['edge'].mean() * 100)

plt.figure(figsize=(10, 6))

sns.lineplot(df, x='time_to_expiry', y='result')

plt.title('Calibration Over Time (99c)')

plt.xlabel("Time to Expiry (minutes)")
plt.ylabel("Converted (%)")

plt.tight_layout()

plt.show()
plt.clf()
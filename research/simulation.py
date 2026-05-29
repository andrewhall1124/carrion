import polars as pl
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt

random.seed(42)

BUY_PRICE = 0.99
TRUE_PROBABILITY = 0.995
N_ITERATIONS = 96 * 7 * 4 # 1 day
N_PATHS = 500
BANKROLL = 200.0
KELLY_FRACTION = 0.25
FEE_MULTIPLIER = 0.07

def kelly_criterion(edge: float, odds: float) -> float:
    return edge / odds

def kalshi_fee(price: float, n_contracts: int) -> float:
    raw_fee = FEE_MULTIPLIER * n_contracts * price * (1 - price)
    return math.ceil(raw_fee * 100) / 100

results = []
for path in range(N_PATHS):
    current_bankroll = BANKROLL
    results.append({
        'path': path,
        'iteration': 0,
        'bankroll': current_bankroll,
        'pnl': 0,
        'raw_pnl': 0,
        'fee': 0
    })
    for iteration in range(1, N_ITERATIONS + 1):
        # Calculate bet size
        edge = (TRUE_PROBABILITY - BUY_PRICE) / BUY_PRICE
        odds = (1 - BUY_PRICE) / BUY_PRICE
        f_star = kelly_criterion(edge, odds)
        fractional_f_star = KELLY_FRACTION * f_star
        bet_size = fractional_f_star * current_bankroll
        n_contracts = int(bet_size // BUY_PRICE)
        fee = kalshi_fee(BUY_PRICE, n_contracts)

        # Sample conversion
        sample = random.uniform(0, 1)
        convert = sample < TRUE_PROBABILITY

        # Calculate pnl
        raw_pnl = n_contracts * (1 - BUY_PRICE) if convert else -n_contracts * BUY_PRICE

        # Append results
        pnl = raw_pnl - fee
        current_bankroll += pnl
        results.append({
            'path': path,
            'iteration': iteration,
            'bankroll': current_bankroll,
            'pnl': pnl,
            'raw_pnl': raw_pnl,
            'fee': fee
        })   
  
results_df = pl.DataFrame(results)

ending_bankrolls = results_df.group_by('path').agg(pl.col('bankroll').last())['bankroll']
p_profitable = ending_bankrolls.gt(BANKROLL).sum() / N_PATHS

plt.figure(figsize=(10, 6))

sns.lineplot(results_df, x='iteration', y='bankroll', hue='path', legend=False)

plt.title(
    "Bankroll Paths Under Fractional Kelly Sizing\n"
    f"Percent Profitable: {p_profitable:.2%}"
)
plt.xlabel("Iteration")
plt.ylabel("Bankroll")

plt.axhline(y=BANKROLL, linestyle='--', color='black')

caption = (
    f"Paths: {N_PATHS}  |  Iterations: {N_ITERATIONS}  |  "
    f"Buy price: {BUY_PRICE}  |  True prob: {TRUE_PROBABILITY}  |  "
    f"Kelly fraction: {KELLY_FRACTION}"
)
plt.figtext(0.5, 0.01, caption, ha='center', fontsize=8, color='gray')

plt.tight_layout(rect=[0, 0.03, 1, 1])

plt.savefig("results/monte-carlo-simulations.png", dpi=300)




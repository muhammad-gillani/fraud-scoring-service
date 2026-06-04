"""
Generate synthetic fraud transaction data.
50,000 transactions, ~2% fraud rate with realistic signal separation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_TRANSACTIONS = 50_000
FRAUD_RATE = 0.02


def norm(weights):
    """Normalize a list of weights to sum exactly to 1.0."""
    total = sum(weights)
    return [w / total for w in weights]


def generate(output_path: str = "data/raw/transactions.csv"):
    rng = np.random.default_rng(SEED)
    n_fraud = int(N_TRANSACTIONS * FRAUD_RATE)
    n_legit = N_TRANSACTIONS - n_fraud

    def make_transactions(n, fraud: bool):
        return {
            "transaction_id": [f"{'F' if fraud else 'L'}{i:06d}" for i in range(n)],
            # Fraud = higher amounts
            "amount": rng.lognormal(mean=6.5 if fraud else 4.5, sigma=1.2, size=n),
            # Fraud = late night (hours 0-4 heavier)
            "hour": rng.choice(
                range(24), size=n,
                p=norm([0.08]*5 + [0.02]*6 + [0.03]*5 + [0.05]*4 + [0.04]*4)
                if fraud else
                norm([0.01]*5 + [0.04]*6 + [0.07]*5 + [0.06]*4 + [0.02]*4)
            ),
            "day_of_week": rng.integers(0, 7, size=n),
            # Fraud = farther from home
            "distance_from_home_km": rng.exponential(scale=80 if fraud else 15, size=n),
            "distance_from_last_transaction_km": rng.exponential(scale=50 if fraud else 10, size=n),
            # Fraud = less likely to use chip/pin
            "used_chip": rng.choice([0, 1], size=n, p=[0.7, 0.3] if fraud else [0.2, 0.8]),
            "used_pin": rng.choice([0, 1], size=n, p=[0.8, 0.2] if fraud else [0.3, 0.7]),
            "online_order": rng.choice([0, 1], size=n, p=[0.3, 0.7] if fraud else [0.6, 0.4]),
            "merchant_category": rng.choice(
                ["grocery", "gas", "restaurant", "online", "travel", "electronics"],
                size=n,
                p=norm([0.05, 0.05, 0.05, 0.40, 0.25, 0.20]) if fraud else
                  norm([0.30, 0.20, 0.25, 0.15, 0.05, 0.05])
            ),
            # Fraud = newer/less-established merchants
            "merchant_age_days": rng.exponential(scale=100 if fraud else 800, size=n),
            "repeat_merchant": rng.choice([0, 1], size=n, p=[0.8, 0.2] if fraud else [0.3, 0.7]),
            "is_fraud": int(fraud),
        }

    fraud_df = pd.DataFrame(make_transactions(n_fraud, fraud=True))
    legit_df = pd.DataFrame(make_transactions(n_legit, fraud=False))

    df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} transactions ({n_fraud} fraud, {n_legit} legit) → {output_path}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    generate()

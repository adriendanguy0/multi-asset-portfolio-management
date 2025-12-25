"""
project3_monte_carlo_black_scholes.py

A small "quant tool" script, kept intentionally simple and interpretable.

In an asset management context, the purpose is not to price exotic derivatives,
but to understand how option value reacts to:
- volatility
- time to maturity
- interest rates
- underlying price moves

We compare:
1) Black–Scholes (analytical reference for European options)
2) Monte Carlo (simulation-based estimate + confidence interval)

Key idea:
Monte Carlo gives a price *and* reminds us that estimation has uncertainty.
"""

import os
import math
import numpy as np
import pandas as pd


# -----------------------------
# Math helpers
# -----------------------------

def norm_cdf(x: float) -> float:
    """Standard normal CDF using error function (no external dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# -----------------------------
# Black–Scholes (European options)
# -----------------------------

def black_scholes_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
) -> float:
    """
    European option price under Black–Scholes assumptions.

    Assumptions (simplified):
    - lognormal underlying returns
    - constant volatility
    - frictionless markets
    - continuous compounding rate r

    Even if assumptions are not perfect in real life,
    BS is a useful reference point for intuition.
    """
    if T <= 0 or sigma <= 0:
        # If no time or no volatility, option is basically its intrinsic value.
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type.lower() == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


# -----------------------------
# Monte Carlo pricing (European options)
# -----------------------------

def monte_carlo_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 200_000,
    option_type: str = "call",
    seed: int = 42,
) -> tuple[float, float]:
    """
    Monte Carlo price for a European option.

    Returns:
    - estimated discounted price
    - standard error of the estimate (so we can build a confidence interval)

    This is useful pedagogically:
    you see the distribution of outcomes rather than a single closed-form number.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)

    # Risk-neutral evolution of the underlying at maturity
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)

    if option_type.lower() == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    disc_payoff = math.exp(-r * T) * payoff
    price = float(disc_payoff.mean())

    # Standard error of the mean (for uncertainty)
    se = float(disc_payoff.std(ddof=1) / math.sqrt(n_paths))
    return price, se


# -----------------------------
# A tiny scenario sweep (optional, but AM-friendly)
# -----------------------------

def vol_sensitivity_table(
    S: float, K: float, r: float, T: float, vol_grid: list[float], option_type: str
) -> pd.DataFrame:
    """
    Builds a small table showing how option price changes with volatility.
    This is a nice, intuitive "Greeks without Greeks" view.
    """
    rows = []
    for sigma in vol_grid:
        bs = black_scholes_price(S, K, r, sigma, T, option_type=option_type)
        mc, se = monte_carlo_price(S, K, r, sigma, T, option_type=option_type, n_paths=120_000, seed=7)
        rows.append(
            {
                "sigma": sigma,
                "BS_price": bs,
                "MC_price": mc,
                "MC_95pct_CI_low": mc - 1.96 * se,
                "MC_95pct_CI_high": mc + 1.96 * se,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = "outputs_project3"
    os.makedirs(out_dir, exist_ok=True)

    # A simple at-the-money example
    S = 100.0
    K = 100.0
    r = 0.02
    T = 1.0
    sigma = 0.20

    for option_type in ["call", "put"]:
        bs = black_scholes_price(S, K, r, sigma, T, option_type=option_type)
        mc, se = monte_carlo_price(S, K, r, sigma, T, option_type=option_type, n_paths=250_000, seed=123)

        print(f"\n=== {option_type.upper()} option ===")
        print(f"Black–Scholes price : {bs:.4f}")
        print(f"Monte Carlo price   : {mc:.4f}  (± {1.96*se:.4f} at ~95% confidence)")
        print("Interpretation: the MC result should be close to BS, and the CI gives a sense of estimation noise.")

    # Small volatility sweep (this is a very human way to show intuition)
    vol_grid = [0.10, 0.15, 0.20, 0.25, 0.30]
    table = vol_sensitivity_table(S, K, r, T, vol_grid, option_type="call")

    table.to_csv(f"{out_dir}/vol_sensitivity_call.csv", index=False)

    print(f"\nSaved a small volatility sensitivity table in: {out_dir}/vol_sensitivity_call.csv")
    print("This table is often more useful than a lot of theory: it shows how prices move when vol changes.\n")


if __name__ == "__main__":
    main()

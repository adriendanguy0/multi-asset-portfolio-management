"""
project1_portfolio_monitoring.py

A simple portfolio monitoring script, written with an asset management mindset.

What this script tries to do (in plain words):
- simulate a small multi-asset universe (Equity / Bonds / Alternatives)
- build a simple strategic allocation (e.g., 60/30/10)
- monitor how the portfolio behaves through time:
  performance, volatility, drawdowns, and a couple of intuitive risk indicators

Important note:
This is not meant to forecast markets.
The point is to observe portfolio mechanics with transparent assumptions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Small helper functions
# -----------------------------

def simulate_gbm_prices(
    dates: pd.DatetimeIndex,
    start_price: float,
    mu: float,
    sigma: float,
    seed: int | None = None,
) -> pd.Series:
    """
    Simulates a price series using a simple Geometric Brownian Motion (GBM).

    Why GBM here?
    - It's a standard toy model
    - It keeps assumptions explicit (drift + volatility)
    - It's good enough to demonstrate monitoring & risk concepts
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dt = 1 / 252  # business-day convention
    z = rng.standard_normal(len(dates))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    prices = start_price * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=dates)


def max_drawdown(curve: pd.Series) -> float:
    """
    Maximum drawdown on an equity curve.
    Returned as a negative number (e.g., -0.23 means -23%).
    """
    peak = curve.cummax()
    drawdown = curve / peak - 1.0
    return float(drawdown.min())


def annualized_return(curve: pd.Series) -> float:
    """
    Annualized return computed from start/end curve values.
    """
    n_days = len(curve)
    if n_days <= 1:
        return np.nan
    return float((curve.iloc[-1] / curve.iloc[0]) ** (252 / n_days) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    """
    Annualized volatility from daily returns.
    """
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(252))


# -----------------------------
# Main workflow
# -----------------------------

def main() -> None:
    # A clean output folder keeps the repo readable.
    # (You can ignore it in .gitignore if you don't want to version outputs.)
    out_dir = "outputs_project1"
    os.makedirs(out_dir, exist_ok=True)

    # Timeline: ~3 years of business days
    dates = pd.bdate_range("2022-01-03", periods=3 * 252, freq="B")

    # Asset universe (toy parameters)
    # These numbers are not "true" assumptions: they are reasonable placeholders for a demo.
    assets = {
        "Equity":       {"mu": 0.08, "sigma": 0.18, "start": 100.0, "seed": 1},
        "Bonds":        {"mu": 0.03, "sigma": 0.06, "start": 100.0, "seed": 2},
        "Alternatives": {"mu": 0.06, "sigma": 0.12, "start": 100.0, "seed": 3},
    }

    # Simulate prices
    prices = pd.DataFrame(
        {
            name: simulate_gbm_prices(
                dates=dates,
                start_price=spec["start"],
                mu=spec["mu"],
                sigma=spec["sigma"],
                seed=spec["seed"],
            )
            for name, spec in assets.items()
        }
    )

    # Compute daily returns
    rets = prices.pct_change().dropna()

    # Simple strategic allocation (weights sum to 1)
    # Here the idea is "boring on purpose": a stable allocation is easy to interpret.
    weights = pd.Series({"Equity": 0.60, "Bonds": 0.30, "Alternatives": 0.10})

    # Portfolio daily returns and equity curve (base 100)
    port_ret = (rets * weights).sum(axis=1)
    port_curve = (1.0 + port_ret).cumprod() * 100.0

    # A couple of intuitive risk/performance metrics
    ann_ret = annualized_return(port_curve)
    ann_vol = annualized_volatility(port_ret)
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    mdd = max_drawdown(port_curve)

    # Optional: rolling volatility (useful in AM to "see risk regimes")
    rolling_vol_63d = port_ret.rolling(63).std(ddof=1) * np.sqrt(252)  # ~3 months

    summary = pd.Series(
        {
            "Annualized Return": ann_ret,
            "Annualized Volatility": ann_vol,
            "Sharpe (rf=0)": sharpe,
            "Max Drawdown": mdd,
        },
        name="Summary",
    )

    # Save outputs (CSV + charts)
    prices.to_csv(f"{out_dir}/prices.csv", index_label="Date")
    rets.to_csv(f"{out_dir}/returns.csv", index_label="Date")
    port_curve.to_csv(f"{out_dir}/portfolio_curve.csv", index_label="Date", header=["Portfolio"])
    rolling_vol_63d.to_csv(f"{out_dir}/rolling_vol_63d.csv", index_label="Date", header=["RollingVol(63d)"])
    summary.to_csv(f"{out_dir}/summary_metrics.csv", header=False)

    # Charts: simple, readable
    # 1) Normalized asset growth
    normalized_assets = prices / prices.iloc[0]
    normalized_assets.plot(title="Asset Growth (normalized to 1.0)")
    plt.xlabel("Date")
    plt.ylabel("Growth index")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/asset_growth.png", dpi=160)
    plt.close()

    # 2) Portfolio curve
    port_curve.plot(title="Portfolio Growth (base 100)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/portfolio_growth.png", dpi=160)
    plt.close()

    # 3) Rolling volatility
    rolling_vol_63d.plot(title="Rolling Volatility (63 business days, annualized)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/rolling_vol_63d.png", dpi=160)
    plt.close()

    # Console output (human-readable)
    print("\n=== Portfolio Monitoring Summary ===")
    print(f"Annualized return     : {ann_ret:.2%}")
    print(f"Annualized volatility : {ann_vol:.2%}")
    print(f"Sharpe (rf=0)         : {sharpe:.2f}")
    print(f"Max drawdown          : {mdd:.2%}")
    print(f"\nSaved outputs in: {out_dir}\n")


if __name__ == "__main__":
    main()

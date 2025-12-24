"""
project4_rebalancing_simulation.py

A practical rebalancing simulation, written from an asset management perspective.

What we want to illustrate:
- A strategic allocation is not only a set of weights,
  it's a discipline that needs maintenance.
- Rebalancing can control risk and prevent the portfolio from drifting too far.
- But it has a cost (transaction costs, turnover).

We compare two simple portfolios:
1) Buy & Hold: rebalance only at the start (then let weights drift)
2) Monthly Rebalance: bring weights back to target each month (with transaction costs)

All data is simulated (toy model) to keep the focus on mechanics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Simulation helper
# -----------------------------

def simulate_gbm_prices(dates, start, mu, sigma, seed=None):
    """
    Simple GBM simulation for a price series.
    Kept explicit: drift (mu), volatility (sigma), business-day convention.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    z = rng.standard_normal(len(dates))
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    return pd.Series(start * np.exp(np.cumsum(log_ret)), index=dates)


# -----------------------------
# Portfolio mechanics
# -----------------------------

def perf_stats(curve: pd.Series) -> dict:
    """
    Basic performance stats (simple but readable).
    """
    rets = curve.pct_change().dropna()
    ann_ret = (curve.iloc[-1] / curve.iloc[0]) ** (252 / len(curve)) - 1
    ann_vol = rets.std(ddof=1) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    peak = curve.cummax()
    mdd = (curve / peak - 1.0).min()

    return {
        "Annualized Return": float(ann_ret),
        "Annualized Volatility": float(ann_vol),
        "Sharpe (rf=0)": float(sharpe),
        "Max Drawdown": float(mdd),
    }


def run_buy_and_hold(prices: pd.DataFrame, target_w: pd.Series, start_value: float = 100.0) -> pd.Series:
    """
    Buy & Hold portfolio:
    we set initial weights and then do nothing.
    This is the natural baseline to compare rebalancing against.
    """
    first = prices.iloc[0]
    alloc = target_w * start_value
    units = alloc / first

    values = prices.mul(units, axis=1)
    curve = values.sum(axis=1)
    curve.name = "BuyHold"
    return curve


def run_monthly_rebalance(
    prices: pd.DataFrame,
    target_w: pd.Series,
    start_value: float = 100.0,
    tc_bps: float = 10.0,
) -> pd.Series:
    """
    Monthly rebalancing with simple transaction costs.

    tc_bps is applied on traded notional (round-trip simplified).
    The point is not microstructure accuracy,
    but to acknowledge that rebalancing is not free.
    """
    cash = 0.0
    units = pd.Series(0.0, index=prices.columns)
    curve = []

    prev_date = None
    for dt, px in prices.iterrows():
        if prev_date is None:
            # Initial allocation (first rebalance)
            port_val = start_value
            desired_alloc = target_w * port_val
            desired_units = desired_alloc / px

            trades = desired_units - units
            traded_notional = (trades.abs() * px).sum()
            cost = traded_notional * (tc_bps / 10000.0)

            units = desired_units
            cash = port_val - (units * px).sum() - cost

        else:
            # Portfolio value at current prices
            port_val = float((units * px).sum() + cash)

            # Rebalance only when month changes
            if dt.month != prev_date.month:
                desired_alloc = target_w * port_val
                desired_units = desired_alloc / px

                trades = desired_units - units
                traded_notional = (trades.abs() * px).sum()
                cost = traded_notional * (tc_bps / 10000.0)

                units = desired_units
                cash = port_val - (units * px).sum() - cost

        curve.append(float((units * px).sum() + cash))
        prev_date = dt

    curve = pd.Series(curve, index=prices.index, name="MonthlyRebalanced")
    return curve


def compute_turnover(prices: pd.DataFrame, target_w: pd.Series, start_value: float = 100.0) -> float:
    """
    A rough turnover estimate for monthly rebalancing.
    Not perfect, but gives a sense of how much trading is required.
    """
    units = pd.Series(0.0, index=prices.columns)
    cash = 0.0
    turnover = 0.0
    prev_date = None

    for dt, px in prices.iterrows():
        if prev_date is None:
            port_val = start_value
            desired_alloc = target_w * port_val
            desired_units = desired_alloc / px
            trades = desired_units - units
            turnover += float((trades.abs() * px).sum()) / port_val
            units = desired_units
            cash = port_val - (units * px).sum()

        else:
            port_val = float((units * px).sum() + cash)
            if dt.month != prev_date.month:
                desired_alloc = target_w * port_val
                desired_units = desired_alloc / px
                trades = desired_units - units
                turnover += float((trades.abs() * px).sum()) / port_val
                units = desired_units
                cash = port_val - (units * px).sum()

        prev_date = dt

    return float(turnover)


# -----------------------------
# Main
# -----------------------------

def main():
    out_dir = "outputs_project4"
    os.makedirs(out_dir, exist_ok=True)

    dates = pd.bdate_range("2022-01-03", periods=3 * 252, freq="B")

    # Simple multi-asset universe (toy parameters)
    assets = {
        "Equity":       (0.08, 0.18, 100.0, 1),
        "Bonds":        (0.03, 0.06, 100.0, 2),
        "Alternatives": (0.06, 0.12, 100.0, 3),
    }

    prices = pd.DataFrame(
        {
            name: simulate_gbm_prices(dates, start, mu, sigma, seed=seed)
            for name, (mu, sigma, start, seed) in assets.items()
        }
    )

    # Target allocation: a simple strategic mix
    target_w = pd.Series({"Equity": 0.60, "Bonds": 0.30, "Alternatives": 0.10})

    # Run both approaches
    buy_hold = run_buy_and_hold(prices, target_w, start_value=100.0)
    rebalanced = run_monthly_rebalance(prices, target_w, start_value=100.0, tc_bps=10.0)

    # Stats
    stats_bh = perf_stats(buy_hold)
    stats_rb = perf_stats(rebalanced)
    turnover_est = compute_turnover(prices, target_w, start_value=100.0)

    summary = pd.DataFrame([stats_bh, stats_rb], index=["Buy&Hold", "MonthlyRebalanced"])
    summary["Estimated Turnover (sum)"] = [0.0, turnover_est]

    # Save outputs
    prices.to_csv(f"{out_dir}/simulated_prices.csv", index_label="Date")
    buy_hold.to_csv(f"{out_dir}/curve_buy_and_hold.csv", index_label="Date", header=["BuyHold"])
    rebalanced.to_csv(f"{out_dir}/curve_monthly_rebalanced.csv", index_label="Date", header=["MonthlyRebalanced"])
    summary.to_csv(f"{out_dir}/rebalancing_summary.csv")

    # Plot: both curves on the same chart (simple comparison)
    pd.DataFrame({"Buy&Hold": buy_hold, "MonthlyRebalanced": rebalanced}).plot(
        title="Portfolio Growth: Buy & Hold vs Monthly Rebalancing"
    )
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (base 100)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/buyhold_vs_rebalanced.png", dpi=160)
    plt.close()

    # Console output (human-readable)
    print("\n=== Rebalancing Comparison ===")
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))
    print("\nInterpretation:")
    print("- Buy & Hold can drift: risk may increase if one asset dominates.")
    print("- Rebalancing enforces discipline but comes with trading costs and turnover.")
    print(f"\nSaved outputs in: {out_dir}\n")


if __name__ == "__main__":
    main()

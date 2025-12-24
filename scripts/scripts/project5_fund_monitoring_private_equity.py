"""
project5_fund_monitoring_private_equity.py

A simple "alternatives monitoring" script.

Why this exists in a multi-asset / asset management repository:
- Alternatives (including private equity) are often part of diversified portfolios.
- They come with a different type of monitoring: cash flows, NAV, and fund-level multiples.
- The goal is not to model a real fund perfectly, but to practice the mechanics:
  capital calls, distributions, residual NAV, and performance metrics such as IRR / DPI / TVPI.

All cash flows are simulated for demonstration purposes.
"""

import os
import numpy as np
import pandas as pd


# -----------------------------
# IRR helper (simple Newton method)
# -----------------------------

def irr_periodic(cashflows, guess=0.05, max_iter=200, tol=1e-8):
    """
    Periodic IRR (for equally spaced cash flows).
    This is a simple Newton-Raphson implementation.

    In real life you would often use XIRR (irregular dates),
    but periodic IRR is enough here because we simulate quarterly cash flows.
    """
    cf = np.array(cashflows, dtype=float)
    r = guess

    for _ in range(max_iter):
        # NPV and derivative
        t = np.arange(len(cf))
        denom = (1 + r) ** t
        npv = np.sum(cf / denom)
        d_npv = np.sum(-t * cf / ((1 + r) ** (t + 1)))

        # If derivative is too small, stop (avoid division by tiny number)
        if abs(d_npv) < 1e-14:
            break

        r_new = r - npv / d_npv
        if abs(r_new - r) < tol:
            return float(r_new)
        r = r_new

    return float(r)


# -----------------------------
# Simulation logic
# -----------------------------

def simulate_pe_cashflows(periods=32, seed=7):
    """
    Simulates a simple PE-style cash flow pattern over quarterly periods:
    - calls mainly in early years
    - distributions mainly in later years

    This is intentionally stylized. The point is the monitoring logic.
    """
    rng = np.random.default_rng(seed)

    calls = np.zeros(periods)
    dists = np.zeros(periods)

    for t in range(periods):
        # Early investment phase: capital calls
        if t < 10:  # ~2.5 years
            calls[t] = rng.uniform(100_000, 900_000)

        # Later phase: distributions pick up
        if t >= 10:
            dists[t] = rng.uniform(0, 900_000)

    # LP perspective: calls are negative cash flows, distributions positive
    net_cf = dists - calls
    return calls, dists, net_cf


def compute_nav_series(calls, dists):
    """
    Very simple NAV proxy:
    NAV(t) = cum_calls - cum_distributions
    (This is not how a real fund reports NAV, but it is a clean mechanical proxy.)
    """
    return np.cumsum(calls) - np.cumsum(dists)


# -----------------------------
# Main workflow
# -----------------------------

def main():
    out_dir = "outputs_project5"
    os.makedirs(out_dir, exist_ok=True)

    periods = 32  # 8 years of quarters
    dates = pd.period_range("2022Q1", periods=periods, freq="Q").to_timestamp(how="end")

    calls, dists, net_cf = simulate_pe_cashflows(periods=periods, seed=7)
    nav = compute_nav_series(calls, dists)

    df = pd.DataFrame(
        {
            "Capital_Calls": calls,
            "Distributions": dists,
            "Net_Cash_Flow": net_cf,
            "NAV_Proxy": nav,
        },
        index=dates,
    )
    df.index.name = "Date"

    # Aggregate metrics
    paid_in = float(np.sum(calls))
    distributions = float(np.sum(dists))
    residual_nav = float(max(nav[-1], 0.0))  # keep it non-negative for the toy example

    # IRR: add residual NAV as terminal distribution
    cf_for_irr = net_cf.copy()
    cf_for_irr[-1] += residual_nav

    irr_q = irr_periodic(cf_for_irr, guess=0.05)
    irr_a = (1 + irr_q) ** 4 - 1  # quarterly -> annualized

    dpi = distributions / paid_in if paid_in > 0 else np.nan
    rvpi = residual_nav / paid_in if paid_in > 0 else np.nan
    tvpi = (distributions + residual_nav) / paid_in if paid_in > 0 else np.nan

    summary = pd.Series(
        {
            "Paid-In Capital (PIC)": paid_in,
            "Distributions": distributions,
            "Residual NAV (proxy)": residual_nav,
            "IRR (quarterly)": irr_q,
            "IRR (annualized)": irr_a,
            "DPI": dpi,
            "RVPI": rvpi,
            "TVPI": tvpi,
        },
        name="PE_Metrics",
    )

    # Save outputs (simple + readable)
    df.to_csv(f"{out_dir}/pe_cashflows.csv", index_label="Date")
    summary.to_csv(f"{out_dir}/pe_metrics.csv", header=False)

    # Console output (human readable)
    print("\n=== Private Equity Fund Monitoring (Toy Example) ===")
    print("This is a simulated cash flow pattern intended to illustrate monitoring mechanics.\n")
    print(summary.to_string(float_format=lambda x: f"{x:,.6f}"))
    print(f"\nSaved outputs in: {out_dir}\n")

    print("Interpretation (simple):")
    print("- DPI tells you how much has been returned vs paid-in capital.")
    print("- RVPI is what remains 'on paper' (residual NAV proxy).")
    print("- TVPI is the total multiple (distributed + residual) / paid-in.")
    print("- IRR is sensitive to timing (cash flow schedule matters).\n")


if __name__ == "__main__":
    main()

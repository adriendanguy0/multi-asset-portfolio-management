"""
project2_risk_metrics.py

A practical risk-metrics script, written in an asset-management spirit.

The goal is not to produce a perfect risk engine,
but to compute a small set of intuitive indicators that help answer questions like:

- How risky is this portfolio on a normal day?
- What does "a bad day" look like (VaR)?
- If things go wrong, how bad can it get on average (CVaR / Expected Shortfall)?
- Is risk stable through time (rolling volatility)?

Inputs:
- By default, the script tries to load returns produced by project1 (outputs_project1/returns.csv).
- If not found, it falls back to a simple simulated return series (so the script always runs).

Outputs:
- CSV with a readable risk summary in outputs_project2/
"""

import os
import numpy as np
import pandas as pd


# -----------------------------
# Helper functions (kept simple on purpose)
# -----------------------------

def load_returns(
    path: str = "outputs_project1/returns.csv",
    portfolio_weights: dict | None = None,
) -> pd.Series:
    """
    Loads asset returns from project1 output and builds a portfolio return series.
    If the file doesn't exist, returns a simulated portfolio return series.

    This makes the script usable even if the user runs it independently.
    """
    if os.path.exists(path):
        df = pd.read_csv(path, index_col="Date", parse_dates=True)

        # If the file contains multiple assets, we build a portfolio return
        if portfolio_weights is None:
            # same weights as project1 (boring, but consistent)
            portfolio_weights = {"Equity": 0.60, "Bonds": 0.30, "Alternatives": 0.10}

        w = pd.Series(portfolio_weights)
        # Keep only columns that exist (robust if names change slightly)
        common_cols = [c for c in df.columns if c in w.index]
        if not common_cols:
            # If something is off, we just take the first column
            s = df.iloc[:, 0].astype(float)
            s.name = "Portfolio"
            return s

        port_ret = (df[common_cols].astype(float) * w[common_cols]).sum(axis=1)
        port_ret.name = "Portfolio"
        return port_ret

    # Fallback: simulated daily returns (simple, transparent)
    rng = np.random.default_rng(42)
    sim = rng.normal(loc=0.0003, scale=0.009, size=756)  # ~3y, 252*3
    idx = pd.bdate_range("2022-01-03", periods=len(sim), freq="B")
    return pd.Series(sim, index=idx, name="Portfolio")


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(252))


def annualized_return_from_daily(returns: pd.Series) -> float:
    """
    Annualized return from daily arithmetic returns.
    """
    if len(returns) < 2:
        return np.nan
    curve = (1 + returns).cumprod()
    return float((curve.iloc[-1] / curve.iloc[0]) ** (252 / len(curve)) - 1)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """
    Sharpe ratio using annualized return and annualized volatility.
    rf is expressed as annual rate (default 0 for simplicity).
    """
    ann_ret = annualized_return_from_daily(returns)
    ann_vol = annualized_volatility(returns)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return float((ann_ret - rf) / ann_vol)


def var_historical(returns: pd.Series, level: float = 0.95) -> float:
    """
    Historical Value at Risk (VaR), returned as a negative number.
    Example: -0.02 means "a 2% loss threshold".
    """
    return float(np.quantile(returns.dropna().values, 1 - level))


def cvar_historical(returns: pd.Series, level: float = 0.95) -> float:
    """
    Historical CVaR / Expected Shortfall:
    average loss given that we are in the worst (1-level) tail.
    """
    r = returns.dropna().values
    threshold = np.quantile(r, 1 - level)
    tail = r[r <= threshold]
    return float(tail.mean()) if len(tail) > 0 else np.nan


def var_parametric_normal(returns: pd.Series, level: float = 0.95) -> float:
    """
    Parametric VaR under a normality assumption.

    We keep it here because:
    - it is widely referenced
    - it is fast to compute
    - it provides a benchmark vs. historical VaR

    But we should remember it can underestimate tail risk in real markets.
    """
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    if sigma == 0:
        return np.nan

    # Try SciPy for accuracy; fallback to a simple approximation if not available.
    try:
        from scipy.stats import norm
        z = float(norm.ppf(1 - level))
    except Exception:
        # quick-and-clean approximation via numpy percent point using a dense grid
        grid = np.linspace(-6, 6, 20001)
        cdf = 0.5 * (1 + np.erf(grid / np.sqrt(2)))
        z = float(grid[np.argmin(np.abs(cdf - (1 - level)))])

    return mu + z * sigma


def rolling_volatility(returns: pd.Series, window: int = 63) -> pd.Series:
    """
    Rolling annualized volatility (default ~3 months = 63 business days).
    """
    return returns.rolling(window).std(ddof=1) * np.sqrt(252)


# -----------------------------
# Main workflow
# -----------------------------

def main() -> None:
    out_dir = "outputs_project2"
    os.makedirs(out_dir, exist_ok=True)

    port_ret = load_returns()

    # Basic stats
    ann_ret = annualized_return_from_daily(port_ret)
    ann_vol = annualized_volatility(port_ret)
    sharpe = sharpe_ratio(port_ret, rf=0.0)

    # Downside risk (daily)
    var_95_h = var_historical(port_ret, 0.95)
    var_99_h = var_historical(port_ret, 0.99)
    cvar_95_h = cvar_historical(port_ret, 0.95)
    cvar_99_h = cvar_historical(port_ret, 0.99)

    var_95_p = var_parametric_normal(port_ret, 0.95)
    var_99_p = var_parametric_normal(port_ret, 0.99)

    # Rolling vol (useful to "see" risk regimes)
    roll_vol_63 = rolling_volatility(port_ret, 63)

    summary = pd.Series(
        {
            "Annualized Return": ann_ret,
            "Annualized Volatility": ann_vol,
            "Sharpe (rf=0)": sharpe,
            "Hist VaR 95% (daily)": var_95_h,
            "Hist VaR 99% (daily)": var_99_h,
            "Hist CVaR 95% (daily)": cvar_95_h,
            "Hist CVaR 99% (daily)": cvar_99_h,
            "Param VaR 95% (daily, Normal)": var_95_p,
            "Param VaR 99% (daily, Normal)": var_99_p,
        },
        name="RiskSummary",
    )

    # Save outputs (keep them easy to read)
    summary.to_csv(f"{out_dir}/risk_summary.csv", header=False)
    port_ret.to_csv(f"{out_dir}/portfolio_returns.csv", index_label="Date", header=["PortfolioReturn"])
    roll_vol_63.to_csv(f"{out_dir}/rolling_vol_63d.csv", index_label="Date", header=["RollingVol(63d)"])

    # Console output (human-readable)
    print("\n=== Risk Metrics Summary (Portfolio) ===")
    print(f"Annualized return     : {ann_ret:.2%}")
    print(f"Annualized volatility : {ann_vol:.2%}")
    print(f"Sharpe (rf=0)         : {sharpe:.2f}")
    print("\nDownside risk (daily):")
    print(f"Hist VaR 95%          : {var_95_h:.2%}")
    print(f"Hist VaR 99%          : {var_99_h:.2%}")
    print(f"Hist CVaR 95%         : {cvar_95_h:.2%}")
    print(f"Hist CVaR 99%         : {cvar_99_h:.2%}")
    print("\nParametric VaR (Normal, daily):")
    print(f"Param VaR 95%         : {var_95_p:.2%}")
    print(f"Param VaR 99%         : {var_99_p:.2%}")
    print(f"\nSaved outputs in: {out_dir}\n")


if __name__ == "__main__":
    main()

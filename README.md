# Multi-Asset Portfolio Management – A Practical Notebook

This repository is a personal asset management project.

The goal is not to build sophisticated quantitative models or to optimize performance at all costs,
but to practice how a multi-asset portfolio can be monitored, analysed, and rebalanced
using simple tools and realistic assumptions.

This project is written from the perspective of a junior asset management analyst,
focusing on understanding portfolio behaviour, risk exposure, and the impact of allocation decisions
over time.

## Repository structure

```plaintext
Projects_Demonstration.ipynb      # Main notebook guiding the project and linking all analyses
├── scripts/
│   ├── project1_portfolio_monitoring.py
│   │   # Portfolio performance monitoring and drawdown analysis
│   ├── project2_risk_metrics.py
│   │   # Core risk indicators (volatility, Sharpe ratio, Value at Risk)
│   ├── project3_monte_carlo_black_scholes.py
│   │   # Simple option pricing and simulation tools used as analytical support
│   ├── project4_rebalancing_simulation.py
│   │   # Multi-asset allocation discipline and rebalancing logic with transaction costs
│   └── project5_fund_monitoring_private_equity.py
│       # Basic monitoring of alternative investments (IRR, DPI, RVPI, TVPI)
├── requirements.txt              # Project dependencies
├── README.md                     # Project overview, methodology and assumptions
└── .gitignore                    # Ignored files and folders
```

# What this project focuses on

Rather than covering many topics superficially, the project focuses on a few core ideas
that are central to asset management:

Multi-asset portfolio behaviour
Understanding how equities, bonds and alternative exposures interact over time.

Risk awareness
Monitoring volatility, drawdowns and downside risk instead of focusing only on returns.

Allocation discipline
Observing the impact of periodic rebalancing and transaction costs.

Simple quantitative tools
Using simulations and models as decision-support tools, not as black boxes.

## Project overview

1. Portfolio monitoring  
   Observation of portfolio performance, volatility and drawdowns.

2. Risk metrics  
   Computation of key risk indicators (volatility, Sharpe ratio, VaR).

3. Quantitative tools  
   Monte Carlo simulations and Black–Scholes pricing used as analytical support.

4. Rebalancing logic  
   Periodic rebalancing with transaction costs to illustrate allocation discipline.

5. Alternatives monitoring  
   Simple private equity-style cash flow monitoring (IRR, DPI, TVPI).

# How to run the project

This project is meant to be read and understood first, rather than executed immediately.

Option 1 – Jupyter Notebook (recommended)

The main notebook portfolio_management_demo.ipynb walks through the analysis step by step
and is the best way to understand the project logic.

You can run it:

locally with Python

or online using Google Colab

Option 2 – Individual scripts

Each script in the scripts/ folder can also be executed independently
to focus on a specific aspect of portfolio management (risk, rebalancing, simulations).

Option 3 - Environment

If you wish to run the code locally, using a virtual environment is recommended
to keep dependencies isolated and avoid conflicts.

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# or .\.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Tech stack

Python (NumPy, Pandas, Matplotlib, SciPy)  
Jupyter Notebook for analysis and visualisation


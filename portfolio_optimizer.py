
import yfinance as yf
import pandas as pd
import numpy as np
import time

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import (
    DiscreteAllocation,
    get_latest_prices
)

# ==========================================
# CONFIGURATION
# ==========================================

TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ITC.NS"
]

RISK_FREE_RATE = 0.06
INVESTMENT_AMOUNT = 10000
MAX_RETRIES = 3
TRADING_DAYS_PER_YEAR = 252

# ==========================================
# 1. GET USER INPUT
# ==========================================

print("====================================")
print("      PORTFOLIO OPTIMIZER")
print("====================================")

try:
    years = int(input("\nEnter historical period (1, 3 or 5 years): "))

    if years == 1:
        DATA_PERIOD = "1y"

    elif years == 3:
        DATA_PERIOD = "3y"

    elif years == 5:
        DATA_PERIOD = "5y"

    else:
        raise ValueError("Please enter only 1, 3 or 5.")

except ValueError as e:
    raise SystemExit(f"Invalid input: {e}")

expected_trading_days = years * TRADING_DAYS_PER_YEAR

print("\nSelected period:", DATA_PERIOD)
print("Expected trading days:", expected_trading_days)

# ==========================================
# 2. DOWNLOAD STOCK DATA
# ==========================================

print("\nDownloading stock data...")

stock_data = []

for ticker in TICKERS:

    print(f"\nDownloading {ticker}...")

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            history = yf.download(
                ticker,
                period=DATA_PERIOD,
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=30
            )

            if history.empty:
                raise ValueError("No historical data received.")

            close = history["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = pd.to_numeric(close, errors="coerce")
            close = close.dropna()

            if close.empty:
                raise ValueError("Close price data is empty.")

            if (close <= 0).any():
                raise ValueError("Invalid stock prices detected.")

            close.name = ticker
            stock_data.append(close)

            print(f"Successfully downloaded {ticker}")
            print("Available trading days:", len(close))
            print("Start date:", close.index.min())
            print("End date:", close.index.max())

            break

        except Exception as e:

            print(f"Attempt {attempt} failed: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(2)

    else:
        print(f"Skipping {ticker}: Download failed.")

# ==========================================
# 3. COMBINE AND VALIDATE DATA
# ==========================================

if len(stock_data) < 2:
    raise RuntimeError(
        "Not enough stocks downloaded. At least 2 are required."
    )

prices = pd.concat(stock_data, axis=1)

prices = prices.replace([np.inf, -np.inf], np.nan)

# Keep only dates available for all selected stocks
prices = prices.dropna()

if prices.shape[1] < 2:
    raise RuntimeError("Not enough valid stocks.")

if len(prices) < 60:
    raise RuntimeError("Insufficient historical data.")

print("\n====================================")
print("          DATA SUMMARY")
print("====================================")

print("Downloaded data shape:", prices.shape)
print("Selected stocks:", list(prices.columns))

actual_trading_days = len(prices)

print("\nExpected trading days:", expected_trading_days)
print("Actual trading days:", actual_trading_days)

if actual_trading_days < expected_trading_days * 0.80:
    print("\nWARNING: Downloaded data is significantly shorter")
    print("than the expected historical period.")

# Check actual calendar coverage
calendar_days = (
    prices.index.max() - prices.index.min()
).days

actual_years = calendar_days / 365.25

print("\nActual calendar coverage:", round(actual_years, 2), "years")

if actual_years < years * 0.80:
    print("WARNING: Historical date coverage is incomplete.")

# ==========================================
# 4. DATA DIAGNOSTICS
# ==========================================

print("\n====================================")
print("         DATA DIAGNOSTICS")
print("====================================")

print("\nDate Range:")
print("Start Date:", prices.index.min())
print("End Date:", prices.index.max())
print("Total Trading Days:", len(prices))

print("\nFirst 5 Prices:")
print(prices.head())

print("\nLatest 5 Prices:")
print(prices.tail())

# First and latest price comparison

print("\nFIRST VS LATEST PRICE:")

comparison = pd.DataFrame({
    "First Price": prices.iloc[0],
    "Latest Price": prices.iloc[-1],
    "Total Return (%)": (
        (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    ),
    "Annualized Return (%)": (
        (
            prices.iloc[-1] / prices.iloc[0]
        ) ** (365.25 / calendar_days) - 1
    ) * 100
})

print(comparison.round(2))

# Check abnormal daily movements

daily_returns = prices.pct_change()

print("\nLARGEST DAILY PRICE MOVEMENT (%):")

print(
    (daily_returns.abs().max() * 100).round(2)
)

print("\nMOVEMENTS ABOVE 15%:")

abnormal = daily_returns[
    daily_returns.abs() > 0.15
].dropna(how="all")

if abnormal.empty:
    print("No abnormal movements detected.")
else:
    print(abnormal)

# ==========================================
# 5. CALCULATE EXPECTED RETURNS
# ==========================================

print("\nCalculating expected returns...")

try:
    mu = expected_returns.mean_historical_return(prices)

    if not np.isfinite(mu.to_numpy()).all():
        raise ValueError("Invalid expected return values.")

    print("\nEXPECTED ANNUAL RETURNS")
    print("-----------------------")
    print((mu * 100).round(2).astype(str) + "%")

except Exception as e:
    raise RuntimeError(
        f"Expected return calculation failed: {e}"
    )

# ==========================================
# 6. CALCULATE COVARIANCE MATRIX
# ==========================================

print("\nCalculating covariance matrix...")

try:
    S = risk_models.CovarianceShrinkage(
        prices
    ).ledoit_wolf()

    if not np.isfinite(S.to_numpy()).all():
        raise ValueError("Invalid covariance matrix detected.")

except Exception as e:
    raise RuntimeError(
        f"Covariance calculation failed: {e}"
    )

# ==========================================
# 7. PORTFOLIO OPTIMIZATION
# ==========================================

print("\nOptimizing portfolio...")

cleaned_weights = None
optimization_method = None

# Strategy 1: Maximum Sharpe Ratio

if (mu > RISK_FREE_RATE).any():

    try:
        ef = EfficientFrontier(
            mu,
            S,
            weight_bounds=(0, 1),
            solver="CLARABEL"
        )

        ef.max_sharpe(
            risk_free_rate=RISK_FREE_RATE
        )

        cleaned_weights = ef.clean_weights()
        optimization_method = "Maximum Sharpe Ratio"

    except Exception as e:
        print("\nMaximum Sharpe optimization failed:")
        print(e)

else:
    print(
        "\nNo stock has expected returns above "
        f"{RISK_FREE_RATE * 100:.2f}%."
    )

# Strategy 2: Minimum Volatility

if cleaned_weights is None:

    print("\nUsing Minimum Volatility strategy...")

    try:
        ef = EfficientFrontier(
            mu,
            S,
            weight_bounds=(0, 1),
            solver="CLARABEL"
        )

        ef.min_volatility()

        cleaned_weights = ef.clean_weights()
        optimization_method = "Minimum Volatility"

    except Exception as e:
        raise RuntimeError(
            f"All optimization strategies failed: {e}"
        )

print("\nSelected strategy:", optimization_method)

# ==========================================
# 8. DISPLAY OPTIMIZED PORTFOLIO
# ==========================================

print("\nOPTIMIZED PORTFOLIO")
print("-------------------")

for stock, weight in cleaned_weights.items():
    print(f"{stock}: {weight * 100:.2f}%")

# ==========================================
# 9. PORTFOLIO PERFORMANCE
# ==========================================

print("\nPORTFOLIO PERFORMANCE")
print("---------------------")

try:
    ef.portfolio_performance(
        verbose=True,
        risk_free_rate=RISK_FREE_RATE
    )

except Exception as e:
    print(f"Performance calculation failed: {e}")

# ==========================================
# 10. DISCRETE STOCK ALLOCATION
# ==========================================

print(
    f"\nSTOCK ALLOCATION FOR ₹{INVESTMENT_AMOUNT:,}"
)
print("--------------------------------------------")

try:
    latest_prices = get_latest_prices(prices)

    da = DiscreteAllocation(
        cleaned_weights,
        latest_prices,
        total_portfolio_value=INVESTMENT_AMOUNT
    )

    allocation, leftover = da.greedy_portfolio()

    if allocation:
        for stock, quantity in allocation.items():
            print(f"{stock}: {quantity} shares")
    else:
        print("No shares could be allocated.")

    print(f"\nRemaining money: ₹{leftover:.2f}")

except Exception as e:
    print(f"Stock allocation failed: {e}")

# ==========================================
# 11. SAVE PORTFOLIO WEIGHTS
# ==========================================

try:
    pd.Series(cleaned_weights).to_csv(
        f"portfolio_weights_{DATA_PERIOD}.csv"
    )

    print("\nPortfolio weights saved successfully!")

except Exception as e:
    print(f"CSV saving failed: {e}")

print("\n====================================")
print(" PORTFOLIO OPTIMIZATION COMPLETED")
print("====================================")

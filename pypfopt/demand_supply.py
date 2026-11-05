"""Historical demand and supply zone detection from OHLCV price data.

This module is intentionally independent from PyPortfolioOpt's return, risk and
allocation calculations.  It is a technical-analysis helper that can be used
alongside an optimiser, not as an optimisation input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _validate_ohlcv(ohlcv: pd.DataFrame, minimum_rows: int) -> pd.DataFrame:
    """Return cleaned OHLCV data or raise a clear validation error."""
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    missing = set(_REQUIRED_COLUMNS).difference(ohlcv.columns)
    if missing:
        raise ValueError("ohlcv is missing required columns: " + ", ".join(sorted(missing)))

    data = ohlcv.loc[:, list(_REQUIRED_COLUMNS)].copy()
    for column in _REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < minimum_rows:
        raise ValueError(
            f"At least {minimum_rows} complete OHLCV rows are required; got {len(data)}"
        )
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (data["Volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    if (data["High"] < data[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError("High must be at least Open, Close, and Low on every row")
    if (data["Low"] > data[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError("Low must be at most Open, Close, and High on every row")
    return data


def _average_true_range(data: pd.DataFrame, window: int) -> pd.Series:
    previous_close = data["Close"].shift(1)
    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=1).mean().clip(
        lower=np.finfo(float).eps
    )


def _zone_strength(
    reaction_atr: float, volume_ratio: float, age: int, length: int
) -> float:
    """Score a zone from 0 to 100 using reaction, volume, and recency."""
    reaction_score = min(reaction_atr / 4.0, 1.0) * 50.0
    volume_score = min(volume_ratio / 2.0, 1.0) * 30.0
    recency_score = max(0.0, 1.0 - age / max(length, 1)) * 20.0
    return round(reaction_score + volume_score + recency_score, 2)


def _find_zones(
    data: pd.DataFrame,
    zone_type: str,
    swing_window: int,
    reaction_window: int,
    volume_window: int,
    volume_threshold: float,
    min_reaction_atr: float,
) -> list[dict[str, Any]]:
    """Find one direction of zones after data has been validated."""
    atr = _average_true_range(data, window=14)
    volume_baseline = (
        data["Volume"].rolling(volume_window, min_periods=1).median().shift(1)
    )
    zones: list[dict[str, Any]] = []

    for position in range(swing_window, len(data) - swing_window - reaction_window):
        window = data.iloc[position - swing_window : position + swing_window + 1]
        pivot = data.iloc[position]
        is_demand = zone_type == "demand"
        pivot_price = float(pivot["Low"] if is_demand else pivot["High"])
        extreme = float(window["Low"].min() if is_demand else window["High"].max())
        extremes = window["Low"] if is_demand else window["High"]
        if not np.isclose(pivot_price, extreme) or np.isclose(extremes, extreme).sum() != 1:
            continue

        baseline = volume_baseline.iloc[position]
        if not np.isfinite(baseline) or baseline <= 0:
            continue
        volume_ratio = float(pivot["Volume"] / baseline)
        if volume_ratio < volume_threshold:
            continue

        reaction_data = data.iloc[position + 1 : position + reaction_window + 1]
        if is_demand:
            reaction_price = float(reaction_data["High"].max())
            reaction_atr = (reaction_price - float(pivot["Close"])) / float(atr.iloc[position])
            lower, upper = float(pivot["Low"]), float(
                max(pivot["Open"], pivot["Close"])
            )
        else:
            reaction_price = float(reaction_data["Low"].min())
            reaction_atr = (float(pivot["Close"]) - reaction_price) / float(atr.iloc[position])
            lower, upper = float(min(pivot["Open"], pivot["Close"])), float(pivot["High"])
        if reaction_atr < min_reaction_atr:
            continue

        subsequent = data.iloc[position + reaction_window + 1 :]
        if is_demand:
            invalid = subsequent["Close"] < lower
        else:
            invalid = subsequent["Close"] > upper
        invalidation_date = None
        if invalid.any():
            invalid_position = int(np.flatnonzero(invalid.to_numpy())[0])
            invalidation_date = subsequent.index[invalid_position]

        zones.append(
            {
                "type": zone_type,
                "date": data.index[position],
                "lower": round(lower, 4),
                "upper": round(upper, 4),
                "reaction_price": round(reaction_price, 4),
                "reaction_atr": round(reaction_atr, 3),
                "volume_ratio": round(volume_ratio, 3),
                "strength": _zone_strength(
                    reaction_atr, volume_ratio, len(data) - position - 1, len(data)
                ),
                "invalidated": bool(invalid.any()),
                "invalidation_date": invalidation_date,
            }
        )
    return zones


def detect_demand_supply_zones(
    ohlcv: pd.DataFrame,
    swing_window: int = 3,
    reaction_window: int = 10,
    volume_window: int = 20,
    volume_threshold: float = 1.2,
    min_reaction_atr: float = 1.5,
) -> dict[str, Any]:
    """Detect historical demand and supply zones from daily OHLCV data.

    A demand zone is a local swing low followed by an upward move of at least
    ``min_reaction_atr`` ATRs; a supply zone is the inverse.  Both require pivot
    volume at least ``volume_threshold`` times the preceding median volume.  A
    demand zone is invalidated by a later close below its lower bound and a
    supply zone by a later close above its upper bound.

    Parameters are deliberately explicit so callers can tune the detector to a
    market or timeframe without affecting portfolio optimisation calculations.

    Returns
    -------
    dict
        ``demand_zones``, ``supply_zones``, and the strongest valid zones as
        ``preferred_buying_zone`` and ``preferred_selling_zone``. Zone bounds
        are inclusive price ranges.
    """
    if swing_window < 1 or reaction_window < 1 or volume_window < 1:
        raise ValueError(
            "swing_window, reaction_window, and volume_window must be positive"
        )
    if volume_threshold <= 0 or min_reaction_atr <= 0:
        raise ValueError("volume_threshold and min_reaction_atr must be positive")

    minimum_rows = 2 * swing_window + reaction_window + 2
    data = _validate_ohlcv(ohlcv, minimum_rows)
    common_args = {
        "swing_window": swing_window,
        "reaction_window": reaction_window,
        "volume_window": volume_window,
        "volume_threshold": volume_threshold,
        "min_reaction_atr": min_reaction_atr,
    }
    demand_zones = _find_zones(data, "demand", **common_args)
    supply_zones = _find_zones(data, "supply", **common_args)

    valid_demand = [zone for zone in demand_zones if not zone["invalidated"]]
    valid_supply = [zone for zone in supply_zones if not zone["invalidated"]]
    best = lambda zones: max(zones, key=lambda zone: zone["strength"], default=None)
    return {
        "demand_zones": demand_zones,
        "supply_zones": supply_zones,
        "preferred_buying_zone": best(valid_demand),
        "preferred_selling_zone": best(valid_supply),
    }

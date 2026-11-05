import pandas as pd
import pytest

from pypfopt.demand_supply import detect_demand_supply_zones


def _ohlcv(rows):
    """Build deterministic daily OHLCV input from compact test rows."""
    return pd.DataFrame(
        rows,
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=pd.date_range("2024-01-01", periods=len(rows), freq="D"),
    )


def test_detects_valid_demand_zone_and_preferred_buying_range():
    data = _ohlcv(
        [
            (100, 101, 99, 100, 100),
            (99, 100, 98, 99, 100),
            (98, 99, 90, 96, 300),
            (97, 104, 96, 103, 110),
            (103, 108, 102, 107, 110),
            (107, 110, 106, 109, 110),
            (109, 111, 108, 110, 110),
            (110, 112, 109, 111, 110),
        ]
    )

    result = detect_demand_supply_zones(
        data, swing_window=1, reaction_window=2, volume_window=2, min_reaction_atr=1
    )

    zone = result["preferred_buying_zone"]
    assert zone is not None
    assert zone["type"] == "demand"
    assert zone["lower"] == 90.0
    assert zone["upper"] == 98.0
    assert zone["invalidated"] is False
    assert zone["strength"] > 0


def test_detects_valid_supply_zone_and_invalidated_demand_zone():
    data = _ohlcv(
        [
            (100, 101, 99, 100, 100),
            (101, 102, 100, 101, 100),
            (103, 110, 102, 104, 300),
            (103, 104, 96, 97, 110),
            (97, 98, 92, 94, 110),
            (94, 96, 89, 91, 110),
            (91, 94, 88, 93, 110),
            (93, 95, 90, 94, 110),
        ]
    )

    result = detect_demand_supply_zones(
        data, swing_window=1, reaction_window=2, volume_window=2, min_reaction_atr=1
    )

    zone = result["preferred_selling_zone"]
    assert zone is not None
    assert zone["type"] == "supply"
    assert zone["lower"] == 103.0
    assert zone["upper"] == 110.0
    assert zone["invalidated"] is False


def test_marks_demand_zone_invalid_after_a_close_below_its_lower_bound():
    data = _ohlcv(
        [
            (100, 101, 99, 100, 100),
            (99, 100, 98, 99, 100),
            (98, 99, 90, 96, 300),
            (97, 104, 96, 103, 110),
            (103, 108, 102, 107, 110),
            (92, 94, 88, 89, 110),
            (89, 91, 87, 90, 110),
            (90, 92, 89, 91, 110),
        ]
    )

    result = detect_demand_supply_zones(
        data, swing_window=1, reaction_window=2, volume_window=2, min_reaction_atr=1
    )

    zone = result["demand_zones"][0]
    assert zone["invalidated"] is True
    assert zone["invalidation_date"] == pd.Timestamp("2024-01-06")
    assert result["preferred_buying_zone"] is None


def test_rejects_missing_ohlcv_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        detect_demand_supply_zones(pd.DataFrame({"Close": [1] * 20}))

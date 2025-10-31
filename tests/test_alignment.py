from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.data.labels import LabelGenerator
from backend.eval.metrics import join_pred_actual, leak_guard, xcorr_peak_lag
from backend.ml.predict import InMemoryPredictionStore, PredictionBundle
from backend.oms.costs import CostModel


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=8, freq="1min")
    close = pd.Series(range(100, 108), index=idx, dtype=float)
    return pd.DataFrame({"close": close})


def test_valid_at_shift(sample_ohlcv: pd.DataFrame) -> None:
    horizon = 2
    generator = LabelGenerator([horizon], epsilon_bp=5, candle_time_convention="end")
    ret_df = generator.make_returns(sample_ohlcv)
    store = InMemoryPredictionStore()
    for made_at in sample_ohlcv.index[:-horizon]:
        valid_at = made_at + pd.Timedelta(minutes=horizon)
        ret = ret_df.loc[made_at, f"ret_h{horizon}"]
        bundle = PredictionBundle(
            made_at=made_at,
            yhat_reg={horizon: float(ret)},
            probs_cls={horizon: {"up": 0.6, "flat": 0.3, "down": 0.1}},
            valid_at={horizon: valid_at},
        )
        store.append(bundle)
    joined = join_pred_actual(store, ret_df, horizon=horizon)
    pred_series = joined.set_index("made_at")["yhat"]
    actual_series = joined.set_index("valid_at")[f"actual_ret_h{horizon}"]
    lag = xcorr_peak_lag(pred_series, actual_series)
    assert lag == horizon


def test_metric_join_uses_valid_at(sample_ohlcv: pd.DataFrame) -> None:
    horizon = 1
    generator = LabelGenerator([horizon], epsilon_bp=5, candle_time_convention="end")
    ret_df = generator.make_returns(sample_ohlcv)
    store = InMemoryPredictionStore()
    for made_at in sample_ohlcv.index[:-horizon]:
        valid_at = made_at + pd.Timedelta(minutes=horizon)
        bundle = PredictionBundle(
            made_at=made_at,
            yhat_reg={horizon: 0.01},
            probs_cls={horizon: {"up": 0.5, "flat": 0.4, "down": 0.1}},
            valid_at={horizon: valid_at},
        )
        store.append(bundle)
    joined = join_pred_actual(store, ret_df, horizon=horizon)
    assert (joined["valid_at"] - joined["made_at"]).eq(pd.Timedelta(minutes=horizon)).all()


def test_leak_guard_blocks_future_cols() -> None:
    with pytest.raises(ValueError):
        leak_guard(["close", "lead_1_ret"])


def test_costs_positive_and_scaling() -> None:
    params = {
        "brokerage_bps": 3,
        "stt_bps": 10,
        "gst_bps": 1.8,
        "stamp_bps": 1,
        "exchange_bps": 3,
        "slippage_half_spread_bp": 2,
        "impact_coeff": 0.15,
    }
    model = CostModel(params)
    small = model.apply_all(price=100, qty=10, side="buy", symbol="XYZ", t=pd.Timestamp("2024-01-01"))
    big = model.apply_all(price=100, qty=100, side="buy", symbol="XYZ", t=pd.Timestamp("2024-01-01"))
    assert big.total_value > small.total_value > 0

from datetime import datetime

import numpy as np

from services.inference.next_bar_service import NextBarService


def test_next_bar_payload_includes_horizon_and_times():
    service = NextBarService()
    bars = [
        {"timestamp": "2023-01-01T00:05:00Z", "close": 105.0},
        {"timestamp": "2023-01-01T00:00:00Z", "close": 100.0},
    ]
    snapshot = {
        "symbol": "TEST",
        "bars": bars,
        "y_reg_hat": 0.01,
        "logits": [0.1, 0.0, -0.1],
    }

    result = service.predict(snapshot)

    assert result["horizon"] == 1
    obs_time = datetime.fromisoformat(result["obs_time"].replace("Z", "+00:00"))
    target_time = datetime.fromisoformat(result["target_time"].replace("Z", "+00:00"))
    assert target_time > obs_time

    expected_next_close = snapshot["bars"][1]["close"] * np.exp(snapshot["y_reg_hat"])
    assert abs(result["next_close_hat"] - expected_next_close) < 1e-9

    bands = result["bands"]
    assert abs(bands["med"] - result["next_close_hat"]) < 1e-9
    assert bands["lo"] < bands["med"] < bands["hi"]

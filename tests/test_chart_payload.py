from pathlib import Path

from services.inference.next_bar_service import NextBarService


def test_service_payload_contains_times_and_bands():
    service = NextBarService()
    snapshot = {
        "symbol": "TEST",
        "obs_time": "2023-01-01T00:00:00Z",
        "target_time": "2023-01-01T00:05:00Z",
        "close": 100.0,
        "y_reg_hat": 0.0,
        "logits": [0.0, 0.0, 0.0],
    }
    result = service.predict(snapshot)
    assert result["obs_time"] == "2023-01-01T00:00:00Z"
    assert result["target_time"] == "2023-01-01T00:05:00Z"
    assert result["c_t"] == 100.0
    assert set(result["bands"].keys()) == {"lo", "med", "hi"}
    assert "calibration" in result and {"a", "b"} <= set(result["calibration"].keys())


def test_ui_component_avoids_date_math():
    content = Path("apps/ui/src/components/NextBarPanel.tsx").read_text(encoding="utf-8")
    assert "target_time" in content
    assert ".shift(" not in content
    assert "new Date(" not in content

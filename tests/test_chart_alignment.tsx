from pathlib import Path


def test_chart_alignment_uses_target_time():
    content = Path("apps/ui/src/components/NextBarPanel.tsx").read_text(encoding="utf-8")
    assert "new Date(point.target_time)" in content
    assert ".shift(" not in content
    assert "i + 1" not in content

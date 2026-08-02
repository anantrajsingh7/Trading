"""Dashboard smoke test.

Uses Streamlit's own ``AppTest`` runner so the script is genuinely executed
rather than merely imported. Missing artefacts must produce informational
messages, never a traceback: a dashboard that crashes on a fresh checkout is
worse than no dashboard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"

pytest.importorskip("streamlit", reason="dashboard extra not installed")
pytest.importorskip("plotly", reason="dashboard extra not installed")

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["Overview", "Live scanner", "Backtest results", "Event explorer", "Trade journal", "Data & status"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(page):
    app = AppTest.from_file(str(APP), default_timeout=90)
    app.run()
    assert not app.exception, f"initial render raised: {app.exception}"

    radios = [r for r in app.radio if page in r.options]
    if not radios:
        pytest.skip("page selector not found")
    radios[0].set_value(page).run()
    assert not app.exception, f"page {page!r} raised: {app.exception}"


def test_synthetic_toggle_shows_a_warning():
    app = AppTest.from_file(str(APP), default_timeout=90)
    app.run()
    if not app.toggle:
        pytest.skip("synthetic toggle not present")
    app.toggle[0].set_value(True).run()
    assert not app.exception
    banners = [e.value for e in app.error]
    assert any("SYNTHETIC" in str(b).upper() for b in banners), (
        "synthetic mode must be visually unmistakable"
    )

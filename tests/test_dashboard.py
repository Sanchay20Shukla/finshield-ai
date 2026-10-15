"""Smoke each delivered Streamlit route using only the included sample."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("page",["Executive dashboard","Alert queue","Score transaction","Investigations","Model monitoring","Transaction 360°"])
def test_dashboard_page(page, monkeypatch):
    monkeypatch.setenv("PAYSIM_PATH",str(ROOT/"data/sample/paysim_demo.csv"))
    monkeypatch.delenv("DATABASE_URL",raising=False)
    from src import config
    monkeypatch.setattr(config,"CSV_PATH",ROOT/"data/sample/paysim_demo.csv")
    app=AppTest.from_file(str(ROOT/"dashboard/app.py"),default_timeout=60).run()
    if page!="Executive dashboard":
        app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [error.message for error in app.exception]
    if page=="Score transaction":
        app.button[0].click().run()
        assert not app.exception
        assert len(app.metric)==3

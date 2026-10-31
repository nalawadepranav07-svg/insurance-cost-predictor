"""CoverWise — Streamlit entry point.  Launch with:  streamlit run frontend/app.py"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the project root importable regardless of the directory Streamlit is launched from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="CoverWise — Medical cost estimator", page_icon="🛡️", layout="wide")

from frontend import components as ui  # noqa: E402
from frontend import views  # noqa: E402

NAV = {"home": "Home", "predict": "Predict", "insights": "Insights", "about": "About"}
RENDER = {"home": views.render_home, "predict": views.render_predict, "result": views.render_result,
          "insights": views.render_insights, "about": views.render_about}


def main() -> None:
    ui.inject_css()
    st.session_state.setdefault("page", "home")
    page = st.session_state["page"]
    ui.nav_bar(NAV, "predict" if page == "result" else page, views.go)
    try:
        RENDER.get(page, views.render_home)()
    except Exception:  # noqa: BLE001 - last-resort guard so users never see a raw traceback
        logging.getLogger("coverwise.app").exception("Unhandled error while rendering page '%s'", page)
        ui.error_box("Something went wrong", "The page could not be displayed. Please reload and try again.")


main()

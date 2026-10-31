"""Reusable CoverWise UI components. Each returns/renders HTML built from trusted values."""
from __future__ import annotations

from html import escape
from typing import Callable, Iterable

import streamlit as st

from frontend.styles import CSS
from src.utils.formatting import format_usd, mean_std

DISCLAIMER_TEXT = "Educational estimate only — not an insurer quote."


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def brand() -> None:
    _html('<div class="cw-brand"><div class="cw-logo" aria-hidden="true">C</div><div>'
          '<div class="cw-wordmark">CoverWise</div><div class="cw-sub">Medical cost estimator</div></div></div>')


def nav_bar(pages: dict[str, str], active: str, on_select: Callable[[str], None]) -> None:
    """Top navigation: brand on the left, one button per page."""
    cols = st.columns([2.2] + [1] * len(pages))
    with cols[0]:
        brand()
    for col, (key, label) in zip(cols[1:], pages.items()):
        with col:
            st.button(label, key=f"nav_{key}", type="primary" if key == active else "secondary",
                      use_container_width=True, on_click=on_select, args=(key,))
    _html('<div class="cw-navrule"></div>')


def section_title(title: str, lead: str = "") -> None:
    _html(f'<div class="cw-h2">{escape(title)}</div>' + (f'<p class="cw-lead">{escape(lead)}</p>' if lead else ""))


def feature_card(icon: str, title: str, body: str) -> None:
    _html(f'<div class="cw-card"><div class="cw-icon">{escape(icon)}</div><h3>{escape(title)}</h3>'
          f'<p>{escape(body)}</p></div>')


def metric_card(label: str, value: str, hint: str = "") -> None:
    _html(f'<div class="cw-metric"><div class="label">{escape(label)}</div>'
          f'<div class="value">{escape(value)}</div>'
          + (f'<div class="hint">{escape(hint)}</div>' if hint else "") + "</div>")


def disclaimer_banner(text: str = DISCLAIMER_TEXT) -> None:
    _html(f'<div class="cw-disclaimer" role="note">⚠ {escape(text)}</div>')


def error_box(title: str, detail: str) -> None:
    _html(f'<div class="cw-error" role="alert"><b>{escape(title)}</b><br>{escape(detail)}</div>')


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="cw-pill {kind}">{escape(text)}</span>'


def estimate_card(amount: float, unit: str, model_name: str, model_id: str) -> None:
    _html(f'<div class="cw-estimate"><div class="label">Estimated annual medical charges</div>'
          f'<div class="amount">{escape(format_usd(amount))}</div><div class="unit">{escape(unit)}</div>'
          f'<div style="margin-top:16px">{pill(model_name, "info")} {pill("model " + model_id, "ok")}</div></div>')


def estimate_placeholder(message: str) -> None:
    _html(f'<div class="cw-placeholder"><div class="big">$ — — —</div><p>{escape(message)}</p></div>')


def input_summary(items: Iterable[tuple[str, str]]) -> None:
    cells = "".join(f"<div><span>{escape(k)}</span><b>{escape(str(v))}</b></div>" for k, v in items)
    _html(f'<div class="cw-summary">{cells}</div>')


def comparison_table(rows: list[dict]) -> None:
    """Model comparison from real saved CV results (mean ± std)."""
    head = "<tr><th>Model</th><th>MAE (USD)</th><th>RMSE (USD)</th><th>R²</th><th>Status</th></tr>"
    body = ""
    for r in rows:
        cv = r["cv"]
        status = pill("Selected", "ok") if r["selected"] else ""
        body += (f'<tr class="{"sel" if r["selected"] else ""}"><td>{escape(r["model"])}</td>'
                 f'<td>{mean_std(cv["mae"]["mean"], cv["mae"]["std"])}</td>'
                 f'<td>{mean_std(cv["rmse"]["mean"], cv["rmse"]["std"])}</td>'
                 f'<td>{mean_std(cv["r2"]["mean"], cv["r2"]["std"], 3)}</td><td>{status}</td></tr>')
    _html(f'<div class="cw-table-wrap"><table class="cw-table"><thead>{head}</thead><tbody>{body}</tbody></table></div>')


def footer() -> None:
    _html('<div class="cw-foot">CoverWise is an educational machine-learning project. '
          'Estimates are not insurance quotes or medical advice.</div>')

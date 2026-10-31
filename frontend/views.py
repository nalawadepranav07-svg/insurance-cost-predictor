"""CoverWise screens. Each ``render_*`` function draws one page; prediction logic stays in src/."""
from __future__ import annotations

import streamlit as st

from frontend import components as ui
from frontend.data import get_config, get_predictor, load_report_json, load_subgroups
from src.prediction.predictor import PredictionError, ValidationError
from src.utils.config import resolve
from src.utils.formatting import format_usd

LABELS = {"sex": {"female": "Female", "male": "Male"},
          "smoker": {"no": "Non-smoker", "yes": "Smoker"},
          "region": {"northeast": "Northeast", "northwest": "Northwest",
                     "southeast": "Southeast", "southwest": "Southwest"}}


def go(page: str) -> None:
    st.session_state["page"] = page


def _meta() -> dict | None:
    return load_report_json(get_config()["paths"]["metadata_path"])


def _comparison() -> list[dict] | None:
    data = load_report_json("reports/model_comparison/model_comparison.json")
    return data["models"] if data else None


# ----------------------------------------------------------------------------- Home
def render_home() -> None:
    predictor, error = get_predictor()
    ui._html(
        '<div class="cw-hero"><span class="cw-eyebrow">Educational ML project</span>'
        "<h1>Understand what medical insurance charges can look like.</h1>"
        "<p>Enter six simple details and CoverWise returns a machine-learning estimate of annual medical "
        "charges, learned from a public historical dataset — with the model’s accuracy shown openly.</p>"
        '<div class="cw-hero-notice">Educational estimate only — not an insurer quote.</div></div>')
    st.write("")
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        st.button("Start prediction →", type="primary", on_click=go, args=("predict",), use_container_width=True)
    with c2:
        st.button("See model insights", on_click=go, args=("insights",), use_container_width=True)
    if error:
        st.write("")
        ui.error_box("Model not available", error)

    ui.section_title("How the estimate is made",
                     "A transparent three-step pipeline — nothing is hard-coded, every number comes from the trained model.")
    a, b, c = st.columns(3)
    with a:
        ui.feature_card("1", "You enter six details", "Age, sex, BMI, number of children, smoking status and region — "
                        "the same fields as the dataset.")
    with b:
        ui.feature_card("2", "A trained model estimates", "A scikit-learn pipeline, selected with 5-fold "
                        "cross-validation, predicts historical annual charges in US dollars.")
    with c:
        ui.feature_card("3", "You see the context", "Results include the model’s typical error on unseen data, "
                        "so the estimate is never presented as exact.")

    meta = _meta()
    if meta:
        ui.section_title("Model at a glance")
        m = meta["holdout_metrics"]
        k1, k2, k3, k4 = st.columns(4)
        with k1: ui.metric_card("Selected model", meta["model_name"], f"ID {meta['model_id']}")
        with k2: ui.metric_card("Hold-out MAE", format_usd(m["mae"]), "average absolute error, unseen data")
        with k3: ui.metric_card("Hold-out R²", f"{m['r2']:.3f}", "share of variance explained")
        with k4: ui.metric_card("Training rows", f"{meta['train_rows']:,}", f"of {meta['dataset_rows']:,} records")
    ui.footer()


# ----------------------------------------------------------------------------- Predict
def render_predict() -> None:
    cfg = get_config()
    predictor, error = get_predictor()
    bounds = cfg["schema"]["bounds"]
    prev = st.session_state.get("last_inputs", {})
    cats = cfg["schema"]["categories"]

    ui.section_title("Your details", "All fields are required. Values outside the dataset’s range are rejected.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (years)", bounds["age"][0], bounds["age"][1],
                                      int(prev.get("age", 35)), 1,
                                      help=f"Whole years, {bounds['age'][0]}–{bounds['age'][1]} (the dataset’s adult range).")
                bmi = st.number_input("BMI (kg/m²)", float(bounds["bmi"][0]), float(bounds["bmi"][1]),
                                      float(prev.get("bmi", 27.0)), 0.1, format="%.1f",
                                      help="Body-mass index = weight (kg) ÷ height (m)². Accepted range "
                                           f"{bounds['bmi'][0]}–{bounds['bmi'][1]}.")
                smoker = st.radio("Smoking status", cats["smoker"], horizontal=True,
                                  index=cats["smoker"].index(prev.get("smoker", "no")),
                                  format_func=LABELS["smoker"].get,
                                  help="Whether the person currently smokes.")
            with c2:
                sex = st.selectbox("Sex", cats["sex"], index=cats["sex"].index(prev.get("sex", "female")),
                                   format_func=LABELS["sex"].get,
                                   help="Categories match the dataset (female / male).")
                children = st.number_input("Children covered", bounds["children"][0], bounds["children"][1],
                                           int(prev.get("children", 0)), 1,
                                           help=f"Number of dependants, {bounds['children'][0]}–{bounds['children'][1]}.")
                region = st.selectbox("Residential region (US)", cats["region"],
                                      index=cats["region"].index(prev.get("region", "northeast")),
                                      format_func=LABELS["region"].get)
            submitted = st.form_submit_button("Estimate charges", type="primary", use_container_width=True,
                                              disabled=predictor is None)
        if predictor is None:
            ui.error_box("Model not available", error or "Unknown error.")
        if submitted and predictor is not None:
            raw = {"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region}
            try:
                result = predictor.predict(raw)
            except ValidationError as exc:
                ui.error_box("Please check your inputs",
                             " • ".join(f"{k}: {v}" for k, v in exc.errors.items()))
            except PredictionError as exc:
                ui.error_box("Prediction failed", f"{exc} Please try again or retrain the model.")
            else:
                st.session_state["result"] = result
                st.session_state["last_inputs"] = result.inputs
                go("result")
                st.rerun()
    with right:
        ui.estimate_placeholder("Your estimate will appear here after you submit the form.")
        st.write("")
        ui.disclaimer_banner()
    ui.footer()


# ----------------------------------------------------------------------------- Result
def render_result() -> None:
    result = st.session_state.get("result")
    if result is None:
        ui.section_title("No estimate yet", "Submit your details first.")
        st.button("Go to the form", type="primary", on_click=go, args=("predict",))
        return
    meta = _meta() or {}
    ui.section_title("Your estimate")
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.estimate_card(result.estimate, result.target_unit, result.model_name, result.model_id)
        st.write("")
        ui.disclaimer_banner(result.disclaimer)
    with right:
        i = result.inputs
        st.markdown("#### Details you submitted")
        ui.input_summary([
            ("Age", f"{i['age']} yrs"), ("Sex", LABELS["sex"][i["sex"]]), ("BMI", f"{i['bmi']:.1f}"),
            ("Children", i["children"]), ("Smoking", LABELS["smoker"][i["smoker"]]),
            ("Region", LABELS["region"][i["region"]]),
        ])
        if meta.get("holdout_metrics"):
            mae = meta["holdout_metrics"]["mae"]
            st.write("")
            st.markdown("#### How much to trust it")
            st.markdown(
                f"On data the model had never seen, its predictions were off by **{format_usd(mae)}** on average "
                f"(mean absolute error), so treat this figure as a rough guide rather than an exact amount. "
                f"Individual errors can be much larger, especially for higher-cost cases.")
    st.write("")
    a, b, _ = st.columns([1.2, 1.2, 3])
    with a:
        st.button("Make another prediction", type="primary", on_click=go, args=("predict",), use_container_width=True)
    with b:
        st.button("View model insights", on_click=go, args=("insights",), use_container_width=True)
    ui.footer()


# ----------------------------------------------------------------------------- Insights
def _interpretation(comparison: list[dict], meta: dict) -> list[str]:
    """Plain-language statements generated from the real saved numbers (no hand-typed results)."""
    ranked = sorted(comparison, key=lambda r: r["cv"]["rmse"]["mean"])
    best, worst = ranked[0], ranked[-1]
    linear = next((r for r in comparison if r["model"] == "Linear Regression"), None)
    lines = [
        f"**{best['model']}** had the lowest mean 5-fold CV RMSE "
        f"({format_usd(best['cv']['rmse']['mean'])}), versus {format_usd(worst['cv']['rmse']['mean'])} for "
        f"{worst['model']}, the highest."
    ]
    if linear and best["model"] != linear["model"]:
        gap = linear["cv"]["rmse"]["mean"] - best["cv"]["rmse"]["mean"]
        lines.append(f"Compared with plain Linear Regression, the selected model lowers CV RMSE by {format_usd(gap)}.")
    top_std = best["cv"]["rmse"]["std"]
    runner = ranked[1]
    if abs(runner["cv"]["rmse"]["mean"] - best["cv"]["rmse"]["mean"]) < top_std:
        lines.append(f"The runner-up ({runner['model']}) is within one fold-to-fold standard deviation "
                     f"({format_usd(top_std)}) of the winner, so the ranking between them is not clear-cut.")
    imp = meta.get("feature_importance", {}).get("values", [])
    if imp:
        lines.append(f"Permutation importance ranks **{imp[0]['feature']}** first, then **{imp[1]['feature']}** — "
                     "these describe what the model relies on to predict, not what causes higher costs.")
    return lines


def render_insights() -> None:
    meta, comparison = _meta(), _comparison()
    ui.section_title("Model insights", "Everything below is read from the saved training and evaluation reports.")
    if not meta or not comparison:
        ui.error_box("No evaluation reports found",
                     "Run `python -m scripts.train_model --overwrite` to train the models and create the reports.")
        ui.footer()
        return

    h = meta["holdout_metrics"]
    ci = meta["holdout_metrics_95ci"]
    st.markdown("#### Final hold-out test results (selected model, unseen data)")
    c1, c2, c3 = st.columns(3)
    with c1: ui.metric_card("MAE", format_usd(h["mae"]), f"95% CI {format_usd(ci['mae'][0])}–{format_usd(ci['mae'][1])}")
    with c2: ui.metric_card("RMSE", format_usd(h["rmse"]), f"95% CI {format_usd(ci['rmse'][0])}–{format_usd(ci['rmse'][1])}")
    with c3: ui.metric_card("R²", f"{h['r2']:.3f}", f"95% CI {ci['r2'][0]:.3f}–{ci['r2'][1]:.3f}")
    st.caption(f"{meta['holdout_rows']} hold-out records, never used for model selection or tuning. "
               "R² is not an accuracy percentage.")

    st.markdown("#### Model comparison — 5-fold cross-validation on the training split")
    ui.comparison_table(comparison)
    st.caption("Mean ± standard deviation across the same 5 shuffled folds for every model. Lower MAE/RMSE and "
               "higher R² are better. Tuned models use nested cross-validation.")
    for line in _interpretation(comparison, meta):
        st.markdown(f"- {line}")

    left, right = st.columns(2, gap="large")
    imp = meta.get("feature_importance", {}).get("values", [])
    with left:
        st.markdown("#### Feature importance")
        if imp:
            import pandas as pd
            df = pd.DataFrame(imp).set_index("feature")[["mean"]].rename(columns={"mean": "RMSE increase when shuffled (USD)"})
            st.bar_chart(df, color="#2F5BEA", horizontal=True)
            st.caption(f"Method: {meta['feature_importance']['method']}. Predictive association only — not causation.")
    with right:
        st.markdown("#### Predicted vs actual (hold-out)")
        fig = resolve(get_config(), "reports/figures/holdout_predicted_vs_actual.png")
        if fig.exists():
            st.image(str(fig), use_container_width=True)

    sub = load_subgroups()
    if sub is not None:
        with st.expander("Error by subgroup (hold-out) — where the model is less reliable"):
            show = sub.rename(columns={"group": "Group", "level": "Level", "n": "Records", "mae": "MAE (USD)",
                                       "rmse": "RMSE (USD)", "r2": "R²", "mean_error": "Mean error (USD)",
                                       "small_sample": "Small sample (<30)"})
            st.dataframe(show.round(2), use_container_width=True, hide_index=True)
            st.caption("Positive mean error = over-prediction. Subgroup differences are signals to investigate, "
                       "not proof of unfairness or a fairness certification.")
    ui.disclaimer_banner()
    ui.footer()


# ----------------------------------------------------------------------------- About
def render_about() -> None:
    meta = _meta() or {}
    ui.section_title("About CoverWise")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("""
**Project overview.** CoverWise is an educational machine-learning project that predicts historical
medical insurance charges from six personal attributes and shows how well that prediction works.

**Dataset.** The *Medical Cost Personal Dataset* (Kaggle `mirichoi0218/insurance`, originally from
Brett Lantz’s *Machine Learning with R*): 1,338 records with `age`, `sex`, `bmi`, `children`, `smoker`,
`region` and `charges` (USD). It is a small, US-based public teaching dataset and is not representative of any
insurer’s book of business.
""")
        if meta:
            st.markdown(f"**Trained model:** {meta['model_name']} · ID `{meta['model_id']}` · "
                        f"trained {meta['trained_utc'][:10]} · scikit-learn {meta['versions']['scikit-learn']}")
    with b:
        st.markdown("""
**Methodology.**
1. Validate the raw data (schema, ranges, duplicates, missing values).
2. Hold out 20 % of rows for a final test, stratified by smoking status.
3. Compare Linear Regression, a log-target Linear Regression, Random Forest and Gradient Boosting on identical
   shuffled 5-fold splits, with preprocessing fitted inside each fold.
4. Tune the tree models with randomised search nested inside the cross-validation.
5. Select by lowest mean CV RMSE, then score once on the hold-out set.
""")
    ui.disclaimer_banner("Educational disclaimer: outputs are model estimates learned from a small historical "
                         "dataset. They are not insurance quotes, premiums, medical advice or predictions of "
                         "your actual costs.")
    with st.expander("Limitations and fairness considerations"):
        st.markdown("""
- Only six features; real costs depend on health history, plan design, provider prices and much more.
- Small dataset (1,338 rows); some subgroups have few records, so their error estimates are noisy.
- `sex`, `smoker` and `region` are used as predictive inputs because they are in the dataset; observed
  associations are not causal, and using such attributes in real pricing may be restricted by law.
- Heavy-tailed charges: the model is least reliable for unusually high-cost cases.
""")
    ui.footer()

# Design implementation status
**Figma:** no Figma integration was available in the build environment, so **no Figma file was created**. The design brief
(`README.md`, `FIGMA_PROMPT.md`) was implemented directly in Streamlit: design tokens in `frontend/styles.py`
(navy/cobalt palette, muted green = valid, amber = disclaimer, 16 px card radius, 8 px spacing, subtle shadows), reusable
components in `frontend/components.py`. Use the brief with Figma AI/a designer if an editable canvas is needed.

**Deviations from the brief (deliberate):** the predictor is a multi-page flow (Home → Predict → Result, plus Insights and
About) as requested in the project prompt, rather than the brief's single two-panel page; the Predict page keeps the
brief's left form / right placeholder layout. The form label for `sex` is "Sex" to match the dataset's category.
**Not verified:** the layout was verified by automated headless tests and a live server start, but not by visual inspection
in a browser (none was available). Please review it in your browser and adjust `styles.py` as needed.

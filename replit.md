# AIRECRUIT — Intelligence Platform

## Project Overview

**AIRECRUIT** is an AI-powered recruitment intelligence web application built with Streamlit. It automates resume screening, role classification, skill extraction, and candidate-job matching using multiple machine learning models.

**Tech stack:** Python 3.12 · Streamlit 1.41.1 · Scikit-Learn 1.6.1 · XGBoost · NLTK · Plotly · pdfplumber

**Run command:** `streamlit run app.py` (port 5000)

**Main file:** `app.py` (~4,100 lines) — single-file Streamlit app containing all pages, CSS, JS patches, and ML logic.

---

## App Structure

The app has 8 pages navigated via sidebar radio:

| Page | Key Features |
|---|---|
| 🏠 Home | Overview stats, feature cards, system status |
| 🧠 Resume Intelligence | Single Screener · Batch Upload · CV Compare · History · Shortlist |
| 💼 Recommendation Engine | 3-tab interface: Live Matcher · JD Analyzer · Salary Estimator |
| 📊 Candidate Analytics | Category distribution · Skill analysis · Live session · Candidate Lookup |
| 🔬 Workflow | 8-stage pipeline timeline with detail cards |
| ⚙️ ML Models | Model comparison · Hyperparameter Tuning · Feature Importance · Leaderboard · Live Predictor |
| 📈 Performance Metrics | Validation results · Confusion matrices · ROC curves |
| 👥 About | Team, supervisor, project info |

---

## Models

All in `models/` directory:

| Key | File | Accuracy |
|---|---|---|
| `svm` | `svm_pipeline.pkl` | 98.75% |
| `lr` | `logistic_regression_pipeline.pkl` | 99.22% |
| `xgb` | `xgboost_pipeline.pkl` | 97.35% |
| `rf` | `random_forest_pipeline.pkl` | 98.44% |
| `tfidf_match` | `tfidf_match.pkl` | — (matching) |
| `kmeans` | `kmeans.pkl` | — (clustering) |

---

## Features Added (Current Version)

- **Candidate Fit Score Radar** — 5-dimension radar (Confidence, ATS Readiness, Skill Coverage, Resume Depth, Role Certainty) in Single Screener
- **Export HTML Report** — Downloadable styled candidate report from Single Screener
- **Model Agreement Heatmap** — In Live Predictor (ML Models page)
- **RF model** — Random Forest now loaded and participates in Live Predictor, Leaderboard, Model Comparison
- **Recommendation Engine — 3-tab restructure:**
  - Tab 1 (Live Matcher): Match Score Breakdown chart · Skill Coverage Matrix heatmap · Shortlist Manager · Candidate Comparison (side-by-side table + radar overlay + winner declaration)
  - Tab 2 (JD Analyzer): Skill detection by category · Seniority detection · Experience extraction · Role category prediction · JD Keyword Frequency chart
  - Tab 3 (Salary Estimator): PKR + USD salary range by role/experience/education/skills · Estimation factors table

---

## CSS / JS Architecture

- All global CSS in one `st.markdown("""<style>...</style>""", unsafe_allow_html=True)` block (~800 lines)
- Theme: Dark Gold Luxury (`#0B0D10` background, `#D6B25E` gold accent)
- **Sidebar proxy button fix** via `streamlit.components.v1.html(...)` at startup:
  - Creates `<button id="_airecruit_sb_proxy">` directly on `document.body` (outside React VDOM)
  - Proxy delegates click to the real hidden `[data-testid="collapsedControl"]` element
  - Since the proxy is NOT in React's component tree, VDOM reconciliation can never remove it
  - Polls every 150ms + MutationObserver for immediate response
  - Also styles the `«` collapse button inside the sidebar

---

## Known Constraints

- Streamlit 1.41.1 deprecation warnings (non-breaking): `use_container_width`, `st.components.v1.html`, empty radio labels
- RF model was pickled with sklearn 1.9.0 vs installed 1.6.1 — loads with `InconsistentVersionWarning` but works
- `[class*="css"] { background-color: #0B0D10 !important; }` is a broad override — be careful when adding new UI elements that need different backgrounds

---

## User Preferences

- Dark gold luxury aesthetic — never use light backgrounds or bright/neon colors
- No emojis in code comments
- Keep all pages in `app.py` (single file architecture)
- University project: SE-CD-638 · Dr. Aamir Arsalan · Team: Hania, Areeba, Hajra Sarwar

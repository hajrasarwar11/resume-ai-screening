# AIRECRUIT — Intelligence Platform

## Project Overview

**AIRECRUIT** is an AI-powered recruitment intelligence web application built with Streamlit. It automates resume screening, role classification, skill extraction, and candidate-job matching using multiple machine learning models.

**Tech stack:** Python 3.12 · Streamlit 1.41.1 · Scikit-Learn 1.6.1 · XGBoost · NLTK · Plotly · pdfplumber

**Run command:** `streamlit run app.py` (port 5000)

**Main file:** `app.py` (~3,800 lines) — single-file Streamlit app containing all pages, CSS, JS patches, and ML logic.

---

## App Structure

The app has 8 pages navigated via sidebar radio:

| Page | Key Features |
|---|---|
| 🏠 Home | Overview stats, feature cards, system status |
| 🧠 Resume Intelligence | Single Screener · Batch Upload · CV Compare · History · Shortlist |
| 💼 Recommendation Engine | JD matcher · Score breakdown · Skill Coverage Matrix · Shortlist Manager · Candidate Comparison |
| 📊 Candidate Analytics | Category distribution · Skill analysis · Live session · Candidate Lookup |
| 🔬 Workflow | 8-stage pipeline timeline with detail cards |
| ⚙️ ML Models | Model comparison · Hyperparameter Tuning · Feature Importance · Leaderboard |
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

## Recent Features Added

- **Candidate Fit Score Radar** — 5-dimension radar (Confidence, ATS Readiness, Skill Coverage, Resume Depth, Role Certainty) in Single Screener
- **Export HTML Report** — Downloadable styled candidate report from Single Screener
- **Model Agreement Heatmap** — In Live Predictor (ML Models page)
- **RF model** — Random Forest now loaded and participates in Live Predictor, Leaderboard, Model Comparison
- **Recommendation Engine enrichment:**
  - Match Score Breakdown (stacked bar: cosine / experience / skills)
  - Skill Coverage Matrix (heatmap: JD skills × top candidates)
  - Shortlist Manager (add, view, export, clear)
  - Candidate Comparison (side-by-side table + radar overlay + winner declaration)

---

## CSS / JS Architecture

- All global CSS in one `st.markdown("""<style>...</style>""", unsafe_allow_html=True)` block (~800 lines)
- Theme: Dark Gold Luxury (`#0B0D10` background, `#D6B25E` gold accent)
- Sidebar JS patch via `streamlit.components.v1.html(...)` at startup:
  - Changes `«»` icons on collapse/expand buttons
  - **Moves `[data-testid="collapsedControl"]` to `document.body`** (escapes sidebar transform container) so the `»` expand button stays visible when sidebar is collapsed
  - Runs every 200ms + MutationObserver

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

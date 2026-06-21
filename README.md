<div align="center">

# ◆ AIRECRUIT
### AI-Powered Intelligent Resume Screening & Candidate Role Recommendation System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-gold?style=for-the-badge&logo=streamlit&logoColor=white&color=D6B25E)](https://resume-ai-screening-mynrd3qwpwnijtjrzhhhs3.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

*University Project · SE-CD-638 Machine Learning · Dr. Aamir Arsalan · Semester VI · 2026–2027*

---

</div>

## 📌 Overview

**AIRECRUIT** is an end-to-end ML-powered recruitment intelligence platform that automates resume screening, role classification, skill extraction, and candidate-job matching. It eliminates manual bias in hiring by using NLP and multiple classification algorithms to objectively rank and recommend candidates.

**🔗 Live App:** [resume-ai-screening-mynrd3qwpwnijtjrzhhhs3.streamlit.app](https://resume-ai-screening-mynrd3qwpwnijtjrzhhhs3.streamlit.app/)

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Resume Intelligence** | NLP pipeline: clean → tokenize → TF-IDF vectorize → classify |
| 🔍 **Single Screener** | Paste or upload a single resume for instant role prediction |
| 📦 **Batch Upload** | Screen multiple PDF resumes simultaneously |
| 🔄 **CV Compare** | Side-by-side comparison of two candidates with a winner declaration |
| 💼 **Recommendation Engine** | Match any job description against 2,400+ resumes using cosine similarity |
| 📊 **Candidate Analytics** | Category distribution, skill frequency, experience histograms, radar profiles |
| 🔴 **Live Session Tracking** | Real-time dashboard of all resumes screened in the current session |
| 🔍 **Candidate Lookup** | Search the full candidate pool by skill keyword or role category |
| 📈 **Performance Metrics** | Full model evaluation — accuracy, F1, ROC-AUC, confusion matrices |
| 📌 **Shortlist & Export** | Pin candidates, add recruiter notes, export to CSV |

---

## 🔄 Full End-to-End Process Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AIRECRUIT PIPELINE                                 │
│                                                                             │
│  INPUT                    PROCESSING                        OUTPUT          │
│  ─────                    ──────────                        ──────          │
│                                                                             │
│  Raw Resume Text  ──▶  1. Text Cleaning (regex, lower-case)                 │
│  (paste / PDF)         2. Tokenisation & Stop-word removal                  │
│                         3. TF-IDF Vectorisation (3K features)               │
│                         4. Feature Engineering                              │
│                            · Experience Years (regex)                       │
│                            · Skill Count (vocab match)                      │
│                            · Education Level (keyword)                      │
│                         5. ML Classification                                │
│                            · SVM  ──────── 98.75% accuracy  ──▶  Role Label │
│                            · Logistic Regression  99.22%                    │
│                            · XGBoost              97.35%                    │
│                            · Random Forest        98.44%                    │
│                         6. ATS Score (skill-keyword density)  ──▶  ATS %   │
│                         7. Top-5 Probabilities  ──────────────▶  Ranked     │
│                                                                  Predictions │
│                                                                             │
│  Job Description  ──▶  1. TF-IDF Vectorisation (5K features)               │
│                         2. Cosine Similarity vs 2,400+ resume vectors       │
│                         3. Category Filter (optional)                       │
│                         4. Composite Ranking                                │
│                            60% Match Score + 20% Experience + 20% Skills   │
│                         ──────────────────────────────────────▶  Ranked     │
│                                                                  Candidate   │
│                                                                  Shortlist  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Project Structure

```
airecruit/
│
├── app.py                          # Main Streamlit application (3000+ lines)
│
├── models/                         # Serialised trained models
│   ├── svm_pipeline.pkl            # SVM + TF-IDF pipeline (best accuracy: 98.75%)
│   ├── logistic_regression_pipeline.pkl  # LR pipeline (99.22%)
│   ├── xgboost_pipeline.pkl        # XGBoost pipeline
│   ├── xgboost_tuned_randomsearch.pkl    # XGBoost tuned via RandomizedSearchCV
│   ├── kmeans.pkl                  # K-Means clustering (k=10)
│   ├── tfidf_match.pkl             # TF-IDF vectoriser for semantic matching
│   ├── preprocessor.pkl            # ColumnTransformer for feature engineering
│   └── cluster_preprocessor.pkl   # Preprocessor for clustering
│
├── data/                           # Datasets
│   ├── cleaned_resume_data.csv     # 2,400+ cleaned resumes (25 categories)
│   └── processed_jobs.csv          # Processed job descriptions
│
├── outputs/                        # Model outputs & evaluation artefacts
│   └── clustered_resumes.csv       # K-Means cluster assignments
│
├── plots/                          # EDA & evaluation visualisations
│   ├── roc_curves.png              # ROC-AUC curves for all models
│   ├── confusion_matrices.png      # Per-model confusion matrices
│   └── ...                        # Additional EDA plots
│
├── notebooks/                      # Jupyter ML pipeline
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_Preprocessing_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb     # GridSearchCV + RandomizedSearchCV
│   └── 04_Evaluation_Recommendation_System.ipynb
│
├── project_report.html             # Full academic project report
├── requirements.txt                # Python dependencies
└── .streamlit/config.toml          # Streamlit server configuration
```

---

## 🤖 ML Models & Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **99.22%** | 99.3% | 99.2% | 99.2% |
| **SVM (Linear Kernel)** | **98.75%** | 98.8% | 98.7% | 98.7% |
| **Random Forest** | **98.44%** | 98.5% | 98.4% | 98.4% |
| **XGBoost (Tuned)** | **97.35%** | 97.4% | 97.3% | 97.3% |

All models trained on 80/20 stratified split · Cross-validated (cv=5) · Tuned via GridSearchCV / RandomizedSearchCV

---

## 🔍 How to Search / Use the App

### Single Resume Screening
1. Go to **Resume Intelligence** → **Single Screener** tab
2. Paste resume text or upload a PDF
3. Select a model (SVM recommended)
4. Click **Analyse Resume** → get role prediction, ATS score, skill breakdown, top-5 predictions

### Batch Screening
1. Go to **Resume Intelligence** → **Batch Upload** tab
2. Upload multiple PDF files
3. System screens all resumes and ranks them by confidence score
4. Export results as CSV or shortlist top candidates

### Job Description Matching
1. Go to **Recommendation Engine**
2. Paste a job description (or select a sample JD)
3. Optionally filter by role category
4. Set how many top candidates you want
5. Click **Find Best Candidates** → ranked shortlist with match %, shared keywords, and missing skills

### Candidate Analytics & Search
1. Go to **Candidate Analytics**
2. Use **Category Distribution**, **Skill Analysis**, or **Radar Profile** tabs for dataset insights
3. Use **Live Session** tab to see all resumes screened in the current session with live charts
4. Use **Candidate Lookup** tab to search the full 2,400+ dataset by skill or role category

---

## 🔁 Where Does Output Go?

```
Resume Screened
      │
      ├──▶  Session History (Resume Intelligence → History tab)
      │          Stores: role, confidence %, ATS %, skills, top-5, model used
      │
      ├──▶  Candidate Analytics → 🔴 Live Session tab
      │          Shows: role distribution chart, confidence vs ATS scatter, per-candidate log
      │
      ├──▶  Shortlist (📌 Add to Shortlist button)
      │          Stores: pinned candidates with recruiter notes
      │          Export: "Export Shortlist (CSV)" button in History tab
      │
      └──▶  Recommendation Engine (🔍 Find Similar button)
                 Sends resume text as JD to find similar candidates from the full pool
```

---

## 🧠 NLP & Feature Engineering Details

### Text Cleaning Pipeline
```python
raw_text
  → lowercase
  → remove URLs, emails, special characters
  → remove stop words (NLTK)
  → TF-IDF vectorisation (max_features=3000 for classification, 5000 for matching)
```

### Feature Engineering
| Feature | Method |
|---|---|
| **Experience Years** | Regex: finds first number before "year"/"yr" |
| **Skill Count** | Vocab match against 45-term skill list |
| **Education Level** | Keyword detection: PhD=3, Master=2, Bachelor=1, HS=0 |

### Recommendation Scoring Formula
```
Composite Score = 0.60 × CosineSimilarity(JD_tfidf, Resume_tfidf)
               + 0.20 × normalize(Experience_Years)
               + 0.20 × normalize(Skill_Count)
```

---

## 📊 Dataset

- **Source:** Kaggle Resume Dataset
- **Size:** 2,400+ labelled resumes
- **Categories:** 25 job roles (Data Science, Java Developer, Python Developer, HR, Marketing, etc.)
- **Balance:** Roughly uniform distribution across categories
- **Format:** Raw resume text → cleaned → TF-IDF vectorised

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | Streamlit 1.41.1 · Custom CSS (Dark Gold Luxury Theme) |
| **ML Framework** | Scikit-Learn 1.6.1 · XGBoost 2.1.3 |
| **NLP** | NLTK · TF-IDF Vectoriser · Cosine Similarity |
| **Data** | Pandas 2.2.2 · NumPy 2.2.0 · SciPy 1.15.0 |
| **Visualisation** | Plotly 5.22.0 · Matplotlib 3.10.0 · Seaborn 0.13.2 |
| **PDF Parsing** | pdfplumber 0.10.3 |
| **Model Serialisation** | Joblib 1.4.2 |
| **Language** | Python 3.12 |

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone <repo-url>
cd airecruit

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (first run only)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 4. Launch the app
streamlit run app.py
```

App will be available at `http://localhost:8501`

---

## 🔬 Notebook Pipeline

Run notebooks in order to reproduce the full ML pipeline:

| Notebook | Purpose |
|---|---|
| `01_EDA.ipynb` | Explore dataset — category distribution, text length, skill frequency |
| `02_Preprocessing_Feature_Engineering.ipynb` | Clean text, extract features, prepare training data |
| `03_Model_Training.ipynb` | Train SVM, LR, RF, XGBoost with hyperparameter tuning |
| `04_Evaluation_Recommendation_System.ipynb` | Evaluate models, build recommendation engine, export results |

---

## 👥 Team

| Member | Role |
|---|---|
| **Hania** | ML Model Development, Feature Engineering, XGBoost Tuning |
| **Areeba** | NLP Pipeline, TF-IDF Implementation, Recommendation Engine |
| **Hajra Sarwar** | EDA, Data Preprocessing, Model Evaluation & Reporting |

*Supervised by Dr. Aamir Arsalan · SE-CD-638 Machine Learning · Semester VI · 2026–2027*

---

## 📄 Academic Report

The full academic report is available as `project_report.html` in the repository root. Open it in any browser for a complete writeup including mathematical formulations, model comparisons, and future work.

---

<div align="center">

**◆ AIRECRUIT · Intelligence Platform**
*AI-Powered Intelligent Resume Screening & Candidate Role Recommendation using Machine Learning*

</div>

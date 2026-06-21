<div align="center">

# ◆ AIRECRUIT
### AI-Powered Intelligent Resume Screening & Candidate Role Recommendation System

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.3-green?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.75%25-gold?style=for-the-badge)](https://github.com)

*University Project · SE-CD-638 Machine Learning · Dr. Aamir Arsalan · Semester VI · 2026–2027*

---

</div>

## 📌 Overview

**AIRECRUIT** is an end-to-end, AI-powered recruitment intelligence platform that automates resume screening, role classification, skill extraction, and candidate-job matching. It eliminates manual bias in hiring by using NLP and four classification algorithms to objectively rank and recommend candidates across **42 job categories** from a corpus of **2,400+ labelled resumes**, achieving **98.75% accuracy** with the SVM model.

The system is deployed as a full Streamlit web application with 9 interactive pages, covering everything from single resume screening to a complete Recruiter Workspace with pipeline tracking.

---

## 🤖 Model Results

| Rank | Model | Accuracy | Precision | Recall | F1-Score | Notes |
|---|---|---|---|---|---|---|
| 🥇 | **SVM (LinearSVC + Calibration)** | **98.75%** | **98.76%** | **98.75%** | **98.74%** | Production model |
| 🥈 | Random Forest | 98.15% | 98.17% | 98.15% | 98.15% | GridSearchCV cv=5 |
| 🥉 | Logistic Regression | 97.00% | 97.03% | 97.00% | 97.00% | L2 / saga solver |
| ④ | XGBoost | 96.50% | 96.52% | 96.50% | 96.50% | RandomizedSearchCV |

> **Version Compatibility:** Models were trained with scikit-learn 1.9.0. A `safe_predict()` helper handles cross-version compatibility for LR (`multi_class` attribute removal) and XGBoost (`__sklearn_tags__` incompatibility) using native API fallbacks.

---

## ✨ Application Pages

| Page | What It Does | Key Features |
|---|---|---|
| 🏠 **Home** | Platform overview | Stats, problem statement, animated neural net, objectives |
| 🔬 **Workflow** | How the system works | Interactive pipeline, step-by-step architecture, feature engineering flow |
| 🧠 **Resume Intelligence** | Screen any resume | Single screener, batch upload, CV compare, ATS score, interview Q generator, HTML export |
| 💼 **Recommendation Engine** | Find the best candidate | JD matcher, top-N ranked candidates, skill gap matrix, shortlist manager, salary estimator |
| ⚙️ **ML Models** | Understand the algorithms | Model cards, live predictor (all 4 models), hyperparameter config, feature importance |
| 📊 **Candidate Analytics** | Dataset insights | Category distribution, skill frequency, experience histograms, candidate lookup |
| 📈 **Performance Metrics** | Model comparison | Leaderboard cards, comparison radar, confusion matrix, per-category F1, ROC curves |
| 🚀 **Recruiter Workspace** | Manage hiring pipeline | Bookmark candidates, pipeline board, interview notes + ratings, CSV export, to-do list |
| 👥 **About** | Project context | Team profiles, tech stack, project structure, future enhancements |

---

## 🔄 End-to-End Pipeline

```
Raw Resume (PDF / Text)
        │
        ▼
  Text Extraction (pdfplumber)
        │
        ▼
  NLP Preprocessing (NLTK)
  ├── Lowercasing
  ├── URL & email removal
  ├── Punctuation removal
  ├── Stopword removal
  └── Lemmatization (WordNetLemmatizer)
        │
        ▼
  Feature Engineering
  ├── TF-IDF Vectorizer (3,000 features, ngram 1-2)
  ├── Experience Years (regex extraction)
  ├── Skill Count (vocabulary matching, 80+ terms)
  └── Education Level (keyword-to-ordinal encoding)
        │
        ▼
  ColumnTransformer Pipeline
  ├── TF-IDF branch (sparse text features)
  └── StandardScaler branch (numeric features)
        │
        ▼
  Classification (4 Models)
  ├── SVM (LinearSVC + CalibratedClassifierCV) — 98.75% ← Production
  ├── Random Forest (GridSearchCV)              — 98.15%
  ├── Logistic Regression (saga, L2)            — 97.00%
  └── XGBoost (RandomizedSearchCV)              — 96.50%
        │
        ▼
  Recommendation Engine
  ├── Cosine Similarity — TF-IDF match (60%)
  ├── Experience Score (20%)
  └── Skill Coverage Score (20%)
        │
        ▼
  Output: Role · Confidence · ATS Score · Top-5 · Skill Gap · Salary Range
```

---

## 🧠 Feature Engineering Details

| Feature | Method | Dimensions |
|---|---|---|
| TF-IDF Text | `TfidfVectorizer(max_features=3000, ngram_range=(1,2), sublinear_tf=True)` | 3,000 |
| Experience Years | Regex pattern matching on resume text | 1 |
| Skill Count | Vocabulary matching against 80+ technical terms | 1 |
| Education Level | Keyword detection → ordinal (0–4) | 1 |
| **Total Input** | `ColumnTransformer` combined output | **3,003** |

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Kaggle Resume Dataset + custom augmentation |
| Total Records | 2,400+ labelled resumes |
| Job Categories | 42 |
| Train Split | 80% stratified (~1,920 samples) |
| Test Split | 20% stratified (~480 samples) |
| Language | English |

**42 Categories include:** Data Science · Python Developer · Java Developer · Web Designing · DevOps Engineer · Network Security Engineer · SQL Developer · Blockchain · Testing · Business Analyst · Mechanical Engineer · Civil Engineer · Electrical Engineering · HR · Sales · Finance · Project Manager · Teacher · and 24 more.

---

## 🚀 Recruiter Workspace

A fully integrated hiring management module — no external ATS needed:

| Tab | Features |
|---|---|
| **➕ Add Candidate** | Name, role, experience, education, confidence %, skills, stage, priority, source, salary, email, note |
| **📋 Pipeline Board** | Visual funnel bar · Search/filter · Stage updates (Applied → Offer) · Remove candidates |
| **📝 Interview Notes** | Per-candidate notes with stage context · Quick rating sliders (Technical, Communication, Culture Fit, Problem Solving) |
| **📤 Export & Reports** | CSV shortlist download · Pipeline donut chart · Hiring to-do list |

KPI strip at top: Total Bookmarked · In Interview · Offers Made · Rejected

---

## 🔮 Future Enhancements

| # | Feature | Category |
|---|---|---|
| 🗄️ | **Persistent Cloud Database** — Supabase PostgreSQL for permanent data across sessions | Infrastructure |
| 📧 | **Automated Shortlisting Emails** — Auto-draft + recruiter approval before Gmail SMTP send | Automation |
| 🧠 | **GPT-Powered Resume Feedback** — LLM-generated personalised CV improvement suggestions | AI / LLM |
| 🌐 | **LinkedIn Profile Scraper** — Paste URL → auto-extract candidate details | Integration |
| 🎥 | **Video Interview Analyser** — Whisper STT + sentiment scoring of recorded interviews | AI / NLP |
| ⚖️ | **Bias & Fairness Auditor** — Detect demographic bias in shortlisting decisions | Ethics / ML |
| 📱 | **Mobile Candidate Portal** — Candidate-facing app for submission & status tracking | UX / Product |
| 🔁 | **Active Learning & Retraining** — Recruiter feedback loop to continuously improve models | MLOps |

---

## 📁 Project Structure

```
AIRECRUIT/
├── app.py                                     # Streamlit application (4,600+ lines)
├── requirements.txt                           # Pinned dependencies
├── PROJECT_REPORT.md                          # Full academic project report
│
├── .streamlit/
│   └── config.toml                            # Dark theme + port 5000 config
│
├── notebooks/
│   ├── 01_EDA.ipynb                           # Exploratory Data Analysis
│   ├── 02_Preprocessing_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb                # Training + hyperparameter tuning
│   └── 04_Evaluation_Recommendation_System.ipynb
│
├── models/
│   ├── svm_pipeline.pkl                       # Production model — 98.75%
│   ├── logistic_regression_pipeline.pkl
│   ├── random_forest_pipeline.pkl
│   ├── xgboost_pipeline.pkl
│   ├── tfidf_match.pkl                        # Recommendation engine TF-IDF
│   ├── kmeans.pkl                             # Clustering model (k=10)
│   └── preprocessor.pkl
│
├── data/
│   ├── cleaned_resume_data.csv                # 2,400+ labelled resumes
│   ├── processed_jobs.csv
│   ├── train_df.csv / test_df.csv
│   └── ...
│
├── outputs/
│   ├── final_resume_data_with_clusters.csv
│   ├── final_jobs_data.csv
│   └── model_comparison_results.csv
│
└── plots/
    ├── confusion_matrix.png
    ├── roc_curves.png
    └── ...
```

---

## 🛠 Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.12 |
| Web Framework | Streamlit | 1.41.1 |
| ML / Classification | Scikit-Learn | 1.6.1 |
| Gradient Boosting | XGBoost | 2.1.3 |
| NLP | NLTK | 3.8.1 |
| Data Processing | Pandas | 2.2.2 |
| Numerical | NumPy | 2.2.0 |
| Scientific | SciPy | 1.15.0 |
| Visualisations | Plotly | 5.22.0 |
| Visualisations | Matplotlib + Seaborn | 3.10.0 / 0.13.2 |
| PDF Parsing | pdfplumber | 0.10.3 |
| Model Persistence | Joblib | 1.4.2 |

---

## ⚙️ Running Locally

```bash
# 1. Clone the repository
git clone <repository-url>
cd AIRECRUIT

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (first run only)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# 4. Launch the app
streamlit run app.py --server.port 5000 --server.address 0.0.0.0
```

Then open `http://localhost:5000` in your browser.

---

## 👥 Team

| Member | Role | Key Contributions |
|---|---|---|
| **Hania** | Data Engineer & Preprocessing Lead | Dataset curation, NLP pipeline, EDA, TF-IDF strategy, ColumnTransformer design |
| **Areeba** | Machine Learning Engineer | SVM, RF, LR, XGBoost implementation, hyperparameter tuning, clustering, evaluation |
| **Hajra Sarwar** | System Integration & Deployment Lead | Streamlit app (9 pages), recommendation engine, UI/UX, Recruiter Workspace, deployment |

**Supervisor:** Dr. Aamir Arsalan &nbsp;·&nbsp; **Course:** SE-CD-638 Machine Learning &nbsp;·&nbsp; **Semester:** VI, 2026–2027

---

## 📖 Documentation

For the full academic project report including literature review, methodology, mathematical formulations, experimental analysis, and references, see **[PROJECT_REPORT.md](PROJECT_REPORT.md)**.

---

<div align="center">

*AIRECRUIT · SE-CD-638 · Machine Learning Research Project · 2026–2027*

</div>

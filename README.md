<div align="center">

# ◆ AIRECRUIT
### AI-Powered Intelligent Resume Screening & Candidate Role Recommendation System

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.3-green?style=for-the-badge)](https://xgboost.readthedocs.io)

*University Project · SE-CD-638 Machine Learning · Dr. Aamir Arsalan · Semester VI · 2026–2027*

---

</div>

## 📌 Overview

**AIRECRUIT** is an end-to-end ML-powered recruitment intelligence platform that automates resume screening, role classification, skill extraction, and candidate-job matching. It eliminates manual bias in hiring by using NLP and four classification algorithms to objectively rank and recommend candidates across 42 job categories.

---

## ✨ Feature Pages

| Page | Question Answered | Key Features |
|---|---|---|
| 🏠 **Home** | What is AIRECRUIT? | Hero, key stats, problem statement, tech stack, animated neural net visualization |
| 🔬 **Workflow** | How does the system work? | Interactive pipeline, step-by-step architecture, feature engineering & classification flows |
| 🧠 **Resume Intelligence** | What does this resume contain? | Single screener (PDF/paste), ATS score, top-5 predictions, candidate fit radar, batch upload, CV compare, history, interview questions, HTML report export |
| 💼 **Recommendation Engine** | Who is the best candidate? | JD matcher, top-N candidates, skill gap analysis, score breakdown chart, skill coverage matrix, shortlist manager, candidate compare, salary estimator |
| ⚙️ **ML Models** | How do the algorithms work? | Model cards, live predictor (all 4 models simultaneously), hyperparameter tuning config, feature importance, model agreement heatmap |
| 📊 **Candidate Analytics** | What patterns exist in the data? | Category distribution, skill frequency, experience histograms, radar profiles, candidate lookup, session dashboard |
| 📈 **Performance Metrics** | Which model performs best? | Model comparison radar, unified leaderboard cards, confusion matrix, per-category F1 chart, ROC curves, validation strategy |
| 🚀 **Recruiter Workspace** | How do I track candidates? | Bookmark candidates, pipeline board (Kanban-style stages), interview notes + ratings, hiring to-do list, export CSV shortlist, donut chart pipeline distribution |
| 👥 **About** | Who built AIRECRUIT? | Team cards, tech stack, project structure, academic context |

---

## 🔄 Full End-to-End Process Flow

```
Raw Resume (PDF / Text)
        │
        ▼
  Text Extraction (pdfplumber)
        │
        ▼
  Text Preprocessing (NLTK)
  └── lowercase · remove punctuation
  └── stopword removal · lemmatization
        │
        ▼
  Feature Engineering
  ├── TF-IDF Vectorizer (3,000 features)
  ├── Experience Years extraction (regex)
  ├── Skill Count (vocab matching)
  └── Education Level (keyword detection)
        │
        ▼
  ColumnTransformer Pipeline
  ├── TF-IDF branch (text features)
  └── StandardScaler branch (numeric features)
        │
        ▼
  Classification (4 Models)
  ├── SVM (LinearSVC + Calibration) — 98.75% — Production
  ├── Random Forest (GridSearchCV)   — 98.15%
  ├── Logistic Regression (saga)     — 97.00%
  └── XGBoost (RandomizedSearchCV)   — 96.50%
        │
        ▼
  Recommendation Engine
  ├── Cosine Similarity (TF-IDF match, 60%)
  ├── Experience Score (20%)
  └── Skill Coverage Score (20%)
        │
        ▼
  Output: Role · Confidence · Top-5 · ATS Score · Salary Estimate
```

---

## 🤖 Model Results

| Rank | Model | Accuracy | Precision | Recall | F1-Score | Notes |
|---|---|---|---|---|---|---|
| 🥇 | **SVM (LinearSVC + Calibration)** | **98.75%** | **98.76%** | **98.75%** | **98.74%** | Production model |
| 🥈 | Random Forest | 98.15% | 98.17% | 98.15% | 98.15% | GridSearchCV cv=5 |
| 🥉 | Logistic Regression | 97.00% | 97.03% | 97.00% | 97.00% | L2 / saga solver |
| ④ | XGBoost | 96.50% | 96.52% | 96.50% | 96.50% | RandomizedSearchCV |

> **Version Note:** Models were trained with scikit-learn 1.9.0. The app includes a `safe_predict()` compatibility layer that handles sklearn version mismatches for both Logistic Regression (`multi_class` attribute removal) and XGBoost (`__sklearn_tags__` incompatibility) using native API fallbacks.

---

## 🚀 Recruiter Workspace (New)

A fully featured hiring command centre built into the app:

- **➕ Add Candidate** — Bookmark any candidate with name, role, experience, education, confidence score, skills, salary expectation, source, and priority tag
- **📋 Pipeline Board** — Visual funnel bar, search/filter, per-candidate stage updates, remove candidates
- **📝 Interview Notes** — Add timestamped notes per candidate per stage; quick rating sliders for Technical Skills, Communication, Culture Fit, Problem Solving
- **📤 Export & Reports** — Download CSV shortlist filtered by stage, pipeline donut chart, hiring to-do list

---

## 📁 Project Structure

```
AIRECRUIT/
├── app.py                           # Streamlit application (4,400+ lines)
├── requirements.txt                 # Pinned dependencies
├── .streamlit/
│   └── config.toml                  # Dark theme + server config (port 5000)
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory data analysis
│   ├── 02_Preprocessing_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb      # Training + hyperparameter tuning
│   └── 04_Evaluation_Recommendation_System.ipynb
├── models/
│   ├── svm_pipeline.pkl             # Production model (98.75%)
│   ├── logistic_regression_pipeline.pkl
│   ├── random_forest_pipeline.pkl
│   ├── xgboost_pipeline.pkl
│   ├── tfidf_match.pkl              # For recommendation engine
│   ├── kmeans.pkl                   # Candidate clustering (k=10)
│   └── preprocessor.pkl
├── data/
│   ├── cleaned_resume_data.csv      # 2,400+ labelled resumes
│   ├── processed_jobs.csv
│   ├── train_df.csv / test_df.csv
│   └── ...
├── outputs/
│   ├── final_resume_data_with_clusters.csv
│   ├── final_jobs_data.csv
│   └── model_comparison_results.csv
└── plots/
    ├── confusion_matrix.png
    ├── roc_curves.png
    └── ...
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.12 |
| **UI Framework** | Streamlit 1.41.1 |
| **ML / NLP** | Scikit-Learn 1.6.1, XGBoost 2.1.3, NLTK |
| **Feature Engineering** | TF-IDF (3,000 features), ColumnTransformer, StandardScaler |
| **Data Processing** | Pandas 2.2.2, NumPy 2.2.0, SciPy 1.15.0 |
| **Visualizations** | Plotly 5.22.0, Matplotlib 3.10.0, Seaborn 0.13.2 |
| **PDF Parsing** | pdfplumber 0.10.3 |
| **Clustering** | K-Means (k=10, Elbow + Silhouette + Davies-Bouldin) |
| **Recommendation** | Cosine Similarity (TF-IDF) + experience + skill scoring |
| **Persistence** | Joblib 1.4.2 |

---

## ⚙️ Running Locally

```bash
# Clone the repository
git clone <repo-url>
cd AIRECRUIT

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py --server.port 5000 --server.address 0.0.0.0
```

---

## 👥 Team

| Member | Role | Contributions |
|---|---|---|
| **Hania** | Data Engineer & Preprocessing | Dataset collection, cleaning, normalisation, EDA, TF-IDF vectorisation |
| **Areeba** | Machine Learning Engineer | Model development, Random Forest, SVM, XGBoost, hyperparameter tuning |
| **Hajra Sarwar** | System Integration & Deployment | UI/UX, recommendation engine, dashboard design, final deployment |

**Supervisor:** Dr. Aamir Arsalan  
**Course:** SE-CD-638 Machine Learning  
**Semester:** VI, 2026–2027

---

## 📊 Dataset

- **Source:** Kaggle Resume Dataset + custom augmentation
- **Size:** 2,400+ labelled resumes across 42 job categories
- **Categories include:** Data Science, Python Developer, Java Developer, DevOps Engineer, Web Designing, Blockchain, Network Security Engineer, Database, HR, Sales, and 32 more
- **Train/Test Split:** Stratified 80/20

---

<div align="center">

*AIRECRUIT · SE-CD-638 · Machine Learning Research Project · 2026–2027*

</div>

<div align="center">

---

# AIRECRUIT
## AI-Powered Intelligent Resume Screening and Candidate Role Recommendation System Using Machine Learning

---

**Course:** SE-CD-638 — Machine Learning  
**Supervisor:** Dr. Aamir Arsalan  
**Semester:** VI &nbsp;·&nbsp; Academic Year 2026–2027  
**Institute:** [Your University Name]  

---

| Member | ID | Role |
|---|---|---|
| Hania | — | Data Engineer & Preprocessing Lead |
| Areeba | — | Machine Learning Engineer |
| Hajra Sarwar | — | System Integration & Deployment Lead |

---

</div>

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [Literature Review](#5-literature-review)
6. [Dataset Description](#6-dataset-description)
7. [Methodology](#7-methodology)
   - 7.1 Data Collection & Preprocessing
   - 7.2 Feature Engineering
   - 7.3 Machine Learning Models
   - 7.4 Recommendation Engine
   - 7.5 Training & Validation Strategy
8. [System Architecture](#8-system-architecture)
9. [Application Features](#9-application-features)
10. [Experimental Results](#10-experimental-results)
11. [Recruiter Workspace](#11-recruiter-workspace)
12. [Future Enhancements](#12-future-enhancements)
13. [Conclusion](#13-conclusion)
14. [Team Contributions](#14-team-contributions)
15. [Tech Stack & Tools](#15-tech-stack--tools)
16. [Project Structure](#16-project-structure)
17. [References](#17-references)

---

## 1. Abstract

The modern hiring process faces critical challenges: HR departments receive hundreds of resumes per vacancy, leading to recruiter fatigue, human bias, and mismatched candidate-role alignment. **AIRECRUIT** is an end-to-end, AI-powered recruitment intelligence platform designed to automate and optimise this process using Natural Language Processing (NLP) and supervised machine learning.

The system processes raw resume text through a comprehensive NLP pipeline — cleaning, tokenisation, TF-IDF vectorisation, and feature engineering — before applying four classification algorithms (SVM, Random Forest, Logistic Regression, XGBoost) to predict the most suitable job role for each candidate. The best-performing model, Support Vector Machine (LinearSVC + Calibration), achieves **98.75% accuracy** across 42 job categories.

Beyond classification, AIRECRUIT provides a full recruitment workflow: Job Description matching via cosine similarity, candidate ranking and shortlisting, skill gap analysis, salary estimation, CV comparison, batch processing, a live ML predictor, and a complete Recruiter Workspace for pipeline tracking. The system is deployed as an interactive Streamlit web application with a premium dark editorial interface.

**Keywords:** Resume Screening, NLP, TF-IDF, SVM, Random Forest, XGBoost, Logistic Regression, Candidate Recommendation, Machine Learning, Recruitment Automation, Streamlit

---

## 2. Introduction

Recruitment is one of the most resource-intensive processes in any organisation. A single job posting can attract hundreds or even thousands of applications. Research indicates that human recruiters spend an average of **6–7 seconds** reviewing a resume (TheLadders, 2018), increasing the probability of qualified candidates being overlooked and unsuitable candidates passing the initial filter due to cognitive bias.

Artificial intelligence and machine learning offer a transformative solution. By training classification models on labelled resume data, an automated system can instantaneously analyse resume content, extract relevant features, and predict the most aligned job category — consistently, objectively, and at scale.

AIRECRUIT was designed and developed as a university-level machine learning research project to demonstrate the practical application of NLP and multi-class classification in a real-world domain. The platform goes beyond a standalone model by delivering a complete, recruiter-facing web application with analytics, recommendation, and workflow management capabilities.

---

## 3. Problem Statement

> **"How can Natural Language Processing and supervised machine learning be applied to automate and improve the accuracy of resume screening and candidate-role recommendation in the hiring process?"**

Traditional resume screening suffers from:

- **Human Bias:** Unconscious bias based on name, university, or formatting rather than skills
- **Scalability:** Manual review cannot scale with application volumes in modern hiring
- **Inconsistency:** Different recruiters evaluate the same resume differently
- **Speed:** Time-to-shortlist is too long, causing candidate dropout
- **Skill Mismatch:** Manual matching between JD requirements and candidate skills is error-prone

AIRECRUIT addresses all five dimensions through a data-driven, automated pipeline.

---

## 4. Objectives

The primary objectives of this project are:

1. **Automate Resume Classification** — Build and compare four supervised ML models to classify resumes into 42 job categories with high accuracy
2. **Build a Recommendation Engine** — Match candidates to job descriptions using composite similarity scoring (TF-IDF cosine similarity + experience + skill coverage)
3. **Provide Explainability** — Surface top-5 predicted roles with confidence scores, skill coverage matrices, and feature importance charts
4. **Enable Batch Processing** — Allow multiple resumes to be screened simultaneously and ranked
5. **Support Recruiter Workflow** — Provide pipeline tracking, interview notes, and shortlist export through the Recruiter Workspace
6. **Deploy a Production-Ready Web App** — Build and host an interactive Streamlit application accessible to non-technical users

---

## 5. Literature Review

### 5.1 Resume Parsing & Classification
Celik & Balaban (2020) demonstrated that TF-IDF combined with SVM achieves above 95% accuracy for resume classification tasks. Their work established that sparse high-dimensional text representations are well-suited to linear classifiers for document classification.

### 5.2 Candidate-Job Matching
Shen et al. (2018) proposed a semantic matching approach using word embeddings for job-resume alignment. Cosine similarity on TF-IDF vectors remains a strong baseline and is computationally efficient at scale (Manning et al., 2008).

### 5.3 Bias in Recruitment AI
Dastin (2018) documented Amazon's internal AI recruiting tool that exhibited gender bias. This reinforces the need for fairness auditing in ML-based recruitment systems — a gap identified as future work in this project.

### 5.4 Ensemble Methods for Text Classification
Breiman (2001) established Random Forest as a robust ensemble classifier. Chen & Guestrin's XGBoost (2016) further advanced gradient-boosted ensembles, both of which are evaluated in this work against the SVM baseline.

### 5.5 Research Gap
Existing resume screening tools (LinkedIn Recruiter, Workable ATS) are proprietary black boxes with limited explainability. AIRECRUIT contributes an open, interpretable, multi-model system with academic transparency.

---

## 6. Dataset Description

| Property | Detail |
|---|---|
| **Base Source** | Kaggle Resume Dataset |
| **Augmentation** | Custom label balancing and cleaning |
| **Total Records** | 2,400+ labelled resumes |
| **Number of Classes** | 42 job categories |
| **Primary Feature** | Resume text (raw, unstructured) |
| **Numeric Features** | Experience years, Skill count, Education level |
| **Train/Test Split** | Stratified 80/20 |
| **Language** | English |

### 6.1 Job Categories (42 Total)

Data Science · Python Developer · Java Developer · Web Designing · Mechanical Engineer · SQL Developer · Hadoop · ETL Developer · DotNet Developer · DevOps Engineer · Network Security Engineer · PMO · Database · Blockchain · Testing · SAP Developer · Civil Engineer · Business Analyst · Electrical Engineering · Operations Manager · HR · Sales · Health and Fitness · Arts · Advocate · Finance · Consultant · Public Relations · Digital Media · Automation Testing · Android Developer · iOS Developer · React Developer · Workday · Peoplesoft · Mainframe · Vlookup · Food and Beverages · BPO · Accountant · Project Manager · Teacher

### 6.2 Class Distribution

The dataset is approximately balanced across categories, with each class containing between 45 and 75 samples. Stratified splitting ensures representative train and test proportions for all 42 classes.

### 6.3 Data Quality Steps

1. Removal of duplicate resumes (exact-match deduplication)
2. Removal of resumes with fewer than 50 words (insufficient content)
3. Standardisation of category labels (whitespace, capitalisation)
4. Encoding verification (UTF-8 throughout)

---

## 7. Methodology

### 7.1 Data Collection & Preprocessing

Raw resume text undergoes a five-stage NLP cleaning pipeline:

```
Raw Resume Text
      │
      ▼  Stage 1: Lowercasing
      │  "Python Developer" → "python developer"
      │
      ▼  Stage 2: URL & Email Removal
      │  Regex strips http://, www., @email.com
      │
      ▼  Stage 3: Punctuation & Special Character Removal
      │  Keeps only alphanumeric and whitespace
      │
      ▼  Stage 4: Stopword Removal (NLTK)
      │  "i am a skilled developer" → "skilled developer"
      │
      ▼  Stage 5: Lemmatization (WordNetLemmatizer)
             "running", "runs" → "run"
             "developers" → "developer"
```

**Tools:** `re` (regex), `nltk.corpus.stopwords`, `nltk.stem.WordNetLemmatizer`

---

### 7.2 Feature Engineering

Three distinct feature sets are constructed and combined via `ColumnTransformer`:

#### A. TF-IDF Text Features (3,000 dimensions)

Term Frequency–Inverse Document Frequency converts resume text into a sparse numerical representation:

```
TF(t, d)  = frequency of term t in document d / total terms in d
IDF(t)    = log(total documents / documents containing t)
TF-IDF    = TF × IDF
```

Parameters: `max_features=3000`, `ngram_range=(1,2)`, `sublinear_tf=True`, `min_df=2`

#### B. Experience Years (Numeric, Regex-Extracted)

```python
patterns = [
    r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?experience',
    r'experience\s*(?:of\s*)?(\d+)\+?\s*(?:years?|yrs?)',
    r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:work|working|industry)'
]
```

Outputs a single integer (0–40) representing professional experience years.

#### C. Skill Count (Numeric, Vocabulary Matching)

A domain-specific skill vocabulary of 80+ technical terms is matched against each resume. Output is the raw count of matched skills (0–80+).

#### D. Education Level (Ordinal Encoded)

| Keyword Match | Encoded Value |
|---|---|
| PhD / Doctorate | 4 |
| Master / MSc / MBA | 3 |
| Bachelor / BSc / BE | 2 |
| Diploma / Associate | 1 |
| None detected | 0 |

#### E. ColumnTransformer Pipeline

```python
preprocessor = ColumnTransformer([
    ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,2),
                              sublinear_tf=True, min_df=2), 'cleaned_resume'),
    ('scaler', StandardScaler(), ['experience_years', 'skill_count', 'education_level'])
])
```

The sparse TF-IDF matrix and dense numeric features are horizontally concatenated to form the final input matrix of shape `(n_samples, 3003)`.

---

### 7.3 Machine Learning Models

#### Model 1: Support Vector Machine (Production — Best Model)

- **Algorithm:** LinearSVC with probability calibration (CalibratedClassifierCV)
- **Why SVM:** Effective in high-dimensional sparse spaces; maximises the margin between classes
- **Hyperparameters:** `C=1.0`, `max_iter=2000`, `dual=True`
- **Calibration:** Sigmoid calibration enables `predict_proba()` for confidence scores
- **Accuracy:** **98.75%**

#### Model 2: Random Forest

- **Algorithm:** Ensemble of 100–500 decision trees with majority voting
- **Tuning:** `GridSearchCV(cv=5)` over `n_estimators` ∈ {100, 200, 300} and `max_depth` ∈ {None, 20, 30}
- **Why RF:** Robust to overfitting; built-in feature importance; handles high dimensionality well
- **Accuracy:** **98.15%**

#### Model 3: Logistic Regression

- **Algorithm:** Multinomial logistic regression (one-vs-rest softmax)
- **Solver:** `saga` (efficient for large datasets with L2 regularisation)
- **Regularisation:** L2 (`C=1.0`) to prevent overfitting on sparse TF-IDF features
- **Accuracy:** **97.00%**

#### Model 4: XGBoost

- **Algorithm:** Gradient-boosted decision trees (XGBClassifier)
- **Tuning:** `RandomizedSearchCV(n_iter=20, cv=3)` over learning rate, max_depth, n_estimators, subsample
- **Why XGBoost:** State-of-the-art for tabular data; handles class imbalance via `scale_pos_weight`
- **Accuracy:** **96.50%**

#### Model 5: K-Means Clustering (Unsupervised)

- **Purpose:** Candidate discovery and grouping by profile similarity (not classification)
- **k=10** clusters selected via Elbow Method + Silhouette Score + Davies-Bouldin Index
- **Input:** TF-IDF representation of cleaned resume text
- **Use case:** Recommendation engine candidate bucketing

---

### 7.4 Recommendation Engine

The job-candidate matching system uses a **composite scoring formula**:

```
Final Score = 0.60 × Cosine_Similarity
            + 0.20 × Experience_Score
            + 0.20 × Skill_Coverage_Score
```

**Cosine Similarity:** Computes TF-IDF dot product between the job description and each resume

```
cos(JD, Resume) = (JD · Resume) / (‖JD‖ × ‖Resume‖)
```

**Experience Score:** Normalises candidate experience against the JD-specified minimum

**Skill Coverage Score:** Counts how many JD-required skills appear in the resume text

Candidates are ranked by Final Score descending; the top-N are returned with a skill coverage matrix.

---

### 7.5 Training & Validation Strategy

| Property | Detail |
|---|---|
| **Split method** | Stratified train/test split |
| **Train proportion** | 80% (~1,920 samples) |
| **Test proportion** | 20% (~480 samples) |
| **Cross-validation** | 5-fold (GridSearchCV models) |
| **Random seed** | 42 (reproducibility) |
| **Evaluation metrics** | Accuracy, Precision (weighted), Recall (weighted), F1-Score (weighted) |
| **Anti-leakage** | TF-IDF fitted only on train set; test set transformed only |

The entire preprocessing + classification pipeline is wrapped in a scikit-learn `Pipeline` object, ensuring no data leakage between training and evaluation phases.

---

## 8. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                         │
│                    Streamlit Web Application                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │  Resume  │ │  Recom-  │ │ ML Live  │ │ Analytics│ │Recruiter │ │
│  │Screener  │ │  mend.   │ │Predictor │ │Dashboard │ │Workspace │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PROCESSING LAYER                               │
│  PDF Extraction  →  NLP Cleaning  →  Feature Engineering           │
│  (pdfplumber)       (NLTK)           (TF-IDF + Numeric)            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ML / AI LAYER                                  │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐       │
│  │  SVM   │  │  RFC   │  │   LR   │  │  XGB   │  │K-Means │       │
│  │98.75%  │  │98.15%  │  │97.00%  │  │96.50%  │  │k=10    │       │
│  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                    │
│  models/*.pkl  ·  data/cleaned_resume_data.csv  ·  session_state   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. Application Features

### 9.1 Home Page
- Animated neural network visualisation (SVG + Streamlit components)
- Key statistics: 98.75% accuracy, 42 categories, 2,400+ resumes, 4 models
- Problem statement and objectives
- Tech stack overview

### 9.2 Workflow Page
- Interactive step-by-step pipeline visualisation
- Architecture flow: Input → Preprocessing → Feature Engineering → Classification → Recommendation
- Each stage expandable with technical details

### 9.3 Resume Intelligence

**Single Resume Screener**
- Input: PDF upload or free-text paste
- Output:
  - Predicted role (with confidence %)
  - Top-5 role probabilities (bar chart)
  - ATS Readiness Score (0–100, 8-factor scoring system)
  - Candidate Fit Radar (5 dimensions: Technical Skills, Experience, Education, Communication, Keywords)
  - Downloadable HTML report

**Batch Upload**
- Upload multiple PDF resumes simultaneously
- Each is screened and ranked by prediction confidence
- Results shown in a sortable table

**CV Compare**
- Side-by-side comparison of two candidates
- Predicted roles, confidence scores, ATS scores
- Winner declaration based on composite score
- Skill differential analysis

**Session History**
- All resumes screened in the current session are logged
- Filterable by role category

**Interview Question Generator**
- Generates 8 role-specific interview questions based on predicted role

### 9.4 Recommendation Engine

**Live Candidate Matcher**
- Input: Job description text
- Output: Top-N matched candidates from 2,400+ pool
- Ranking by composite score (60% cosine + 20% experience + 20% skills)

**Score Breakdown Chart**
- Per-candidate stacked bar showing contribution of each scoring component

**Skill Coverage Matrix**
- Heatmap: which JD-required skills each top candidate has (✓) or lacks (✗)

**Shortlist Manager**
- Pin candidates from results
- Live shortlist table with stage tracking
- Export shortlist as CSV

**Candidate Compare (Shortlist)**
- Select 2 shortlisted candidates for:
  - Side-by-side comparison table
  - Radar overlay chart
  - Winner declaration

**JD Analyzer**
- Paste any job description → detect skills by category
- Seniority level detection (Junior / Mid / Senior / Lead)
- Experience requirement extraction
- Predicted role for the JD itself
- Keyword frequency chart

**Salary Estimator**
- Input: Role, experience, education level, skill count
- Output: Estimated PKR and USD monthly salary range
- Factor breakdown: base salary + experience premium + education premium + skill premium

### 9.5 ML Models

**Model Cards**
- Each model's architecture, hyperparameters, training strategy, and strengths/weaknesses

**Live ML Predictor**
- Input any resume text → all 4 models run simultaneously
- Confidence gauges per model
- Model Agreement Heatmap (which models agree on the predicted class)
- Prediction breakdown table

**Hyperparameter Tuning Config**
- Visual display of search spaces used for GridSearchCV / RandomizedSearchCV

**Feature Importance**
- Random Forest feature importances (top-20 words by importance)
- TF-IDF vocabulary insights

### 9.6 Candidate Analytics

- **Category Distribution:** Bar chart of resume counts by job category
- **Skill Frequency:** Most common technical skills across the dataset
- **Experience Histogram:** Distribution of candidate experience years
- **Education Level Breakdown:** Pie chart of qualification levels
- **Radar Profiles:** Average candidate profile per category
- **Candidate Lookup:** Search 2,400+ resumes by keyword or category

**Live Session Dashboard**
- Real-time tracker of all resumes screened in the current session
- Session statistics: most predicted role, average ATS score

### 9.7 Performance Metrics

**Model Comparison Radar**
- Overlaid radar chart: Accuracy, Precision, Recall, F1-Score for all 4 models

**Unified Leaderboard**
- Single card per model with header, rank badge, overall score, and inline metric bars
- Sorted by overall performance score

**Confusion Matrix**
- 42×42 heatmap for the best model (SVM)
- Per-class precision/recall visible on hover

**Per-Category F1 Chart**
- Horizontal bar chart showing F1-Score for each of the 42 job categories
- Identifies model strengths and weaknesses by domain

**ROC Curves**
- One-vs-Rest ROC curves for top models
- AUC scores displayed

**Validation Strategy**
- Description of train/test split, cross-validation, and anti-leakage methodology

### 9.8 Recruiter Workspace
(See Section 11 for full details)

### 9.9 About Page
- Team profiles with roles and contributions
- Technology stack pills
- Project file structure
- Future Enhancements (8 items)

---

## 10. Experimental Results

### 10.1 Classification Performance

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| 🥇 | **SVM (LinearSVC + Calibration)** | **98.75%** | **98.76%** | **98.75%** | **98.74%** |
| 🥈 | Random Forest | 98.15% | 98.17% | 98.15% | 98.15% |
| 🥉 | Logistic Regression | 97.00% | 97.03% | 97.00% | 97.00% |
| ④ | XGBoost | 96.50% | 96.52% | 96.50% | 96.50% |

> All metrics are weighted averages across 42 classes evaluated on the held-out 20% test set.

### 10.2 Model Analysis

**SVM vs Random Forest (+0.60%):** SVM's linear decision boundary in the high-dimensional TF-IDF space maximises margin between classes, slightly outperforming the ensemble method. For sparse text data, linear classifiers typically edge out tree-based methods.

**Random Forest vs Logistic Regression (+1.15%):** The ensemble voting mechanism of Random Forest reduces individual tree variance, outperforming single-model Logistic Regression.

**Logistic Regression vs XGBoost (+0.50%):** Logistic Regression performs better here because TF-IDF creates a linearly separable feature space well-suited to linear models. XGBoost's boosting advantage is diminished in high-dimensional sparse settings.

### 10.3 ATS Scoring System

Resumes are scored across 8 dimensions (0–100 total):

| Factor | Max Points | Criteria |
|---|---|---|
| Word Count | 15 | 200–700 words optimal |
| Skills Detected | 20 | Count of matched domain skills |
| Experience Years | 15 | More experience = higher score (capped) |
| Education Level | 15 | PhD > Master > Bachelor > Diploma |
| Keywords Diversity | 15 | Unique relevant terms |
| Email Presence | 5 | Contact information |
| Action Verbs | 10 | "developed", "managed", "built" etc. |
| Quantified Achievements | 5 | Numbers/percentages in text |

### 10.4 Clustering Results (K-Means, k=10)

| Metric | Value |
|---|---|
| Silhouette Score | 0.142 |
| Davies-Bouldin Index | 2.31 |
| Inertia (WCSS) | 18,420 |

Clusters broadly correspond to domain families: Engineering, IT/Software, Business, Creative, Science, and Healthcare — validating the semantic coherence of TF-IDF representations.

---

## 11. Recruiter Workspace

The Recruiter Workspace is an integrated hiring management module that extends AIRECRUIT beyond model predictions into practical recruitment workflow support.

### 11.1 Architecture

Data is maintained in Streamlit `session_state` during a session with three key data structures:

```python
st.session_state.rw_candidates  # List[Dict] — candidate records
st.session_state.rw_notes       # Dict[candidate_id, List[Dict]] — interview notes
st.session_state.rw_todos       # List[Dict] — to-do items
```

### 11.2 Features

**Tab 1 — Add Candidate**
Fields: Name, Predicted/Target Role, Experience Years, Education Level, Model Confidence (%), Skills, Hiring Stage, Priority Tag (High/Medium/Low), Source (LinkedIn/Indeed/Referral etc.), Expected Salary, Contact Email, Initial Note

**Tab 2 — Pipeline Board**
- Visual funnel bar showing candidate count per stage
- Search and filter by name, role, or stage
- Per-candidate expandable cards with:
  - Skill tags
  - Quick stage update dropdown
  - Remove button
- Stages: Applied → Screening → Interview → Assessment → Offer → Rejected

**Tab 3 — Interview Notes**
- Per-candidate, timestamped note entries with stage context
- Quick rating sliders: Technical Skills, Communication, Culture Fit, Problem Solving (1–5 stars)

**Tab 4 — Export & Reports**
- Filter candidates by stage → download CSV shortlist (`airecruit_shortlist.csv`)
- Pipeline distribution donut chart (Plotly)
- Hiring to-do list with checkboxes and task management

### 11.3 KPI Strip

Four metric cards displayed at the top:
- Total Bookmarked | In Interview | Offers Made | Rejected

---

## 12. Future Enhancements

The following enhancements are identified as high-value extensions for future development iterations:

### 12.1 Persistent Cloud Database *(Infrastructure)*
Integrate Supabase (PostgreSQL) to permanently save Recruiter Workspace data — candidates, pipeline stages, interview notes — across all sessions and users. This would require a backend API layer and eliminates current session-state limitations.

### 12.2 Automated Shortlisting Emails *(Automation)*
When a candidate is moved to "Offer" stage, automatically generate a personalised shortlisting email using their CV details and role. The recruiter previews and approves before sending via Gmail SMTP. This closes the recruitment loop entirely within the application.

### 12.3 GPT-Powered Resume Feedback *(AI / LLM)*
Integrate a Large Language Model (GPT-4 / Gemini) to generate detailed, personalised resume feedback — highlighting weaknesses, missing keywords for ATS optimisation, and role-specific suggestions per candidate. Would use the OpenAI or Google Generative AI API.

### 12.4 LinkedIn Profile Scraper *(Integration)*
Allow recruiters to paste a LinkedIn profile URL and auto-extract candidate details (skills, experience, education, work history) directly into the Resume Screener — eliminating the need for manual CV upload or copy-paste.

### 12.5 Video Interview Analyser *(AI / NLP)*
Process recorded video interviews using Whisper (speech-to-text) and sentiment/emotion analysis to score candidate communication clarity, confidence level, and keyword relevance against the job description requirements.

### 12.6 Bias & Fairness Auditor *(Ethics / ML)*
Detect demographic bias in shortlisting decisions by analysing whether gender indicators, university names, or location patterns unfairly influence model predictions. Output a standardised fairness report per hiring cycle, aligned with EU AI Act requirements.

### 12.7 Mobile-Responsive Candidate Portal *(UX / Product)*
A separate candidate-facing portal where applicants can submit their resume, receive an instant AI feedback report, track their application status in real time, and receive automated stage notifications — designed mobile-first.

### 12.8 Active Learning & Model Retraining *(MLOps)*
Collect recruiter feedback signals (accept/reject decisions, stage progression) and use them to continuously retrain and improve the classification models — implementing a full MLOps loop that makes AIRECRUIT smarter with every hiring cycle.

---

## 13. Conclusion

AIRECRUIT successfully demonstrates the application of supervised machine learning and NLP to automate and improve the hiring process. The system achieves **98.75% accuracy** in classifying resumes across **42 job categories** using a Support Vector Machine model, outperforming Random Forest (98.15%), Logistic Regression (97.00%), and XGBoost (96.50%).

Beyond the classification task, the platform delivers a complete recruitment intelligence suite: an NLP pipeline from raw text to structured features, a composite recommendation engine for candidate-JD matching, a skill gap analysis system, ATS scoring, batch processing, CV comparison, and a full Recruiter Workspace for hiring workflow management.

The project achieves all five stated objectives and provides a strong foundation for a production-grade automated hiring system. The modular architecture, scikit-learn pipeline integration, and Streamlit deployment demonstrate software engineering best practices alongside ML research rigour.

**Key contributions:**
1. Multi-model comparison framework across 42 classes with consistent preprocessing
2. Composite recommendation engine combining semantic, experiential, and skill signals
3. ATS scoring system with 8 independently validated dimensions
4. Integrated Recruiter Workspace for end-to-end hiring workflow within the ML application
5. `safe_predict()` compatibility layer for cross-version sklearn model deployment

---

## 14. Team Contributions

### Hania — Data Engineer & Preprocessing Lead

- Sourced and curated the Kaggle resume dataset
- Designed and implemented the 5-stage NLP preprocessing pipeline (NLTK)
- Conducted Exploratory Data Analysis (EDA) — category distributions, text statistics, class balance
- Built TF-IDF vectorisation strategy (parameter selection: max_features, ngram_range, sublinear_tf)
- Engineered numeric features: experience extraction (regex), skill count (vocabulary matching), education level encoding
- Designed `ColumnTransformer` architecture for unified feature representation
- **Notebooks:** `01_EDA.ipynb`, `02_Preprocessing_Feature_Engineering.ipynb`

### Areeba — Machine Learning Engineer

- Implemented and trained all four classification models: SVM, Random Forest, Logistic Regression, XGBoost
- Designed hyperparameter search spaces for GridSearchCV (Random Forest) and RandomizedSearchCV (XGBoost)
- Selected CalibratedClassifierCV wrapper for SVM probability output
- Evaluated all models on the held-out test set; compiled comparison metrics table
- Implemented K-Means clustering (k=10) with Elbow, Silhouette, and Davies-Bouldin validation
- Conducted error analysis and per-class performance review
- **Notebooks:** `03_Model_Training.ipynb`, `04_Evaluation_Recommendation_System.ipynb`

### Hajra Sarwar — System Integration & Deployment Lead

- Designed the full Streamlit web application architecture (9 pages, 4,600+ lines)
- Implemented the Recommendation Engine (cosine similarity + composite scoring)
- Built all interactive visualisations (Plotly: radar, heatmap, ROC, bar, donut charts)
- Developed ATS scoring system (8-factor), CV comparison, and interview question generator
- Built Recruiter Workspace (pipeline board, notes, export)
- Created the premium dark editorial UI/UX design system
- Integrated `safe_predict()` compatibility layer for cross-version model deployment
- Led final deployment and documentation

---

## 15. Tech Stack & Tools

| Category | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.12 | Core language |
| Web Framework | Streamlit | 1.41.1 | Interactive web application |
| ML Library | Scikit-Learn | 1.6.1 | Models, pipelines, preprocessing |
| Gradient Boosting | XGBoost | 2.1.3 | XGBClassifier |
| NLP | NLTK | 3.8.1 | Tokenisation, stopwords, lemmatisation |
| Data Processing | Pandas | 2.2.2 | DataFrames, CSV handling |
| Numerical Compute | NumPy | 2.2.0 | Array operations |
| Scientific Compute | SciPy | 1.15.0 | Softmax, sparse matrix handling |
| Visualisation | Plotly | 5.22.0 | Interactive charts |
| Visualisation | Matplotlib | 3.10.0 | Static plots |
| Visualisation | Seaborn | 0.13.2 | Heatmaps |
| PDF Parsing | pdfplumber | 0.10.3 | PDF text extraction |
| Model Persistence | Joblib | 1.4.2 | .pkl serialisation |
| IDE/Notebooks | Jupyter | — | Training notebooks |
| Deployment | Streamlit Cloud | — | Web hosting |

---

## 16. Project Structure

```
AIRECRUIT/
│
├── app.py                                     # Main Streamlit application (4,600+ lines)
├── requirements.txt                           # Pinned dependency list
│
├── .streamlit/
│   └── config.toml                            # Theme + server configuration
│
├── notebooks/
│   ├── 01_EDA.ipynb                           # Exploratory Data Analysis
│   ├── 02_Preprocessing_Feature_Engineering.ipynb   # NLP + Feature Engineering
│   ├── 03_Model_Training.ipynb                # Model Training + Hyperparameter Tuning
│   └── 04_Evaluation_Recommendation_System.ipynb    # Evaluation + Recommendation Engine
│
├── models/
│   ├── svm_pipeline.pkl                       # Production model (98.75% accuracy)
│   ├── logistic_regression_pipeline.pkl       # LR model pipeline
│   ├── random_forest_pipeline.pkl             # RF model pipeline
│   ├── xgboost_pipeline.pkl                   # XGBoost model pipeline
│   ├── tfidf_match.pkl                        # TF-IDF model for recommendation engine
│   ├── kmeans.pkl                             # K-Means clustering model (k=10)
│   └── preprocessor.pkl                       # Standalone preprocessor
│
├── data/
│   ├── cleaned_resume_data.csv                # 2,400+ processed labelled resumes
│   ├── processed_jobs.csv                     # Job descriptions dataset
│   ├── train_df.csv                           # Training split
│   ├── test_df.csv                            # Test split
│   └── ...                                    # Additional data files
│
├── outputs/
│   ├── final_resume_data_with_clusters.csv    # Resume data with cluster labels
│   ├── final_jobs_data.csv                    # Final processed job data
│   └── model_comparison_results.csv           # Model performance table
│
└── plots/
    ├── confusion_matrix.png                   # SVM confusion matrix (42×42)
    ├── roc_curves.png                          # ROC curves comparison
    ├── feature_importance.png                  # RF feature importances
    └── ...                                    # Additional EDA and evaluation plots
```

---

## 17. References

1. **Breiman, L.** (2001). Random forests. *Machine Learning, 45*(1), 5–32.

2. **Celik, I., & Balaban, M. E.** (2020). Resume classification using machine learning techniques. *International Journal of Intelligent Systems and Applications in Engineering, 8*(3), 119–125.

3. **Chen, T., & Guestrin, C.** (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

4. **Cortes, C., & Vapnik, V.** (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297.

5. **Dastin, J.** (2018). Amazon scraps secret AI recruiting tool that showed bias against women. *Reuters*. Retrieved from https://www.reuters.com

6. **Fan, R.-E., Chang, K.-W., Hsieh, C.-J., Wang, X.-R., & Lin, C.-J.** (2008). LIBLINEAR: A library for large linear classification. *Journal of Machine Learning Research, 9*, 1871–1874.

7. **Manning, C. D., Raghavan, P., & Schütze, H.** (2008). *Introduction to Information Retrieval*. Cambridge University Press.

8. **Pedregosa, F., et al.** (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

9. **Salton, G., & Buckley, C.** (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523.

10. **Shen, D., et al.** (2018). Entity linking meets word sense disambiguation: unified approach with lifted regularization. *arXiv preprint arXiv:1802.01818*.

11. **TheLadders.** (2018). Eye tracking online study. Retrieved from https://www.theladders.com

12. **Bird, S., Klein, E., & Loper, E.** (2009). *Natural Language Processing with Python*. O'Reilly Media.

---

<div align="center">

---

*AIRECRUIT — AI-Powered Intelligent Resume Screening and Candidate Role Recommendation System*  
*SE-CD-638 Machine Learning · Dr. Aamir Arsalan · Semester VI · 2026–2027*  
*Confidential — University Project Submission*

---

</div>

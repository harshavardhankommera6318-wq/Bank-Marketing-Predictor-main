# Bank Marketing Predictor

**Internship project:** Machine Learning  
**Intern ID:** `CITS9471`  
**Full name:** `Chilaka Pankaj Dheeraj`  
**Duration (weeks):** `12 weeks`  
**Project scope:** Analyze campaign records and train a classification model to estimate whether a customer may subscribe to a term deposit.

## Overview
A Streamlit dashboard built with Python, Pandas, Matplotlib, Seaborn and scikit-learn. It includes 1,200 fictional records, campaign visualizations, filters, CSV export, Logistic Regression classification, evaluation metrics and an interactive probability prediction.

> All records are synthetic. This educational project is not a real banking model and must not be used for lending, credit, eligibility, financial advice or decisions about real customers.

## Features
- Data validation, missing-value handling, categorical normalization and duplicate removal.
- Campaign summary and subscription-rate visualizations.
- Customer record filtering and CSV download.
- Logistic Regression binary classification pipeline.
- One-hot encoding for categorical fields and standardization for numeric fields.
- Accuracy, precision, recall, F1 and ROC-AUC evaluation.
- Interactive YES/NO prediction with estimated probability.
- Automated tests.

## Folder structure
```text
Bank-Marketing-Predictor/
├── app.py
├── data/bank_marketing.csv
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   └── model.py
├── tests/test_project.py
├── screenshots/
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset fields
`age`, `job`, `marital`, `education`, `account_balance`, `housing_loan`, `personal_loan`, `contact_type`, `last_contact_month`, `campaign_contacts`, `days_since_previous_contact`, `previous_contacts`, and `subscribed` (target: yes/no). All records are generated synthetic examples.

## Machine Learning workflow
1. Load and validate the CSV with Pandas.
2. Clean numeric and categorical fields; remove duplicate rows and invalid labels.
3. Split data into stratified 80% training and 20% test sets.
4. Impute missing values, standardize numeric features and one-hot encode categorical features inside a scikit-learn pipeline.
5. Train Logistic Regression with balanced class weights.
6. Evaluate on the held-out test set using accuracy, precision, recall, F1 and ROC-AUC.
7. Predict a sample customer's subscription class and estimated probability.

## Requirements
Windows 10/11, Python 3.13 or later, VS Code, and internet access for initial package installation.

## Run in VS Code on Windows
Open this folder in VS Code and run commands in Terminal from the project root.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
If `py -3.13` is unavailable, use `python -m venv .venv`. For Command Prompt activation use `.venv\Scripts\activate.bat`. Open the local URL printed by Streamlit, usually `http://localhost:8501`. Stop with `Ctrl+C`.

## Test
```powershell
python -m pytest -q
```
Run after activating the virtual environment.

## Dashboard walkthrough
1. Campaign overview: review campaign totals and charts.
2. Customer records: filter by job category and download visible rows.
3. Predict subscription: enter fictional customer details and submit.
4. Model evaluation: review classification metrics.

## Limitations and future improvements
Synthetic records cannot represent actual banking customers or campaign behavior. Metrics are demonstration results only. Future work could compare other classifiers and use a properly licensed public dataset for academic benchmarking, with suitable privacy and fairness checks.

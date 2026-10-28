# Customer Churn & Billing Intelligence System

An end-to-end Machine Learning web application and analytics dashboard developed to predict customer churn risk, forecast monthly billing charges, and optimize customer retention strategies.

## Business Objective

Identify customers at high risk of cancelling their subscriptions, determine the underlying service and contract drivers behind customer churn, and provide management with data-driven recommendations and revenue-at-risk assessments to improve customer retention and protect profitability.

## Dashboard Pages

### 1. Single Customer Predictor

Enables retention and support representatives to configure individual customer attributes in real time, evaluate estimated monthly charges via regression, assess churn risk probability via classification, and receive instant alert guidance.

<img width="992" height="603" alt="01_Single_Customer_Predictor" src="screenshots/Single_Customer_Predictor/Screenshot%202026-10-28%20150034.png" />


### 2. Executive Retention Summary

Provides management with a high-level summary of total evaluated accounts, count of customers requiring retention action, average churn risk, and total monthly and annualized revenue at risk.

<img width="986" height="611" alt="02_Executive_Retention_Summary" src="screenshots/Executive_Retention_Summary/Screenshot%202026-10-28%20150110.png" />


### 3. Actionable Retention Outreach List

Provides an interactive data table with quick-filtering options to isolate high-risk customers, assign automated retention offers, and export actionable CSV campaign files.

<img width="987" height="612" alt="03_Scored_Customer_Records" src="screenshots/Scored_Customer_Records/Screenshot%202026-10-28%20150128.png" />


### 4. Visual Analytics Dashboard

Compares customer risk against predicted monthly charges using interactive Plotly scatter plots with a 60% retention action threshold, alongside monthly revenue loss by contract type.

<img width="993" height="607" alt="04_Visual_Analytics_Dashboard" src="screenshots/Visual_Analytics_Dashboard/Screenshot%202026-10-28%20150146.png" />


## Key Performance Indicators

- Total Evaluated Customers
- High-Risk Retention Count (Risk ≥ 60%)
- Average Churn Risk %
- Monthly Revenue at Risk ($)
- Annualized Revenue Impact ($)
- Churn Risk Probability Score (%)
- Estimated Monthly Charges ($)
- Classification ROC-AUC Score
- Model Prediction Accuracy

## Dataset

This project uses the Telco Customer Churn dataset.

- Approximately 7,043 records
- 21 columns
- Includes customer demographics, account tenure, contract terms, payment methods, monthly charges, internet services, and churn status

## Project Workflow

1. Imported and explored the raw customer dataset
2. Cleaned and transformed data by handling missing values in total charges and removing non-predictive identifiers
3. Encoded target variables and performed one-hot encoding on categorical attributes
4. Applied feature scaling using `StandardScaler`
5. Trained and tuned a K-Nearest Neighbors (KNN) Classifier for churn risk prediction
6. Trained and tuned a K-Nearest Neighbors (KNN) Regressor to estimate monthly billing charges
7. Evaluated models using Confusion Matrices, ROC-AUC curves, MAE, MSE, and R² scores
8. Serialized the trained pipeline artifacts using Joblib
9. Developed an interactive Streamlit web application with Single and Batch prediction modes
10. Built business retention rules, value tier segmentation, and automated CSV export capabilities

## Tools and Skills

- Python
- Streamlit
- Scikit-Learn
- Pandas & NumPy
- Plotly Express
- Matplotlib
- Data Cleaning
- Data Preprocessing
- Classification & Regression
- Model Serialization (Joblib)
- KPI Development
- Business Insights & Retention Analysis

## Project Setup & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/bandapallijoshnasree71/telco-churn-predictor.git](https://github.com/bandapallijoshnasree71/telco-churn-predictor.git)
cd telco-churn-predictor
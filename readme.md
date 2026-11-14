# Customer Churn Prediction – ML DevOps Project

An end-to-end Machine Learning and MLOps project implementing customer churn prediction using **MLflow, pytest, and Evidently**.

---

## 1. Project Overview

### Problem Statement

Customer churn is a major challenge for telecommunication companies. Losing existing customers affects revenue and business growth.

This project aims to predict whether a customer is likely to leave a telecommunications service using historical customer information.

The project demonstrates an end-to-end ML workflow, including data analysis, preprocessing, model training, experiment tracking, automated testing, and data drift monitoring.

### Domain
Telecommunications

### Machine Learning Task
Binary Classification

### Model Used
Logistic Regression

---

## 2. Dataset Information

### Dataset Name
IBM Telco Customer Churn Dataset

### Dataset Source
IBM Sample Dataset Repository:

https://github.com/IBM/telco-customer-churn-on-icp4d

Direct CSV download:

https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

### Dataset Description

The dataset contains historical telecommunications customer information, including their service subscriptions, account information, billing details, and churn status.

| Attribute | Information |
|---|---|
| Total Records | 7,043 |
| Total Columns | 21 |
| Target Variable | Churn |
| Problem Type | Binary Classification |
| Domain | Telecommunications |
| File Format | CSV |

### Important Features

- Customer demographics: gender, SeniorCitizen, Partner, Dependents
- Services: PhoneService, InternetService, OnlineSecurity, OnlineBackup, TechSupport
- Account information: tenure, Contract, PaymentMethod
- Billing information: MonthlyCharges, TotalCharges
- Target variable: Churn

### Target Variable

`Churn`

- Yes = Customer left the service
- No = Customer continued the service

### Data Preprocessing

The following preprocessing operations are performed:

1. Remove the customerID column.
2. Convert TotalCharges into numeric format.
3. Handle missing values using SimpleImputer.
4. Encode categorical variables using OneHotEncoder.
5. Scale numerical variables using StandardScaler.
6. Split the dataset into training and testing sets.
7. Use an 80:20 train-test split.

---

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Matplotlib | Data visualization |
| MLflow | Experiment tracking |
| pytest | Automated testing |
| Evidently | Data drift monitoring |
| SQLite | MLflow tracking database |
| VS Code | Development environment |
| Git & GitHub | Version control |

---

## 4. Project Structure

```text
MLOPS_external/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── src/
│   ├── eda.py
│   ├── train.py
│   └── monitor.py
│
├── tests/
│   └── test_model.py
│
├── reports/
│   └── data_drift_report.html
│
├── .gitignore
├── requirements.txt
├── README.md
└── churn_model.pkl
```

Note: Generated files, datasets, model files, and local tracking databases may be excluded from Git using `.gitignore`. The structure above describes the complete local project.

---

# 5. Step-by-Step Implementation

## Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/mlops-customer-churn.git
cd mlops-customer-churn
```

## Step 2: Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

## Step 3: Install Dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install project dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Step 4: Download the Dataset

Download the IBM Telco Customer Churn CSV from:

https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Place the downloaded file inside the `data` directory.

Required filename:

```text
Telco-Customer-Churn.csv
```

## Step 5: Perform Exploratory Data Analysis

Execute:

```powershell
python src/eda.py
```

The EDA script performs:

- Dataset overview
- Shape and data types
- Missing-value analysis
- Duplicate analysis
- Target variable distribution
- Numerical statistics
- TotalCharges conversion check

## Step 6: Train the Machine Learning Model

Execute:

```powershell
python src/train.py
```

This script performs:

- Data cleaning
- Feature identification
- Data preprocessing
- Train-test splitting
- Logistic Regression model training
- Model evaluation

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

The trained model is saved locally as:

```text
churn_model.pkl
```

---

# 6. MLflow – Experiment Tracking

MLflow is used to track machine learning experiments, parameters, evaluation metrics, and model artifacts.

### Features Implemented

- Experiment creation
- Parameter logging
- Metric logging
- Model artifact logging
- SQLite-based experiment tracking

### Start MLflow UI

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open the following address in your browser:

http://localhost:5000

### Experiment Name

`Customer_Churn_Prediction`

### Logged Parameters

- Model name
- Test size
- Random state
- Maximum iterations
- Class weights

### Logged Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

---

# 7. pytest – Automated Testing

pytest is used to validate the functionality of the machine learning pipeline.

### Test Cases Implemented

1. Dataset loading test
2. Model file existence test
3. Model loading test
4. Model prediction test
5. Prediction value validation
6. Missing prediction validation
7. Prediction probability validation

### Execute Tests

```powershell
python -m pytest tests/test_model.py -v
```

Expected result:

```text
7 passed
```

The tests help identify errors and ensure that the trained model behaves as expected.

---

# 8. Evidently – Data Drift Monitoring

Evidently is used to monitor changes in incoming data by comparing a reference dataset against a current dataset.

### Implementation

1. Load the original customer dataset.
2. Create reference and current datasets.
3. Simulate changes in customer features.
4. Generate a data drift report.
5. Save the report in HTML format.

### Simulated Drift

- MonthlyCharges increased by 30%.
- Tenure reduced by 30%.

These modifications simulate changes in incoming customer data.

### Execute Monitoring

```powershell
python src/monitor.py
```

### Generated Report

```text
reports/data_drift_report.html
```

Open the report:

```powershell
Invoke-Item .\reports\data_drift_report.html
```

The report provides:

- Overall data drift analysis
- Feature-level drift information
- Reference and current data comparisons
- Distribution changes

Note: This implementation demonstrates simulated data drift, not actual production customer behaviour.

---

# 9. Complete Execution Commands

Execute the following commands in sequence:

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Exploratory Data Analysis
python src/eda.py

# Train the ML model
python src/train.py

# Run automated tests
python -m pytest tests/test_model.py -v

# Generate Evidently report
python src/monitor.py

# Start MLflow dashboard
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

# 10. MLOps Workflow

The project demonstrates three important MLOps capabilities:

| Tool | Responsibility |
|---|---|
| MLflow | Machine learning experiment tracking |
| pytest | Automated testing |
| Evidently | Data drift monitoring |

Together, these tools support model development, validation, experiment management, and monitoring.

---

# 11. Author

**Rushikesh**

B.Tech – Computer Science and Engineering (AIML & Data Science)

KIT's College of Engineering, Kolhapur

Project: ML DevOps Laboratory – POE

---

# 12. License and Dataset Attribution

Dataset: IBM Telco Customer Churn Sample Dataset.

Please refer to the original IBM repository for dataset attribution and applicable usage terms.

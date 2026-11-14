
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# ==========================================
# 2. DATA CLEANING
# ==========================================

df = df.drop(columns=["customerID"], errors="ignore")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

df = df.dropna(subset=["Churn"])

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

print("Missing values before imputation:")
print(X.isnull().sum())

# ==========================================
# 3. IDENTIFY COLUMN TYPES
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)

# ==========================================
# 4. PREPROCESSING PIPELINES
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 6. BUILD MODEL
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# ==========================================
# 7. MLflow CONFIGURATION
# ==========================================


# MLflow SQLite configuration

DATABASE_PATH = ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{DATABASE_PATH.as_posix()}"
)

mlflow.set_experiment("Customer_Churn_Prediction")

print("MLflow tracking URI:", mlflow.get_tracking_uri())


# ==========================================
# 8. MODEL TRAINING AND TRACKING
# ==========================================

with mlflow.start_run(run_name="Logistic_Regression"):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    # Log parameters
    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_param("test_size", 0.20)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("class_weight", "balanced")

    # Log evaluation metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Save trained model in MLflow
  
    mlflow.sklearn.log_model(
    sk_model=model,
    name="churn_model",
    skops_trusted_types=["numpy.dtype"]
)


    print("\n========== MODEL PERFORMANCE ==========")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC:", roc_auc)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

# ==========================================
# 9. SAVE MODEL LOCALLY
# ==========================================

import joblib

joblib.dump(model, ROOT / "churn_model.pkl")

print("\nModel saved successfully!")
print("MLflow experiment tracking completed!")

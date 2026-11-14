
import pandas as pd
import joblib
import pytest

from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "churn_model.pkl"


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def dataset():
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture
def sample_data(dataset):
    X = dataset.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    X["TotalCharges"] = pd.to_numeric(
        X["TotalCharges"],
        errors="coerce"
    )

    return X.head(10)


# ==========================================
# TEST 1: DATASET LOADING
# ==========================================

def test_dataset_loading(dataset):
    assert dataset is not None
    assert len(dataset) > 0
    assert "Churn" in dataset.columns


# ==========================================
# TEST 2: MODEL FILE EXISTENCE
# ==========================================

def test_model_exists():
    assert MODEL_PATH.exists()


# ==========================================
# TEST 3: MODEL LOADING
# ==========================================

def test_model_loading(model):
    assert model is not None


# ==========================================
# TEST 4: PREDICTION
# ==========================================

def test_model_prediction(model, sample_data):

    predictions = model.predict(sample_data)

    assert len(predictions) == len(sample_data)


# ==========================================
# TEST 5: VALID PREDICTION VALUES
# ==========================================

def test_prediction_values(model, sample_data):

    predictions = model.predict(sample_data)

    assert set(predictions).issubset({0, 1})


# ==========================================
# TEST 6: NO MISSING PREDICTIONS
# ==========================================

def test_no_missing_predictions(model, sample_data):

    predictions = model.predict(sample_data)

    assert not pd.isnull(predictions).any()


# ==========================================
# TEST 7: PREDICTION PROBABILITIES
# ==========================================

def test_prediction_probabilities(model, sample_data):

    probabilities = model.predict_proba(sample_data)

    assert probabilities.shape == (len(sample_data), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


print("Pytest test cases loaded successfully!")

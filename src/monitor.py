
import pandas as pd
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
REPORT_PATH = ROOT / "reports" / "data_drift_report.html"

# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)

df = df.drop(
    columns=["customerID", "Churn"],
    errors="ignore"
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# ==========================================
# 3. CREATE REFERENCE AND CURRENT DATA
# ==========================================

reference_data = df.sample(
    frac=0.7,
    random_state=42
).copy()

current_data = df.drop(
    reference_data.index
).copy()

# ==========================================
# 4. SIMULATE DATA DRIFT
# ==========================================

# Simulate an increase in monthly charges
current_data["MonthlyCharges"] = (
    current_data["MonthlyCharges"] * 1.30
)

# Simulate a change in customer tenure
current_data["tenure"] = (
    current_data["tenure"] * 0.70
)

print("Reference dataset:", reference_data.shape)
print("Current dataset:", current_data.shape)

print("\nSimulated changes:")
print("MonthlyCharges increased by 30%")
print("Tenure reduced by 30%")

# ==========================================
# 5. GENERATE EVIDENTLY REPORT
# ==========================================

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

result = report.run(
    reference_data=reference_data,
    current_data=current_data
)

# ==========================================
# 6. SAVE HTML REPORT
# ==========================================

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

result.save_html(str(REPORT_PATH))

print("\n========== EVIDENTLY MONITORING ==========")
print("Data drift analysis completed!")
print("Report saved at:", REPORT_PATH)

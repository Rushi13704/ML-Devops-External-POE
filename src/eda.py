
import pandas as pd

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("\n========== DATASET OVERVIEW ==========")

print("\nShape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== DATA INFORMATION ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())

print("\n========== TARGET DISTRIBUTION ==========")
print(df["Churn"].value_counts())

print("\n========== TARGET PERCENTAGE ==========")
print(df["Churn"].value_counts(normalize=True) * 100)

print("\n========== NUMERICAL STATISTICS ==========")
print(df.describe())

print("\n========== TOTAL CHARGES CHECK ==========")
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("Invalid TotalCharges:", df["TotalCharges"].isnull().sum())

print("\n========== EDA COMPLETED ==========")

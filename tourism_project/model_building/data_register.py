import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# These are the dataset fields listed in the project data dictionary.
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "OwnCar", "NumberOfChildrenVisiting", "Designation",
    "MonthlyIncome", "PitchSatisfactionScore", "ProductPitched",
    "NumberOfFollowups", "DurationOfPitch"
]

df = pd.read_csv(RAW_PATH)

missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
if missing_columns:
    raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

if "ProdTaken" not in df.columns:
    raise ValueError("Target column 'ProdTaken' is missing.")

if df["ProdTaken"].isna().any():
    raise ValueError("Target column 'ProdTaken' contains missing values.")

unexpected_target_values = set(df["ProdTaken"].unique()) - {0, 1}
if unexpected_target_values:
    raise ValueError(
        f"'ProdTaken' must contain only 0 and 1; found {unexpected_target_values}"
    )

print("Dataset registered and validated successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("Column names:", list(df.columns))
print("\nMissing values by column:")
print(df.isna().sum())
print("\nTarget distribution:")
print(df["ProdTaken"].value_counts(dropna=False).sort_index())
print("\nDataset information:")
print(df.info())

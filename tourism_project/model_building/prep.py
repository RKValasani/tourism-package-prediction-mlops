import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "tourism_project/data/tourism.csv"
TARGET_COLUMN = "ProdTaken"
IDENTIFIER_COLUMNS = ["Unnamed: 0", "CustomerID"]

# Load the registered source dataset.
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print("Original dataset shape:", df.shape)

# Check required target before making changes.
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")

# Report missing values before cleaning; no rows are silently removed.
print("\nMissing values before cleaning:")
print(df.isna().sum())

# Clean a documented spelling/spacing inconsistency in Gender.
# 'Fe Male' is normalized to the existing 'Female' category.
if "Gender" in df.columns:
    print("\nGender values before cleaning:")
    print(df["Gender"].value_counts(dropna=False))
    df["Gender"] = df["Gender"].replace({
        "Fe Male": "Female",
        "Fe male": "Female",
        "FE MALE": "Female"
    })
    print("\nGender values after cleaning:")
    print(df["Gender"].value_counts(dropna=False))

# Remove only the agreed identifier columns; retain all other predictors.
columns_to_drop = [column for column in IDENTIFIER_COLUMNS if column in df.columns]
df = df.drop(columns=columns_to_drop)
print("\nRemoved identifier columns:", columns_to_drop)

# Validate target values and missingness after cleaning.
if df[TARGET_COLUMN].isna().any():
    raise ValueError("Target column contains missing values.")
unexpected_target_values = set(df[TARGET_COLUMN].unique()) - {0, 1}
if unexpected_target_values:
    raise ValueError(
        f"'ProdTaken' must contain only 0 and 1; found {unexpected_target_values}"
    )

# Separate features and target.
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].astype(int)

# Stratified 70:30 split, preserving class proportions.
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Save the four files for the next GitHub Actions job.
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_frame(name=TARGET_COLUMN).to_csv("ytrain.csv", index=False)
ytest.to_frame(name=TARGET_COLUMN).to_csv("ytest.csv", index=False)

print("\nData cleaning and preparation completed.")
print("Cleaned dataset shape:", df.shape)
print(f"Training rows: {len(Xtrain)} ({len(Xtrain) / len(df):.0%})")
print(f"Testing rows: {len(Xtest)} ({len(Xtest) / len(df):.0%})")
print("\nTraining target distribution:")
print(ytrain.value_counts(normalize=True).sort_index())
print("\nTesting target distribution:")
print(ytest.value_counts(normalize=True).sort_index())
print("\nSaved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

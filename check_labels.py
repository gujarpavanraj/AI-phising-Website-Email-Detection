import pandas as pd

DATASET_PATH = "database/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("DATASET LABEL CHECK")
print("=" * 60)

print("\nColumns:")
print(df.columns.tolist())

print("\nRaw label distribution:")
print(df["label"].value_counts())

print("\nSample URLs with labels:")
print(df[["URL", "label"]].head(20).to_string(index=False))

print("\nExamples for label 0:")
print(
    df[df["label"] == 0][["URL", "label"]]
    .head(10)
    .to_string(index=False)
)

print("\nExamples for label 1:")
print(
    df[df["label"] == 1][["URL", "label"]]
    .head(10)
    .to_string(index=False)
)
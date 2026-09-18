import pandas as pd
import os

INPUT_FILE = "database/data_bal - 20000.xlsx"
OUTPUT_FILE = "database/url_dataset.csv"

print("Reading Excel dataset...")

df = pd.read_excel(INPUT_FILE)

print("\nColumns found:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# Try to find URL column
url_column = None
for column in df.columns:
    if str(column).strip().lower() in ["url", "urls", "link", "website"]:
        url_column = column
        break

# Try to find label column
label_column = None
for column in df.columns:
    if str(column).strip().lower() in ["label", "labels", "class", "target"]:
        label_column = column
        break

if url_column is None or label_column is None:
    print("\nCould not automatically find URL/label columns.")
    print("Please send me the column names shown above.")
    exit()

# Keep only required columns
output_df = df[[url_column, label_column]].copy()

# Rename them to what train_url_model.py expects
output_df.columns = ["url", "label"]

# Remove empty rows
output_df = output_df.dropna(subset=["url", "label"])

# Save CSV
output_df.to_csv(OUTPUT_FILE, index=False)

print("\nConversion successful! ✅")
print("Total URLs:", len(output_df))
print("Saved to:", OUTPUT_FILE)
print("\nLabel distribution:")
print(output_df["label"].value_counts())
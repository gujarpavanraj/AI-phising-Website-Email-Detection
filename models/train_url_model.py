import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = "database/PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_PATH = "models/url_text_model.pkl"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("PHISHGUARD - URL MODEL TRAINING")
print("=" * 60)

print("\nLoading PhiUSIIL dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Total records:", len(df))
print("Total columns:", len(df.columns))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["URL", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' was not found in the dataset."
        )

print("\nRequired columns found:")
print("✓ URL")
print("✓ label")


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

print("\nChecking missing values...")

df = df.dropna(subset=["URL", "label"])

print("Records after cleaning:", len(df))


# ============================================================
# URL DATA
# ============================================================

X = df["URL"].astype(str)

# Original PhiUSIIL labels:
# 0 = Phishing
# 1 = Legitimate
#
# Convert to PhishGuard labels:
# 0 = Legitimate
# 1 = Phishing

y = 1 - df["label"].astype(int)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nLabel distribution:")
print(y.value_counts())

print("\nLabel percentages:")
print((y.value_counts(normalize=True) * 100).round(2))


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training URLs:", len(X_train))
print("Testing URLs :", len(X_test))


# ============================================================
# CHARACTER-LEVEL URL MODEL
# ============================================================

print("\nCreating character-level URL model...")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining model...")
print("This may take some time because the dataset contains",
      len(X_train), "URLs.")

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ============================================================
# TEST MODEL
# ============================================================

print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Phishing"]
))


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("Model location:", MODEL_PATH)

print("\nPhishGuard URL model is ready!")
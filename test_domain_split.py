import os
import joblib
import pandas as pd

from urllib.parse import urlparse

from sklearn.model_selection import GroupShuffleSplit
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
MODEL_PATH = "models/url_domain_split_model.pkl"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PHISHGUARD - DOMAIN SPLIT URL TEST")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Total records:", len(df))


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = ["URL", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' was not found."
        )

print("\nRequired columns found:")
print("✓ URL")
print("✓ label")


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(subset=["URL", "label"])

df["URL"] = df["URL"].astype(str)

print("\nRecords after cleaning:", len(df))


# ============================================================
# EXTRACT DOMAIN
# ============================================================

def extract_domain(url):

    try:
        parsed = urlparse(url)

        hostname = parsed.hostname

        if hostname is None:
            return "unknown"

        hostname = hostname.lower()

        # Remove www.
        if hostname.startswith("www."):
            hostname = hostname[4:]

        parts = hostname.split(".")

        # Handle domains such as:
        # example.com
        # example.co.uk
        # example.org
        if len(parts) >= 3:

            common_two_part_tlds = {
                "co.uk",
                "org.uk",
                "ac.uk",
                "gov.uk",
                "com.au",
                "net.au",
                "org.au",
                "co.in",
                "com.br",
                "co.jp",
                "com.cn"
            }

            last_two = ".".join(parts[-2:])

            if last_two in common_two_part_tlds:
                return ".".join(parts[-3:])

        if len(parts) >= 2:
            return ".".join(parts[-2:])

        return hostname

    except Exception:
        return "unknown"


# ============================================================
# CREATE DOMAIN GROUPS
# ============================================================

print("\nExtracting domains...")

df["domain"] = df["URL"].apply(extract_domain)

print("Unique domains:", df["domain"].nunique())


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df["URL"]

# Original PhiUSIIL:
#
# 0 = Phishing
# 1 = Legitimate
#
# PhishGuard:
#
# 0 = Legitimate
# 1 = Phishing

y = 1 - df["label"].astype(int)

groups = df["domain"]


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nPhishGuard label distribution:")

print(
    y.value_counts()
    .rename(index={
        0: "Legitimate",
        1: "Phishing"
    })
)


# ============================================================
# DOMAIN-BASED TRAIN / TEST SPLIT
# ============================================================

print("\nCreating domain-based split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

train_domains = set(
    groups.iloc[train_indices]
)

test_domains = set(
    groups.iloc[test_indices]
)


# ============================================================
# VERIFY DOMAIN SEPARATION
# ============================================================

overlap = train_domains.intersection(test_domains)

print("\nDomain split completed!")

print("Training URLs:", len(X_train))
print("Testing URLs :", len(X_test))

print("Training domains:", len(train_domains))
print("Testing domains :", len(test_domains))

print("Domain overlap:", len(overlap))


if len(overlap) == 0:
    print("✓ No domain leakage detected!")
else:
    print("WARNING: Domain overlap detected!")


# ============================================================
# CREATE MODEL
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

print(
    "Training on",
    len(X_train),
    "URLs from",
    len(train_domains),
    "domains."
)

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\nEvaluating on unseen domains...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("DOMAIN SPLIT MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Phishing"
        ]
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DOMAIN SPLIT TEST COMPLETED")
print("=" * 70)

print("\nModel saved at:")
print(MODEL_PATH)

print("\nThis evaluation tests the model on domains")
print("that were not present during training.")

print("\nPhishGuard domain-split evaluation completed!")
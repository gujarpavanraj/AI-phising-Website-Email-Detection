import os
import re
import joblib
import pandas as pd

from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from scipy.sparse import hstack


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = "../database/PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_PATH = "models/hybrid_url_model.pkl"


# ============================================================
# URL FEATURE EXTRACTION
# ============================================================

def extract_url_features(url):

    url = str(url)

    try:
        parsed = urlparse(url)
    except Exception:
        parsed = None

    hostname = parsed.hostname if parsed and parsed.hostname else ""

    features = []

    # URL length
    features.append(len(url))

    # Hostname length
    features.append(len(hostname))

    # Number of dots
    features.append(url.count("."))

    # Number of hyphens
    features.append(url.count("-"))

    # Number of slashes
    features.append(url.count("/"))

    # Number of question marks
    features.append(url.count("?"))

    # Number of equals
    features.append(url.count("="))

    # Number of @ symbols
    features.append(url.count("@"))

    # Number of digits
    features.append(sum(c.isdigit() for c in url))

    # Number of special characters
    special_chars = "@?=&%_"
    features.append(
        sum(c in special_chars for c in url)
    )

    # HTTPS
    features.append(
        1 if url.lower().startswith("https://") else 0
    )

    # IP address
    has_ip = bool(
        re.search(
            r"https?://(?:\d{1,3}\.){3}\d{1,3}",
            url
        )
    )

    features.append(1 if has_ip else 0)

    # Suspicious keywords
    suspicious_words = [
        "login",
        "signin",
        "verify",
        "verification",
        "account",
        "secure",
        "security",
        "update",
        "password",
        "credential",
        "bank",
        "paypal",
        "confirm",
        "authentication",
        "recover",
        "unlock"
    ]

    url_lower = url.lower()

    suspicious_count = sum(
        word in url_lower
        for word in suspicious_words
    )

    features.append(suspicious_count)

    # Subdomain count
    subdomain_count = 0

    if hostname:
        parts = hostname.split(".")

        if len(parts) > 2:
            subdomain_count = len(parts) - 2

    features.append(subdomain_count)

    # URL depth
    path = parsed.path if parsed else ""
    features.append(
        len([x for x in path.split("/") if x])
    )

    return features


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("PHISHGUARD - HYBRID URL MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Total records:", len(df))


# ============================================================
# CHECK COLUMNS
# ============================================================

if "URL" not in df.columns:
    raise ValueError("URL column not found.")

if "label" not in df.columns:
    raise ValueError("label column not found.")


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=["URL", "label"]
)

df["URL"] = df["URL"].astype(str)

# Original:
# 0 = Phishing
# 1 = Legitimate
#
# PhishGuard:
# 0 = Legitimate
# 1 = Phishing

y = 1 - df["label"].astype(int)

X = df["URL"]


print("\nPhishGuard label distribution:")
print(
    y.value_counts().rename(
        {
            0: "Legitimate",
            1: "Phishing"
        }
    )
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating train/test split...")

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
# CHARACTER TF-IDF
# ============================================================

print("\nCreating character-level TF-IDF...")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=100000,
    sublinear_tf=True
)


X_train_text = vectorizer.fit_transform(X_train)

X_test_text = vectorizer.transform(X_test)


# ============================================================
# HANDCRAFTED FEATURES
# ============================================================

print("Extracting URL features...")

X_train_features = pd.DataFrame(
    [
        extract_url_features(url)
        for url in X_train
    ]
)

X_test_features = pd.DataFrame(
    [
        extract_url_features(url)
        for url in X_test
    ]
)


# ============================================================
# COMBINE FEATURES
# ============================================================

print("Combining TF-IDF + URL features...")

X_train_combined = hstack(
    [
        X_train_text,
        X_train_features.values
    ]
)

X_test_combined = hstack(
    [
        X_test_text,
        X_test_features.values
    ]
)


# ============================================================
# CLASSIFIER
# ============================================================

print("\nCreating Logistic Regression classifier...")

classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining hybrid model...")

classifier.fit(
    X_train_combined,
    y_train
)

print("Model training completed!")


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating model...")

y_pred = classifier.predict(
    X_test_combined
)

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("HYBRID MODEL PERFORMANCE")
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


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# SAVE EVERYTHING
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

model_package = {
    "vectorizer": vectorizer,
    "classifier": classifier
}

joblib.dump(
    model_package,
    MODEL_PATH
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("HYBRID MODEL SAVED")
print("=" * 70)

print(
    "Model location:",
    MODEL_PATH
)

print("\nPhishGuard hybrid URL model is ready!")
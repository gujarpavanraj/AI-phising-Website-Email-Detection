import pandas as pd
import joblib

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


# =========================
# 1. Dataset path
# =========================

DATASET_PATH = "database/email_dataset.csv"


# =========================
# 2. Load dataset
# =========================

print("📂 Loading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    usecols=["body", "label"]
)

df = df.dropna(subset=["body", "label"])

# Convert labels to integers
df["label"] = df["label"].astype(int)

# Shuffle dataset
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("Dataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())


# =========================
# 3. Hashing Vectorizer
# =========================

vectorizer = HashingVectorizer(
    n_features=2**18,
    alternate_sign=False,
    ngram_range=(1, 2),
    analyzer="word",
    lowercase=True,
    stop_words="english"
)


# =========================
# 4. ML Model
# =========================

model = SGDClassifier(
    loss="log_loss",
    max_iter=1,
    learning_rate="optimal",
    random_state=42
)


# =========================
# 5. Train in chunks
# =========================

chunk_size = 5000

first_chunk = True

print("\n🚀 Starting email model training...")


for start in range(0, len(df), chunk_size):

    chunk = df.iloc[
        start:start + chunk_size
    ]

    texts = chunk["body"].astype(str)

    labels = chunk["label"]

    # Convert text to numerical features
    X = vectorizer.transform(texts)

    # Train model
    if first_chunk:

        model.partial_fit(
            X,
            labels,
            classes=[0, 1]
        )

        first_chunk = False

    else:

        model.partial_fit(
            X,
            labels
        )

    print(
        f"✅ Processed "
        f"{min(start + chunk_size, len(df))}"
        f"/{len(df)} emails"
    )


# =========================
# 6. Save model
# =========================

joblib.dump(
    model,
    "models/email_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/email_vectorizer.pkl"
)


print("\n🎉 EMAIL MODEL TRAINING COMPLETE!")

print(
    "Model saved: "
    "models/email_model.pkl"
)

print(
    "Vectorizer saved: "
    "models/email_vectorizer.pkl"
)
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


DATASET_PATH = "database/url_dataset.csv"
MODEL_PATH = "models/url_text_model.pkl"


print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Total URLs:", len(df))


# URL strings
X = df["url"].astype(str)

# Labels
y = df["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training character-level URL model...")


model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000
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


model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


print("\nModel Training Complete!")
print("-----------------------------")
print("Accuracy:", accuracy)
print("-----------------------------")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)
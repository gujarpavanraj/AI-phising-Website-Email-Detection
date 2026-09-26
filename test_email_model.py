import joblib
import pandas as pd


# ==============================
# Load model and vectorizer
# ==============================

email_model = joblib.load(
    "models/email_model.pkl"
)

email_vectorizer = joblib.load(
    "models/email_vectorizer.pkl"
)


# ==============================
# Load email dataset
# ==============================

df = pd.read_csv(
    "database/email_dataset.csv"
)


# Take first 30 emails
test_emails = df.head(30)


# ==============================
# Test 30 emails
# ==============================

correct = 0

print("\n========================================")
print("       TESTING 30 EMAILS")
print("========================================")


for index, row in test_emails.iterrows():

    email = row["body"]

    actual_label = int(row["label"])


    # Convert email into vector
    email_vector = email_vectorizer.transform(
        [email]
    )


    # Model prediction
    prediction = email_model.predict(
        email_vector
    )[0]


    # Probability
    probability = email_model.predict_proba(
        email_vector
    )[0][1] * 100


    # Convert prediction
    predicted_label = int(prediction)


    # Check whether prediction is correct
    if predicted_label == actual_label:
        result = "CORRECT"
        correct += 1
    else:
        result = "WRONG"


    print("\n----------------------------------------")
    print(f"Email #{index + 1}")

    print("Email:")
    print(email[:500])

    print(
        "\nActual:",
        "Phishing" if actual_label == 1 else "Legitimate"
    )

    print(
        "Predicted:",
        "Phishing" if predicted_label == 1 else "Legitimate"
    )

    print(
        f"Phishing Probability: {probability:.2f}%"
    )

    print("Result:", result)


# ==============================
# Final accuracy
# ==============================

accuracy = (correct / len(test_emails)) * 100


print("\n========================================")
print("           FINAL RESULT")
print("========================================")

print(f"Total Emails Tested: {len(test_emails)}")
print(f"Correct Predictions: {correct}")
print(f"Wrong Predictions: {len(test_emails) - correct}")
print(f"Accuracy: {accuracy:.2f}%")

print("========================================")
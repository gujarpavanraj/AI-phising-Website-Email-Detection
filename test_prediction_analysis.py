import joblib


# ============================================================
# PHISHGUARD - PREDICTION ANALYSIS
# ============================================================

MODEL_PATH = "models/url_text_model.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# TEST URLS
# ============================================================

test_urls = {

    "LEGITIMATE": [
        "https://google.com",
        "https://www.google.com",
        "https://youtube.com",
        "https://github.com",
        "https://microsoft.com",
        "https://amazon.com",
        "https://wikipedia.org",
        "https://stackoverflow.com",
        "https://chatgpt.com",
        "https://linkedin.com",
        "https://apple.com",
        "https://reddit.com",
        "https://instagram.com",
        "https://facebook.com",
        "https://netflix.com",
    ],

    "PHISHING": [
        "http://google-login-example.com",
        "http://paypal-login-example.com",
        "http://secure-login-example.com",
        "http://account-verify-example.com",
        "http://login-account-security-example.com/update",
        "http://192.168.1.100/login",
        "http://secure-account-verification-example.com/login",
        "http://paypal.com.login-example.com/verify",
        "http://account-security-example.com/verify",
        "http://verify-account-example.com/login",
        "http://secure-paypal-login-example.com",
        "http://google-account-verify-example.com",
        "http://login-security-example.com/update",
        "http://bank-account-verification-example.com",
        "http://password-reset-example.com/login",
    ]
}


# ============================================================
# ANALYSIS
# ============================================================

print("=" * 75)
print("PHISHGUARD - PREDICTION ANALYSIS")
print("=" * 75)


results = []


for category, urls in test_urls.items():

    print("\n" + "=" * 75)
    print(category)
    print("=" * 75)

    for url in urls:

        prediction = model.predict([url])[0]

        probabilities = model.predict_proba([url])[0]

        legitimate_probability = probabilities[0] * 100
        phishing_probability = probabilities[1] * 100

        if prediction == 1:
            predicted_label = "PHISHING"
        else:
            predicted_label = "LEGITIMATE"

        expected_label = category

        correct = predicted_label == expected_label

        if correct:
            result = "✓ CORRECT"
        else:
            result = "✗ WRONG"

        results.append({
            "url": url,
            "expected": expected_label,
            "predicted": predicted_label,
            "legitimate_probability": legitimate_probability,
            "phishing_probability": phishing_probability,
            "correct": correct
        })

        print("\nURL:")
        print(url)

        print("Expected :", expected_label)
        print("Predicted:", predicted_label)

        print(
            f"Legitimate Probability: "
            f"{legitimate_probability:.2f}%"
        )

        print(
            f"Phishing Probability: "
            f"{phishing_probability:.2f}%"
        )

        print("Result:", result)

        print("-" * 75)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("PREDICTION ANALYSIS SUMMARY")
print("=" * 75)


total = len(results)

correct = sum(
    result["correct"]
    for result in results
)

wrong = total - correct

accuracy = (correct / total) * 100


print("\nTotal URLs tested:", total)
print("Correct predictions:", correct)
print("Wrong predictions:", wrong)

print(
    f"Overall test accuracy: {accuracy:.2f}%"
)


# ============================================================
# LEGITIMATE URL ANALYSIS
# ============================================================

legitimate_results = [
    r for r in results
    if r["expected"] == "LEGITIMATE"
]

legitimate_correct = sum(
    r["correct"]
    for r in legitimate_results
)

legitimate_accuracy = (
    legitimate_correct /
    len(legitimate_results)
) * 100


print("\n" + "=" * 75)
print("LEGITIMATE URL ANALYSIS")
print("=" * 75)

print(
    f"Correct: {legitimate_correct}/"
    f"{len(legitimate_results)}"
)

print(
    f"Legitimate URL accuracy: "
    f"{legitimate_accuracy:.2f}%"
)


# ============================================================
# PHISHING URL ANALYSIS
# ============================================================

phishing_results = [
    r for r in results
    if r["expected"] == "PHISHING"
]

phishing_correct = sum(
    r["correct"]
    for r in phishing_results
)

phishing_accuracy = (
    phishing_correct /
    len(phishing_results)
) * 100


print("\n" + "=" * 75)
print("PHISHING URL ANALYSIS")
print("=" * 75)

print(
    f"Correct: {phishing_correct}/"
    f"{len(phishing_results)}"
)

print(
    f"Phishing URL accuracy: "
    f"{phishing_accuracy:.2f}%"
)


# ============================================================
# WRONG PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("WRONG PREDICTIONS")
print("=" * 75)


wrong_results = [
    r for r in results
    if not r["correct"]
]


if len(wrong_results) == 0:

    print("\nNo wrong predictions found!")

else:

    for r in wrong_results:

        print("\nURL:")
        print(r["url"])

        print("Expected :", r["expected"])
        print("Predicted:", r["predicted"])

        print(
            f"Legitimate Probability: "
            f"{r['legitimate_probability']:.2f}%"
        )

        print(
            f"Phishing Probability: "
            f"{r['phishing_probability']:.2f}%"
        )

        print("-" * 75)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("ANALYSIS COMPLETED")
print("=" * 75)
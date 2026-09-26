import joblib


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "models/url_text_model.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# REAL-WORLD TEST URLs
# ============================================================

test_urls = {

    # --------------------------------------------------------
    # LEGITIMATE URLS
    # --------------------------------------------------------

    "Legitimate": [
        "https://google.com",
        "https://youtube.com",
        "https://github.com",
        "https://microsoft.com",
        "https://amazon.com",
        "https://wikipedia.org",
        "https://stackoverflow.com",
        "https://chatgpt.com",
        "https://linkedin.com",
        "https://apple.com",
    ],


    # --------------------------------------------------------
    # SUSPICIOUS / PHISHING-LIKE URLS
    # --------------------------------------------------------

    "Phishing": [
        "http://google-login-example.com",
        "http://paypal-login-example.com",
        "http://secure-login-example.com",
        "http://account-verify-example.com",
        "http://login-account-security-example.com/update",
        "http://192.168.1.100/login",
        "http://secure-account-verification-example.com/login",
        "http://paypal.com.login-example.com/verify",
        "http://google-login-example.com",
        "http://account-security-example.com/verify",
    ]
}


# ============================================================
# TEST MODEL
# ============================================================

print("=" * 70)
print("PHISHGUARD - REAL WORLD URL TEST")
print("=" * 70)


correct = 0
total = 0


for category, urls in test_urls.items():

    print("\n" + "=" * 70)
    print(category.upper() + " URLS")
    print("=" * 70)

    for url in urls:

        prediction = model.predict([url])[0]

        probabilities = model.predict_proba([url])[0]

        legitimate_probability = probabilities[0] * 100
        phishing_probability = probabilities[1] * 100

        predicted_label = (
            "PHISHING"
            if prediction == 1
            else "LEGITIMATE"
        )

        expected_label = category.upper()

        if predicted_label == expected_label:
            correct += 1

        total += 1

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

        print("-" * 70)


# ============================================================
# FINAL RESULT
# ============================================================

accuracy = (correct / total) * 100

print("\n" + "=" * 70)
print("REAL-WORLD TEST RESULT")
print("=" * 70)

print(f"Correct predictions: {correct}/{total}")
print(f"Real-world test accuracy: {accuracy:.2f}%")

print("=" * 70)
import joblib


# ============================================================
# PHISHGUARD - URL MODEL TESTING
# ============================================================

MODEL_PATH = "models/url_text_model.pkl"


print("=" * 70)
print("PHISHGUARD - URL MODEL TESTING")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained URL model...")

model = joblib.load(MODEL_PATH)

print("✓ Model loaded successfully!")


# ============================================================
# TEST URLS
# ============================================================

test_urls = [

    # --------------------------------------------------------
    # LEGITIMATE WEBSITES
    # --------------------------------------------------------

    "https://google.com",

    "https://youtube.com",

    "https://github.com",

    "https://microsoft.com",

    "https://amazon.in",

    "https://linkedin.com",

    "https://wikipedia.org",

    "https://stackoverflow.com",

    "https://chatgpt.com",

    "https://chatgpt.com/g/g-p-123456789/example/conversation/abcdef123456789",

    "https://github.com/microsoft/vscode",

    "https://stackoverflow.com/questions/123456/example-question",

    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",


    # --------------------------------------------------------
    # SUSPICIOUS / PHISHING-LIKE URLs
    # --------------------------------------------------------

    "http://paypal-login-example.com",

    "http://secure-login-example.com",

    "http://account-verify-example.com",

    "http://google-login-example.com",

    "http://microsoft-security-example.com/login",

    "http://192.168.1.100/login",

    "http://secure-account-verification-example.com/login",

    "http://paypal.com.login-example.com/verify",

    "http://login-account-security-example.com/update",

]


# ============================================================
# PREDICTION
# ============================================================

print("\n")
print("=" * 70)
print("MODEL PREDICTIONS")
print("=" * 70)


for url in test_urls:

    prediction = model.predict([url])[0]

    probabilities = model.predict_proba([url])[0]

    legitimate_probability = probabilities[0] * 100
    phishing_probability = probabilities[1] * 100


    if prediction == 1:
        result = "PHISHING"
    else:
        result = "LEGITIMATE"


    print("\nURL:")
    print(url)

    print("Prediction:", result)

    print(
        f"Legitimate Probability: {legitimate_probability:.2f}%"
    )

    print(
        f"Phishing Probability:   {phishing_probability:.2f}%"
    )

    print("-" * 70)


print("\nTesting completed!")
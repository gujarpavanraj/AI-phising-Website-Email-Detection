import re
import joblib
import pandas as pd

from urllib.parse import urlparse
from scipy.sparse import hstack


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/hybrid_url_model.pkl"


# ============================================================
# URL FEATURE EXTRACTION
# MUST MATCH TRAINING CODE EXACTLY
# ============================================================

def extract_url_features(url):

    url = str(url)

    try:
        parsed = urlparse(url)
    except Exception:
        parsed = None

    hostname = parsed.hostname if parsed and parsed.hostname else ""

    features = []

    # 1. URL length
    features.append(len(url))

    # 2. Hostname length
    features.append(len(hostname))

    # 3. Number of dots
    features.append(url.count("."))

    # 4. Number of hyphens
    features.append(url.count("-"))

    # 5. Number of slashes
    features.append(url.count("/"))

    # 6. Number of question marks
    features.append(url.count("?"))

    # 7. Number of equals
    features.append(url.count("="))

    # 8. Number of @ symbols
    features.append(url.count("@"))

    # 9. Number of digits
    features.append(
        sum(c.isdigit() for c in url)
    )

    # 10. Number of special characters
    special_chars = "@?=&%_"

    features.append(
        sum(c in special_chars for c in url)
    )

    # 11. HTTPS
    features.append(
        1 if url.lower().startswith("https://") else 0
    )

    # 12. IP address
    has_ip = bool(
        re.search(
            r"https?://(?:\d{1,3}\.){3}\d{1,3}",
            url
        )
    )

    features.append(
        1 if has_ip else 0
    )

    # 13. Suspicious keywords
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

    # 14. Subdomain count
    subdomain_count = 0

    if hostname:

        parts = hostname.split(".")

        if len(parts) > 2:
            subdomain_count = len(parts) - 2

    features.append(subdomain_count)

    # 15. URL depth
    path = parsed.path if parsed else ""

    features.append(
        len(
            [
                x
                for x in path.split("/")
                if x
            ]
        )
    )

    return features


# ============================================================
# LOAD HYBRID MODEL
# ============================================================

print("=" * 70)
print("PHISHGUARD - HYBRID URL MODEL REAL-WORLD TEST")
print("=" * 70)

print("\nLoading hybrid model...")

model = joblib.load(MODEL_PATH)

print("Hybrid model loaded successfully!")


# ============================================================
# CHECK MODEL
# ============================================================

print("\nMODEL TYPE:")
print(type(model))

print("\nMODEL CONTENT:")
print(model.keys())

vectorizer = model["vectorizer"]
classifier = model["classifier"]

print("\nVectorizer:")
print(type(vectorizer))

print("\nClassifier:")
print(type(classifier))


# ============================================================
# CHECK FEATURE COUNT
# ============================================================

print("\nMODEL FEATURE INFORMATION")
print("-" * 70)

print(
    "TF-IDF vocabulary size:",
    len(vectorizer.vocabulary_)
)

print(
    "Classifier expected features:",
    classifier.n_features_in_
)

print(
    "Handcrafted URL features:",
    len(extract_url_features("https://example.com"))
)


# ============================================================
# REAL-WORLD TEST URLS
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
        "https://reddit.com",
        "https://instagram.com",
        "https://facebook.com",
        "https://netflix.com",
        "https://www.google.com",

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
        "http://account-security-example.com/verify",
        "http://verify-your-account-example.com/login",
        "http://secure-paypal-login-example.com",
        "http://google-account-security-example.com/verify",
        "http://bank-login-verification-example.com",
        "http://amazon-account-verify-example.com/login",
        "http://facebook-security-verification-example.com",

    ]
}


# ============================================================
# TESTING
# ============================================================

print("\n" + "=" * 70)
print("TESTING REAL-WORLD URLS")
print("=" * 70)


correct = 0
total = 0

legitimate_correct = 0
legitimate_total = 0

phishing_correct = 0
phishing_total = 0


# ============================================================
# URL TEST LOOP
# ============================================================

for category, urls in test_urls.items():

    print("\n" + "=" * 70)
    print(category.upper() + " URLS")
    print("=" * 70)

    for url in urls:

        # ----------------------------------------------------
        # CREATE TF-IDF FEATURES
        # ----------------------------------------------------

        tfidf_features = vectorizer.transform([url])

        # ----------------------------------------------------
        # CREATE HANDCRAFTED FEATURES
        # ----------------------------------------------------

        url_features = extract_url_features(url)

        url_features_df = pd.DataFrame(
            [url_features]
        )

        # ----------------------------------------------------
        # COMBINE TF-IDF + HANDCRAFTED FEATURES
        # ----------------------------------------------------

        combined_features = hstack(
            [
                tfidf_features,
                url_features_df.values
            ]
        )

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if combined_features.shape[1] != classifier.n_features_in_:

            print("\nFEATURE MISMATCH!")
            print(
                "Generated features:",
                combined_features.shape[1]
            )
            print(
                "Expected features:",
                classifier.n_features_in_
            )

            raise ValueError(
                "Feature count mismatch between "
                "training and testing."
            )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = classifier.predict(
            combined_features
        )[0]

        probabilities = classifier.predict_proba(
            combined_features
        )[0]

        legitimate_probability = probabilities[0] * 100
        phishing_probability = probabilities[1] * 100

        # ----------------------------------------------------
        # CONVERT PREDICTION TO LABEL
        # ----------------------------------------------------

        if prediction == 1:

            predicted_label = "PHISHING"

        else:

            predicted_label = "LEGITIMATE"

        expected_label = category.upper()

        # ----------------------------------------------------
        # CHECK CORRECTNESS
        # ----------------------------------------------------

        if predicted_label == expected_label:

            correct += 1

            if expected_label == "LEGITIMATE":

                legitimate_correct += 1

            else:

                phishing_correct += 1

        total += 1

        # ----------------------------------------------------
        # TOTAL COUNTS
        # ----------------------------------------------------

        if expected_label == "LEGITIMATE":

            legitimate_total += 1

        else:

            phishing_total += 1

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        print("\nURL:")
        print(url)

        print("\nExpected :",
              expected_label)

        print("Predicted:",
              predicted_label)

        print(
            f"Legitimate Probability: "
            f"{legitimate_probability:.2f}%"
        )

        print(
            f"Phishing Probability: "
            f"{phishing_probability:.2f}%"
        )

        if predicted_label == expected_label:

            print("Result   : CORRECT")

        else:

            print("Result   : WRONG")

        print("-" * 70)


# ============================================================
# FINAL RESULTS
# ============================================================

overall_accuracy = (
    correct / total
) * 100

legitimate_accuracy = (
    legitimate_correct / legitimate_total
) * 100

phishing_accuracy = (
    phishing_correct / phishing_total
) * 100


print("\n" + "=" * 70)
print("HYBRID MODEL REAL-WORLD TEST RESULT")
print("=" * 70)

print(
    f"\nTotal URLs tested       : {total}"
)

print(
    f"Correct predictions     : {correct}"
)

print(
    f"Wrong predictions       : {total - correct}"
)

print(
    f"\nOverall accuracy        : "
    f"{overall_accuracy:.2f}%"
)


print("\nLEGITIMATE URL ANALYSIS")
print("-" * 70)

print(
    f"Correct: "
    f"{legitimate_correct}/{legitimate_total}"
)

print(
    f"Legitimate URL accuracy: "
    f"{legitimate_accuracy:.2f}%"
)


print("\nPHISHING URL ANALYSIS")
print("-" * 70)

print(
    f"Correct: "
    f"{phishing_correct}/{phishing_total}"
)

print(
    f"Phishing URL accuracy: "
    f"{phishing_accuracy:.2f}%"
)


print("\n" + "=" * 70)
print("HYBRID MODEL TEST COMPLETED")
print("=" * 70)
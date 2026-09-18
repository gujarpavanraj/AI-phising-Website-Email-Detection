import re


def extract_email_features(email):
    features = {}

    email_lower = email.lower()

    # 1. Email length
    features["email_length"] = len(email)

    # 2. Number of URLs
    urls = re.findall(r"https?://\S+|www\.\S+", email_lower)
    features["url_count"] = len(urls)

    # 3. Suspicious keywords
    suspicious_keywords = [
        "verify",
        "verification",
        "password",
        "account",
        "login",
        "urgent",
        "suspended",
        "click here",
        "confirm",
        "security alert",
        "payment",
        "bank",
        "otp",
        "refund"
    ]

    keyword_count = 0

    for keyword in suspicious_keywords:
        if keyword in email_lower:
            keyword_count += 1

    features["suspicious_keyword_count"] = keyword_count

    # 4. Urgency indicators
    urgency_words = [
        "urgent",
        "immediately",
        "as soon as possible",
        "within 24 hours",
        "action required"
    ]

    urgency_count = 0

    for word in urgency_words:
        if word in email_lower:
            urgency_count += 1

    features["urgency_count"] = urgency_count

    # 5. Credential-related words
    credential_words = [
        "password",
        "username",
        "login",
        "otp",
        "pin",
        "credentials"
    ]

    credential_count = 0

    for word in credential_words:
        if word in email_lower:
            credential_count += 1

    features["credential_count"] = credential_count

    # 6. @ symbols
    features["at_symbol_count"] = email.count("@")

    # 7. Exclamation marks
    features["exclamation_count"] = email.count("!")

    return features
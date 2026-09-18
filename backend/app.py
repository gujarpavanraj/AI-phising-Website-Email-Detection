from flask import Flask, request, render_template
from backend.services.url_features import extract_url_features
from backend.services.email_features import extract_email_features

import joblib
from pathlib import Path

app = Flask(__name__)

# -----------------------------
# Load trained ML model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "url_text_model.pkl"

url_model = joblib.load(MODEL_PATH)

print("Character-level URL ML model loaded successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/check", methods=["POST"])
def check():
    url = request.form.get("url", "").strip()

    if not url:
        return "Please enter a URL.", 400

    # -----------------------------
    # Extract URL features
    # -----------------------------
    features = extract_url_features(url)

    # -----------------------------
    # AI / ML prediction
    # -----------------------------
    # Character-level model works directly with URL text
    prediction = int(url_model.predict([url])[0])

    phishing_probability = float(
        url_model.predict_proba([url])[0][1]
    )

    # Clean percentage values for the UI
    phishing_probability_percent = round(
        phishing_probability * 100, 2
    )

    legitimate_probability_percent = round(
        100 - phishing_probability_percent, 2
    )

    if prediction == 1:
        ai_status = "AI: Potential Phishing"
    else:
        ai_status = "AI: Likely Legitimate"

    # -----------------------------
    # Rule-Based Risk Scoring
    # -----------------------------
    risk_score = 0
    indicators = []

    # HTTPS
    if features["uses_https"] == 0:
        risk_score += 15
        indicators.append("Website does not use HTTPS")

    # IP address instead of domain
    if features["has_ip"] == 1:
        risk_score += 25
        indicators.append(
            "URL uses an IP address instead of a domain name"
        )

    # @ symbol
    if features["has_at_symbol"] == 1:
        risk_score += 20
        indicators.append("URL contains an @ symbol")

    # Suspicious keywords
    if features["suspicious_keyword_count"] > 0:
        score = min(
            features["suspicious_keyword_count"] * 10,
            25
        )

        risk_score += score

        indicators.append(
            f"Contains {features['suspicious_keyword_count']} "
            f"suspicious keyword(s)"
        )

    # Too many subdomains
    if features["subdomain_count"] >= 3:
        risk_score += 15
        indicators.append("URL contains multiple subdomains")

    # Long URL
    if features["url_length"] > 75:
        risk_score += 10
        indicators.append("URL is unusually long")

    # URL shortener
    if features["is_shortened"] == 1:
        risk_score += 20
        indicators.append(
            "URL appears to use a URL shortening service"
        )

    # Special characters
    if features["special_char_count"] >= 5:
        risk_score += 10
        indicators.append(
            "URL contains many special characters"
        )

    # Keep rule-based score between 0 and 100
    rule_risk_score = min(risk_score, 100)

    # -----------------------------
    # Combine ML + Rule-Based Risk
    # -----------------------------

    # ML phishing probability converted to 0-100
    ml_risk_score = phishing_probability * 100

    # Weighted combined risk score
    final_risk_score = (
        (ml_risk_score * 0.70) +
        (rule_risk_score * 0.30)
    )

    final_risk_score = round(final_risk_score)

    # -----------------------------
    # Final Risk Level
    # -----------------------------
    if final_risk_score >= 70:
        risk_level = "High Risk"
        status = "Potential Phishing"

    elif final_risk_score >= 40:
        risk_level = "Medium Risk"
        status = "Suspicious"

    else:
        risk_level = "Low Risk"
        status = "Likely Safe"

    # If no indicators were found
    if not indicators:
        indicators.append(
            "No major phishing indicators detected"
        )

    # -----------------------------
    # Analysis Explanation
    # -----------------------------
    explanation = (
        f"The ML model estimated a phishing probability of "
        f"{phishing_probability_percent}%. "
        f"The rule-based analysis produced a risk score of "
        f"{rule_risk_score}/100. "
        f"After combining both analyses, the final risk score is "
        f"{final_risk_score}/100, classified as "
        f"{risk_level.lower()}."
    )

    # -----------------------------
    # Send results to webpage
    # -----------------------------
    return render_template(
        "result.html",
        risk_score=final_risk_score,
        ml_risk_score=round(ml_risk_score, 2),
        rule_risk_score=rule_risk_score,
        risk_level=risk_level,
        status=status,
        ai_status=ai_status,
        ai_confidence=phishing_probability_percent,
        legitimate_probability=legitimate_probability_percent,
        analyzed_input=url,
        indicators=indicators,
        explanation=explanation,
        features=features
    )


@app.route("/check-email", methods=["POST"])
def check_email():
    email_text = request.form.get("email", "").strip()

    if not email_text:
        return "Please enter an email.", 400

    # -----------------------------
    # Extract email features
    # -----------------------------
    features = extract_email_features(email_text)

    # -----------------------------
    # Risk scoring
    # -----------------------------
    risk_score = 0
    indicators = []

    # Suspicious keywords
    if features["suspicious_keyword_count"] > 0:
        score = min(
            features["suspicious_keyword_count"] * 8,
            30
        )

        risk_score += score

        indicators.append(
            f"Email contains "
            f"{features['suspicious_keyword_count']} "
            f"suspicious keyword(s)"
        )

    # Urgency
    if features["urgency_count"] > 0:
        risk_score += min(
            features["urgency_count"] * 10,
            20
        )

        indicators.append(
            "Email uses urgent or pressure-based language"
        )

    # Credential requests
    if features["credential_count"] > 0:
        risk_score += min(
            features["credential_count"] * 10,
            25
        )

        indicators.append(
            "Email contains credential-related terms"
        )

    # Links
    if features["url_count"] > 0:
        risk_score += min(
            features["url_count"] * 5,
            15
        )

        indicators.append(
            f"Email contains {features['url_count']} link(s)"
        )

    # Excessive exclamation marks
    if features["exclamation_count"] >= 3:
        risk_score += 10

        indicators.append(
            "Email contains excessive exclamation marks"
        )

    # Keep score between 0 and 100
    risk_score = min(risk_score, 100)

    # -----------------------------
    # Risk level
    # -----------------------------
    if risk_score >= 70:
        risk_level = "High Risk"
        status = "Potential Phishing"

    elif risk_score >= 40:
        risk_level = "Medium Risk"
        status = "Suspicious"

    else:
        risk_level = "Low Risk"
        status = "Likely Safe"

    # If no indicators were found
    if not indicators:
        indicators.append(
            "No major phishing indicators detected"
        )

    explanation = (
        f"The email received a risk score of {risk_score}/100. "
        f"The system classified it as {risk_level.lower()}."
    )

    return render_template(
        "result.html",
        risk_score=risk_score,
        risk_level=risk_level,
        status=status,
        analyzed_input=email_text,
        indicators=indicators,
        explanation=explanation,
        features=features
    )


if __name__ == "__main__":
    app.run(debug=True)
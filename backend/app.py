import os
import re
import joblib

from flask import Flask, request, render_template, send_file

from services.url_features import extract_url_features
from services.email_features import extract_email_features

from io import BytesIO
from datetime import datetime
from urllib.parse import urlparse

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

import uuid
from pathlib import Path

from scipy.sparse import hstack, csr_matrix


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# TRUSTED DOMAINS
# =========================================================

TRUSTED_DOMAINS = {
    "chatgpt.com",
    "openai.com",
    "google.com",
    "youtube.com",
    "github.com",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "x.com",
}


def is_trusted_domain(url):

    try:

        if not url.startswith(("http://", "https://")):
            url_to_parse = "https://" + url
        else:
            url_to_parse = url

        parsed = urlparse(url_to_parse)

        hostname = parsed.hostname

        if not hostname:
            return False

        hostname = hostname.lower().strip()

        return hostname in TRUSTED_DOMAINS

    except Exception:

        return False


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# LOAD HYBRID URL MODEL
# =========================================================

URL_MODEL_PATH = BASE_DIR / "models" / "hybrid_url_model.pkl"

url_model_package = joblib.load(URL_MODEL_PATH)

url_vectorizer = url_model_package["vectorizer"]
url_classifier = url_model_package["classifier"]

print("==============================================")
print("PHISHGUARD")
print("==============================================")
print("Hybrid URL model loaded successfully!")
print("Vectorizer loaded successfully!")
print("Classifier loaded successfully!")


# =========================================================
# LOAD EMAIL ML MODEL
# =========================================================

EMAIL_MODEL_PATH = os.path.join(BASE_DIR, "models", "email_model.pkl")

EMAIL_VECTORIZER_PATH = (
    BASE_DIR / "models" / "email_vectorizer.pkl"
)

email_model = joblib.load(
    EMAIL_MODEL_PATH
)

email_vectorizer = joblib.load(
    EMAIL_VECTORIZER_PATH
)

print("Email ML model loaded successfully!")


# =========================================================
# HYBRID URL FEATURE EXTRACTION
#
# IMPORTANT:
# This MUST match the exact order used during
# train_hybrid_url_model.py
#
# Total handcrafted features = 15
# =========================================================

def extract_hybrid_numeric_features(url):

    url = str(url)

    try:

        parsed = urlparse(url)

    except Exception:

        parsed = None

    hostname = (
        parsed.hostname
        if parsed and parsed.hostname
        else ""
    )

    # -----------------------------------------------------
    # 1. URL length
    # -----------------------------------------------------

    url_length = len(url)

    # -----------------------------------------------------
    # 2. Hostname length
    # -----------------------------------------------------

    hostname_length = len(hostname)

    # -----------------------------------------------------
    # 3. Number of dots
    # -----------------------------------------------------

    dot_count = url.count(".")

    # -----------------------------------------------------
    # 4. Number of hyphens
    # -----------------------------------------------------

    hyphen_count = url.count("-")

    # -----------------------------------------------------
    # 5. Number of slashes
    # -----------------------------------------------------

    slash_count = url.count("/")

    # -----------------------------------------------------
    # 6. Number of question marks
    # -----------------------------------------------------

    question_count = url.count("?")

    # -----------------------------------------------------
    # 7. Number of equals
    # -----------------------------------------------------

    equals_count = url.count("=")

    # -----------------------------------------------------
    # 8. Number of @ symbols
    # -----------------------------------------------------

    at_count = url.count("@")

    # -----------------------------------------------------
    # 9. Number of digits
    # -----------------------------------------------------

    digit_count = sum(
        c.isdigit()
        for c in url
    )

    # -----------------------------------------------------
    # 10. Number of special characters
    # -----------------------------------------------------

    special_chars = "@?=&%_"

    special_char_count = sum(
        c in special_chars
        for c in url
    )

    # -----------------------------------------------------
    # 11. HTTPS
    # -----------------------------------------------------

    uses_https = (
        1
        if url.lower().startswith("https://")
        else 0
    )

    # -----------------------------------------------------
    # 12. IP address
    # -----------------------------------------------------

    has_ip = bool(
        re.search(
            r"https?://(?:\d{1,3}\.){3}\d{1,3}",
            url
        )
    )

    has_ip = 1 if has_ip else 0

    # -----------------------------------------------------
    # 13. Suspicious keywords
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # 14. Subdomain count
    # -----------------------------------------------------

    subdomain_count = 0

    if hostname:

        parts = hostname.split(".")

        if len(parts) > 2:

            subdomain_count = len(parts) - 2

    # -----------------------------------------------------
    # 15. URL depth
    # -----------------------------------------------------

    path = (
        parsed.path
        if parsed
        else ""
    )

    path_depth = len(
        [
            x
            for x in path.split("/")
            if x
        ]
    )

    # -----------------------------------------------------
    # RETURN FEATURES
    #
    # EXACT SAME ORDER AS TRAINING
    # -----------------------------------------------------

    return [

        url_length,

        hostname_length,

        dot_count,

        hyphen_count,

        slash_count,

        question_count,

        equals_count,

        at_count,

        digit_count,

        special_char_count,

        uses_https,

        has_ip,

        suspicious_count,

        subdomain_count,

        path_depth

    ]


# =========================================================
# HYBRID URL PREDICTION
# =========================================================

def predict_hybrid_url(url):

    """
    The hybrid model was trained using:

        Character TF-IDF
                +
        15 handcrafted URL features

    TF-IDF = 100000 features

    Handcrafted = 15 features

    Total = 100015 features
    """

    # -----------------------------------------------------
    # STEP 1
    # Character-level TF-IDF
    # -----------------------------------------------------

    tfidf_features = (
        url_vectorizer.transform(
            [url]
        )
    )

    # -----------------------------------------------------
    # STEP 2
    # Handcrafted URL features
    # -----------------------------------------------------

    numeric_features = (
        extract_hybrid_numeric_features(
            url
        )
    )

    # -----------------------------------------------------
    # STEP 3
    # Convert handcrafted features
    # to sparse matrix
    # -----------------------------------------------------

    numeric_matrix = csr_matrix(
        [numeric_features],
        dtype=float
    )

    # -----------------------------------------------------
    # STEP 4
    # Combine TF-IDF + handcrafted features
    # -----------------------------------------------------

    combined_features = hstack(
        [
            tfidf_features,
            numeric_matrix
        ],
        format="csr"
    )

    # -----------------------------------------------------
    # SAFETY CHECK
    # -----------------------------------------------------

    expected_features = getattr(
        url_classifier,
        "n_features_in_",
        None
    )

    if (
        expected_features is not None
        and combined_features.shape[1]
        != expected_features
    ):

        raise ValueError(
            "HYBRID FEATURE MISMATCH: "
            f"Model expects "
            f"{expected_features} features, "
            f"but application created "
            f"{combined_features.shape[1]} features."
        )

    # -----------------------------------------------------
    # STEP 5
    # Prediction
    # -----------------------------------------------------

    prediction = int(
        url_classifier.predict(
            combined_features
        )[0]
    )

    # -----------------------------------------------------
    # STEP 6
    # Probability
    # -----------------------------------------------------

    probabilities = (
        url_classifier.predict_proba(
            combined_features
        )[0]
    )

    # -----------------------------------------------------
    # Map probabilities using actual class labels
    # -----------------------------------------------------

    class_probabilities = dict(
        zip(
            url_classifier.classes_,
            probabilities
        )
    )

    legitimate_probability = float(
        class_probabilities.get(
            0,
            0.0
        )
    )

    phishing_probability = float(
        class_probabilities.get(
            1,
            0.0
        )
    )

    return (
        prediction,
        phishing_probability,
        legitimate_probability
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# URL CHECKING
# =========================================================

@app.route(
    "/check",
    methods=["POST"]
)
def check():

    url = request.form.get(
        "url",
        ""
    ).strip()

    if not url:

        return (
            "Please enter a URL.",
            400
        )

    # -----------------------------------------------------
    # Extract URL Features
    # -----------------------------------------------------

    features = extract_url_features(
        url
    )

    # -----------------------------------------------------
    # HYBRID ML PREDICTION
    # -----------------------------------------------------

    (
        prediction,
        phishing_probability,
        legitimate_probability
    ) = predict_hybrid_url(
        url
    )

    # -----------------------------------------------------
    # Convert probabilities
    # -----------------------------------------------------

    phishing_probability_percent = round(
        phishing_probability * 100,
        2
    )

    legitimate_probability_percent = round(
        legitimate_probability * 100,
        2
    )

    # -----------------------------------------------------
    # AI STATUS
    # -----------------------------------------------------

    if prediction == 1:

        ai_status = (
            "AI: Potential Phishing"
        )

    else:

        ai_status = (
            "AI: Likely Legitimate"
        )

    # =====================================================
    # RULE-BASED URL RISK SCORING
    # =====================================================

    risk_score = 0

    indicators = []

    # -----------------------------------------------------
    # HTTPS
    # -----------------------------------------------------

    if features["uses_https"] == 0:

        risk_score += 15

        indicators.append(
            "Website does not use HTTPS"
        )

    # -----------------------------------------------------
    # IP address
    # -----------------------------------------------------

    if features["has_ip"] == 1:

        risk_score += 25

        indicators.append(
            "URL uses an IP address "
            "instead of a domain name"
        )

    # -----------------------------------------------------
    # @ symbol
    # -----------------------------------------------------

    if features["has_at_symbol"] == 1:

        risk_score += 20

        indicators.append(
            "URL contains an @ symbol"
        )

    # -----------------------------------------------------
    # Suspicious keywords
    # -----------------------------------------------------

    if (
        features[
            "suspicious_keyword_count"
        ] > 0
    ):

        score = min(
            features[
                "suspicious_keyword_count"
            ] * 10,
            25
        )

        risk_score += score

        indicators.append(
            f"Contains "
            f"{features['suspicious_keyword_count']} "
            f"suspicious keyword(s)"
        )

    # -----------------------------------------------------
    # Multiple subdomains
    # -----------------------------------------------------

    if (
        features["subdomain_count"] >= 3
    ):

        risk_score += 15

        indicators.append(
            "URL contains multiple subdomains"
        )

    # -----------------------------------------------------
    # Long URL
    # -----------------------------------------------------

    if (
        features["url_length"] > 75
    ):

        risk_score += 10

        indicators.append(
            "URL is unusually long"
        )

    # -----------------------------------------------------
    # URL shortener
    # -----------------------------------------------------

    if (
        features["is_shortened"] == 1
    ):

        risk_score += 20

        indicators.append(
            "URL appears to use a "
            "URL shortening service"
        )

    # -----------------------------------------------------
    # Special characters
    # -----------------------------------------------------

    if (
        features["special_char_count"] >= 5
    ):

        risk_score += 10

        indicators.append(
            "URL contains many "
            "special characters"
        )

    # -----------------------------------------------------
    # Limit rule score
    # -----------------------------------------------------

    rule_risk_score = min(
        risk_score,
        100
    )

    # =====================================================
    # COMBINE ML + RULE-BASED RISK
    # =====================================================

    ml_risk_score = (
        phishing_probability * 100
    )

    # 70% ML + 30% rules

    final_risk_score = (

        (
            ml_risk_score
            * 0.70
        )

        +

        (
            rule_risk_score
            * 0.30
        )

    )

    final_risk_score = round(
        final_risk_score
    )

    # -----------------------------------------------------
    # Final Risk Level
    # -----------------------------------------------------

    if final_risk_score >= 70:

        risk_level = "High Risk"

        status = (
            "Potential Phishing"
        )

    elif final_risk_score >= 40:

        risk_level = "Medium Risk"

        status = "Suspicious"

    else:

        risk_level = "Low Risk"

        status = "Likely Safe"

    # -----------------------------------------------------
    # Indicators
    # -----------------------------------------------------

    if not indicators:

        indicators.append(
            "No major phishing indicators detected"
        )

    # -----------------------------------------------------
    # Explanation
    # -----------------------------------------------------

    explanation = (

        f"The hybrid ML model estimated "
        f"a phishing probability of "
        f"{phishing_probability_percent}%. "

        f"The rule-based analysis produced "
        f"a risk score of "
        f"{rule_risk_score}/100. "

        f"After combining both analyses, "
        f"the final risk score is "
        f"{final_risk_score}/100, "
        f"classified as "
        f"{risk_level.lower()}."

    )

    # -----------------------------------------------------
    # Report ID
    # -----------------------------------------------------

    report_id = (

        "PG-"

        + datetime.now().strftime(
            "%Y%m%d"
        )

        + "-"

        + uuid.uuid4().hex[
            :6
        ].upper()

    )

    # -----------------------------------------------------
    # SEND RESULTS TO RESULT PAGE
    # -----------------------------------------------------

    return render_template(

        "result.html",

        risk_score=final_risk_score,

        ml_risk_score=round(
            ml_risk_score,
            2
        ),

        rule_risk_score=rule_risk_score,

        risk_level=risk_level,

        status=status,

        ai_status=ai_status,

        ai_confidence=(
            phishing_probability_percent
        ),

        legitimate_probability=(
            legitimate_probability_percent
        ),

        analyzed_input=url,

        indicators=indicators,

        explanation=explanation,

        features=features,

        input_type="Website URL",

        report_id=report_id

    )


# =========================================================
# EMAIL CHECKING
# =========================================================

@app.route(
    "/check-email",
    methods=["POST"]
)
def check_email():

    email_text = request.form.get(
        "email",
        ""
    ).strip()

    if not email_text:

        return (
            "Please enter an email.",
            400
        )

    # -----------------------------------------------------
    # Extract Email Features
    # -----------------------------------------------------

    features = (
        extract_email_features(
            email_text
        )
    )

    # -----------------------------------------------------
    # Convert email into TF-IDF
    # -----------------------------------------------------

    email_vector = (
        email_vectorizer.transform(
            [email_text]
        )
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = int(
        email_model.predict(
            email_vector
        )[0]
    )

    # -----------------------------------------------------
    # Probability
    # -----------------------------------------------------

    probabilities = (
        email_model.predict_proba(
            email_vector
        )[0]
    )

    email_class_probabilities = dict(
        zip(
            email_model.classes_,
            probabilities
        )
    )

    email_legitimate_probability = float(
        email_class_probabilities.get(
            0,
            0.0
        )
    )

    email_phishing_probability = float(
        email_class_probabilities.get(
            1,
            0.0
        )
    )

    # -----------------------------------------------------
    # Convert probability
    # -----------------------------------------------------

    email_phishing_probability_percent = round(
        email_phishing_probability * 100,
        2
    )

    email_legitimate_probability_percent = round(
        email_legitimate_probability * 100,
        2
    )

    # -----------------------------------------------------
    # AI status
    # -----------------------------------------------------

    if prediction == 1:

        ai_status = (
            "AI: Potential Phishing"
        )

    else:

        ai_status = (
            "AI: Likely Legitimate"
        )

    # =====================================================
    # RULE-BASED EMAIL RISK
    # =====================================================

    risk_score = 0

    indicators = []

    # -----------------------------------------------------
    # Suspicious keywords
    # -----------------------------------------------------

    if (
        features[
            "suspicious_keyword_count"
        ] > 0
    ):

        score = min(
            features[
                "suspicious_keyword_count"
            ] * 8,
            30
        )

        risk_score += score

        indicators.append(
            f"Email contains "
            f"{features['suspicious_keyword_count']} "
            f"suspicious keyword(s)"
        )

    # -----------------------------------------------------
    # Urgency
    # -----------------------------------------------------

    if (
        features["urgency_count"] > 0
    ):

        urgency_score = min(
            features["urgency_count"] * 10,
            20
        )

        risk_score += urgency_score

        indicators.append(
            "Email uses urgent or "
            "pressure-based language"
        )

    # -----------------------------------------------------
    # Credential requests
    # -----------------------------------------------------

    if (
        features["credential_count"] > 0
    ):

        credential_score = min(
            features["credential_count"] * 10,
            25
        )

        risk_score += credential_score

        indicators.append(
            "Email contains "
            "credential-related terms"
        )

    # -----------------------------------------------------
    # Links
    # -----------------------------------------------------

    if (
        features["url_count"] > 0
    ):

        link_score = min(
            features["url_count"] * 5,
            15
        )

        risk_score += link_score

        indicators.append(
            f"Email contains "
            f"{features['url_count']} link(s)"
        )

    # -----------------------------------------------------
    # Excessive exclamation marks
    # -----------------------------------------------------

    if (
        features["exclamation_count"] >= 3
    ):

        risk_score += 10

        indicators.append(
            "Email contains excessive "
            "exclamation marks"
        )

    # -----------------------------------------------------
    # Rule score limit
    # -----------------------------------------------------

    rule_risk_score = min(
        risk_score,
        100
    )

    # =====================================================
    # COMBINE EMAIL ML + RULES
    # =====================================================

    ml_risk_score = (
        email_phishing_probability * 100
    )

    final_risk_score = (

        (
            ml_risk_score
            * 0.70
        )

        +

        (
            rule_risk_score
            * 0.30
        )

    )

    final_risk_score = round(
        final_risk_score
    )

    risk_score = final_risk_score

    # =====================================================
    # FINAL EMAIL RISK LEVEL
    # =====================================================

    if risk_score >= 70:

        risk_level = "High Risk"

        status = (
            "Potential Phishing"
        )

    elif risk_score >= 40:

        risk_level = "Medium Risk"

        status = "Suspicious"

    else:

        risk_level = "Low Risk"

        status = "Likely Safe"

    # -----------------------------------------------------
    # Indicators
    # -----------------------------------------------------

    if not indicators:

        indicators.append(
            "No major phishing indicators detected"
        )

    # -----------------------------------------------------
    # Explanation
    # -----------------------------------------------------

    explanation = (

        f"The ML model estimated "
        f"a phishing probability of "
        f"{email_phishing_probability_percent}%. "

        f"The rule-based analysis produced "
        f"a risk score of "
        f"{rule_risk_score}/100. "

        f"After combining both analyses, "
        f"the final risk score is "
        f"{risk_score}/100, classified as "
        f"{risk_level.lower()}."

    )

    # -----------------------------------------------------
    # Report ID
    # -----------------------------------------------------

    report_id = (

        "PG-"

        + datetime.now().strftime(
            "%Y%m%d"
        )

        + "-"

        + uuid.uuid4().hex[
            :6
        ].upper()

    )

    # -----------------------------------------------------
    # RESULT PAGE
    # -----------------------------------------------------

    return render_template(

        "result.html",

        risk_score=risk_score,

        ml_risk_score=round(
            ml_risk_score,
            2
        ),

        rule_risk_score=rule_risk_score,

        risk_level=risk_level,

        status=status,

        ai_status=ai_status,

        ai_confidence=(
            email_phishing_probability_percent
        ),

        legitimate_probability=(
            email_legitimate_probability_percent
        ),

        analyzed_input=email_text,

        indicators=indicators,

        explanation=explanation,

        features=features,

        input_type="Email",

        report_id=report_id

    )


# =========================================================
# DOWNLOAD SECURITY REPORT
# =========================================================

@app.route(
    "/download-report",
    methods=["POST"]
)
def download_report():

    input_type = request.form.get(
        "input_type",
        "Email"
    )

    analyzed_input = request.form.get(
        "analyzed_input",
        ""
    )

    risk_score = request.form.get(
        "risk_score",
        "0"
    )

    risk_level = request.form.get(
        "risk_level",
        ""
    )

    status = request.form.get(
        "status",
        ""
    )

    ai_status = request.form.get(
        "ai_status",
        ""
    )

    ai_confidence = request.form.get(
        "ai_confidence",
        "0"
    )

    legitimate_probability = request.form.get(
        "legitimate_probability",
        "0"
    )

    rule_risk_score = request.form.get(
        "rule_risk_score",
        "0"
    )

    explanation = request.form.get(
        "explanation",
        ""
    )

    indicators = request.form.getlist(
        "indicators"
    )

    # -----------------------------------------------------
    # Create PDF in memory
    # -----------------------------------------------------

    buffer = BytesIO()

    doc = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=40,

        leftMargin=40,

        topMargin=40,

        bottomMargin=40

    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(

        "ReportTitle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontSize=20,

        spaceAfter=10

    )

    subtitle_style = ParagraphStyle(

        "Subtitle",

        parent=styles["Normal"],

        alignment=TA_CENTER,

        fontSize=10,

        spaceAfter=20

    )

    heading_style = ParagraphStyle(

        "Heading",

        parent=styles["Heading2"],

        fontSize=13,

        spaceBefore=12,

        spaceAfter=8

    )

    normal_style = ParagraphStyle(

        "NormalCustom",

        parent=styles["Normal"],

        fontSize=9,

        leading=13

    )

    story = []

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "PHISHGUARD",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Email / Website Security Analysis Report",
            subtitle_style
        )
    )

    # -----------------------------------------------------
    # REPORT INFORMATION
    # -----------------------------------------------------

    analysis_time = datetime.now().strftime(
        "%d %B %Y, %I:%M %p"
    )

    report_info = [

        [
            "Analysis Date",
            analysis_time
        ],

        [
            "Input Type",
            input_type
        ],

        [
            "Risk Score",
            f"{risk_score} / 100"
        ],

        [
            "Final Result",
            risk_level
        ],

        [
            "Status",
            status
        ]

    ]

    table = Table(
        report_info,
        colWidths=[
            130,
            350
        ]
    )

    table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                "Helvetica"
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

        ])

    )

    story.append(table)

    # -----------------------------------------------------
    # ANALYZED INPUT
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Analyzed Input",
            heading_style
        )
    )

    story.append(
        Paragraph(
            analyzed_input.replace(
                "\n",
                "<br/>"
            ),
            normal_style
        )
    )

    # -----------------------------------------------------
    # AI ANALYSIS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "AI Analysis",
            heading_style
        )
    )

    ai_table = Table(

        [

            [
                "AI Status",
                ai_status
            ],

            [
                "Phishing Probability",
                f"{ai_confidence}%"
            ],

            [
                "Legitimate Probability",
                f"{legitimate_probability}%"
            ],

            [
                "Rule-Based Risk Score",
                f"{rule_risk_score} / 100"
            ]

        ],

        colWidths=[
            180,
            300
        ]

    )

    ai_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

        ])

    )

    story.append(ai_table)

    # -----------------------------------------------------
    # DETECTION INDICATORS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Detection Indicators",
            heading_style
        )
    )

    if indicators:

        for indicator in indicators:

            story.append(
                Paragraph(
                    "• " + indicator,
                    normal_style
                )
            )

            story.append(
                Spacer(
                    1,
                    4
                )
            )

    else:

        story.append(
            Paragraph(
                "No major phishing indicators detected.",
                normal_style
            )
        )

    # -----------------------------------------------------
    # ANALYSIS EXPLANATION
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Analysis Explanation",
            heading_style
        )
    )

    story.append(
        Paragraph(
            explanation,
            normal_style
        )
    )

    # -----------------------------------------------------
    # DISCLAIMER
    # -----------------------------------------------------

    story.append(
        Spacer(
            1,
            20
        )
    )

    story.append(
        Paragraph(

            "<b>Disclaimer:</b> This report is generated by "
            "an automated phishing detection system. The result "
            "should be reviewed by a qualified cybersecurity "
            "professional before taking action.",

            normal_style

        )
    )

    # -----------------------------------------------------
    # BUILD PDF
    # -----------------------------------------------------

    doc.build(story)

    buffer.seek(0)

    return send_file(

        buffer,

        as_attachment=True,

        download_name=(
            "PhishGuard_Security_Report.pdf"
        ),

        mimetype="application/pdf"

    )


# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
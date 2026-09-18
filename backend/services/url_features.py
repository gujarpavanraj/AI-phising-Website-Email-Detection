from urllib.parse import urlparse
import re
import ipaddress
import math
from collections import Counter


def calculate_entropy(text):
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0

    counts = Counter(text)
    length = len(text)

    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def extract_url_features(url):

    parsed = urlparse(url)

    hostname = parsed.netloc
    path = parsed.path
    query = parsed.query

    features = {}

    # 1. URL length
    features["url_length"] = len(url)

    # 2. Hostname length
    features["hostname_length"] = len(hostname)

    # 3. Number of dots
    features["dot_count"] = url.count(".")

    # 4. Number of hyphens
    features["hyphen_count"] = url.count("-")

    # 5. Number of special characters
    features["special_char_count"] = len(
        re.findall(r"[@?=&%_]", url)
    )

    # 6. HTTPS usage
    features["uses_https"] = int(parsed.scheme == "https")

    # 7. Presence of @ symbol
    features["has_at_symbol"] = int("@" in url)

    # 8. Presence of IP address
    try:
        ipaddress.ip_address(hostname)
        features["has_ip"] = 1
    except ValueError:
        features["has_ip"] = 0

    # 9. Number of subdomains
    features["subdomain_count"] = max(
        0, len(hostname.split(".")) - 2
    )

    # 10. Suspicious keywords
    suspicious_words = [
        "login",
        "verify",
        "account",
        "update",
        "secure",
        "bank",
        "signin",
        "confirm",
        "password",
        "credential"
    ]

    features["suspicious_keyword_count"] = sum(
        word in url.lower()
        for word in suspicious_words
    )

    # 11. URL shortener detection
    shorteners = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "is.gd",
    "ow.ly"
}

    features["is_shortened"] = int(
    hostname.lower().strip(".") in shorteners
)

    # 12. Path length
    features["path_length"] = len(path)

    # 13. Number of digits
    features["digit_count"] = sum(char.isdigit() for char in url)

    # 14. Number of letters
    features["letter_count"] = sum(char.isalpha() for char in url)

    # 15. Digit ratio
    features["digit_ratio"] = (
        features["digit_count"] / len(url)
        if url else 0
    )

    # 16. Query length
    features["query_length"] = len(query)

    # 17. Number of query parameters
    features["query_parameter_count"] = (
        len(query.split("&"))
        if query else 0
    )

    # 18. Path depth
    features["path_depth"] = len(
        [part for part in path.split("/") if part]
    )

    # 19. URL entropy
    features["url_entropy"] = calculate_entropy(url)

    # 20. Hostname entropy
    features["hostname_entropy"] = calculate_entropy(hostname)

    # 21. Suspicious double slash in path
    features["double_slash_path"] = int("//" in path)

    return features
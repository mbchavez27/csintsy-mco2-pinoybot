import numpy as np


def get_ortographic_features(token: str):
    """
    Extract orthographic features for a given token.
    Args:
        token: The word token (string).
    Returns:
        features: A numpy array of orthographic features.
    """
    features = []

    token_lower = token.lower()
    length = len(token)

    # Basic orthographic features
    features.append(length)  # Length of the token
    features.append(int(token.islower()))  # All lowercase
    features.append(int(token.isupper()))  # All uppercase
    features.append(int(token.istitle()))  # Title case
    features.append(int(token.isdigit()))  # Numeric
    features.append(int(any(c.isdigit() for c in token)))  # Contains digit
    features.append(int(any(c.isalpha() for c in token)))  # Contains alphabetic
    features.append(int(any(not c.isalnum() for c in token)))  # Contains special char

    # Character composition
    vowels = set("aeiouAEIOU")
    vowel_ratio = sum(c in vowels for c in token) / max(length, 1)
    features.append(vowel_ratio)

    # Common Filipino / English letter patterns
    patterns = ["ng", "mga", "th", "sh", "ay", "ts", "ch"]
    features.extend(int(p in token_lower) for p in patterns)

    # Prefixes / suffixes (morphological clues)
    filipino_prefixes = ["mag", "nag", "pag", "ka", "pa", "ma", "pinaka"]
    filipino_suffixes = ["in", "an", "ang", "han", "hin", "ing"]
    english_suffixes = [
        "ing",
        "tion",
        "ness",
        "ment",
        "ly",
        "ed",
        "ee",
        "er",
        "ist",
        "able",
    ]

    features.append(int(any(token_lower.startswith(p) for p in filipino_prefixes)))
    features.append(
        int(any(token_lower.endswith(s) for s in filipino_suffixes + english_suffixes))
    )

    # Explicit "-ee" ending
    features.append(int(token_lower.endswith("ee")))

    # Non-ASCII characters
    features.append(int(any(ord(c) > 127 for c in token)))

    return np.array(features)

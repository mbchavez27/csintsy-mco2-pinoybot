import calamancy

# Load English and Tagalog pipelines
nlp = calamancy.load("tl_calamancy_md-0.1.0")

abbr_pattern = r"^([A-Z]\.){2,}$|^[A-Z]{2,5}$"


def is_abbreviation(token):
    if token.isupper():
        if 2 <= len(token) <= 5 and re.match(abbr_pattern, token):
            return True
        else:
            return False
    return False


def classify_if_is_ne(token: str) -> str:
    """
    Classifies if a token is a named entity.
    Args:
        token: The word token (string).
    Returns:
        "ABB_NE" -> abbreviation named entity
        "NE" -> named entity
        "NONE" -> neither
        "EXPR" -> expression
    """
    doc = nlp(token)
    ent_type = doc[0].ent_type_
    is_ne = bool(ent_type)
    is_abbr = is_abbreviation(token)

    if is_ne and is_abbr:
        return "ABB_NE"
    elif is_ne:
        return "NE"
    elif is_abbr:
        return "ABB"
    else:
        return "NONE"

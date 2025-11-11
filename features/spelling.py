import calamancy
import pandas as pd
from spellchecker import SpellChecker

# Load English and Tagalog pipelines
nlp = calamancy.load("tl_calamancy_md-0.1.0")

# Generate set of correctly spelled words
spellings = pd.read_csv("data/final_annotations.csv")

correct_words = set(
    spellings.loc[spellings["is_spelling_correct"] == True, "word"].str.lower()
)

spell = SpellChecker()


def classify_if_is_spelling_correct(token: str) -> int:
    """
    Classifies if a token is spelled correctly.
    Args:
        token: The word token (string).
    Returns:
        is_correct: 1 if spelled correctly, 0 otherwise.
    """
    doc = nlp(token)
    tokens = [t.text.lower() for t in doc if t.text.strip()]

    def is_word_correct(word):
        if word in correct_words:
            return True
        if word in spell:
            return True
        return False

    return 1 if all(is_word_correct(t) for t in tokens) else 0

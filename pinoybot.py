"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
import sys
from typing import List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
from nltk import word_tokenize
import spacy
import re


MODEL_PATH = "models/language_rf.pkl"
PCA_PATH = "models/language_pca.pkl"


name_checker = spacy.load("xx_ent_wiki_sm")


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
    abbr_pattern = r"^([A-Z0-9]\.?)+$"

    doc = name_checker(token)

    is_ne = bool(doc[0].ent_type_)
    is_abbr = bool(re.match(abbr_pattern, token))

    if is_ne and is_abbr:
        return "ABB_NE"
    elif is_ne:
        return "NE"
    elif is_abbr:
        return "ABB"
    else:
        return "NONE"


# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Please train the model first."
        )

    # Load model and PCA
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(PCA_PATH, "rb") as f:
        pca = pickle.load(f)

    # Generate embeddings for input tokens
    embedder = SentenceTransformer("all-mpnet-base-v2")

    features = []
    ne_categories = ["ABB", "ABB_EXPR", "ABB_NE", "EXPR", "NE", "NONE"]

    for token in tokens:
        embedding = embedder.encode([str(token)])[0].reshape(1, -1)

        # classify if token is a Named Entity
        ne_token = classify_if_is_ne(token)

        # Create one-hot encoding for NE feature
        is_ne_onehot = np.zeros((1, len(ne_categories)))
        if ne_token in ne_categories:
            is_ne_onehot[0, ne_categories.index(ne_token)] = 1

        # is_spelling = classify_if_is_spelling_correct(token)

        combined = np.hstack([embedding, is_ne_onehot])
        features.append(combined)

    # Stack all token feature vectors
    X = np.vstack(features)

    # Apply PCA
    X = pca.transform(X)

    # Predict tags
    predicted = model.predict(X)

    return [str(tag) for tag in predicted]


if __name__ == "__main__":
    # Download NLTK resources
    nltk.download("punkt")

    if len(sys.argv) > 1:
        sentence = sys.argv[1]
    else:
        sentence = input("Enter a sentence: ")

    print("\n-----\n")

    tokens = word_tokenize(sentence)
    tags = tag_language(tokens)
    print("\n-----\n")
    print("Tokens:", tokens)
    print("\n-----\n")
    print("Tags:", tags)
    print("\n-----\n")

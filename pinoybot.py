"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

# Standard Libraries
import os
import pickle
import sys
from typing import List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
from nltk import word_tokenize

# Import feature extraction functions
from features.named_entity import classify_if_is_ne
from features.spelling import classify_if_is_spelling_correct
from features.ortographic import get_ortographic_features


# Load Classifier Model and PCA
MODEL_PATH = "artifacts/language_xgboost.pkl"
PCA_PATH = "artifacts/language_pca.pkl"

# Load MPNet model for embeddings
mpnet_model = SentenceTransformer("all-mpnet-base-v2")

# Setup Named Entity Categories
ne_categories = ["ABB", "ABB_EXPR", "ABB_NE", "EXPR", "NE", "NONE"]


# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    """

    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Please train the model first."
        )

    # Load Classifier Model and PCA
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(PCA_PATH, "rb") as f:
        pca = pickle.load(f)

    # Extract features for each token
    features = []

    for token in tokens:
        # Token embedding
        token_embedding = mpnet_model.encode([token], convert_to_tensor=False)[0]

        # Classify if token is a Named Entity
        ne_token = classify_if_is_ne(token)

        # Create one-hot encoding for NE feature
        ne_features = np.zeros((1, len(ne_categories)))
        if ne_token in ne_categories:
            ne_features[0, ne_categories.index(ne_token)] = 1

        # Get Spelling Correctness Feature
        spelling_features = classify_if_is_spelling_correct(token)

        # Get Orthographic Features
        ortographic_features = np.array(get_ortographic_features(token))

        combined = np.hstack(
            [token_embedding, ortographic_features, ne_features[0], spelling_features]
        )
        features.append(combined)

    # Stack all token feature vectors
    X = np.vstack(features)

    # Apply PCA
    X = pca.transform(X)

    # Predict tags
    predicted = clf.predict(X)

    return [str(tag) for tag in predicted]


if __name__ == "__main__":
    # Download NLTK resources
    nltk.download("punkt", quiet=True)

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

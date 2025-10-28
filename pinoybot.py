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


MODEL_PATH = "models/language_rf.pkl"
PCA_PATH = "models/language_pca.pkl"


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

    for token in tokens:
        embedding = embedder.encode([str(token)])[0]

        # is_spelling = 1  # 1 Placeholder for spelling correctness feature

        combined = np.hstack([embedding])
        features.append(combined)

    # Stack all token feature vectors
    X = np.vstack(features)

    # Apply PCA
    X = pca.transform(X)

    # Predict tags
    predicted = model.predict(X)

    return [str(tag) for tag in predicted]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentence = sys.argv[1]
    else:
        sentence = input("Enter a sentence: ")
    print("\n-----\n")

    tokens = sentence.split()
    tags = tag_language(tokens)
    print("\n-----\n")
    print("Tokens:", tokens)
    print("\n-----\n")
    print("Tags:", tags)
    print("\n-----\n")

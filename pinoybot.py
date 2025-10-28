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


MODEL_PATH = "random_forest_model.pkl"
PCA_PATH = "pca.pkl"


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
    with open("pca.pkl", "rb") as f:
        pca = pickle.load(f)

    # Generate embeddings for input tokens
    embedder = SentenceTransformer("all-mpnet-base-v2")
    embeddings = np.vstack(embedder.encode([str(t) for t in tokens]))

    # Apply PCA transformation
    embeddings = pca.transform(embeddings)

    # Predict tags
    predicted = model.predict(embeddings)

    return [str(tag) for tag in predicted]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentence = sys.argv[1]
    else:
        sentence = input("Enter a sentence: ")
    print("-----")

    tokens = sentence.split()
    tags = tag_language(tokens)
    print("Tokens:", tokens)
    print("-----")
    print("Tags:", tags)

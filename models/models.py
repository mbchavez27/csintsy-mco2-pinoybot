import os
import pickle
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer

MODEL_PATH = "random_forest_model.pkl"


def train_language_model(data: str = "data/final_annotations.csv"):
    """
    Trains the RandomForest language classifier and saves it as a pickle file.
    """

    print("Loading dataset")
    language = pd.read_csv(data)  # loads csv

    language["word"] = language["word"].astype(
        str
    )  # converts to string even as numberes

    print("Generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    language["embeddings"] = list(model.encode(language["word"].tolist()))

    X = np.vstack(language["embeddings"])
    y = language["label"]

    print("Splitting dataset...")
    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print("Training Random Forest model...")
    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train, y_train)

    # Save Model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print("Model trained and saved to", MODEL_PATH)


if __name__ == "__main__":
    train_language_model()

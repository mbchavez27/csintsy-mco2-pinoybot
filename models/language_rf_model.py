import os
import pickle
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

MODEL_PATH = "models/language_rf.pkl"
PCA_PATH = "models/language_pca.pkl"


def train_language_model(data: str = "data/final_annotations.csv"):
    """
    Trains the RandomForest language classifier and saves it as a pickle file.
    """

    print("Loading dataset\n")
    language = pd.read_csv(data)  # loads csv

    language["word"] = language["word"].astype(
        str
    )  # converts to string even as numberes

    print("Getting spelling correctness feature...\n")
    is_spelling_correct = (
        language["is_spelling_correct"].astype(int).to_numpy().reshape(-1, 1)
    )

    print("Getting is named entity feature...\n")
    language["is_ne"] = language["is_ne"].fillna("NONE")
    is_ne = pd.get_dummies(language["is_ne"], prefix="is_ne")

    print("Generating embeddings...\n")
    model = SentenceTransformer("all-mpnet-base-v2")
    language["embeddings"] = list(
        model.encode(
            language["word"].tolist(), convert_to_tensor=False, show_progress_bar=True
        )
    )

    print("Splitting dataset...\n")

    X = np.hstack([np.vstack(language["embeddings"])])
    # X = np.hstack(
    #     [np.vstack(language["embeddings"]), is_spelling_correct, is_ne.values]
    # )
    # X = np.hstack([np.vstack(language["embeddings"]), is_ne.values])
    y = language["label"]

    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print("Apply PCA...\n")
    pca = PCA(n_components=0.95, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    print(f"PCA retained {np.sum(pca.explained_variance_ratio_):.2%} of variance\n")

    print("Training Random Forest model...\n")
    clf = RandomForestClassifier(n_estimators=300, random_state=42, verbose=1)
    clf.fit(X_train, y_train)

    # Evaluate Model
    print("Evaluating model...\n")
    y_val_pred = clf.predict(X_val)
    print("Validation Performance:\n")
    print(classification_report(y_val, y_val_pred))

    y_test_pred = clf.predict(X_test)
    print("\nTest Performance:")
    print(classification_report(y_test, y_test_pred))
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

    # Save Model and PCA
    with open(PCA_PATH, "wb") as f:
        pickle.dump(pca, f)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print("\nPCA saved to", PCA_PATH)
    print("\nModel trained and saved to", MODEL_PATH)


if __name__ == "__main__":
    train_language_model()

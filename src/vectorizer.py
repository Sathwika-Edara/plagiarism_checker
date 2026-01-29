from typing import List, Iterable

from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np

from .preprocessing import preprocess_text

VECTORIZER =TfidfVectorizer(analyzer=preprocess_text)

def fit_vectorizer(documents: List[str]) ->TfidfVectorizer:
    print("Fitting the TF-IDF vectorizer on the document corpus...")
    VECTORIZER.fit(documents)
    print("Vectorizer fitting complete.")
    return VECTORIZER


def transform_documents(documents: Iterable[str]) -> spmatrix:
    print(f"Transforming {len(list(documents))} document(s) into TF-IDF vectors...")
    vectors = VECTORIZER.transform(documents)

    print("Transformation complete.")
    return vectors

def vectorize_corpus(corpus: List[str]) -> spmatrix:
    print("Fitting vectorizer and transforming corpus in one step...")
    vectors = VECTORIZER.fit_transform(corpus)
    print("Corpus vectorization complete.")
    return vectors


def save_vectorizer(vectorizer: TfidfVectorizer, file_path: str):
    try:
        print(f"Saving vectorizer to {file_path}...")
        joblib.dump(vectorizer, file_path)
        print("Vectorizer saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving the vectorizer: {e}")

def load_vectorizer(file_path: str) -> TfidfVectorizer | None:
    try:
        print(f"Loading vectorizer from {file_path}...")
        vectorizers = joblib.load(file_path)
        print("Vectorizer loaded successfully.")
        return vectorizers
    except FileNotFoundError:
        print(f"Error: Vectorizer file not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the vectorizer: {e}")
        return None

if __name__ == "__main__":
    print("--- Demonstrating Sparse Matrix Handling ---")

    # 1. Create a small sample corpus of documents.
    sample_corpus = [
        "The first document is about Python.",
        "This second document is about programming in Python.",
        "And the third one is about programming languages.",
        "Is this the first document again?"
    ]

    vectorizer_path = "tfidf_vectorizer.joblib"
    fitted_vectorizer=fit_vectorizer(sample_corpus)
    save_vectorizer(fitted_vectorizer,vectorizer_path)
    print("\n--- Simulating a new application session ---")
    loaded_vectorizer = load_vectorizer(vectorizer_path)
    if loaded_vectorizer:
        VECTORIZER=loaded_vectorizer
        new_doc="Is Python a good programming language?"
        print(f"\\nTransforming a new, unseen document: '{new_doc}'")
        new_vector = transform_documents([new_doc])
        print("Transformation successful with the loaded vectorizer.")
        print(f"Shape of the new vector: {new_vector.shape}")
        print("Sparse vector content:")
        print(new_vector)
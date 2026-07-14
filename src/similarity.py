"""
similarity.py

Calculate TF-IDF and Sentence Transformer similarity
"""

import pickle

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer


# ==========================================================
# Load Models Once
# ==========================================================

with open("models/tfidf_vectorizer.pkl", "rb") as file:
    tfidf_vectorizer = pickle.load(file)

sentence_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================================
# TF-IDF Similarity
# ==========================================================

def calculate_tfidf_similarity(
    resume_text,
    job_description
):

    vectors = tfidf_vectorizer.transform(
        [
            resume_text,
            job_description
        ]
    )

    score = cosine_similarity(
        vectors[0],
        vectors[1]
    )[0][0]

    return round(score * 100, 2)


# ==========================================================
# Sentence Transformer Similarity
# ==========================================================

def calculate_sentence_similarity(
    resume_text,
    job_description
):

    embeddings = sentence_model.encode(
        [
            resume_text,
            job_description
        ]
    )

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return round(score * 100, 2)
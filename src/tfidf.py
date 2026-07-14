"""
tfidf.py

Feature Engineering using TF-IDF
"""

import os
import pickle

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================================
# Create Models Folder
# ==========================================================

os.makedirs("models", exist_ok=True)


# ==========================================================
# Load Prepared Dataset
# ==========================================================

print("=" * 70)
print("Loading Prepared Dataset...")
print("=" * 70)

job_df = pd.read_csv(
    "data/prepared_job_resume_fit_dataset.csv",
    keep_default_na=False
)

job_df["clean_resume_text"] = (
    job_df["clean_resume_text"]
    .fillna("")
    .astype(str)
)

job_df["clean_job_text"] = (
    job_df["clean_job_text"]
    .fillna("")
    .astype(str)
)
# ==========================================================
# Create Combined Corpus
# ==========================================================

print("\nCreating Combined Corpus...")

corpus = pd.concat(
    [
        job_df["clean_resume_text"],
        job_df["clean_job_text"]
    ],
    ignore_index=True
)

print(f"Total Documents : {len(corpus)}")


# ==========================================================
# Initialize TF-IDF Vectorizer
# ==========================================================

print("\nTraining TF-IDF Vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)


# ==========================================================
# Learn Vocabulary + Transform
# ==========================================================

tfidf_matrix = vectorizer.fit_transform(corpus)


# ==========================================================
# Separate Resume & Job Vectors
# ==========================================================

num_resumes = len(job_df)

resume_vectors = tfidf_matrix[:num_resumes]

job_vectors = tfidf_matrix[num_resumes:]


# ==========================================================
# Save Vectorizer
# ==========================================================

with open("models/tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


# ==========================================================
# Save TF-IDF Matrices
# ==========================================================

with open("models/resume_vectors.pkl", "wb") as file:
    pickle.dump(resume_vectors, file)

with open("models/job_vectors.pkl", "wb") as file:
    pickle.dump(job_vectors, file)


# ==========================================================
# Information
# ==========================================================

print("\n" + "=" * 70)
print("TF-IDF Summary")
print("=" * 70)

print("Vocabulary Size :", len(vectorizer.vocabulary_))

print("TF-IDF Matrix Shape :", tfidf_matrix.shape)

print("Resume Matrix Shape :", resume_vectors.shape)

print("Job Matrix Shape :", job_vectors.shape)

print("\nTF-IDF Vectorizer Saved Successfully.")

print("Resume Vectors Saved Successfully.")

print("Job Vectors Saved Successfully.")
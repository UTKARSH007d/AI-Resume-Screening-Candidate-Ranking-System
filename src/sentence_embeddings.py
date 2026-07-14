"""
sentence_embeddings.py

Semantic Resume-Job Matching using Sentence Transformers
"""

import os
import pickle

import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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
# Load Sentence Transformer Model
# ==========================================================

print("\nLoading Sentence Transformer Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================================
# Generate Resume Embeddings
# ==========================================================

print("\nGenerating Resume Embeddings...")

resume_embeddings = model.encode(
    job_df["clean_resume_text"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)


# ==========================================================
# Generate Job Embeddings
# ==========================================================

print("\nGenerating Job Embeddings...")

job_embeddings = model.encode(
    job_df["clean_job_text"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)


# ==========================================================
# Semantic Similarity
# ==========================================================

print("\nCalculating Semantic Similarity...")

semantic_scores = []

for i in range(len(job_df)):

    score = cosine_similarity(
        [resume_embeddings[i]],
        [job_embeddings[i]]
    )[0][0]

    semantic_scores.append(round(score * 100, 2))


job_df["sentence_transformer_score"] = semantic_scores


# ==========================================================
# Save Embeddings
# ==========================================================

with open("models/resume_embeddings.pkl", "wb") as file:
    pickle.dump(resume_embeddings, file)

with open("models/job_embeddings.pkl", "wb") as file:
    pickle.dump(job_embeddings, file)


# ==========================================================
# Save Updated Dataset
# ==========================================================

job_df.to_csv(
    "data/final_candidate_matching.csv",
    index=False
)


# ==========================================================
# Preview
# ==========================================================

print("\n" + "=" * 70)
print("Sentence Transformer Summary")
print("=" * 70)

print(
    job_df[
        [
            "ID",
            "sentence_transformer_score"
        ]
    ].head(10)
)

print("\nResume Embeddings Shape :", resume_embeddings.shape)
print("Job Embeddings Shape    :", job_embeddings.shape)

print("\nEmbeddings Saved Successfully.")
print("Updated Dataset Saved Successfully.")
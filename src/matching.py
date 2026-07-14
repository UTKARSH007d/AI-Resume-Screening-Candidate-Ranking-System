"""
matching.py

Candidate Matching using TF-IDF Cosine Similarity
"""

import ast
import pickle
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# Load Dataset
# ==========================================================

job_df = pd.read_csv("data/job_resume_fit.csv")


# ==========================================================
# Load TF-IDF Vectors
# ==========================================================

with open("models/resume_vectors.pkl", "rb") as file:
    resume_vectors = pickle.load(file)

with open("models/job_vectors.pkl", "rb") as file:
    job_vectors = pickle.load(file)


# ==========================================================
# TF-IDF Cosine Similarity
# ==========================================================

scores = []

for i in range(len(job_df)):

    score = cosine_similarity(
        resume_vectors[i],
        job_vectors[i]
    )[0][0]

    scores.append(round(score * 100, 2))

job_df["tfidf_match_score"] = scores


# ==========================================================
# Matching Skills
# ==========================================================

matching_skills = []
missing_skills = []
matched_percentage = []

for _, row in job_df.iterrows():

    # Resume Skills
    resume_skills = row["resume_skill_list"]

    if isinstance(resume_skills, str):
        resume_skills = ast.literal_eval(resume_skills)
    else:
        resume_skills = []

    # Job Skills
    job_skills = row["job_required_skills"]

    if isinstance(job_skills, str):
        job_skills = ast.literal_eval(job_skills)
    else:
        job_skills = []

    resume_set = {skill.lower().strip() for skill in resume_skills}
    job_set = {skill.lower().strip() for skill in job_skills}

    matched = sorted(list(resume_set & job_set))
    missing = sorted(list(job_set - resume_set))

    matching_skills.append(matched)
    missing_skills.append(missing)

    if len(job_set) == 0:
        matched_percentage.append(0)
    else:
        matched_percentage.append(
            round((len(matched) / len(job_set)) * 100, 2)
        )


job_df["matching_skills"] = matching_skills
job_df["missing_skills"] = missing_skills
job_df["skill_match_percentage"] = matched_percentage


# ==========================================================
# Final Ranking
# ==========================================================
job_df["final_match_score"] = (
    0.7 * job_df["tfidf_match_score"]
    + 0.3 * job_df["skill_match_percentage"]
).round(2)

job_df = job_df.sort_values(
    by="final_match_score",
    ascending=False
)
# ==========================================================
# Save Results
# ==========================================================

job_df.to_csv(
    "data/final_candidate_matching.csv",
    index=False
)


# ==========================================================
# Preview
# ==========================================================

print("=" * 70)
print("Top 10 Candidates")
print("=" * 70)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print("=" * 70)

for i in range(10):

    print(f"\nCandidate {i+1}")
    print("-" * 70)

    print("ID :", job_df.iloc[i]["ID"])

    print("TF-IDF Score :", job_df.iloc[i]["tfidf_match_score"])

    print("Skill Match % :", job_df.iloc[i]["skill_match_percentage"])

    print("Matching Skills :")
    print(job_df.iloc[i]["matching_skills"])

    print("\nMissing Skills :")
    print(job_df.iloc[i]["missing_skills"])
print("\nSaved Successfully!")
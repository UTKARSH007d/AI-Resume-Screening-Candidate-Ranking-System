import pandas as pd

from preprocessing import preprocess_text

from information_extraction import (
    extract_skills,
    extract_education,
    extract_experience,
    extract_certifications
)

# ==========================================================
# Load Datasets
# ==========================================================

resume_df = pd.read_csv("data/resume.csv")
job_df = pd.read_csv("data/job_resume_fit.csv")


# ==========================================================
# Resume Dataset Preprocessing
# ==========================================================

print("=" * 70)
print("Preprocessing Resume Dataset...")
print("=" * 70)

resume_df["clean_resume"] = resume_df["Resume_str"].apply(preprocess_text)


# ==========================================================
# Job Resume Fit Dataset Preprocessing
# ==========================================================

print("\n" + "=" * 70)
print("Preprocessing Job Resume Fit Dataset...")
print("=" * 70)

job_df["clean_resume_text"] = job_df["resume_text"].apply(preprocess_text)

job_df["clean_job_text"] = job_df["job_text"].apply(preprocess_text)


# ==========================================================
# Information Extraction - Resume Dataset
# ==========================================================

print("\n" + "=" * 70)
print("Extracting Information from Resume Dataset...")
print("=" * 70)

resume_df["skills"] = resume_df["clean_resume"].apply(extract_skills)

resume_df["education"] = resume_df["clean_resume"].apply(extract_education)

resume_df["experience"] = resume_df["clean_resume"].apply(extract_experience)

resume_df["certifications"] = resume_df["clean_resume"].apply(
    extract_certifications
)


# ==========================================================
# Information Extraction - Job Resume Fit Dataset
# ==========================================================

print("\n" + "=" * 70)
print("Extracting Information from Job Resume Fit Dataset...")
print("=" * 70)

job_df["resume_skills"] = job_df["clean_resume_text"].apply(
    extract_skills
)

job_df["job_skills"] = job_df["clean_job_text"].apply(
    extract_skills
)

job_df["resume_education"] = job_df["clean_resume_text"].apply(
    extract_education
)

job_df["resume_experience"] = job_df["clean_resume_text"].apply(
    extract_experience
)

job_df["resume_certifications"] = job_df["clean_resume_text"].apply(
    extract_certifications
)


# ==========================================================
# Display Sample Output
# ==========================================================

print("\n" + "=" * 70)
print("Resume Dataset Preview")
print("=" * 70)

print(
    resume_df[
        [
            "clean_resume",
            "skills",
            "education",
            "experience",
            "certifications",
        ]
    ].head()
)

print("\n" + "=" * 70)
print("Job Resume Fit Dataset Preview")
print("=" * 70)

print(
    job_df[
        [
            "clean_resume_text",
            "resume_skills",
            "resume_education",
            "resume_experience",
            "resume_certifications",
            "clean_job_text",
            "job_skills",
        ]
    ].head()
)


# ==========================================================
# Save Prepared Datasets
# ==========================================================

resume_df.to_csv(
    "data/prepared_resume_dataset.csv",
    index=False,
)

job_df.to_csv(
    "data/prepared_job_resume_fit_dataset.csv",
    index=False,
)

print("\n" + "=" * 70)
print("Datasets Prepared Successfully")
print("=" * 70)

print("Resume Dataset Shape :", resume_df.shape)
print("Job Dataset Shape    :", job_df.shape)

print("\nPrepared datasets saved successfully.")


# Load datasets
# Apply preprocessing
# Save cleaned datasets
import pandas as pd
from information_extraction import extract_skills
# Load Resume Dataset
resume_df = pd.read_csv("data/Resume.csv")

# Load Job Resume Fit Dataset
job_df = pd.read_csv("data/job_resume_fit.csv")

from preprocessing import preprocess_text

resume_df["clean_resume"] = resume_df["Resume_str"].apply(preprocess_text)

job_df["clean_resume_text"] = job_df["resume_text"].apply(preprocess_text)

job_df["clean_job_text"] = job_df["job_text"].apply(preprocess_text)



print(resume_df[["Resume_str", "clean_resume"]].head())

print(job_df[["resume_text", "clean_resume_text"]].head())

print(job_df[["job_text", "clean_job_text"]].head())
# -------------------------------
# Resume Dataset Information
# -------------------------------

print("=" * 50)
print("Resume Dataset")
print("=" * 50)

print("Shape:", resume_df.shape)
print("\nColumns:")
print(resume_df.columns.tolist())

print("\nData Types:")
print(resume_df.dtypes)

print("\nMissing Values:")
print(resume_df.isnull().sum())

print("\nDuplicate Rows:")
print(resume_df.duplicated().sum())

print("\nFirst Five Rows:")
print(resume_df.head())


# -------------------------------
# Job Resume Fit Dataset Information
# -------------------------------

print("\n" + "=" * 50)
print("Job Resume Fit Dataset")
print("=" * 50)

print("Shape:", job_df.shape)

print("\nColumns:")
print(job_df.columns.tolist())

print("\nData Types:")
print(job_df.dtypes)

print("\nMissing Values:")
print(job_df.isnull().sum())

print("\nDuplicate Rows:")
print(job_df.duplicated().sum())

print("\nFirst Five Rows:")
print(job_df.head())
resume_df.to_csv("data/cleaned_resume.csv", index=False)

job_df.to_csv("data/cleaned_job_resume_fit.csv", index=False)
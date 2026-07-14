import pandas as pd

df = pd.read_csv("data/prepared_resume_dataset.csv")

print(df["Category"].nunique())
print(df["Category"].value_counts())
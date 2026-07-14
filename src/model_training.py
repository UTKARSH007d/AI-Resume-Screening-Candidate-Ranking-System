"""
model_training.py

Resume Category Classification
"""

import os
import pickle

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# Create Models Folder
# ==========================================================

os.makedirs("models", exist_ok=True)


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 70)
print("Loading Resume Dataset...")
print("=" * 70)

resume_df = pd.read_csv(
    "data/prepared_resume_dataset.csv",
    keep_default_na=False
)


# ==========================================================
# Features and Labels
# ==========================================================

X = resume_df["clean_resume"]

y = resume_df["Category"]


# ==========================================================
# TF-IDF
# ==========================================================

print("\nCreating TF-IDF Features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=2
)

X = vectorizer.fit_transform(X)


# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


# ==========================================================
# Models
# ==========================================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "Linear SVM":
        LinearSVC(random_state=42)


}


results = {}


print("\n" + "="*70)
print("Training Models...")
print("="*70)


for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    results[name] = accuracy

    print(f"\n{name}")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")

    print(classification_report(
        y_test,
        predictions
    ))

    print("\nConfusion Matrix")

    print(confusion_matrix(
        y_test,
        predictions
    ))

    filename = (
        name.lower()
        .replace(" ","_")
        + ".pkl"
    )

    with open(
        f"models/{filename}",
        "wb"
    ) as file:

        pickle.dump(model, file)


# ==========================================================
# Save TF-IDF Vectorizer
# ==========================================================

with open(
    "models/resume_vectorizer.pkl",
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )


# ==========================================================
# Best Model
# ==========================================================

print("\n" + "="*70)

print("Model Comparison")

print("="*70)

for model, score in results.items():

    print(f"{model:<30} {score:.4f}")

best_model = max(
    results,
    key=results.get
)

print("\nBest Model :", best_model)

print("\nTraining Completed Successfully.")
# ==========================================================
# Save Best Model
# ==========================================================

best_model_object = models[best_model]

with open("models/best_model.pkl", "wb") as file:
    pickle.dump(best_model_object, file)

print(f"\nBest model '{best_model}' saved successfully.")
# ==========================================================
# Save Deployment Model (Linear SVM)
# ==========================================================

deployment_model = models["Linear SVM"]

with open("models/deployment_model.pkl", "wb") as file:
    pickle.dump(deployment_model, file)

print("Deployment model (Linear SVM) saved successfully.")
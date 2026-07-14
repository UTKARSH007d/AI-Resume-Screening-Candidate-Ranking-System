# 🤖 AI Resume Screening & Candidate Ranking System

An AI-powered Resume Screening & Candidate Ranking web application that automatically analyzes resumes, compares them with job descriptions, and generates an ATS-style compatibility score using Natural Language Processing (NLP) and Machine Learning.

---

## 🚀 Features

- 📄 Upload resumes in PDF or DOCX format
- 📝 Enter a custom job description
- 🏷️ Resume category prediction using Linear SVM
- 🔍 Extract candidate information:
  - Skills
  - Education
  - Experience
  - Certifications
- 📊 TF-IDF Similarity Score
- 🧠 Semantic Similarity using Sentence Transformers
- 🎯 Hybrid ATS Match Score
- ✅ Matching Skills Detection
- ❌ Missing Skills Identification
- 💡 Candidate Recommendation
- 🌐 Interactive Streamlit Web Application

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- NLTK
- Sentence Transformers (all-MiniLM-L6-v2)
- Pandas
- NumPy
- Pickle

---

## 📂 Project Structure

```
AI Resume/
│
├── app.py
├── requirements.txt
├── models/
├── data/
├── src/
│   ├── preprocessing.py
│   ├── resume_parser.py
│   ├── information_extraction.py
│   ├── similarity.py
│   ├── matching.py
│   ├── recommendation.py
│   ├── sentence_embeddings.py
│   ├── tfidf.py
│   └── ...
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/UTKARSH007d/AI-Resume-Screening-Candidate-Ranking-System.git
```

Move into the project folder:

```bash
cd AI-Resume-Screening-Candidate-Ranking-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📊 Workflow

Resume Upload
⬇️

Resume Parsing
⬇️

NLP Preprocessing
⬇️

Resume Classification
⬇️

Information Extraction
⬇️

TF-IDF Similarity
⬇️

Sentence Transformer Similarity
⬇️

Hybrid ATS Match Score
⬇️

Skill Matching
⬇️

Hiring Recommendation

---

## 📸 Screenshots

### Home Page

*(Add Screenshot)*

### ATS Match Scores

*(Add Screenshot)*

### Skill Matching

*(Add Screenshot)*

### Resume Information Extraction

*(Add Screenshot)*

### Hiring Recommendation

*(Add Screenshot)*

---

## 🌐 Live Demo

**Streamlit Deployment**

(Add your Streamlit URL here)

---

## 📄 Project Report

Project report explaining the methodology, workflow, implementation, and results is included in this repository.

---

## 🔮 Future Improvements

- Bulk Resume Ranking
- Explainable AI (SHAP/LIME)
- REST API Integration
- Fine-tuned Transformer Models
- Multilingual Resume Support
- Authentication System
- Recruiter Dashboard

---

## 👨‍💻 Developer

**Utkarsh Gupta**

B.Tech CSE (AI & ML)

Manipal University Jaipur

AI/ML Intern – Appit Software Solutions
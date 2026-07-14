import re

# =====================================================
# Technical Skills
# =====================================================

SKILLS = [
    "python", "java", "javascript", "c++",
    "sql", "mysql", "mongodb",
    "machine learning", "deep learning",
    "artificial intelligence", "nlp",
    "computer vision", "tensorflow",
    "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "matplotlib",
    "opencv", "docker", "git", "github",
    "aws", "azure", "gcp",
    "flask", "django", "fastapi",
    "streamlit", "linux", "excel"
]

# =====================================================
# Education Keywords
# =====================================================

EDUCATION = [
    "b.tech",
    "bachelor of technology",
    "b.e",
    "bachelor of engineering",
    "bca",
    "mca",
    "b.sc",
    "bachelor of science",
    "m.sc",
    "master of science",
    "m.tech",
    "master of technology",
    "mba",
    "phd",
    "doctor of philosophy",
    "diploma"
]

# =====================================================
# Certification Keywords
# =====================================================

CERTIFICATIONS = [
    "aws certified",
    "azure",
    "google cloud",
    "oracle certified",
    "cisco",
    "ccna",
    "ccnp",
    "red hat",
    "rhce",
    "tensorflow developer",
    "microsoft certified",
    "coursera",
    "udemy",
    "nptel"
]


# =====================================================
# Skills Extraction
# =====================================================

def extract_skills(text):

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if text.strip() == "":
        return []

    text = text.lower()

    found_skills = []

    for skill in SKILLS:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):
            found_skills.append(skill)

    return found_skills


# =====================================================
# Education Extraction
# =====================================================

def extract_education(text):

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if text.strip() == "":
        return []

    text = text.lower()

    found_education = []

    for degree in EDUCATION:

        pattern = r"\b" + re.escape(degree) + r"\b"

        if re.search(pattern, text):
            found_education.append(degree)

    return found_education


# =====================================================
# Certification Extraction
# =====================================================

def extract_certifications(text):

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if text.strip() == "":
        return []

    text = text.lower()

    found_certifications = []

    for cert in CERTIFICATIONS:

        pattern = r"\b" + re.escape(cert) + r"\b"

        if re.search(pattern, text):
            found_certifications.append(cert)

    return found_certifications


# =====================================================
# Experience Extraction
# =====================================================

def extract_experience(text):

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if text.strip() == "":
        return 0

    text = text.lower()

    patterns = [
        r"(\d+)\+?\s*years",
        r"(\d+)\+?\s*year",
        r"experience\s*[:\-]?\s*(\d+)",
    ]

    years = []

    for pattern in patterns:

        matches = re.findall(pattern, text)

        for match in matches:
            years.append(int(match))

    if years:
        return max(years)

    return 0
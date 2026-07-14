"""
resume_parser.py

Extract text from PDF and DOCX resumes.
"""

import pdfplumber
import docx
import os


# ==========================================================
# PDF Parser
# ==========================================================

def extract_text_from_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    return text


# ==========================================================
# DOCX Parser
# ==========================================================

def extract_text_from_docx(file):

    document = docx.Document(file)

    text = ""

    for paragraph in document.paragraphs:

        text += paragraph.text + "\n"

    return text


# ==========================================================
# Universal Resume Parser
# ==========================================================

def extract_resume_text(file):

    extension = os.path.splitext(file.name)[1].lower()

    if extension == ".pdf":

        return extract_text_from_pdf(file)

    elif extension == ".docx":

        return extract_text_from_docx(file)

    else:

        raise ValueError("Only PDF and DOCX files are supported.")
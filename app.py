"""
app.py

AI Resume Screening & Candidate Ranking System
"""
import pickle
import streamlit as st
from src.resume_parser import extract_resume_text
from src.preprocessing import preprocess_text
from src.information_extraction import (
    extract_skills,
    extract_education,
    extract_experience,
    extract_certifications
)
from src.similarity import (
    calculate_tfidf_similarity,
    calculate_sentence_similarity
)
from src.skill_matching import compare_skills
from src.recommendation import generate_recommendation
# ==========================================================
# Load Resume Classification Model
# ==========================================================

with open("models/deployment_model.pkl", "rb") as file:
    category_model = pickle.load(file)

with open("models/resume_vectorizer.pkl", "rb") as file:
    category_vectorizer = pickle.load(file)
# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)


# ==========================================================
# Title
# ==========================================================

st.title("📄 AI Resume Screening & Candidate Ranking System")

st.markdown(
    """
Upload a resume and enter a job description to evaluate
candidate-job compatibility using NLP and Machine Learning.
"""
)


st.divider()


# ==========================================================
# Resume Upload
# ==========================================================

uploaded_resume = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)


# ==========================================================
# Job Description
# ==========================================================

job_description = st.text_area(
    "Paste Job Description",
    height=220,
    placeholder="Paste the complete job description here..."
)

# ==========================================================
# Analyze Button
# ==========================================================

analyze = st.button(
    "Analyze Resume",
    type="primary",
    use_container_width=True
)
# ==========================================================
# Analyze Resume
# ==========================================================

if analyze:

    # Validation
    if uploaded_resume is None:

        st.error("Please upload a resume.")

    elif job_description.strip() == "":

        st.error("Please enter a job description.")

    else:

        with st.spinner("Analyzing Resume..."):

            # ----------------------------------------
            # Extract Resume Text
            # ----------------------------------------

            resume_text = extract_resume_text(uploaded_resume)

            # ----------------------------------------
            # Preprocess
            # ----------------------------------------

            clean_resume = preprocess_text(resume_text)

            clean_job = preprocess_text(job_description)
            # ==========================================================
            # Resume Category Prediction
            # ==========================================================

            resume_vector = category_vectorizer.transform([clean_resume])

            predicted_category = category_model.predict(resume_vector)[0]
            
            # ==========================================================
            # Similarity Scores
            # ==========================================================

            tfidf_score = calculate_tfidf_similarity(
                clean_resume,
                clean_job
            )

            sentence_score = calculate_sentence_similarity(
                clean_resume,
                clean_job
            )

            TFIDF_WEIGHT = 0.40
            SEMANTIC_WEIGHT = 0.60

            overall_score = round(
                (TFIDF_WEIGHT * tfidf_score) +
                (SEMANTIC_WEIGHT * sentence_score),
                2
            )

            recommendation = generate_recommendation(overall_score)                            
            # ==========================================================
            # Information Extraction
            # ==========================================================

            skills = extract_skills(clean_resume)

            job_skills = extract_skills(clean_job)
            matching_skills, missing_skills, skill_match_percentage = compare_skills(
            skills,
            job_skills
            )
         
            education = extract_education(resume_text)

            experience = extract_experience(resume_text)

            certifications = extract_certifications(resume_text)
            st.success("Resume parsed successfully!")
            # ==========================================================
            # Resume Classification
            # ==========================================================

            st.divider()

            st.header("🤖 Resume Classification")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    label="Resume Category",
                    value=predicted_category
                )

            with col2:

                st.metric(
                    label="Experience",
                    value=f"{experience} Years"
                )
            st.divider()

            st.header("🎯 ATS Match Scores")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "TF-IDF Match",
                    f"{tfidf_score:.2f}%"
                )

            with col2:

                st.metric(
                    "Semantic Match",
                    f"{sentence_score:.2f}%"
                )

            with col3:

                st.metric(
                    "Overall Match",
                    f"{overall_score:.2f}%"
                ) 
            st.progress(overall_score / 100)

            st.caption(f"Overall Resume Match Score : {overall_score:.2f}%")   
            st.divider()

            st.header("🎯 Skill Matching")

            col1, col2 = st.columns(2)

            # ==========================
            # Matching Skills
            # ==========================
            with col1:

                st.subheader("✅ Matching Skills")

                if matching_skills:

                    matching_html = ""

                    for skill in matching_skills:

                        matching_html += f"""
                        <span style="
                            background-color:#1E5631;
                            color:white;
                            padding:6px 12px;
                            border-radius:15px;
                            margin:4px;
                            display:inline-block;
                            font-size:14px;">
                            ✅ {skill.title()}
                        </span>
                        """

                    st.markdown(
                        matching_html,
                        unsafe_allow_html=True
                    )

                else:

                    st.warning("No matching skills found.")


            # ==========================
            # Missing Skills
            # ==========================
            with col2:

                st.subheader("❌ Missing Skills")

                if missing_skills:

                    for skill in missing_skills:

                        st.write(f"❌ {skill.title()}")

                else:

                    st.success("No missing skills!")


            st.metric(
                "Skill Match %",
                f"{skill_match_percentage:.2f}%"
            )

            st.progress(skill_match_percentage / 100)
            # ==========================================================
            # Recommendation
            # ==========================================================

            st.divider()

            st.header("📋 Hiring Recommendation")

            if overall_score >= 85:

                st.success(recommendation)
                
                st.markdown("""
**Reason:**
- Excellent overall match with the job description.
- Most required skills are present.
- Candidate is highly suitable for this role.
""")

            elif overall_score >= 70:

                st.info(recommendation)
                st.markdown("""
**Reason:**
- Good overall compatibility.
- Candidate possesses many required skills.
- Suitable for interview consideration.
""")

            elif overall_score >= 50:

                st.warning(recommendation)
                st.markdown("""
**Reason:**
- Moderate compatibility with the job.
- Some important skills are missing.
- Resume can be improved for better alignment.
""")

            else:

                st.error(recommendation)
                st.markdown("""
**Reason:**
- Low similarity with the job description.
- Several important skills are missing.
- Resume should be tailored for this position.
""")
            st.divider()

            st.header("📊 Analysis Summary")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Resume Category",
                    predicted_category
                )

                st.metric(
                    "Overall Match",
                    f"{overall_score:.2f}%"
                )

                st.metric(
                    "Experience",
                    f"{experience} Years"
                )

            with col2:

                st.metric(
                    "Matching Skills",
                    len(matching_skills)
                )

                st.metric(
                    "Missing Skills",
                    len(missing_skills)
                )

                st.metric(
                    "Recommendation",
                    recommendation
                )     
            st.divider()

            with st.expander("📄 View Original Resume"):

                st.text_area(
                    "Original Resume",
                    resume_text,
                    height=350
                )

            with st.expander("🧹 View Preprocessed Resume"):

                st.text_area(
                    "Preprocessed Resume",
                    clean_resume,
                    height=350
                )
                st.divider()

            st.header("Extracted Resume Information")

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Skills")

                if skills:

                    for skill in skills:

                        st.write(f"✅ {skill.title()}")



                else:

                    st.warning("No skills detected.")

                st.subheader("Education")

                if education:

                    for item in education:

                        st.write(f"🎓 {item.title()}")

                else:

                    st.warning("Education not found.")

            with col2:

                st.subheader("Experience")

                st.metric(
                    "Experience",
                    f"{experience} Years"
                )
                
                st.subheader("Certifications")

                if certifications:

                    for cert in certifications:

                        st.write(f"🏆 {cert}")

                else:

                    st.warning("No certifications found.")
                  
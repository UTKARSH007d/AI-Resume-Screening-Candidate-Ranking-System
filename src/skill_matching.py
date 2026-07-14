"""
skill_matching.py

Compare resume skills with job skills.
"""


def compare_skills(resume_skills, job_skills):

    # Convert to lowercase sets
    resume_set = set(skill.lower() for skill in resume_skills)
    job_set = set(skill.lower() for skill in job_skills)

    # Matching Skills
    matching_skills = sorted(list(resume_set & job_set))

    # Missing Skills
    missing_skills = sorted(list(job_set - resume_set))

    # Skill Match Percentage
    if len(job_set) == 0:

        skill_match_percentage = 0

    else:

        skill_match_percentage = round(
            len(matching_skills) / len(job_set) * 100,
            2
        )

    return (
        matching_skills,
        missing_skills,
        skill_match_percentage
    )
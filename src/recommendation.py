"""
recommendation.py
"""


def generate_recommendation(score):

    if score >= 85:

        return "🟢 Highly Recommended"

    elif score >= 70:

        return "🟡 Recommended"

    elif score >= 50:

        return "🟠 Consider After Review"

    else:

        return "🔴 Needs Improvement"
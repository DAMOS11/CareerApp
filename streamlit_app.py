import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("AI-based Career Recommendation System.csv")
    df.drop(columns=["CandidateID", "Name"], inplace=True)
    df["Combined_Text"] = df["Education"] + " " + df["Skills"] + " " + df["Interests"]
    label_encoder = LabelEncoder()
    df["Career_Label"] = label_encoder.fit_transform(df["Recommended_Career"])
    return df, label_encoder

df, label_encoder = load_data()

# Train model
@st.cache_resource
def train_model():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(df["Combined_Text"], df["Career_Label"])
    return pipeline

pipeline = train_model()

# Skill-resource mapping
skill_resources = {
    "python": "https://www.learnpython.org/",
    "machine learning": "https://www.coursera.org/learn/machine-learning",
    "data analysis": "https://www.kaggle.com/learn/pandas",
    "deep learning": "https://www.deeplearning.ai/",
    "cloud computing": "https://www.coursera.org/specializations/google-cloud-platform",
    "java": "https://www.codecademy.com/learn/learn-java",
    "project management": "https://www.coursera.org/specializations/project-management",
    "ui/ux": "https://www.interaction-design.org/courses",
    "graphic design": "https://www.canva.com/learn/graphic-design/",
    "communication": "https://www.coursera.org/learn/wharton-communication-skills",
    "system design": "https://www.educative.io/courses/grokking-the-system-design-interview"
}

# Extract text from PDF or TXT
def extract_resume_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        text = ""
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    else:
        text = uploaded_file.read().decode("utf-8")
    return text

# Parse fields from text
def parse_resume_info(text):
    text = text.lower()
    known_skills = list(skill_resources.keys())
    education_keywords = ["bachelor", "master", "msc", "bsc", "phd", "degree", "diploma"]
    interest_keywords = ["ai", "technology", "design", "media", "healthcare", "management"]

    education = next((word.title() for word in education_keywords if word in text), "Not Found")
    skills = "; ".join([s.title() for s in known_skills if s in text])
    interests = "; ".join([i.title() for i in interest_keywords if i in text])
    return education, skills, interests

# Recommend careers
def recommend_top_3_with_resources(education, skills, interests):
    text_input = education + " " + skills + " " + interests
    probabilities = pipeline.predict_proba([text_input])[0]
    top_indices = probabilities.argsort()[-3:][::-1]
    top_careers = [(label_encoder.inverse_transform([i])[0], round(probabilities[i]*100, 2)) for i in top_indices]

    result = "### Top 3 Career Recommendations:"
    for i, (career, score) in enumerate(top_careers, 1):
        result += f"\n{i}. **{career}** ({score}%)"

    result += "\n\n### Recommended Learning Resources:"
    user_skills = [s.strip().lower() for s in skills.split(";")]
    matched = False
    for skill in user_skills:
        for key in skill_resources:
            if key in skill:
                result += f"\n- {key.title()}: {skill_resources[key]}"
                matched = True
    if not matched:
        result += "\n- No matching resources found."

    return result

# Streamlit UI
st.title("🚀 AI Career Recommender System")
st.markdown("Upload your resume **or** manually enter your profile to get personalized career suggestions and learning resources.")

option = st.radio("Choose input method:", ("Upload Resume", "Manual Input"))

if option == "Upload Resume":
    uploaded_file = st.file_uploader("Upload your resume (.pdf or .txt)", type=["pdf", "txt"])
    if uploaded_file:
        text = extract_resume_text(uploaded_file)
        education, skills, interests = parse_resume_info(text)
        st.write(f"**Extracted Education:** {education}")
        st.write(f"**Extracted Skills:** {skills}")
        st.write(f"**Extracted Interests:** {interests}")
        if st.button("Get Recommendations"):
            st.markdown(recommend_top_3_with_resources(education, skills, interests))

else:
    education = st.text_input("Education", "Bachelor's in Computer Science")
    skills = st.text_area("Skills (semicolon-separated)", "Python; Machine Learning; Data Analysis")
    interests = st.text_area("Interests (semicolon-separated)", "AI; Technology")
    if st.button("Get Recommendations"):
        st.markdown(recommend_top_3_with_resources(education, skills, interests))
import streamlit as st
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

model = joblib.load('models/student_model.pkl')
le = joblib.load('models/label_encoder.pkl')

st.title("DevLevel Classifier")
st.write("Predict your developer skill level")

leetcode_total = st.number_input("LeetCode Total Solved", min_value=0, value=0)
contest_rating = st.number_input("Contest Rating", min_value=0, value=0)
github_repos = st.number_input("GitHub Repos", min_value=0, value=0)
github_commits = st.number_input("GitHub Commits", min_value=0, value=0)
commit_streak = st.number_input("Commit Streak (days)", min_value=0, value=0)
open_source_prs = st.number_input("Open Source PRs", min_value=0, value=0)

if st.button("Classify My Level"):
    X = np.array([[leetcode_total, contest_rating, github_repos, github_commits, commit_streak, open_source_prs]])
    pred = model.predict(X)
    proba = model.predict_proba(X)[0]
    label = le.inverse_transform(pred)[0]
    classes = le.classes_.tolist()

    colors = {"Beginner": "🟢", "Intermediate": "🟡", "Expert": "🔴"}
    st.success(f"{colors.get(label, '⚪')} Predicted Level: **{label}**")

    st.write("### Confidence Scores")
    for cls, prob in zip(classes, proba):
        st.progress(float(prob), text=f"{cls}: {round(prob*100, 1)}%")

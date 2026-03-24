import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

@st.cache_resource
def train_model():
    df = pd.read_csv('student_performance.csv')
    df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})
    df['Tuition'] = df['Tuition'].map({'Yes': 1, 'No': 0})
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Performance Index'])
    y = df['Performance Index']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X.columns

model, feature_cols = train_model()

st.title('🎓 Student Performance Predictor')

gender          = st.selectbox('Gender', ['Female', 'Male'])
hours_studied   = st.slider('Hours Studied per Week', 1, 9, 5)
previous_scores = st.slider('Previous Scores', 40, 99, 70)
extracurricular = st.selectbox('Extracurricular Activities', ['Yes', 'No'])
tuition         = st.selectbox('Attends Tuition', ['Yes', 'No'])
sleep_hours     = st.slider('Sleep Hours per Night', 4, 9, 7)
sample_papers   = st.slider('Sample Papers Practiced', 0, 9, 3)

if st.button('Predict Score 🎯'):
    input_data = pd.DataFrame([[
        1 if gender == 'Female' else 0,
        hours_studied, previous_scores,
        1 if extracurricular == 'Yes' else 0,
        1 if tuition == 'Yes' else 0,
        sleep_hours, sample_papers
    ]], columns=feature_cols)

    score = model.predict(input_data)[0]
    st.success(f'📊 Predicted Performance Index: **{round(score, 1)} / 100**')

    if score >= 75:
        st.balloons()
        st.info('🌟 Excellent performance expected!')
    elif score >= 50:
        st.info('👍 Good performance — keep it up!')
    else:
        st.warning('⚠️ At risk — consider more study hours or tuition support.')

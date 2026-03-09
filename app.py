import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Title
st.title("🏏 Cricket Match Result Prediction")

# Load dataset
df = pd.read_csv("log_dataset.csv")
df = df.drop_duplicates()

# Features and target
X = df[['powerplay_score','powerplay_wickets']]
y = df['result']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

st.subheader("Enter Powerplay Details")

# Sliders
powerplay_score = st.slider("Powerplay Score", 0, 120, 50)
powerplay_wickets = st.slider("Powerplay Wickets", 0, 10, 2)

# Prediction
if st.button("Predict Result"):

    input_data = np.array([[powerplay_score, powerplay_wickets]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Result: {prediction[0]}")
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival using a Decision Tree.")

model = joblib.load("model.pkl")
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

pclass = st.selectbox("Ticket class", [1, 2, 3])
sex = st.radio("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 80, 30)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 8, 0)
parch = st.number_input("Parents/Children aboard", 0, 6, 0)
fare = st.number_input("Fare paid", 0.0, 520.0, 32.0)

if st.button("Predict"):
    row = pd.DataFrame(
        [[pclass, 1 if sex == "Female" else 0, age, sibsp, parch, fare]],
        columns=features,
    )
    result = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]

    if result == 1:
        st.success(f"Survived (probability {prob:.0%})")
    else:
        st.error(f"Did not survive (survival probability {prob:.0%})")

if st.checkbox("Show decision tree"):
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(model, feature_names=features,
              class_names=["Died", "Survived"], filled=True, ax=ax)
    st.pyplot(fig)
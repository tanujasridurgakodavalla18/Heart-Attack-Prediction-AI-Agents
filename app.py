import streamlit as st
import numpy as np
import pickle

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="CardioCare AI",
    page_icon="🫀",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- HEADER ----------------
st.title("🫀 CardioCare AI - Heart Risk Prediction Dashboard")
st.caption("AI Clinical Decision Support System (Educational Use Only)")
st.markdown("---")

# ---------------- INPUT ----------------
st.sidebar.header("Patient Clinical Data")

age = st.sidebar.number_input("Age", 1, 120, 50)

sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex == "Male" else 0

cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])

trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)

chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)

fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", ["No", "Yes"])
fbs = 1 if fbs == "Yes" else 0

restecg = st.sidebar.selectbox("Rest ECG", [0, 1, 2])

thalach = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)

exang = st.sidebar.selectbox("Exercise Angina", ["No", "Yes"])
exang = 1 if exang == "Yes" else 0

oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 6.0, 1.0)

slope = st.sidebar.selectbox("Slope", [0, 1, 2])

ca = st.sidebar.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])

thal = st.sidebar.selectbox("Thal", [0, 1, 2, 3])

# ---------------- INPUT ARRAY ----------------
input_data = np.array([[
    age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak,
    slope, ca, thal
]])

# ---------------- PREDICTION ----------------
if st.button("🔍 Generate Report"):

    # scale input
    input_scaled = scaler.transform(input_data)

    # predict
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    # ---------------- SAFE CLASS HANDLING ----------------
    classes = list(model.classes_)

    # find index for HIGH risk class (1)
    if 1 in classes:
        high_index = classes.index(1)
        low_index = classes.index(0)
    else:
        high_index = 1
        low_index = 0

    high_risk = proba[high_index]
    low_risk = proba[low_index]

    risk_score = high_risk

    # ---------------- DECISION LOGIC ----------------
    if risk_score >= 0.65:
        label = "🔴 HIGH CARDIAC RISK"
        st.error(label)

    elif risk_score >= 0.35:
        label = "🟡 MODERATE CARDIAC RISK"
        st.warning(label)

    else:
        label = "🟢 LOW CARDIAC RISK"
        st.success(label)

    # ---------------- DISPLAY ----------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Age", age)
    col2.metric("Heart Rate", thalach)
    col3.metric("Cholesterol", chol)

    st.markdown("---")

    st.subheader("📊 AI Risk Analysis")

    st.write(f"**Final Prediction:** {label}")
    st.write(f"**High Risk Probability:** {high_risk*100:.2f}%")
    st.write(f"**Low Risk Probability:** {low_risk*100:.2f}%")

    st.progress(int(risk_score * 100))

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("⚠ Educational AI system only - Not a medical diagnosis")
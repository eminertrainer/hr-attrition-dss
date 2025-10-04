# streamlit_hr_app.py
# HR Attrition Prediction DSS – SQIT5033
# Author: Dr. Izwan Nizal Mohd Shaharanee

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="HR Attrition DSS", layout="centered")
st.title("👥 HR Attrition Decision Support System (Model-Driven DSS)")

@st.cache_resource
def train_pipeline():
    # load data
    df = pd.read_csv("../data/HRDataset_v14.csv")

    # choose features/target (adjust if your file has different names)
    target = "Termd"
    cat_features = ["Department", "Position", "PerformanceScore", "RecruitmentSource"]
    num_features = ["Salary", "EngagementSurvey", "Absences"]

    # minimal cleaning
    df = df[cat_features + num_features + [target]].dropna()

    X = df[cat_features + num_features]
    y = df[target]

    # encoder + model
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features),
        ]
    )
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    pipe = Pipeline([("prep", pre), ("model", clf)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe.fit(X_tr, y_tr)

    # quick metrics for display
    y_hat = pipe.predict(X_te)
    acc = accuracy_score(y_te, y_hat)
    report = classification_report(y_te, y_hat, output_dict=True)

    return pipe, cat_features, num_features, acc, report

pipe, cat_cols, num_cols, acc, report = train_pipeline()
st.success(f"Model trained. Hold-out accuracy: **{acc:.2f}**")

st.markdown("---")
st.header("Enter an Employee Profile")

# simple input UI
inp = {}
inp["Department"] = st.selectbox("Department", ["Sales", "Production", "IT/IS", "Admin Offices", "Executive Office"])
inp["Position"] = st.text_input("Position", "Account Executive")
inp["PerformanceScore"] = st.selectbox("Performance Score", ["Needs Improvement", "Fully Meets", "Exceeds", "PIP"])
inp["RecruitmentSource"] = st.selectbox("Recruitment Source", ["LinkedIn", "Indeed", "Google Search", "Employee Referral"])

inp["Salary"] = st.number_input("Current Salary (RM)", min_value=1000, max_value=50000, value=5000, step=250)
inp["EngagementSurvey"] = st.slider("Engagement Survey (1–5)", 1.0, 5.0, 3.5, 0.1)
inp["Absences"] = st.slider("Absences (past year)", 0, 60, 5)

X_new = pd.DataFrame([inp])

if st.button("Predict Attrition Risk"):
    proba = pipe.predict_proba(X_new)[0][1]
    pred = int(proba >= 0.5)

    if pred == 1:
        st.error(f"⚠️ High Attrition Risk — Probability {proba:.2f}")
        st.markdown("- **Suggested Action:** Retention interview, review compensation, assign mentor, targeted training.")
    else:
        st.success(f"✅ Low Attrition Risk — Probability {proba:.2f}")
        st.markdown("- **Suggested Action:** Maintain engagement plan; schedule periodic check-ins.")

st.markdown("---")
st.caption("This classroom prototype demonstrates a Model-Driven DSS (Data → Model → Interface → Action).")

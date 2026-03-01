import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide")

# ============================================================
# 🎨 PREMIUM CORPORATE STYLING
# ============================================================

st.markdown("""
<style>
body {background-color:#0E1117;}
section[data-testid="stSidebar"] {background-color:#111827;}
h1,h2,h3 {color:white;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL + ARTIFACTS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "attrition_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "models", "feature_columns.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Palo Alto Networks.csv")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_PATH)

df = pd.read_csv(DATA_PATH)

# ============================================================
# HEADER
# ============================================================

st.title("🏢 Employee Attrition Prediction & Risk Scoring System")
st.markdown("AI-Powered Workforce Intelligence for Proactive Retention Strategy")
st.markdown("---")

# ============================================================
# DATA PREPROCESSING
# ============================================================

X = df.drop("Attrition", axis=1)
X = pd.get_dummies(X)
X = X.reindex(columns=feature_columns, fill_value=0)
X_scaled = scaler.transform(X)

probs = model.predict_proba(X_scaled)[:, 1]
df["Attrition_Probability"] = probs

# ============================================================
# RISK THRESHOLD (Problem Statement Requirement)
# ============================================================

st.sidebar.header("⚙ Risk Threshold Control")

threshold = st.sidebar.slider("Set High Risk Threshold",
                              0.1, 0.9, 0.6, 0.05)

def risk_category(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < threshold:
        return "Medium Risk"
    else:
        return "High Risk"

df["Risk_Category"] = df["Attrition_Probability"].apply(risk_category)

# ============================================================
# KPI SECTION
# ============================================================

col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", len(df))
col2.metric("High Risk Employees",
            (df["Risk_Category"]=="High Risk").sum())
col3.metric("Average Attrition Risk",
            f"{df['Attrition_Probability'].mean()*100:.2f}%")

st.markdown("---")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Attrition Risk Dashboard",
    "👤 Employee Risk Profile",
    "🏢 Department Risk View",
    "🔮 What-If Simulator",
    "📈 Explainable AI (SHAP)"
])

# ============================================================
# TAB 1: DASHBOARD
# ============================================================

with tab1:

    st.subheader("Risk Category Distribution")

    risk_counts = df["Risk_Category"].value_counts().reset_index()
    risk_counts.columns = ["Risk", "Count"]

    fig = px.bar(risk_counts,
                 x="Risk",
                 y="Count",
                 color="Risk",
                 template="plotly_dark",
                 title="Employee Risk Segmentation")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Attrition Probability Distribution")

    fig2 = px.histogram(df,
                        x="Attrition_Probability",
                        nbins=20,
                        template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# TAB 2: EMPLOYEE PROFILE
# ============================================================

with tab2:

    st.subheader("Individual Employee Risk Analysis")

    emp_id = st.selectbox("Select Employee Index", df.index)

    prob = df.loc[emp_id, "Attrition_Probability"]
    category = risk_category(prob)

    st.metric("Attrition Probability",
              f"{prob*100:.2f}%")

    st.metric("Risk Category", category)

    st.dataframe(df.loc[[emp_id]])

# ============================================================
# TAB 3: DEPARTMENT VIEW
# ============================================================

with tab3:

    st.subheader("Average Risk by Department")

    dept_risk = df.groupby("Department")["Attrition_Probability"].mean().reset_index()

    fig = px.bar(dept_risk,
                 x="Department",
                 y="Attrition_Probability",
                 color="Attrition_Probability",
                 template="plotly_dark",
                 color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4: WHAT-IF SIMULATOR
# ============================================================

with tab4:

    st.subheader("Simulate Employee Attrition Risk")

    col1, col2 = st.columns(2)

    age = col1.slider("Age", 18, 60, 30)
    income = col2.number_input("Monthly Income", 1000, 20000, 5000)
    overtime = col1.selectbox("OverTime", ["Yes", "No"])
    satisfaction = col2.slider("Job Satisfaction", 1, 4, 3)

    sim = df.iloc[[0]].copy()
    sim["Age"] = age
    sim["MonthlyIncome"] = income
    sim["OverTime"] = overtime
    sim["JobSatisfaction"] = satisfaction

    sim = pd.get_dummies(sim)
    sim = sim.reindex(columns=feature_columns, fill_value=0)
    sim_scaled = scaler.transform(sim)

    sim_prob = model.predict_proba(sim_scaled)[:,1][0]

    st.metric("Simulated Risk",
              f"{sim_prob*100:.2f}%")

# ============================================================
# TAB 5: SHAP EXPLAINABILITY (FULLY FIXED)
# ============================================================

with tab5:

    st.subheader("Explainable AI – Individual Risk Drivers")

    emp_id = st.selectbox("Select Employee for SHAP Explanation",
                          df.index,
                          key="shap_emp")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1][emp_id]
    else:
        shap_vals = shap_values[emp_id]

    shap_vals = np.array(shap_vals).flatten()

    feature_names = feature_columns
    min_len = min(len(feature_names), len(shap_vals))

    shap_vals = shap_vals[:min_len]
    feature_names = feature_names[:min_len]

    # Safe base value
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_value = float(np.array(explainer.expected_value[1]).flatten()[0])
    else:
        base_value = float(explainer.expected_value)

    explanation = shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=X_scaled[emp_id][:min_len],
        feature_names=feature_names
    )

    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor("#0E1117")
    st.pyplot(fig)

    # Plotly Top Features
    importance = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_vals
    })

    importance["Abs"] = np.abs(importance["SHAP Value"])
    importance = importance.sort_values("Abs",
                                        ascending=False).head(10)

    fig2 = px.bar(importance,
                  x="SHAP Value",
                  y="Feature",
                  orientation="h",
                  template="plotly_dark",
                  title="Top Risk Drivers")
    st.plotly_chart(fig2, use_container_width=True)
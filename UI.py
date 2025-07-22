import streamlit as st
import pandas as pd
import requests
import pickle
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# Features for graph
feature_cols = ['Age', 'Academic Pressure', 'CGPA', 'Sleep Duration', 'Work/Study Hours']

sleep_map = {
    "Less than 5": 4,
    "5-6 hours": 5.5,
    "6-7 hours": 6.5,
    "7-8 hours": 7.5,
    "More than 8": 9
}

# Load dataset
df = pd.read_csv("dataset.csv")
df['Sleep Duration'] = df['Sleep Duration'].astype(str).str.strip().str.replace('"', '').str.replace("'", '')
df['Sleep Duration'] = df['Sleep Duration'].replace({
    'Less than 5 hours': 4,
    '5-6 hours': 5.5,
    '6-7 hours': 6.5,
    '7-8 hours': 7.5,
    'More than 8 hours': 9
})
df['Sleep Duration'] = pd.to_numeric(df['Sleep Duration'], errors='coerce')
df['Work/Study Hours'] = pd.to_numeric(df['Work/Study Hours'], errors='coerce')
feature_means = {col: df[col].mean() for col in feature_cols}

# Load model and tools
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoder.pkl", "rb"))

# Page setup
st.set_page_config(page_title="Student Depression Predictor", layout="wide")
st.title("🎓 Student Depression Predictor")

# Sidebar Inputs
st.sidebar.header("📝 Input Student Details")
age = st.sidebar.number_input("Age", min_value=10, max_value=50, step=1)
academic_pressure = st.sidebar.number_input("Academic Pressure (1-5)", min_value=1.0, max_value=5.0, step=0.1)
cgpa = st.sidebar.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)
sleep_duration = st.sidebar.selectbox("Sleep Duration", [
    "Less than 5", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8"
])
suicidal_thoughts = st.sidebar.selectbox("Suicidal Thoughts?", label_encoders['Have you ever had suicidal thoughts ?'].classes_)
work_hours = st.sidebar.number_input("Work/Study Hours", min_value=1.0, max_value=7.0, step=0.1)

# PDP feature selector
pdp_feature = st.sidebar.selectbox("Select Feature for PDP", feature_cols)

# Input dictionary
input_data = {
    'Age': age,
    'Academic Pressure': academic_pressure,
    'CGPA': cgpa,
    'Sleep Duration': sleep_duration,
    'Have you ever had suicidal thoughts ?': suicidal_thoughts,
    'Work/Study Hours': work_hours
}

# Gauge chart function
def plot_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Depression Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if probability > 0.5 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "pink"},
            ],
        }
    ))
    return fig

# Prediction Request
if st.button("🔍 Predict"):
    try:
        response = requests.post("https://ai-project-dy32.onrender.com/predict", json=input_data)

        if response.status_code == 200:
            result_json = response.json()
            st.session_state.prediction_result = {
                "prediction": result_json['prediction'],
                "probability": result_json['probability'],
                "input_data": input_data.copy()
            }

        else:
            st.error(f"❌ Backend Error: {response.status_code}")
            st.json(response.json())

    except Exception as e:
        st.error(f"An error occurred: {e}")

# If prediction exists in session_state, render charts
if "prediction_result" in st.session_state:
    result_data = st.session_state.prediction_result
    prediction = result_data["prediction"]
    depression_prob = result_data["probability"]
    input_data = result_data["input_data"]

    result = "🟢 No Depression Detected" if prediction == 0 else "🔴 Depression Detected"
    st.subheader("📊 Prediction Result")
    st.markdown(
        f"<h2 style='color: {'green' if prediction == 0 else 'red'}'>{result}</h2>",
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Prediction Confidence",
        "📈 Feature Contribution",
        "📊 Input vs Dataset Average",
        f"🧠 PDP: {pdp_feature} vs Depression Probability"
    ])

    with tab1:
        st.markdown("### 🎯 Prediction Confidence")
        st.progress(depression_prob)
        st.markdown(f"**Model Confidence (Depression Detected): `{depression_prob*100:.2f}%`**")
        st.markdown("<span style='color:gray;'>Threshold for Depression: <b>0.5</b></span>", unsafe_allow_html=True)
        st.plotly_chart(plot_gauge(depression_prob), use_container_width=True)

    with tab2:
        st.subheader("📈 Feature Contributions to Prediction")
        input_copy = input_data.copy()
        input_copy['Sleep Duration'] = sleep_map.get(input_copy['Sleep Duration'], 5.5)
        input_df = pd.DataFrame([input_copy])
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
        X_scaled = scaler.transform(input_df)
        coefs = model.coef_[0]
        contributions = X_scaled[0] * coefs
        contrib_df = pd.DataFrame({
            'Feature': input_df.columns,
            'Value': X_scaled[0],
            'Coefficient': coefs,
            'Contribution': contributions
        }).sort_values(by='Contribution', key=abs, ascending=False)
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        colors = ['green' if c > 0 else 'red' for c in contrib_df['Contribution']]
        ax1.barh(contrib_df['Feature'], contrib_df['Contribution'], color=colors)
        ax1.set_xlabel("Contribution to Prediction")
        ax1.set_title("Impact of Each Feature")
        max_contrib = max(abs(contrib_df['Contribution'].max()), abs(contrib_df['Contribution'].min()))
        ax1.set_xlim(-max_contrib * 1.1, max_contrib * 1.1)
        st.pyplot(fig1)

    with tab3:
        st.subheader("📊 Comparison: Your Input vs Dataset Average")
        radar_features = ['Age', 'Academic Pressure', 'CGPA', 'Sleep Duration', 'Work/Study Hours']
        user_vals = [
            input_data['Age'],
            input_data['Academic Pressure'],
            input_data['CGPA'],
            sleep_map[input_data['Sleep Duration']],
            input_data['Work/Study Hours']
        ]
        avg_vals = [feature_means[feat] for feat in radar_features]
        combined = np.array([user_vals, avg_vals])
        min_vals = combined.min(axis=0)
        max_vals = combined.max(axis=0)
        normalized = (combined - min_vals) / (max_vals - min_vals + 1e-5)
        labels = radar_features + [radar_features[0]]
        user_vals_norm = np.append(normalized[0], normalized[0][0])
        avg_vals_norm = np.append(normalized[1], normalized[1][0])
        angles = np.linspace(0, 2 * np.pi, len(labels))
        fig2 = plt.figure(figsize=(6, 6))
        ax = fig2.add_subplot(111, polar=True)
        ax.plot(angles, user_vals_norm, 'r-', linewidth=2, label='User Input')
        ax.fill(angles, user_vals_norm, 'r', alpha=0.25)
        ax.plot(angles, avg_vals_norm, 'b-', linewidth=2, label='Dataset Mean')
        ax.fill(angles, avg_vals_norm, 'b', alpha=0.25)
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        ax.set_title("Radar Chart: Normalized Feature Comparison", y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig2)

    with tab4:
        st.subheader(f"🧠 Partial Dependence Plot (PDP) for {pdp_feature}")
        input_copy = input_data.copy()
        input_copy['Sleep Duration'] = sleep_map.get(input_copy['Sleep Duration'], 5.5)
        val_range = np.linspace(df[pdp_feature].min(), df[pdp_feature].max(), 50)
        pdp_probs = []
        for val in val_range:
            temp_input = input_copy.copy()
            temp_input[pdp_feature] = val
            temp_input_df = pd.DataFrame([temp_input])
            for col, le in label_encoders.items():
                if col in temp_input_df.columns:
                    temp_input_df[col] = le.transform(temp_input_df[col])
            temp_scaled = scaler.transform(temp_input_df)
            prob = model.predict_proba(temp_scaled)[0][1]
            pdp_probs.append(prob)
        fig_pdp, ax_pdp = plt.subplots(figsize=(7, 5))
        ax_pdp.plot(val_range, pdp_probs, color='navy', linewidth=2)
        ax_pdp.set_title(f"{pdp_feature} vs Predicted Depression Probability")
        ax_pdp.set_xlabel(pdp_feature)
        ax_pdp.set_ylabel("Depression Probability")
        ax_pdp.grid(True)
        st.pyplot(fig_pdp)

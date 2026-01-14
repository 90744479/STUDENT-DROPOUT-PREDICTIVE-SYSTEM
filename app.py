import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Configuration ---
st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="centered")

# --- Load Assets ---
def load_assets():
    try:
        model = pickle.load(open('best_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        le = pickle.load(open('label_encoder.pkl', 'rb'))
        return model, scaler, le
    except FileNotFoundError:
        st.error("Error: Missing pickle files. Run your Jupyter Notebook cells to save best_model.pkl, scaler.pkl, and label_encoder.pkl.")
        return None, None, None

model, scaler, le = load_assets()

# --- App UI ---
st.title("🎓 Student Dropout Predictive System")
st.markdown("Enter the student's details below to predict their academic outcome.")

if model:
    with st.form("prediction_form"):
        st.subheader("Personal & Socio-Economic Info")
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            age = st.number_input("Age at enrollment", min_value=17, max_value=70, value=20)
            scholarship = st.selectbox("Scholarship holder", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            tuition = st.selectbox("Tuition fees up to date", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            debtor = st.selectbox("Debtor", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col2:
            marital = st.selectbox("Marital status", [1, 2, 3, 4, 5, 6], help="1: Single, 2: Married...")
            displaced = st.selectbox("Displaced", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            special_needs = st.selectbox("Educational special needs", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            int_student = st.selectbox("International", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            course = st.number_input("Course Code", value=1, step=1)

        st.divider()
        st.subheader("Academic Performance (Units & Grades)")
        col3, col4 = st.columns(2)

        with col3:
            st.info("First Semester")
            c1_enrolled = st.number_input("Units Enrolled (1st Sem)", value=6)
            c1_approved = st.number_input("Units Approved (1st Sem)", value=5)
            c1_grade = st.number_input("Grade Average (1st Sem)", value=12.0)
            c1_eval = st.number_input("Evaluations (1st Sem)", value=6)

        with col4:
            st.info("Second Semester")
            c2_enrolled = st.number_input("Units Enrolled (2nd Sem)", value=6)
            c2_approved = st.number_input("Units Approved (2nd Sem)", value=5)
            c2_grade = st.number_input("Grade Average (2nd Sem)", value=12.0)
            c2_eval = st.number_input("Evaluations (2nd Sem)", value=6)

        # Submit Button
        submitted = st.form_submit_button("Predict Status")

    if submitted:
        # Construct input array - ensuring same 34-feature structure as dataset.csv
        # Note: We fill less critical features with defaults/averages to keep UI clean
        input_data = {
            'Marital status': marital, 'Application mode': 1, 'Application order': 1, 'Course': course,
            'Daytime/evening attendance': 1, 'Previous qualification': 1, 'Nacionality': 1,
            "Mother's qualification": 1, "Father's qualification": 1, "Mother's occupation": 1,
            "Father's occupation": 1, 'Displaced': displaced, 'Educational special needs': special_needs, 
            'Debtor': debtor, 'Tuition fees up to date': tuition, 'Gender': gender, 
            'Scholarship holder': scholarship, 'Age at enrollment': age, 'International': int_student,
            'Curricular units 1st sem (credited)': 0, 'Curricular units 1st sem (enrolled)': c1_enrolled,
            'Curricular units 1st sem (evaluations)': c1_eval, 'Curricular units 1st sem (approved)': c1_approved,
            'Curricular units 1st sem (grade)': c1_grade, 'Curricular units 1st sem (without evaluations)': 0,
            'Curricular units 2nd sem (credited)': 0, 'Curricular units 2nd sem (enrolled)': c2_enrolled,
            'Curricular units 2nd sem (evaluations)': c2_eval, 'Curricular units 2nd sem (approved)': c2_approved,
            'Curricular units 2nd sem (grade)': c2_grade, 'Curricular units 2nd sem (without evaluations)': 0,
            'Unemployment rate': 11.0, 'Inflation rate': 1.0, 'GDP': 0.0
        }
        
        # Convert to DataFrame
        features_df = pd.DataFrame([input_data])
        
        # Scale and Predict
        scaled_features = scaler.transform(features_df)
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)
        
        result = le.inverse_transform(prediction)[0]
        
        # Results Display
        st.divider()
        if result == "Graduate":
            st.success(f"🎉 Predicted Result: **{result}**")
        elif result == "Dropout":
            st.error(f"⚠️ Predicted Result: **{result}**")
        else:
            st.warning(f"📖 Predicted Result: **{result}**")

        # Probability Visualization
        st.write("### Prediction Confidence")
        prob_df = pd.DataFrame(prediction_proba, columns=le.classes_)
        st.bar_chart(prob_df.T)
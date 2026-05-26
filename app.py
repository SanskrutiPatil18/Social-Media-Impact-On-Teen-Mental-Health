import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.title('Teen Mental Health: Depression Prediction')
st.write("Enter the teen's information to predict the likelihood of depression.")

# --- Check if model files exist ---
if not os.path.exists('random_forest_model.joblib') or not os.path.exists('standard_scaler.joblib'):
    st.error("❌ Model or scaler file not found. Please make sure 'random_forest_model.joblib' and 'standard_scaler.joblib' are in the same folder as app.py.")
else:
    # Load the trained model and scaler
    rf_model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('standard_scaler.joblib')

    # Define the order of columns the model expects
    expected_columns = [
        'age', 'daily_social_media_hours', 'sleep_hours',
        'screen_time_before_sleep', 'academic_performance',
        'physical_activity', 'stress_level', 'anxiety_level',
        'addiction_level', 'gender_male', 'platform_usage_Instagram',
        'platform_usage_TikTok', 'social_interaction_level_low',
        'social_interaction_level_medium'
    ]

    # --- Input Features ---
    age = st.slider('Age', min_value=12, max_value=20, value=15)
    gender = st.selectbox('Gender', ['female', 'male'])
    daily_social_media_hours = st.slider('Daily Social Media Hours', 0.0, 10.0, 3.0, 0.1)
    platform_usage = st.selectbox('Primary Platform Usage', ['Instagram', 'TikTok', 'Both'])
    sleep_hours = st.slider('Sleep Hours', 4.0, 12.0, 7.0, 0.1)
    screen_time_before_sleep = st.slider('Screen Time Before Sleep (hours)', 0.0, 5.0, 1.0, 0.1)
    academic_performance = st.slider('Academic Performance (GPA equivalent)', 0.0, 4.0, 2.5, 0.1)
    physical_activity = st.slider('Physical Activity (hours/day)', 0.0, 3.0, 0.5, 0.1)
    social_interaction_level = st.selectbox('Social Interaction Level', ['low', 'medium', 'high'])
    stress_level = st.slider('Stress Level (1-10)', 1, 10, 5)
    anxiety_level = st.slider('Anxiety Level (1-10)', 1, 10, 5)
    addiction_level = st.slider('Addiction Level (1-10)', 1, 10, 5)

    # --- Preprocessing Function ---
    def preprocess_input(input_data):
        df_input = pd.DataFrame([input_data])

        # One-hot encode categorical features
        df_input['gender_male'] = 1 if df_input['gender'].iloc[0] == 'male' else 0
        df_input['platform_usage_Instagram'] = 1 if df_input['platform_usage'].iloc[0] == 'Instagram' else 0
        df_input['platform_usage_TikTok'] = 1 if df_input['platform_usage'].iloc[0] == 'TikTok' else 0
        df_input['social_interaction_level_low'] = 1 if df_input['social_interaction_level'].iloc[0] == 'low' else 0
        df_input['social_interaction_level_medium'] = 1 if df_input['social_interaction_level'].iloc[0] == 'medium' else 0

        # Drop original categorical columns
        df_input = df_input.drop(columns=['gender', 'platform_usage', 'social_interaction_level'])

        # Ensure all expected columns exist
        for col in expected_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        # Reorder columns
        final_input = df_input[expected_columns].copy()

        # Scale numerical features
        numerical_cols = [
            'age', 'daily_social_media_hours', 'sleep_hours',
            'screen_time_before_sleep', 'academic_performance',
            'physical_activity', 'stress_level', 'anxiety_level',
            'addiction_level'
        ]
        final_input[numerical_cols] = scaler.transform(final_input[numerical_cols])

        return final_input

    # --- Prediction ---
    if st.button('Predict'):
        input_data = {
            'age': age,
            'gender': gender,
            'daily_social_media_hours': daily_social_media_hours,
            'platform_usage': platform_usage,
            'sleep_hours': sleep_hours,
            'screen_time_before_sleep': screen_time_before_sleep,
            'academic_performance': academic_performance,
            'physical_activity': physical_activity,
            'social_interaction_level': social_interaction_level,
            'stress_level': stress_level,
            'anxiety_level': anxiety_level,
            'addiction_level': addiction_level,
        }

        try:
            processed_input = preprocess_input(input_data)
            prediction = rf_model.predict(processed_input)
            prediction_proba = rf_model.predict_proba(processed_input)[:, 1]

            st.write('---')
            if prediction[0] == 1:
                st.error(f"Prediction: High likelihood of Depression (Probability: {prediction_proba[0]:.2f})")
                st.write("It is recommended to seek professional help or support.")
            else:
                st.success(f"Prediction: Low likelihood of Depression (Probability: {prediction_proba[0]:.2f})")
                st.write("Continue to monitor mental well-being.")

            st.write('---')
            st.write("Disclaimer: This prediction is based on a machine learning model and should not be used as a substitute for professional medical advice or diagnosis.")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

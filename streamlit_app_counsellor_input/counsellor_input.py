import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import os
import pickle
from pathlib import Path
#from openai import OpenAI
from mistralai import Mistral

model_uri = "runs:/e61c2bcfb80e4477af0cb8d6bcb7d515/random_forest_model"  
vectorizer_model_uri = "runs:/84079bd6b7164fbfbebe4c3aed3e1ebc/vectorizer"

def download_model(model_uri, output_dir):
    output_dir.parent.mkdir(exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=output_dir)

model_dir_path = Path('models/')

rf_model_path = model_dir_path / 'random_forest_model' / 'model.pkl'
download_model(model_uri, model_dir_path)
with open(rf_model_path, 'rb') as file:
    rf_model = pickle.load(file)


vectorizer_model_path = model_dir_path / 'vectorizer' / 'model.pkl'
download_model(vectorizer_model_uri, model_dir_path)
with open(vectorizer_model_path, 'rb') as file:
    vectorizer_model = pickle.load(file)

#vectorizer_model = mlflow.sklearn.load_model(vectorizer_model_uri)
api_key = os.environ.get("mistral_ai_api_key")

client = Mistral(api_key=api_key)




# Model
def process_text_with_model(text, model):
    features = vectorizer_model.transform([text]).toarray()  
    prediction = model.predict(features)
    return prediction

# get suggestions from ChatGPT based on the detected emotion
# Function to get suggestions from ChatGPT (using gpt-3.5-turbo)
def get_emotion_suggestion(emotion):
    try:   
        #response = client.chat.completions.create(
        #    model="gpt-4-turbo",
        #    messages=[
        #        {"role": "system", "content": "You are a mental health counselor."},
        #        {"role": "user", "content": f"Provide actionable suggestions for someone experiencing {emotion}."}
        #    ],
        #    max_tokens=150
        #)
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": "You are a mental health counselor."},
                {"role": "user", "content": f"Provide actionable suggestions for someone experiencing {emotion}."}
            ],
            max_tokens=150
        )
        #suggestion = response['choices'][0]['message']['content'].strip()
        suggestion = response.choices[0].message.content
        return suggestion
    except Exception as e:
        return f"An error occurred while generating suggestions: {e}"

# Streamlit App
st.markdown(
    """
    <h1 style="color: blue; text-align: center;">
        🧑‍⚕️ Legacy: Mental Health Counselling Portal
    </h1>
    """,
    unsafe_allow_html=True,
)

st.write("Welcome, Counselor! This portal allows you to input your observations, notes, or insights regarding a patient's mental health.")

# Text input from user
user_input = st.text_area("Enter your notes or observations here:", "",height=200)


if st.button("Submit"):
    if user_input.strip():
        try:
            prediction = process_text_with_model(user_input, rf_model)
            st.success(f"Emotion sensed: **{prediction[0]}**")
            # Get suggestions from ChatGPT
            suggestion = get_emotion_suggestion(prediction[0])
            st.write(f"**Suggestions:** {suggestion}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text before submitting.")

import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Title of the app
st.title("Symptom Checker")

# Load the scaler
scaler = None
try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

# Load the disease data
disese_data = pd.read_csv("webscrapdata.csv")

# Load the CSV file
df = pd.read_csv("FinalFilteredData.csv")

# Encode the labels
xins = df.iloc[:, 1].values
y = df.iloc[:, 0].values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Create a label mapping
label_mapping = {encoded_label: original_label for encoded_label, original_label in enumerate(label_encoder.classes_)}

# Get input features from the dataframe
inputs = np.array(df.columns)[1:]

# Initialize a dictionary to keep track of symptoms
mapping = {i: 0 for i in inputs}

# Try to load the pre-trained model with error handling
try:
    model = load_model('final4.h5', compile=False)  # Avoid recompiling here
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Text box for inputting symptoms
symptoms = str(st.text_area("Enter your symptoms separated by commas:"))

# Split the symptoms input into a list
symptoms_array = [s.strip() for s in symptoms.split(',')]

# Submit button
if st.button("Submit"):
    # Reset the mapping to all zeros
    mapping = {i: 0 for i in inputs}
    
    for symptom in symptoms_array:
        if symptom in mapping:
            mapping[symptom] = 1  # Set corresponding feature to 1 if symptom is present

    # Convert the mapping to an array and reshape to match model input
    x = np.array(list(mapping.values())).reshape(1, -1)

    # Use the scaler to transform the input data
    try:
        x = scaler.transform(x)
    except Exception as e:
        st.error(f"Error scaling input data: {e}")
        st.stop()
    
    # Debugging: Print reshaped input
    # st.write("Reshaped input:", x)
    
    # Predict using the model
    try:
        output = model.predict(x)
        
        # Debugging: Print raw model output
        # st.write("Model raw output:", output)
        
        # Get the predicted label
        predicted_label = output.argmax(axis=1)[0]  # Assuming a classification problem

        # Display the predicted label
        st.write("Predicted Condition:", label_mapping.get(predicted_label, "Unknown"))
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")

    if symptoms:
        # Display information sections after submitting symptoms
        st.header("Diagnostic method")
        st.write(disese_data["Diagnostic method"].values)

        st.header("Risk Factors")
        st.write(disese_data["Risk factors"].values)

        st.header("Treatments")
        st.write(disese_data['Treatment'].values)

        st.header("Causes")
        st.write(disese_data['Causes'].values)
    else:
        st.write("Please enter some symptoms before submitting.")

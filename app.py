import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Title of the app
st.title("Symptom Checker")

# Load the CSV file
df = pd.read_csv("FinalFilteredData.csv")

# Encode the labels
xins = df[:, 1]
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
    model = load_model('finalannmodel.h5')
    model.compile()  # Recompile the model to set up metrics
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop the script if the model can't be loaded

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
    
    # Debugging: Print reshaped input
    st.write("Reshaped input:", x)
    
    # Predict using the model
    try:
        output = model.predict(x)
        
        # Debugging: Print raw model output
        st.write("Model raw output:", output)
        
        # Get the predicted label
        predicted_label = np.argmax(output, axis=1)[0]  # Assuming a classification problem

        # Display the predicted label
        st.write("Predicted Condition:", label_mapping.get(predicted_label, "Unknown"))
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")

    if symptoms:
        # Display information sections after submitting symptoms
        st.header("Precautions")
        st.write("List of precautions based on the symptoms entered.")

        st.header("Risk Factors")
        st.write("List of risk factors associated with the symptoms.")

        st.header("Treatments")
        st.write("Suggested treatments for the symptoms entered.")

        st.header("Causes")
        st.write("Possible causes for the symptoms entered.")
    else:
        st.write("Please enter some symptoms before submitting.")

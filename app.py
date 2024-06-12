import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import json
import numpy as np

st.header('Lungs Xray Prediction', divider='gray')

class_labels = {
    0: 'COVID-19',
    1: 'Non-COVID',
    2: 'Normal'
}

#Button to upload Xrays Image
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

#Check if a file has been uploaded
if uploaded_file is not None:
    #Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Image uploaded successfully");
else:
    st.write("Please upload an image file");

def predict():
    if uploaded_file is not None:
        image = uploaded_file.getvalue()

        response = requests.post(
            "https://monica22-covid-prediction.hf.space/predict",
            files = {
                "image": BytesIO(image)
            }
        )

        response = json.loads(response._content.decode('utf-8'))
        prediction = json.loads(response["prediction"])
        label = np.argmax(predict)

        # st.write(prediction)
        # st.write(label)
        
        if label == 0:
            st.write(f"Prediction - :red[{class_labels[label]}]")
        elif label == 1:
            st.write(f"Prediction - :orange[{class_labels[label]}]")
        else:
            st.write(f"Prediction - :green[{class_labels[label]}]")

st.button(
    label="Predict",
    type="primary",
    on_click=predict
)
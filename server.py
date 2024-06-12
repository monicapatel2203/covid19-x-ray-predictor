from fastapi import FastAPI, UploadFile, File
import json
from model import load_model
from PIL import Image
from io import BytesIO
import numpy as np


app = FastAPI()

model = load_model('./model3.keras')

@app.get("/")
def greeter():
    return {
        "response": "Lungs Xray Prediction ML-APP"
    }

def image_pipeline(image):
    # step 1: Load the image
    image = Image.open(BytesIO(image))

    # step 2: Resize the image
    image = image.resize((256, 256))

    # Step 3: Conver the image to grayscale
    image = image.convert('L')
    
    # Step 4: Add a Dimension at end stating it as grayscale channel
    image = np.expand_dims(image, axis=-1)

    return image

def get_model_prediction(image):
    image = image_pipeline(image)
    image = np.expand_dims(image, axis=0)

    # Get Model Prediction
    pred = model.predict(image)

    return pred[0].tolist()

# API to predict the X-ray
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image = await image.read()

    prediction = get_model_prediction(image)

    return {
        "prediction": json.dumps(prediction)
    }
    
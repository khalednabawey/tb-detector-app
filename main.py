import uvicorn
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
import os
import kagglehub
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="TB Detection API")

# Add middleware and configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a root endpoint for health check


@app.get("/")
async def root():
    return {"status": "healthy", "message": "TB Detection API is running"}

tuberculosis_count = 3000
normal_count = 3500

# Define class frequencies
freq_pos = tuberculosis_count  # TB images
freq_neg = normal_count  # Normal images

# Compute class weights
pos_weight = freq_neg / freq_pos  # Weight for TB class
neg_weight = freq_pos / freq_neg  # Weight for Normal class

# Convert weights to a tensor for loss function
class_weights = tf.constant([neg_weight, pos_weight], dtype=tf.float32)


def weighted_binary_crossentropy(y_true, y_pred):
    """
    Custom loss function for imbalanced binary classification.
    """
    epsilon = 1e-7  # Avoid log(0)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

    loss = -(
        class_weights[1] * y_true * tf.math.log(y_pred) +
        class_weights[0] * (1 - y_true) * tf.math.log(1 - y_pred)
    )
    return tf.reduce_mean(loss)


# Global model variable
model = None


def load_model_from_kaggle():
    """Load model from Kaggle Hub with error handling"""
    global model
    try:
        MODEL_PATH = kagglehub.model_download(
            "khalednabawi/tb-chest-prediction/keras/v1"
        )
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        model = load_model(os.path.join(
            MODEL_PATH, 'tb_resnet.h5'), compile=False)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Initialize model at startup


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    try:
        logger.info("Starting model initialization...")
        model = load_model_from_kaggle()
        logger.info("Model initialized successfully!")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

# Define class labels
CLASS_LABELS = {0: "Normal", 1: "Tuberculosis"}


def preprocess_image(img):
    """Preprocesses the uploaded image for ResNet50."""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    # Normalize using ResNet50's preprocessing
    img_array = preprocess_input(img_array)
    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receives an image file, preprocesses it, and returns TB classification."""
    try:
        logger.info(f"Receiving prediction request for file: {file.filename}")
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")  # Ensure RGB format
        img_array = preprocess_image(img)

        if model is None:
            raise ValueError("Model not initialized")

        # Make prediction
        prediction = model.predict(img_array)
        # Convert sigmoid output to 0 or 1
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0])  # Confidence score

        logger.info(f"Prediction complete: {CLASS_LABELS[predicted_class]}")
        return {
            "success": True,
            "filename": file.filename,
            "prediction": CLASS_LABELS[predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/test")
async def test():
    """Test endpoint to verify API is running"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tensorflow_version": tf.__version__,
        "python_path": os.environ.get("PYTHONPATH", "Not set")
    }

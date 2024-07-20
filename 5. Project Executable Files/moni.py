import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
from flask import Flask, request, render_template
import cv2

app = Flask(__name__)

# Load the model with the correct custom_objects parameter
def load_model():
    try:
        model = tf.keras.models.load_model(filepath='D:\\Rice-type-classification-CNN-main\\rice.h5', custom_objects={'KerasLayer': hub.KerasLayer})
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        if model is None:
            return "Model is not loaded correctly. Please check the logs."

        # Get the uploaded file from the request
        f = request.files['image']
        # Save the uploaded file to the 'Data/val' directory
        basepath = os.path.dirname(__file__)  # Getting the current path i.e where app.py is present
        filepath = os.path.join(basepath, 'uploads', f.filename)  # Define the path to save the uploaded image
        f.save(filepath)

        # Read and preprocess the uploaded image
        a2 = cv2.imread(filepath)
        a2 = cv2.resize(a2, (224, 224))
        a2 = np.array(a2)
        a2 = a2 / 255.0
        a2 = np.expand_dims(a2, 0)  # Add batch dimension

        # Predict the class
        pred = model.predict(a2)
        pred = pred.argmax()

        # Define the labels
        df_labels = {
            0: 'arborio',
            1: 'basmati',
            2: 'ipsala',
            3: 'jasmine',
            4: 'karacadag'
        }

        # Map the prediction to the corresponding label
        prediction = df_labels.get(pred, "Unknown")
        print(prediction)
        # Render the results template with the predicted label
        return render_template('results.html', prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True)

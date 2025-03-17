import os
import numpy as np
import cv2
import pywt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import RandomRotation as BaseRandomRotation
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

# --- Monkey-patch the RandomRotation layer with the correct signature ---
original_from_config = BaseRandomRotation.from_config

def patched_from_config(cls, config):
    # Remove the unwanted parameter "value_range" from the config, if it exists.
    config.pop("value_range", None)
    return original_from_config.__func__(cls, config)

BaseRandomRotation.from_config = classmethod(patched_from_config)
# ---------------------------------------------------------------------

# Define allowed file extensions.
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function: Preprocess image using Discrete Wavelet Transform (DWT)
def apply_DWT(img):
    # Convert image to grayscale for DWT processing.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.dwt2(img_gray, 'haar')
    LL, _ = coeffs  # Keep only the LL component.
    LL = cv2.resize(LL, (224, 224))  # Resize to 224x224.
    # Convert 2D LL image to a 3-channel image.
    LL = np.repeat(LL[:, :, np.newaxis], 3, axis=-1)
    return LL.astype(np.float32)

# Define the classes used during training.
classes = ['brown spots', 'healthy', 'white scale']

# Initialize the Flask app.
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret key in production.
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model.
model = load_model('date_palm_model.keras')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page route: image upload form and prediction processing.
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read, preprocess, and normalize the image.
            img = cv2.imread(filepath)
            processed_img = apply_DWT(img)
            processed_img = processed_img / 255.0
            input_img = np.expand_dims(processed_img, axis=0)
            
            # Generate predictions.
            predictions = model.predict(input_img)
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions, axis=1)[0]
            
            result = {
                'class': classes[predicted_class_idx],
                'confidence': f"{confidence * 100:.2f}%"
            }
            return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

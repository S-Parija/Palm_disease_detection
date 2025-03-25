import os
import io
import base64
import numpy as np
import cv2
import pywt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import RandomRotation as BaseRandomRotation
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

# --- Monkey-patch the RandomRotation layer to remove "value_range" ---
original_from_config = BaseRandomRotation.from_config

def patched_from_config(cls, config):
    # Remove the unwanted parameter "value_range" if present.
    config.pop("value_range", None)
    return original_from_config.__func__(cls, config)

BaseRandomRotation.from_config = classmethod(patched_from_config)
# ---------------------------------------------------------------------

# Allowed file extensions.
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function: Preprocess image using Discrete Wavelet Transform (DWT)
def apply_DWT(img):
    # Convert to grayscale for DWT.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.dwt2(img_gray, 'haar')
    LL, _ = coeffs  # Keep only the LL component.
    LL = cv2.resize(LL, (224, 224))
    # Convert single-channel image to 3 channels.
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

# Home route: file upload, category selection, prediction and plot generation.
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file is provided.
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)
        # Get the selected original category.
        original_category = request.form.get('original_category')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and preprocess the image.
            img = cv2.imread(filepath)
            processed_img = apply_DWT(img)
            processed_img = processed_img / 255.0
            input_img = np.expand_dims(processed_img, axis=0)
            
            # Generate predictions.
            predictions = model.predict(input_img)
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions, axis=1)[0]
            
            # --- Generate a bar chart for predicted probabilities ---
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend.
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(classes, predictions[0], color='teal')
            ax.set_ylim([0,1])
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            # -------------------------------------------------------
            
            result = {
                'original_category': original_category,
                'predicted_class': classes[predicted_class_idx],
                'confidence': f"{confidence * 100:.2f}%",
                'plot_data': plot_data
            }
            return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

print("Flask server starting up...", flush=True)
import os
from flask import Flask, request, send_file, render_template, send_from_directory
from PIL import Image, ImageOps
import io
import numpy as np
from flask_cors import CORS
from models.zero_dce import ZeroDCE

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

# Initialize Zero-DCE model with pretrained weights
zero_dce = ZeroDCE(device='cpu', weights_path='models/weights/Epoch99.pth')

@app.route('/process', methods=['POST'])
def process_image():
    print("\n\n=== NEW REQUEST RECEIVED ===", flush=True)
    print(f"Request headers: {dict(request.headers)}", flush=True)
    print(f"Request form data: {request.form}", flush=True)
    
    if 'image' not in request.files:
        print("ERROR: No 'image' in request.files", flush=True)
        return 'No image uploaded', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    try:
        print("\n\n--- New image processing request ---", flush=True)
        print(f"Received file: {file.filename} ({file.content_length} bytes)", flush=True)
        print("Starting image processing", flush=True)
        
        # Open and convert image
        img = Image.open(file.stream).convert('RGB')
        print(f"Processing image: size={img.size} mode={img.mode}", flush=True)
        
        # Get parameters from sliders with defaults
        alpha = float(request.form.get('alpha', 0.1))
        gamma = float(request.form.get('gamma', 1.2))
        print(f"Using enhancement parameters - alpha: {alpha}, gamma: {gamma}")
        
        # Enhance with neural network using slider parameters
        corrected_img = zero_dce.enhance(img, alpha=alpha, gamma=gamma)
        
        # Save to bytes buffer
        img_byte_arr = io.BytesIO()
        corrected_img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        return send_file(img_byte_arr, mimetype='image/jpeg')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f'Error processing image: {str(e)}', 500

if __name__ == '__main__':
    print("Checking templates...", flush=True)
    print(f"Template folder: {app.template_folder}", flush=True)
    print(f"Static folder: {app.static_folder}", flush=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

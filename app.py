import torch
import flask
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv5 model
MODEL_PATH = "runs/train/FoodMold/weights/best.pt"  
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

# Define allowed image formats
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Load and process image
    img = cv2.imread(file_path)
    results = model(img)

    # Parse results
    detections = results.pandas().xyxy[0]  
    mold_detected = not detections.empty  
    
    response = {
        "filename": filename,
        "mold_detected": mold_detected,
        "detections": detections.to_dict(orient="records")  
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Initialize Flask App
app = Flask(__name__)

os.environ["GITHUB_TOKEN"] = os.getenv("GITHUB_TOKEN")

# Load YOLOv5 Model
MODEL_PATH = "best.pt"

# ✅ Check if best.pt exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model file 'best.pt' not found! Make sure it is uploaded to Render.")


model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=MODEL_PATH,
    force_reload=True,
    trust_repo=True
)

@app.route("/", methods=["GET"])
def home():
    return "Mold Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if an image is provided
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        # Read Image
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Convert image to OpenCV format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Perform inference using YOLOv5
        results = model(image)

        # Parse Results
        detections = results.pandas().xyxy[0]  # Get results as DataFrame
        mold_detected = not detections.empty  # True if mold is detected

        return jsonify({
            "mold_detected": mold_detected,
            "detections": detections.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render-assigned port
    app.run(host="0.0.0.0", port=port, debug=False)

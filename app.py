from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from PIL import Image, ImageChops
import joblib
import torch
import torchvision.transforms as transforms
import torchvision.models as models

app = Flask(__name__)

# Paths to the saved models
svm_model_path = "svm_model.pkl"
knn_model_path = "knn_model.pkl"

# Load the trained models
svm_model = joblib.load(svm_model_path)
knn_model = joblib.load(knn_model_path)

# Load ResNet-18 model for feature extraction
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Handle pretrained weights (torchvision compatibility)
try:
    resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
except AttributeError:
    resnet_model = models.resnet18(pretrained=True)

resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  # Remove final layer
resnet_model.eval()

# Function to extract features
def extract_features_from_image(image_path):
    """Extract features from an image using ResNet-18."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = resnet_transform(image).unsqueeze(0)
        with torch.no_grad():
            features = resnet_model(image_tensor).squeeze().numpy()
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Function to apply ELA
def apply_ela(image_path, scale_factor=0.5):
    """Apply Error Level Analysis to an image."""
    try:
        original = Image.open(image_path).convert("RGB")
        resized = original.resize(
            (int(original.width * scale_factor), int(original.height * scale_factor)),
            Image.Resampling.LANCZOS
        )
        error_level = ImageChops.difference(original, resized)
        error_level_image = "error_level_image.png"
        error_level.save(error_level_image)
        return error_level_image
    except Exception as e:
        print(f"Error applying ELA: {e}")
        return None

@app.route("/")
def index():
    """Render the homepage."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle video upload and make predictions."""
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded video
    video_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(video_path)

    # Extract the first frame from the video
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if not success:
        return jsonify({"error": "Failed to read video"}), 400

    # Save the extracted frame
    frame_path = "frame.png"
    cv2.imwrite(frame_path, frame)

    # Apply ELA and extract features
    ela_image_path = apply_ela(frame_path)
    if ela_image_path is None:
        return jsonify({"error": "Failed to apply ELA"}), 500

    features = extract_features_from_image(ela_image_path)
    if features is None:
        return jsonify({"error": "Failed to extract features"}), 400

    # Make predictions using SVM and KNN models
    try:
        svm_prediction = svm_model.predict([features])[0]
        knn_prediction = knn_model.predict([features])[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

    # Clean up temporary files
    os.remove(frame_path)
    os.remove(ela_image_path)
    os.remove(video_path)

    return jsonify({
        "svm_prediction": "FAKE" if svm_prediction == 0 else "REAL",
        "knn_prediction": "FAKE" if knn_prediction == 0 else "REAL"
    })

if __name__ == "__main__":
    app.run(debug=True)

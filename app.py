# File: app.py
import os
from flask import Flask, render_template, request, jsonify
import torch
import cv2
import numpy as np
import os
import pathlib

# Use relative paths
BASE_DIR = pathlib.Path(__file__).parent.resolve()
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Rest of your original app.py remains the same
# Import YOLOv5 from Ultralytics
from ultralytics import YOLO

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained YOLOv5 model
model = YOLO('yolov5s.pt')  # You can change to yolov5m, yolov5l for more accuracy

@app.route('/', methods=['GET'])
def index():
    """Render the main upload page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Process uploaded image and perform object detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    try:
        # Perform object detection
        results = model(filepath)
        
        # Process detection results
        detections = []
        for result in results:
            # Get bounding boxes, confidence, and class names
            boxes = result.boxes
            for box in boxes:
                # Extract detection details
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = result.names[int(box.cls[0].item())]
                
                detections.append({
                    'class': cls,
                    'confidence': round(conf, 2),
                    'bbox': [round(x1), round(y1), round(x2), round(y2)]
                })
        
        # Visualize detected objects
        img = cv2.imread(filepath)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class']} {det['confidence']}"
            cv2.putText(img, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save annotated image
        annotated_path = os.path.join(UPLOAD_FOLDER, 'annotated_' + file.filename)
        cv2.imwrite(annotated_path, img)
        
        return jsonify({
            'detections': detections,
            'original_image': filepath.replace('static/', ''),
            'annotated_image': annotated_path.replace('static/', '')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
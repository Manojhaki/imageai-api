from flask import Flask, request, jsonify
from flask_cors import CORS

import os
from PIL import Image, ImageDraw
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor

app = Flask(__name__)
CORS(app)

# Explicitly use CPU
device = torch.device("cpu")

# Load processor and model from Hugging Face (YOLOv5 variant)
model_name = "facebook/detr-resnet-50"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForObjectDetection.from_pretrained(model_name).to(device)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_path = "input.jpg"
    image_file.save(image_path)
    img = Image.open(image_path).convert("RGB")

    # Preprocess image
    inputs = processor(images=img, return_tensors="pt").to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process outputs
    results = processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=[img.size]
    )[0]

     # Draw bounding boxes
    draw = ImageDraw.Draw(img)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(coord, 2) for coord in box.tolist()]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]} {score:.2f}", fill="red")

    # Save output image
    output_path = "output.jpg"
    img.save(output_path)

    # Format results
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append({
            "label": model.config.id2label[label.item()],
            "score": round(score.item(), 4),
            "box": [round(coord, 2) for coord in box.tolist()]
        })

    return jsonify(detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

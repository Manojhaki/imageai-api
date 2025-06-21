from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor

app = Flask(__name__)
CORS(app)

# Load model and processor (using CPU)
device = torch.device("cpu")
model_name = "facebook/detr-resnet-50"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForObjectDetection.from_pretrained(model_name).to(device)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Read image
    image = request.files['image']
    img = Image.open(image).convert("RGB")

    # Preprocess and infer
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=[img.size]
    )[0]

    # Optional: draw bounding boxes on image
    draw = ImageDraw.Draw(img)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(x, 2) for x in box.tolist()]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]} {score:.2f}", fill="red")

    # Save image with bounding boxes
    img.save("output.jpg")

    # Return detection metadata
    return jsonify([
        {
            "label": model.config.id2label[label.item()],
            "score": round(score.item(), 4),
            "box": [round(x, 2) for x in box.tolist()]
        }
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"])
    ])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

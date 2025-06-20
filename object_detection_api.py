from flask import Flask, request, jsonify
from imageai.Detection import ObjectDetection
from flask_cors import CORS

import os

app = Flask(__name__)
CORS(app)

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolo.h5")  # Ensure yolo.h5 is in the same folder
detector.loadModel()

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    image.save("input.jpg")  # Save uploaded image
    output_path = "output.jpg"

    detections = detector.detectObjectsFromImage(
        input_image="input.jpg",
        output_image_path=output_path
    )

    return jsonify(detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

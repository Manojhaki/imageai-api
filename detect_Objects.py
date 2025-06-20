from imageai.Detection import ObjectDetection
import os

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolo.h5")
detector.loadModel()

detections = detector.detectObjectsFromImage(
    input_image="input.jpg",
    output_image_path="output.jpg"
)

for obj in detections:
    print(obj["name"], ":", obj["percentage_probability"])

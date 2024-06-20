import os
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolov8n.yaml")

# Load a model
model = YOLO("yolov8n.pt")


if __name__ == '__main__':
    data_path = os.path.join("datasets", "data.yaml") # Path to data.yaml file
    results = model.train(data=data_path, epochs=50, imgsz=640, device="cuda")
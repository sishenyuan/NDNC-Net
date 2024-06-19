from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolov8n.yaml")

# Load a model
model = YOLO("yolov8n.pt")


if __name__ == '__main__':
    results = model.train(data="datasets/data.yaml", epochs=50, imgsz=640, device="mps") # use 'cuda' for device if you have a Nvidia GPU
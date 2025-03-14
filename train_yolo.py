from ultralytics import YOLO

# Load YOLOv8 model (you can use yolov8n.pt, yolov8s.pt, etc.)
model = YOLO("yolov8n.pt")

# Train on your dataset
model.train(data="dataset/dataset.yaml", epochs=50, imgsz=640, batch=8)

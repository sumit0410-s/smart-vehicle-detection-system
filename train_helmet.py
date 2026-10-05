from ultralytics import YOLO

# Base YOLO model
model = YOLO("yolov8n.pt")

# Train helmet detection model
model.train(
    data="helmet_dataset/data.yaml",
    epochs=20,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    project="runs",
    name="helmet_training"
)

print("Helmet model training completed!")
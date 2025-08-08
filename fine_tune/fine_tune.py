from ultralytics import YOLO

model = YOLO("drowsiness_detection.pt")  


model.train(
    data="dataset.yaml",
    epochs=20,
    imgsz=640,
    batch=8
)

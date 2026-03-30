from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="dataset/data.yaml",
    model="yolo11n.pt",
    epochs=50,
    imgsz=640,
    batch=8,
    patience=20,       # early stopping if val loss stagnates
    translate=0.2,     # shift wave position — helps generalize across machines
)

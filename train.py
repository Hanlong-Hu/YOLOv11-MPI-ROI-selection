from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="dataset/data.yaml",
    model="yolo11s.pt",
    epochs=150,
    imgsz=640,
    batch=4,
    patience=20,       # early stopping if val loss stagnates
    translate=0.2,     # shift wave position — helps generalize across machines
    flipud=0.0,    # don't flip — wave direction matters
    fliplr=0.5,    # horizontal flip is safe                                      
    hsv_s=0.3,     # vary saturation (different machine displays)                 
    degrees=5,     # slight rotation 
)

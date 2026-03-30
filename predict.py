import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MODEL = "runs/detect/train2/weights/best.pt"
CONF = 0.25
# Merge boxes whose vertical overlap ratio exceeds this threshold
MERGE_IOU_THRESHOLD = 0.5


def vertical_overlap_ratio(box1, box2):
    """Return the vertical overlap ratio between two boxes (0–1)."""
    y1_min, y1_max = box1[1], box1[3]
    y2_min, y2_max = box2[1], box2[3]
    overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    union = max(y1_max, y2_max) - min(y1_min, y2_min)
    return overlap / union if union > 0 else 0


def merge_boxes(boxes):
    """Merge horizontally adjacent boxes that share significant vertical overlap."""
    if len(boxes) <= 1:
        return boxes

    merged = True
    while merged:
        merged = False
        result = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            current = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if vertical_overlap_ratio(current, boxes[j]) >= MERGE_IOU_THRESHOLD:
                    # Merge into one encompassing box
                    current = [
                        min(current[0], boxes[j][0]),
                        min(current[1], boxes[j][1]),
                        max(current[2], boxes[j][2]),
                        max(current[3], boxes[j][3]),
                        max(current[4], boxes[j][4]),  # keep highest confidence
                    ]
                    used[j] = True
            result.append(current)
            used[i] = True
        boxes = result
        if len(result) < len(boxes):
            merged = True

    return boxes


def predict(source):
    model = YOLO(MODEL)
    results = model.predict(source=source, conf=CONF, save=False)

    output_dir = Path("runs/detect/predict_merged")
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        img = r.orig_img.copy()
        path = Path(r.path)

        boxes = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            boxes.append([x1, y1, x2, y2, conf])

        merged = merge_boxes(boxes)

        for x1, y1, x2, y2, conf in merged:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            label = f"Ultrasound-Waves {conf:.2f}"
            cv2.putText(img, label, (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out_path = output_dir / path.name
        cv2.imwrite(str(out_path), img)
        print(f"Saved: {out_path}  ({len(r.boxes)} → {len(merged)} boxes)")


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "manual-test"
    predict(source)

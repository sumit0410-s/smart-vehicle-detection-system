import os
import cv2
from ultralytics import YOLO


# ==============================
# HELMET MODEL
# ==============================

MODEL_PATH = "models/best.pt"

model = YOLO(MODEL_PATH)

print("Helmet Model Classes:", model.names)


# ==============================
# SETTINGS
# ==============================

CONFIDENCE = 0.25
IMAGE_SIZE = 640


# ==============================
# HELMET DETECTION
# ==============================

def detect_helmet(image_path):

    print("--------------------------------")
    print("Starting Helmet Detection")
    print("Input:", image_path)
    print("--------------------------------")

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Run YOLO directly on complete image
    results = model.predict(
        source=image,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        iou=0.50,
        max_det=100,
        verbose=False
    )

    result = results[0]

    # Counters
    with_helmet = 0
    without_helmet = 0

    # Copy image for drawing
    output_image = image.copy()

    # ==============================
    # PROCESS DETECTIONS
    # ==============================

    for box in result.boxes:

        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        # Class 0 = With Helmet
        if class_id == 0:
            label = f"With Helmet {confidence:.2f}"
            with_helmet += 1

            color = (0, 255, 0)

        # Class 1 = Without Helmet
        elif class_id == 1:
            label = f"Without Helmet {confidence:.2f}"
            without_helmet += 1

            color = (0, 0, 255)

        else:
            continue

        # Draw bounding box
        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Draw label
        cv2.putText(
            output_image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # ==============================
    # SAVE OUTPUT
    # ==============================

    os.makedirs("data/output", exist_ok=True)

    output_path = "data/output/helmet_detection.jpg"

    cv2.imwrite(
        output_path,
        output_image
    )

    # ==============================
    # PRINT RESULTS
    # ==============================

    print("--------------------------------")
    print("Helmet Detection Completed")
    print("With Helmet:", with_helmet)
    print("Without Helmet:", without_helmet)
    print("Output:", output_path)
    print("--------------------------------")

    return output_path, with_helmet, without_helmet
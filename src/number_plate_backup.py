import cv2
import os
import re
import pytesseract
import easyocr
from ultralytics import YOLO


# -----------------------------
# Models
# -----------------------------

plate_model = YOLO("models/license_plate.pt")

ocr_reader = easyocr.Reader(
    ["en"],
    gpu=False
)


# -----------------------------
# Clean OCR text
# -----------------------------

def clean_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


# -----------------------------
# OCR using EasyOCR
# -----------------------------

def easyocr_text(image):
    results = ocr_reader.readtext(
        image,
        detail=1,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    candidates = []

    for result in results:
        text = clean_text(result[1])
        confidence = float(result[2])

        if len(text) >= 4:
            candidates.append((text, confidence))

    if not candidates:
        return "", 0.0

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[0]


# -----------------------------
# OCR using Tesseract
# -----------------------------

def tesseract_text(image):

    config = (
        "--psm 7 "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    text = pytesseract.image_to_string(
        image,
        config=config
    )

    text = clean_text(text)

    return text


# -----------------------------
# Prepare plate image
# -----------------------------

def preprocess_plate(crop):

    # Resize
    crop = cv2.resize(
        crop,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    # Grayscale
    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    # Noise removal
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Sharpen
    sharpen = cv2.addWeighted(
        gray,
        1.5,
        cv2.GaussianBlur(gray, (0, 0), 3),
        -0.5,
        0
    )

    # Threshold
    _, threshold = cv2.threshold(
        sharpen,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return crop, gray, sharpen, threshold


# -----------------------------
# Main Number Plate Detection
# -----------------------------

def detect_number_plate(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # YOLO license plate detection
    results = plate_model.predict(
        image,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    output_image = image.copy()

    detected_plates = []

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    # Process every detected plate
    for index, box in enumerate(result.boxes):

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        confidence = float(
            box.conf[0]
        )

        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        # Crop plate
        plate_crop = image[
            y1:y2,
            x1:x2
        ]

        if plate_crop.size == 0:
            continue

        # Save original detected plate crop
        crop_path = (
            f"data/output/"
            f"plate_crop_{index + 1}.jpg"
        )

        cv2.imwrite(
            crop_path,
            plate_crop
        )

        # Preprocessing
        original, gray, sharpen, threshold = (
            preprocess_plate(plate_crop)
        )

        # -------------------------
        # EasyOCR
        # -------------------------

        easy_text, easy_conf = easyocr_text(
            original
        )

        # Try sharpened image also
        if not easy_text:
            easy_text, easy_conf = easyocr_text(
                sharpen
            )

        # -------------------------
        # Tesseract
        # -------------------------

        tess_text = tesseract_text(
            threshold
        )

        # -------------------------
        # Select OCR result
        # -------------------------

        if easy_text and easy_conf >= 0.30:

            final_text = easy_text

        elif tess_text:

            final_text = tess_text

        else:

            final_text = "UNKNOWN"

        detected_plates.append(
            final_text
        )

        # -------------------------
        # Draw bounding box
        # -------------------------

        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        label = (
            f"{final_text} "
            f"({confidence:.2f})"
        )

        # Text background
        (text_width, text_height), _ = (
            cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                2
            )
        )

        text_y = max(
            y1 - 10,
            text_height + 10
        )

        cv2.rectangle(
            output_image,
            (x1, text_y - text_height - 10),
            (x1 + text_width + 10, text_y + 5),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            output_image,
            label,
            (x1 + 5, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

    # Output file
    output_path = (
        "data/output/number_plate.jpg"
    )

    cv2.imwrite(
        output_path,
        output_image
    )

    print(
        "Detected Plates:",
        detected_plates
    )

    print(
        "Output:",
        output_path
    )

    return (
        output_path,
        detected_plates
    )
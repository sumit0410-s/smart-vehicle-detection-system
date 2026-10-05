import cv2
import os
import re

from ultralytics import YOLO
from paddleocr import PaddleOCR


# ==========================================
# MODELS
# ==========================================

plate_model = YOLO("models/license_plate.pt")

ocr = PaddleOCR(lang="en")


# ==========================================
# CLEAN PLATE TEXT
# ==========================================

def clean_plate_text(text):
    """
    Remove unwanted characters from OCR text.
    """

    text = str(text).upper().strip()

    # Keep only A-Z, 0-9 and hyphen
    text = re.sub(r"[^A-Z0-9-]", "", text)

    return text


# ==========================================
# PADDLEOCR
# ==========================================

def run_paddleocr(crop):
    """
    Run PaddleOCR on a license plate crop.
    """

    try:

        result = ocr.predict(crop)

        if not result:
            return "", 0.0

        data = result[0]

        texts = []
        scores = []

        # PaddleOCR 3.x
        if isinstance(data, dict):

            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])

        else:

            try:
                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])

            except Exception:
                pass

        if not texts:
            return "", 0.0

        # Combine detected text
        final_text = ""

        for text in texts:

            cleaned = clean_plate_text(text)

            if cleaned:
                final_text += cleaned

        if not final_text:
            return "", 0.0

        # Calculate average OCR confidence
        if scores:

            confidence = sum(
                float(score) for score in scores
            ) / len(scores)

        else:

            confidence = 0.0

        return final_text, confidence

    except Exception as e:

        print("PaddleOCR Error:", e)

        return "", 0.0


# ==========================================
# DETECT NUMBER PLATE
# ==========================================

def detect_number_plate(image_path):

    print()
    print("========================================")
    print("LICENSE PLATE DETECTION + OCR")
    print("========================================")

    # --------------------------------------
    # READ IMAGE
    # --------------------------------------

    image = cv2.imread(image_path)

    if image is None:

        print("ERROR: Image could not be loaded.")

        return None, []

    output_image = image.copy()

    # --------------------------------------
    # CREATE OUTPUT FOLDER
    # --------------------------------------

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    # --------------------------------------
    # YOLO LICENSE PLATE DETECTION
    # --------------------------------------

    results = plate_model.predict(
        image,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    detected_plates = []

    plate_number = 0

    # ======================================
    # PROCESS DETECTED PLATES
    # ======================================

    for result in results:

        boxes = result.boxes

        for box in boxes:

            plate_number += 1

            # ----------------------------------
            # GET BOUNDING BOX
            # ----------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # ----------------------------------
            # YOLO DETECTION CONFIDENCE
            # ----------------------------------

            detection_confidence = float(
                box.conf[0]
            )

            # ----------------------------------
            # IMAGE SIZE
            # ----------------------------------

            h, w = image.shape[:2]

            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w, x2)
            y2 = min(h, y2)

            # ----------------------------------
            # CROP LICENSE PLATE
            # ----------------------------------

            crop = image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            # ----------------------------------
            # SAVE ORIGINAL CROP
            # ----------------------------------

            crop_path = (
                f"data/output/"
                f"plate_crop_{plate_number}.jpg"
            )

            cv2.imwrite(
                crop_path,
                crop
            )

            # ----------------------------------
            # ENLARGE LICENSE PLATE
            # ----------------------------------

            enlarged_crop = cv2.resize(
                crop,
                None,
                fx=5,
                fy=5,
                interpolation=cv2.INTER_CUBIC
            )

            # ----------------------------------
            # OCR
            # ----------------------------------

            text, ocr_confidence = run_paddleocr(
                enlarged_crop
            )

            # ==================================
            # CONFIDENCE LOGIC
            # ==================================

            if text:

                if ocr_confidence >= 0.80:

                    status = "HIGH CONFIDENCE"

                elif ocr_confidence >= 0.50:

                    status = "MEDIUM CONFIDENCE"

                else:

                    status = "LOW CONFIDENCE"

            else:

                text = "UNKNOWN"

                status = "LOW CONFIDENCE"

            # ==================================
            # STORE RESULT
            # ==================================

            detected_plates.append(
                {
                    "plate": text,

                    "ocr_confidence": round(
                        ocr_confidence * 100,
                        2
                    ),

                    "detection_confidence": round(
                        detection_confidence * 100,
                        2
                    ),

                    "status": status,

                    "crop": crop_path
                }
            )

            # ==================================
            # DRAW LICENSE PLATE BOX
            # ==================================

            cv2.rectangle(
                output_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # ----------------------------------
            # LABEL
            # ----------------------------------

            label = (
                f"{text} | "
                f"{ocr_confidence * 100:.1f}%"
            )

            text_y = max(
                30,
                y1 - 10
            )

            # ----------------------------------
            # LABEL BACKGROUND
            # ----------------------------------

            cv2.rectangle(
                output_image,

                (x1, text_y - 30),

                (x2, text_y + 5),

                (0, 0, 0),

                -1
            )

            # ----------------------------------
            # WRITE TEXT
            # ----------------------------------

            cv2.putText(
                output_image,

                label,

                (x1 + 5, text_y),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2
            )

            # ==================================
            # PRINT RESULT
            # ==================================

            print()
            print("----------------------------------------")

            print(
                "Plate:",
                text
            )

            print(
                "OCR Confidence:",
                f"{ocr_confidence * 100:.2f}%"
            )

            print(
                "Detection Confidence:",
                f"{detection_confidence * 100:.2f}%"
            )

            print(
                "Status:",
                status
            )

            print(
                "Crop:",
                crop_path
            )

    # ==========================================
    # SAVE FINAL OUTPUT
    # ==========================================

    output_path = (
        "data/output/number_plate.jpg"
    )

    cv2.imwrite(
        output_path,
        output_image
    )

    # ==========================================
    # FINAL SUMMARY
    # ==========================================

    print()
    print("========================================")

    print(
        "Output:",
        output_path
    )

    print(
        "Total plates:",
        len(detected_plates)
    )

    print(
        "========================================")

    return (
        output_path,
        detected_plates
    )
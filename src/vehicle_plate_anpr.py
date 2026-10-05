import cv2
import os
import re
import subprocess
import imageio_ffmpeg

from ultralytics import YOLO
from paddleocr import PaddleOCR


# ============================================================
# MODELS
# ============================================================

vehicle_model = YOLO("yolo11n.pt")

plate_model = YOLO("models/license_plate.pt")

ocr = PaddleOCR(lang="en")


# ============================================================
# SETTINGS
# ============================================================

VEHICLE_CONFIDENCE = 0.30

PLATE_CONFIDENCE = 0.25

OCR_INTERVAL = 10


# Vehicle classes from YOLO
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def clean_plate_text(text):

    text = str(text).upper().strip()

    text = re.sub(
        r"[^A-Z0-9-]",
        "",
        text
    )

    return text


# ============================================================
# PADDLE OCR
# ============================================================

def run_ocr(crop):

    try:

        if crop is None or crop.size == 0:

            return "", 0.0


        # Enlarge plate
        enlarged = cv2.resize(
            crop,
            None,
            fx=5,
            fy=5,
            interpolation=cv2.INTER_CUBIC
        )


        result = ocr.predict(enlarged)


        if not result:

            return "", 0.0


        data = result[0]


        texts = data.get(
            "rec_texts",
            []
        )

        scores = data.get(
            "rec_scores",
            []
        )


        if not texts:

            return "", 0.0


        final_text = ""


        for text in texts:

            cleaned = clean_plate_text(
                text
            )

            if cleaned:

                final_text += cleaned


        if not final_text:

            return "", 0.0


        if scores:

            confidence = sum(
                float(score)
                for score in scores
            ) / len(scores)

        else:

            confidence = 0.0


        return (
            final_text,
            confidence
        )


    except Exception as e:

        print(
            "OCR Error:",
            e
        )

        return "", 0.0


# ============================================================
# VEHICLE-WISE VIDEO ANPR
# ============================================================

def detect_vehicle_plate_anpr(video_path):

    print()
    print(
        "========================================"
    )

    print(
        "VEHICLE-WISE VIDEO ANPR"
    )

    print(
        "Vehicle Tracking + Plate Detection + OCR"
    )

    print(
        "========================================"
    )


    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        print(
            "ERROR: Could not open video."
        )

        return None, []


    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    if fps <= 0:

        fps = 25


    # ========================================================
    # OUTPUT
    # ========================================================

    os.makedirs(
        "data/output",
        exist_ok=True
    )


    temp_output_path = (
        "data/output/"
        "vehicle_plate_anpr_temp.mp4"
    )

    output_path = (
        "data/output/"
        "vehicle_plate_anpr.mp4"
    )


    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    out = cv2.VideoWriter(
        temp_output_path,
        fourcc,
        fps,
        (width, height)
    )


    # ========================================================
    # VEHICLE INFORMATION
    # ========================================================

    vehicle_data = {}


    # Example:
    #
    # vehicle_data = {
    #     1: {
    #         "type": "Car",
    #         "plate": "29A33185",
    #         "ocr_confidence": 0.95,
    #         "status": "HIGH CONFIDENCE",
    #         "last_ocr_frame": 100
    #     }
    # }


    frame_number = 0


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    while True:

        ret, frame = cap.read()


        if not ret:

            break


        frame_number += 1


        # ====================================================
        # VEHICLE TRACKING
        # ====================================================

        results = vehicle_model.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            conf=VEHICLE_CONFIDENCE,

            classes=list(
                VEHICLE_CLASSES.keys()
            ),

            imgsz=960,

            verbose=False
        )


        # ====================================================
        # PROCESS VEHICLES
        # ====================================================

        for result in results:

            boxes = result.boxes


            if boxes is None:

                continue


            # ------------------------------------------------
            # GET TRACK IDS
            # ------------------------------------------------

            if boxes.id is not None:

                track_ids = (
                    boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

            else:

                track_ids = []


            # ------------------------------------------------
            # EACH VEHICLE
            # ------------------------------------------------

            for index, box in enumerate(boxes):


                if index >= len(track_ids):

                    continue


                vehicle_id = int(
                    track_ids[index]
                )


                # ============================================
                # VEHICLE CLASS
                # ============================================

                class_id = int(
                    box.cls[0]
                )


                vehicle_type = (
                    VEHICLE_CLASSES.get(
                        class_id,
                        "Vehicle"
                    )
                )


                # ============================================
                # VEHICLE CONFIDENCE
                # ============================================

                vehicle_confidence = float(
                    box.conf[0]
                )


                # ============================================
                # VEHICLE COORDINATES
                # ============================================

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )


                x1 = max(
                    0,
                    x1
                )

                y1 = max(
                    0,
                    y1
                )

                x2 = min(
                    width,
                    x2
                )

                y2 = min(
                    height,
                    y2
                )


                # ============================================
                # VEHICLE CROP
                # ============================================

                vehicle_crop = frame[
                    y1:y2,
                    x1:x2
                ]


                if vehicle_crop.size == 0:

                    continue


                # ============================================
                # CREATE VEHICLE RECORD
                # ============================================

                if vehicle_id not in vehicle_data:

                    vehicle_data[vehicle_id] = {

                        "vehicle_id":
                            vehicle_id,

                        "vehicle_type":
                            vehicle_type,

                        "plate":
                            "PLATE NOT READ",

                        "ocr_confidence":
                            0.0,

                        "plate_detection_confidence":
                            0.0,

                        "status":
                            "NOT READ",

                        "last_ocr_frame":
                            0
                    }


                # =================================================
                # OCR INTERVAL
                # =================================================

                should_run_ocr = (

                    frame_number
                    -
                    vehicle_data[vehicle_id][
                        "last_ocr_frame"
                    ]

                    >= OCR_INTERVAL
                )


                if should_run_ocr:

                    vehicle_data[vehicle_id][
                        "last_ocr_frame"
                    ] = frame_number


                    # ============================================
                    # PLATE DETECTION INSIDE VEHICLE CROP
                    # ============================================

                    plate_results = plate_model.predict(

                        vehicle_crop,

                        conf=PLATE_CONFIDENCE,

                        imgsz=640,

                        verbose=False
                    )


                    best_plate = None

                    best_plate_confidence = 0.0


                    # ============================================
                    # FIND BEST PLATE
                    # ============================================

                    for plate_result in plate_results:

                        if plate_result.boxes is None:

                            continue


                        for plate_box in (
                            plate_result.boxes
                        ):

                            plate_confidence = float(
                                plate_box.conf[0]
                            )


                            if (
                                plate_confidence
                                >
                                best_plate_confidence
                            ):

                                best_plate_confidence = (
                                    plate_confidence
                                )

                                best_plate = (
                                    plate_box
                                )


                    # ============================================
                    # IF PLATE FOUND
                    # ============================================

                    if best_plate is not None:

                        px1, py1, px2, py2 = map(
                            int,
                            best_plate.xyxy[0].tolist()
                        )


                        px1 = max(
                            0,
                            px1
                        )

                        py1 = max(
                            0,
                            py1
                        )

                        px2 = min(
                            vehicle_crop.shape[1],
                            px2
                        )

                        py2 = min(
                            vehicle_crop.shape[0],
                            py2
                        )


                        plate_crop = vehicle_crop[
                            py1:py2,
                            px1:px2
                        ]


                        # ========================================
                        # OCR
                        # ========================================

                        text, ocr_confidence = (
                            run_ocr(
                                plate_crop
                            )
                        )


                        # ========================================
                        # SAVE BEST OCR RESULT
                        # ========================================

                        if text:

                            if ocr_confidence >= 0.80:

                                status = (
                                    "HIGH CONFIDENCE"
                                )

                            elif ocr_confidence >= 0.50:

                                status = (
                                    "MEDIUM CONFIDENCE"
                                )

                            else:

                                status = (
                                    "LOW CONFIDENCE"
                                )


                            old_confidence = (
                                vehicle_data[
                                    vehicle_id
                                ][
                                    "ocr_confidence"
                                ]
                            )


                            # Keep the best reading
                            if (
                                ocr_confidence
                                >=
                                old_confidence
                            ):

                                vehicle_data[
                                    vehicle_id
                                ][
                                    "plate"
                                ] = text


                                vehicle_data[
                                    vehicle_id
                                ][
                                    "ocr_confidence"
                                ] = (
                                    ocr_confidence
                                )


                                vehicle_data[
                                    vehicle_id
                                ][
                                    "plate_detection_confidence"
                                ] = (
                                    best_plate_confidence
                                )


                                vehicle_data[
                                    vehicle_id
                                ][
                                    "status"
                                ] = status


                    # If plate was not found,
                    # keep the previous result.


                # =================================================
                # DRAW VEHICLE BOX
                # =================================================

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (255, 0, 0),

                    2
                )


                # =================================================
                # GET CURRENT DATA
                # =================================================

                plate_text = (
                    vehicle_data[
                        vehicle_id
                    ][
                        "plate"
                    ]
                )


                ocr_confidence = (
                    vehicle_data[
                        vehicle_id
                    ][
                        "ocr_confidence"
                    ]
                )


                # =================================================
                # VEHICLE LABEL
                # =================================================

                vehicle_label = (

                    f"ID {vehicle_id} | "

                    f"{vehicle_type}"
                )


                cv2.putText(

                    frame,

                    vehicle_label,

                    (
                        x1,
                        max(
                            25,
                            y1 - 35
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (255, 0, 0),

                    2
                )


                # =================================================
                # PLATE LABEL
                # =================================================

                plate_label = (

                    f"Plate: {plate_text}"
                )


                if ocr_confidence > 0:

                    plate_label += (

                        f" | "
                        f"{ocr_confidence * 100:.1f}%"
                    )


                cv2.putText(

                    frame,

                    plate_label,

                    (
                        x1,
                        max(
                            50,
                            y1 - 8
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.60,

                    (0, 255, 0),

                    2
                )


        # ====================================================
        # WRITE FRAME
        # ====================================================

        out.write(
            frame
        )


        # ====================================================
        # PROGRESS
        # ====================================================

        if frame_number % 30 == 0:

            percentage = 0

            if total_frames > 0:

                percentage = (
                    frame_number
                    /
                    total_frames
                    *
                    100
                )


            print(

                f"Processing: "
                f"{frame_number}/"
                f"{total_frames} "
                f"({percentage:.1f}%)"
            )


    # ========================================================
    # RELEASE
    # ========================================================

    cap.release()

    out.release()


    # ========================================================
    # CONVERT OUTPUT TO BROWSER-COMPATIBLE H.264 MP4
    # ========================================================

    try:

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_exe,
            "-y",
            "-i",
            temp_output_path,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            output_path
        ]

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

        print("H.264 conversion completed.")

    except Exception as e:

        print(
            "H.264 conversion failed:",
            e
        )

        if os.path.exists(temp_output_path):

            if os.path.exists(output_path):
                os.remove(output_path)

            os.rename(
                temp_output_path,
                output_path
            )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "VEHICLE-WISE ANPR COMPLETED"
    )

    print(
        "Total frames:",
        frame_number
    )

    print(
        "Unique vehicles:",
        len(vehicle_data)
    )

    print(
        "Output:",
        output_path
    )

    print(
        "========================================"
    )


    # ========================================================
    # FINAL VEHICLE RESULTS
    # ========================================================

    print()
    print(
        "VEHICLE RESULTS"
    )

    print(
        "----------------------------------------"
    )


    for vehicle_id, data in vehicle_data.items():

        print(

            f"Vehicle ID: "
            f"{vehicle_id} | "

            f"Type: "
            f"{data['vehicle_type']} | "

            f"Plate: "
            f"{data['plate']} | "

            f"OCR: "
            f"{data['ocr_confidence'] * 100:.2f}% | "

            f"Status: "
            f"{data['status']}"
        )


    print(
        "----------------------------------------"
    )


    # ========================================================
    # RETURN
    # ========================================================

    return (
        output_path,
        list(
            vehicle_data.values()
        )
    )
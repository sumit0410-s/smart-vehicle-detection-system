import cv2
import re
import os
import torch
import subprocess
import imageio_ffmpeg

from ultralytics import YOLO
from paddleocr import PaddleOCR

from src.evidence_manager import save_complete_evidence


# ============================================================
# DEVICE
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("========================================")
print("SMART VEHICLE VIOLATION DETECTOR")
print("========================================")

if torch.cuda.is_available():
    print("GPU ENABLED")
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )
else:
    print("CPU MODE")


# ============================================================
# MODELS
# ============================================================

vehicle_model = YOLO(
    "yolo11n.pt"
)

vehicle_model.to(
    DEVICE
)


helmet_model = YOLO(
    "models/best.pt"
)

helmet_model.to(
    DEVICE
)


plate_model = YOLO(
    "models/license_plate.pt"
)

plate_model.to(
    DEVICE
)


ocr = PaddleOCR(
    lang="en"
)


# ============================================================
# SETTINGS
# ============================================================

VEHICLE_CONFIDENCE = 0.30

HELMET_CONFIDENCE = 0.15

PLATE_CONFIDENCE = 0.10

OCR_INTERVAL = 5


# ============================================================
# VEHICLE CLASSES
# ============================================================

VEHICLE_CLASSES = {

    2: "Car",

    3: "Motorcycle",

    5: "Bus",

    7: "Truck"
}


# ============================================================
# HELMET CLASSES
# ============================================================

HELMET_CLASSES = {

    0: "With Helmet",

    1: "Without Helmet"
}


# ============================================================
# CLEAN PLATE TEXT
# ============================================================

def clean_plate_text(text):

    if text is None:
        return ""

    text = str(
        text
    ).upper().strip()

    text = re.sub(
        r"[^A-Z0-9-]",
        "",
        text
    )

    return text


# ============================================================
# OCR
# ============================================================

def read_plate_text(
    plate_crop
):

    if (
        plate_crop is None
        or plate_crop.size == 0
    ):
        return "", 0.0


    try:

        enlarged = cv2.resize(
            plate_crop,
            None,
            fx=5,
            fy=5,
            interpolation=cv2.INTER_CUBIC
        )


        result = ocr.predict(
            enlarged
        )


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
# PLATE STATUS
# ============================================================

def get_plate_status(
    confidence
):

    if confidence >= 0.80:

        return "HIGH"

    elif confidence >= 0.50:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# TRACKING HELPERS
# ============================================================

def box_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter / float(area_a + area_b - inter)


def center_distance(a, b):

    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0

    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


# ============================================================
# DETECT VIOLATIONS
# ============================================================

def detect_violations(
    video_path
):

    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Video open nahi hua: {video_path}"
        )


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
    # OUTPUT VIDEO
    # ========================================================

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    temp_output_path = (
        "data/output/"
        "violation_detection_temp.mp4"
    )

    output_path = (
        "data/output/"
        "violation_detection.mp4"
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

    if not out.isOpened():

        cap.release()

        raise RuntimeError(
            "Output video create nahi hua."
        )


    print(
        "----------------------------------------"
    )

    print(
        "Video:",
        video_path
    )

    print(
        "Frames:",
        total_frames
    )

    print(
        "Resolution:",
        width,
        "x",
        height
    )

    print(
        "----------------------------------------"
    )


    # ========================================================
    # VEHICLE RECORDS
    # ========================================================

    vehicles = {}
    canonical_tracks = {}
    raw_to_canonical = {}
    next_canonical_id = 1


    frame_number = 0


    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

        success, frame = cap.read()


        if not success:

            break


        frame_number += 1


        # ====================================================
        # VEHICLE TRACKING
        # ====================================================

        results = vehicle_model.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            classes=list(
                VEHICLE_CLASSES.keys()
            ),

            conf=VEHICLE_CONFIDENCE,

            imgsz=640,

            device=DEVICE,

            verbose=False
        )


        if not results:

            continue


        result = results[0]


        if result.boxes is None:

            continue


        if result.boxes.id is None:

            continue


        # ====================================================
        # PROCESS VEHICLES
        # ====================================================

        for box, track_id in zip(

            result.boxes,

            result.boxes.id

        ):

            raw_track_id = int(track_id)

            vehicle_class = int(
                box.cls[0]
            )


            vehicle_type = (
                VEHICLE_CLASSES.get(
                    vehicle_class,
                    "Unknown"
                )
            )


            # =================================================
            # COORDINATES
            # =================================================

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
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


            current_bbox = (x1, y1, x2, y2)

            # Keep a stable application-level ID even when YOLO's tracker
            # changes its temporary ID. This prevents one physical vehicle
            # from being counted multiple times.
            vehicle_id = raw_to_canonical.get(raw_track_id)

            if vehicle_id is not None:
                previous = canonical_tracks.get(vehicle_id)
                if previous is None or previous.get("vehicle_class") != vehicle_class:
                    vehicle_id = None

            if vehicle_id is None:
                best_id = None
                best_score = 0.0

                for candidate_id, state in canonical_tracks.items():
                    if state.get("vehicle_class") != vehicle_class:
                        continue

                    gap = frame_number - state.get("last_seen", frame_number)
                    if gap > 45:
                        continue

                    old_bbox = state.get("bbox")
                    if old_bbox is None:
                        continue

                    iou = box_iou(current_bbox, old_bbox)
                    dist = center_distance(current_bbox, old_bbox)
                    old_w = max(1, old_bbox[2] - old_bbox[0])
                    old_h = max(1, old_bbox[3] - old_bbox[1])
                    size = max(1.0, (old_w + old_h) / 2.0)
                    proximity = max(0.0, 1.0 - dist / (size * 2.5))
                    score = max(iou, proximity * 0.55)

                    if (iou >= 0.20 or proximity >= 0.45) and score > best_score:
                        best_score = score
                        best_id = candidate_id

                if best_id is not None:
                    vehicle_id = best_id
                else:
                    vehicle_id = next_canonical_id
                    next_canonical_id += 1

            raw_to_canonical[raw_track_id] = vehicle_id
            canonical_tracks[vehicle_id] = {
                "bbox": current_bbox,
                "vehicle_class": vehicle_class,
                "last_seen": frame_number,
            }

            vehicle_crop = frame[
                y1:y2,
                x1:x2
            ]


            if vehicle_crop.size == 0:

                continue


            # =================================================
            # CREATE VEHICLE RECORD
            # =================================================

            if vehicle_id not in vehicles:

                vehicles[vehicle_id] = {

                    "vehicle_id":
                        vehicle_id,

                    "vehicle_type":
                        vehicle_type,

                    "helmet_status":
                        "NOT CHECKED",

                    "helmet_violation":
                        False,

                    "triple_riding":
                        False,

                    "wrong_side":
                        False,

                    "red_light_violation":
                        False,

                    "plate":
                        "NOT READ",

                    "ocr_confidence":
                        0.0,

                    "plate_detection_confidence":
                        0.0,

                    "plate_status":
                        "NOT READ",

                    "evidence":
                        None,

                    "evidence_saved":
                        False
                }


            # =================================================
            # HELMET DETECTION
            # =================================================

            if vehicle_type == "Motorcycle":

                # Detect helmets on the motorcycle crop. The original
                # implementation ran on the whole frame, which made the
                # rider/head regions too small for reliable detection.
                helmet_results = helmet_model(
                    vehicle_crop,
                    conf=HELMET_CONFIDENCE,
                    imgsz=960,
                    device=DEVICE,
                    verbose=False
                )

                best_with_conf = 0.0
                best_without_conf = 0.0

                for helmet_result in helmet_results:
                    if helmet_result.boxes is None:
                        continue

                    for helmet_box in helmet_result.boxes:
                        helmet_class = int(helmet_box.cls[0])
                        helmet_confidence = float(helmet_box.conf[0])

                        if helmet_class == 0:
                            best_with_conf = max(
                                best_with_conf,
                                helmet_confidence
                            )
                        elif helmet_class == 1:
                            best_without_conf = max(
                                best_without_conf,
                                helmet_confidence
                            )

                # Class 1 = Without Helmet. Once a violation is detected,
                # keep it true even if a later frame is uncertain.
                if best_without_conf >= HELMET_CONFIDENCE:
                    vehicles[vehicle_id]["helmet_status"] = "Without Helmet"
                    vehicles[vehicle_id]["helmet_violation"] = True

                elif best_with_conf >= HELMET_CONFIDENCE:
                    if not vehicles[vehicle_id]["helmet_violation"]:
                        vehicles[vehicle_id]["helmet_status"] = "With Helmet"


            # =================================================
            # NUMBER PLATE
            # =================================================

            if (
                frame_number
                % OCR_INTERVAL
                == 0
            ):

                try:

                    plate_results = (
                        plate_model.predict(

                            vehicle_crop,

                            conf=PLATE_CONFIDENCE,

                            imgsz=960,

                            device=DEVICE,

                            verbose=False
                        )
                    )


                    best_plate = None

                    best_plate_confidence = 0.0


                    for plate_result in (
                        plate_results
                    ):

                        if (
                            plate_result.boxes
                            is None
                        ):
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


                    # =================================================
                    # PLATE FOUND
                    # =================================================

                    if best_plate is not None:

                        px1, py1, px2, py2 = map(
                            int,
                            best_plate.xyxy[0]
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


                        # Add a small margin around the detected plate
                        # to give OCR a little more context.
                        margin_x = max(4, int((px2 - px1) * 0.08))
                        margin_y = max(3, int((py2 - py1) * 0.15))

                        crop_x1 = max(0, px1 - margin_x)
                        crop_y1 = max(0, py1 - margin_y)
                        crop_x2 = min(vehicle_crop.shape[1], px2 + margin_x)
                        crop_y2 = min(vehicle_crop.shape[0], py2 + margin_y)

                        plate_crop = vehicle_crop[
                            crop_y1:crop_y2,
                            crop_x1:crop_x2
                        ]


                        plate_text, ocr_confidence = (
                            read_plate_text(
                                plate_crop
                            )
                        )


                        if plate_text:

                            if (
                                ocr_confidence
                                >
                                vehicles[
                                    vehicle_id
                                ][
                                    "ocr_confidence"
                                ]
                            ):

                                vehicles[
                                    vehicle_id
                                ][
                                    "plate"
                                ] = plate_text


                                vehicles[
                                    vehicle_id
                                ][
                                    "ocr_confidence"
                                ] = (
                                    ocr_confidence
                                )


                                vehicles[
                                    vehicle_id
                                ][
                                    "plate_detection_confidence"
                                ] = (
                                    best_plate_confidence
                                )


                                vehicles[
                                    vehicle_id
                                ][
                                    "plate_status"
                                ] = (
                                    get_plate_status(
                                        ocr_confidence
                                    )
                                )


                except Exception as e:

                    print(
                        "Plate/OCR Error:",
                        e
                    )


            # =================================================
            # EVIDENCE FOR HELMET VIOLATION
            # =================================================

            if (

                vehicles[
                    vehicle_id
                ][
                    "helmet_violation"
                ]

                and

                not vehicles[
                    vehicle_id
                ][
                    "evidence_saved"
                ]

            ):

                violations = [
                    "Riding Without Helmet"
                ]


                plate = vehicles[
                    vehicle_id
                ].get(
                    "plate",
                    "NOT READ"
                )


                try:

                    evidence = (
                        save_complete_evidence(

                            frame=frame.copy(),

                            vehicle_id=vehicle_id,

                            vehicle_type=vehicle_type,

                            plate=plate,

                            violations=violations
                        )
                    )


                    if evidence:

                        vehicles[
                            vehicle_id
                        ][
                            "evidence"
                        ] = evidence


                        vehicles[
                            vehicle_id
                        ][
                            "evidence_saved"
                        ] = True


                        print(
                            "Evidence created for vehicle:",
                            vehicle_id
                        )


                except Exception as e:

                    print(
                        "Evidence Error:",
                        e
                    )


        # ====================================================
        # WRITE PROCESSED FRAME
        # ====================================================

        out.write(
            frame
        )


    # ========================================================
    # RELEASE
    # ========================================================

    cap.release()

    out.release()


    # ========================================================
    # CONVERT TO BROWSER-COMPATIBLE H.264 MP4
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

        if os.path.exists(
            temp_output_path
        ):

            os.remove(
                temp_output_path
            )

        print(
            "Violation output H.264 conversion completed."
        )

    except Exception as e:

        print(
            "H.264 conversion failed:",
            e
        )

        if os.path.exists(
            temp_output_path
        ):

            if os.path.exists(
                output_path
            ):

                os.remove(
                    output_path
                )

            os.rename(
                temp_output_path,
                output_path
            )


    # ========================================================
    # RELEASE
    # ========================================================


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    vehicle_list = list(
        vehicles.values()
    )


    motorcycles = sum(

        1

        for vehicle in vehicle_list

        if vehicle.get(
            "vehicle_type"
        ) == "Motorcycle"

    )


    helmet_violations = sum(

        1

        for vehicle in vehicle_list

        if vehicle.get(
            "helmet_violation",
            False
        )

    )


    plates_read = sum(

        1

        for vehicle in vehicle_list

        if vehicle.get(
            "plate"
        )
        not in [
            "NOT READ",
            "",
            None
        ]

    )


    evidence_count = sum(

        1

        for vehicle in vehicle_list

        if vehicle.get(
            "evidence_saved",
            False
        )

    )


    print()
    print(
        "========================================"
    )

    print(
        "VIOLATION DETECTION COMPLETED"
    )

    print(
        "========================================"
    )

    print(
        "Unique vehicles:",
        len(vehicle_list)
    )

    print(
        "Motorcycles:",
        motorcycles
    )

    print(
        "Helmet violations:",
        helmet_violations
    )

    print(
        "Number plates read:",
        plates_read
    )

    print(
        "Evidence saved:",
        evidence_count
    )

    print(
        "========================================"
    )


    return vehicle_list
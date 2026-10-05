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

plate_model = YOLO("models/license_plate.pt")

ocr = PaddleOCR(lang="en")


# ============================================================
# SETTINGS
# ============================================================

OCR_INTERVAL = 10
PLATE_CONFIDENCE = 0.25


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
# OCR FUNCTION
# ============================================================

def run_paddleocr(crop):

    try:

        result = ocr.predict(crop)

        if not result:
            return "", 0.0

        data = result[0]

        texts = []
        scores = []

        if isinstance(data, dict):

            texts = data.get(
                "rec_texts",
                []
            )

            scores = data.get(
                "rec_scores",
                []
            )

        else:

            try:

                texts = data.get(
                    "rec_texts",
                    []
                )

                scores = data.get(
                    "rec_scores",
                    []
                )

            except Exception:

                pass

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
            "PaddleOCR Error:",
            e
        )

        return "", 0.0


# ============================================================
# VIDEO NUMBER PLATE DETECTION
# ============================================================

def detect_video_number_plate(video_path):

    print()
    print(
        "========================================"
    )
    print(
        "VIDEO NUMBER PLATE DETECTION"
    )
    print(
        "PLATE TRACKING + PADDLEOCR"
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
    # OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        "data/output",
        exist_ok=True
    )


    temp_output_path = (
        "data/output/"
        "video_number_plate_temp.mp4"
    )

    output_path = (
        "data/output/"
        "video_number_plate.mp4"
    )


    # ========================================================
    # VIDEO WRITER
    # ========================================================

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
    # PLATE TRACK INFORMATION
    # ========================================================

    # Each tracking ID gets its own information

    plate_tracks = {}


    # Example:

    # plate_tracks = {
    #     1: {
    #         "text": "29A33185",
    #         "confidence": 0.99,
    #         "status": "HIGH CONFIDENCE"
    #     },
    #
    #     2: {
    #         "text": "UNKNOWN",
    #         "confidence": 0.30,
    #         "status": "LOW CONFIDENCE"
    #     }
    # }


    # ========================================================
    # FINAL OCR RESULTS
    # ========================================================

    detected_plates = []


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
        # RUN PLATE TRACKING
        # ====================================================

        results = plate_model.track(
            frame,
            persist=True,
            conf=PLATE_CONFIDENCE,
            imgsz=640,
            verbose=False
        )


        # ====================================================
        # PROCESS RESULTS
        # ====================================================

        for result in results:

            boxes = result.boxes


            if boxes is None:
                continue


            # ------------------------------------------------
            # CHECK TRACK IDs
            # ------------------------------------------------

            if boxes.id is not None:

                track_ids = (
                    boxes.id.int()
                    .cpu()
                    .tolist()
                )

            else:

                track_ids = []


            # ------------------------------------------------
            # PROCESS EACH PLATE
            # ------------------------------------------------

            for index, box in enumerate(boxes):


                # ============================================
                # TRACK ID
                # ============================================

                if index < len(track_ids):

                    track_id = int(
                        track_ids[index]
                    )

                else:

                    # If tracking ID unavailable,
                    # skip this plate.

                    continue


                # ============================================
                # BOUNDING BOX
                # ============================================

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )


                # ============================================
                # DETECTION CONFIDENCE
                # ============================================

                detection_confidence = float(
                    box.conf[0]
                )


                # ============================================
                # KEEP INSIDE IMAGE
                # ============================================

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
                # CROP LICENSE PLATE
                # ============================================

                crop = frame[
                    y1:y2,
                    x1:x2
                ]


                if crop.size == 0:

                    continue


                # ============================================
                # CREATE TRACK RECORD
                # ============================================

                if track_id not in plate_tracks:

                    plate_tracks[track_id] = {

                        "text": "READING...",

                        "confidence": 0.0,

                        "status": "LOW CONFIDENCE",

                        "last_ocr_frame": 0,

                        "best_confidence": 0.0
                    }


                # ============================================
                # OCR INTERVAL
                # ============================================

                should_run_ocr = (

                    frame_number
                    -
                    plate_tracks[track_id][
                        "last_ocr_frame"
                    ]

                    >= OCR_INTERVAL
                )


                # ============================================
                # RUN OCR
                # ============================================

                if should_run_ocr:

                    enlarged_crop = cv2.resize(
                        crop,
                        None,
                        fx=5,
                        fy=5,
                        interpolation=cv2.INTER_CUBIC
                    )


                    text, ocr_confidence = (
                        run_paddleocr(
                            enlarged_crop
                        )
                    )


                    # ----------------------------------------
                    # UPDATE OCR FRAME
                    # ----------------------------------------

                    plate_tracks[track_id][
                        "last_ocr_frame"
                    ] = frame_number


                    # ----------------------------------------
                    # ONLY ACCEPT REASONABLE OCR
                    # ----------------------------------------

                    if text:

                        # HIGH CONFIDENCE
                        if ocr_confidence >= 0.80:

                            status = (
                                "HIGH CONFIDENCE"
                            )

                        # MEDIUM CONFIDENCE
                        elif ocr_confidence >= 0.50:

                            status = (
                                "MEDIUM CONFIDENCE"
                            )

                        # LOW CONFIDENCE
                        else:

                            status = (
                                "LOW CONFIDENCE"
                            )


                        # ------------------------------------
                        # UPDATE ONLY IF BETTER RESULT
                        # ------------------------------------

                        old_confidence = (
                            plate_tracks[track_id][
                                "best_confidence"
                            ]
                        )


                        if (
                            ocr_confidence
                            >=
                            old_confidence
                        ):

                            plate_tracks[track_id][
                                "text"
                            ] = text


                            plate_tracks[track_id][
                                "confidence"
                            ] = ocr_confidence


                            plate_tracks[track_id][
                                "best_confidence"
                            ] = ocr_confidence


                            plate_tracks[track_id][
                                "status"
                            ] = status


                            # ----------------------------
                            # SAVE RESULT
                            # ----------------------------

                            detected_plates.append(
                                {
                                    "track_id": track_id,

                                    "frame": frame_number,

                                    "plate": text,

                                    "ocr_confidence": round(
                                        ocr_confidence * 100,
                                        2
                                    ),

                                    "detection_confidence": round(
                                        detection_confidence * 100,
                                        2
                                    ),

                                    "status": status
                                }
                            )


                # ============================================
                # GET CURRENT PLATE DATA
                # ============================================

                current_text = (
                    plate_tracks[track_id][
                        "text"
                    ]
                )


                current_confidence = (
                    plate_tracks[track_id][
                        "confidence"
                    ]
                )


                current_status = (
                    plate_tracks[track_id][
                        "status"
                    ]
                )


                # ============================================
                # DRAW PLATE BOX
                # ============================================

                cv2.rectangle(
                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    3
                )


                # ============================================
                # LABEL
                # ============================================

                label = (
                    f"ID {track_id} | "
                    f"{current_text} | "
                    f"{current_confidence * 100:.1f}%"
                )


                text_y = max(
                    30,
                    y1 - 10
                )


                # ============================================
                # LABEL BACKGROUND
                # ============================================

                cv2.rectangle(
                    frame,

                    (
                        x1,
                        text_y - 30
                    ),

                    (
                        min(
                            width,
                            x1 + 500
                        ),
                        text_y + 5
                    ),

                    (0, 0, 0),

                    -1
                )


                # ============================================
                # WRITE LABEL
                # ============================================

                cv2.putText(
                    frame,

                    label,

                    (
                        x1 + 5,
                        text_y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (0, 255, 0),

                    2
                )


        # ====================================================
        # WRITE OUTPUT FRAME
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

        # Keep the OpenCV-generated file as fallback.
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
        "VIDEO ANPR COMPLETED"
    )

    print(
        "Total frames:",
        frame_number
    )

    print(
        "Unique plate IDs:",
        len(plate_tracks)
    )

    print(
        "OCR results:",
        len(detected_plates)
    )

    print(
        "Output:",
        output_path
    )

    print(
        "========================================"
    )


    # ========================================================
    # SHOW FINAL PLATE RESULTS
    # ========================================================

    print()
    print(
        "FINAL PLATE RESULTS"
    )


    for track_id, data in plate_tracks.items():

        print(
            f"ID {track_id}: "
            f"{data['text']} | "
            f"{data['confidence'] * 100:.2f}% | "
            f"{data['status']}"
        )


    # ========================================================
    # RETURN
    # ========================================================

    return (
        output_path,
        detected_plates
    )
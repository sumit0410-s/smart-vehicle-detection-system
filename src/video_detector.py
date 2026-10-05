import cv2
import os
import subprocess
import torch
import imageio_ffmpeg

from ultralytics import YOLO


# =========================================================
# DEVICE
# =========================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"


print("==============================================")
print("SMART VEHICLE + HELMET VIDEO DETECTOR")
print("==============================================")

if torch.cuda.is_available():
    print("GPU ENABLED")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CPU MODE")


# =========================================================
# LOAD MODELS
# =========================================================

vehicle_model = YOLO("yolo11n.pt")
helmet_model = YOLO("models/best.pt")

vehicle_model.to(DEVICE)
helmet_model.to(DEVICE)


# =========================================================
# VEHICLE CLASSES
# =========================================================

vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# =========================================================
# HELMET CLASSES
# =========================================================

helmet_classes = {
    0: "With Helmet",
    1: "Without Helmet"
}


# =========================================================
# VIDEO DETECTION
# =========================================================

def detect_video(video_path):

    print()
    print("==============================================")
    print("STARTING VIDEO DETECTION")
    print("==============================================")

    # -----------------------------------------------------
    # CHECK INPUT VIDEO
    # -----------------------------------------------------

    if not os.path.exists(video_path):

        print("ERROR: Video file not found")
        print("Video path:", video_path)

        return None


    print("Video found:")
    print(video_path)


    # -----------------------------------------------------
    # OPEN VIDEO
    # -----------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("ERROR: Video cannot be opened")

        return None


    # -----------------------------------------------------
    # VIDEO INFORMATION
    # -----------------------------------------------------

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )


    if fps <= 0:
        fps = 25


    print("----------------------------------------------")
    print("FPS:", fps)
    print("Width:", width)
    print("Height:", height)
    print("Total Frames:", total_frames)
    print("----------------------------------------------")


    # -----------------------------------------------------
    # OUTPUT FOLDER
    # -----------------------------------------------------

    os.makedirs(
        "data/output",
        exist_ok=True
    )


    # -----------------------------------------------------
    # OUTPUT FILES
    # -----------------------------------------------------

    temp_output_path = (
        "data/output/detected_video_temp.mp4"
    )

    output_path = (
        "data/output/detected_video.mp4"
    )


    # -----------------------------------------------------
    # REMOVE OLD FILES
    # -----------------------------------------------------

    if os.path.exists(temp_output_path):

        try:
            os.remove(temp_output_path)

        except Exception:
            pass


    if os.path.exists(output_path):

        try:
            os.remove(output_path)

        except Exception:
            pass


    # -----------------------------------------------------
    # OPENCV VIDEO WRITER
    # -----------------------------------------------------

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

        print(
            "ERROR: Output video could not be created."
        )

        cap.release()

        return None


    # =====================================================
    # COUNTERS
    # =====================================================

    frame_number = 0

    total_vehicle_detections = 0

    total_helmet_detections = 0

    with_helmet_count = 0

    without_helmet_count = 0

    unique_vehicle_ids = set()


    # =====================================================
    # PROCESS VIDEO
    # =====================================================

    while True:

        success, frame = cap.read()


        if not success:

            break


        frame_number += 1


        # =================================================
        # VEHICLE DETECTION + TRACKING
        # =================================================

        try:

            vehicle_results = vehicle_model.track(
                source=frame,
                conf=0.35,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(
                    vehicle_classes.keys()
                ),
                imgsz=640,
                device=DEVICE,
                verbose=False
            )

        except Exception as e:

            print(
                "Vehicle detection error:",
                e
            )

            continue


        if not vehicle_results:

            out.write(frame)

            continue


        vehicle_result = vehicle_results[0]


        # =================================================
        # DRAW VEHICLES
        # =================================================

        if vehicle_result.boxes is not None:

            for box in vehicle_result.boxes:

                class_id = int(
                    box.cls[0]
                )


                if class_id not in vehicle_classes:

                    continue


                vehicle_name = vehicle_classes[
                    class_id
                ]


                confidence = float(
                    box.conf[0]
                )


                # -----------------------------------------
                # TRACK ID
                # -----------------------------------------

                if box.id is not None:

                    track_id = int(
                        box.id[0]
                    )

                else:

                    track_id = -1


                if track_id != -1:

                    unique_vehicle_ids.add(
                        track_id
                    )


                # -----------------------------------------
                # BOUNDING BOX
                # -----------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )


                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(width, x2)
                y2 = min(height, y2)


                # -----------------------------------------
                # VEHICLE BOX
                # -----------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                # -----------------------------------------
                # VEHICLE LABEL
                # -----------------------------------------

                label = (
                    f"ID:{track_id} "
                    f"{vehicle_name} "
                    f"{confidence * 100:.1f}%"
                )


                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(y1 - 10, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


                total_vehicle_detections += 1


        # =================================================
        # HELMET DETECTION
        # =================================================

        try:

            helmet_results = helmet_model.predict(
                source=frame,
                conf=0.25,
                imgsz=640,
                device=DEVICE,
                verbose=False
            )

        except Exception as e:

            print(
                "Helmet detection error:",
                e
            )

            helmet_results = []


        frame_with_helmet = 0

        frame_without_helmet = 0


        # =================================================
        # DRAW HELMET DETECTIONS
        # =================================================

        if helmet_results:

            helmet_result = helmet_results[0]


            if helmet_result.boxes is not None:

                for box in helmet_result.boxes:

                    class_id = int(
                        box.cls[0]
                    )


                    if class_id not in helmet_classes:

                        continue


                    confidence = float(
                        box.conf[0]
                    )


                    helmet_name = helmet_classes[
                        class_id
                    ]


                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )


                    # -------------------------------------
                    # HELMET STATUS
                    # -------------------------------------

                    if class_id == 0:

                        frame_with_helmet += 1

                        box_color = (
                            255,
                            0,
                            0
                        )

                    else:

                        frame_without_helmet += 1

                        box_color = (
                            0,
                            0,
                            255
                        )


                    # -------------------------------------
                    # HELMET BOX
                    # -------------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )


                    # -------------------------------------
                    # HELMET LABEL
                    # -------------------------------------

                    label = (
                        f"{helmet_name} "
                        f"{confidence * 100:.1f}%"
                    )


                    cv2.putText(
                        frame,
                        label,
                        (
                            x1,
                            max(y1 - 10, 20)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        box_color,
                        2
                    )


        # =================================================
        # UPDATE HELMET COUNTERS
        # =================================================

        total_helmet_detections += (
            frame_with_helmet
            + frame_without_helmet
        )


        with_helmet_count += (
            frame_with_helmet
        )


        without_helmet_count += (
            frame_without_helmet
        )


        # =================================================
        # DISPLAY COUNTERS
        # =================================================

        cv2.putText(
            frame,
            f"Vehicles: {len(unique_vehicle_ids)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"With Helmet: {with_helmet_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )


        cv2.putText(
            frame,
            f"Without Helmet: {without_helmet_count}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


        # =================================================
        # WRITE FRAME
        # =================================================

        out.write(frame)


        # =================================================
        # PROGRESS
        # =================================================

        if frame_number % 30 == 0:

            print(
                "Frames processed:",
                frame_number,
                "/",
                total_frames
            )


    # =====================================================
    # RELEASE
    # =====================================================

    cap.release()

    out.release()


    # =====================================================
    # CONVERT TO H.264 USING IMAGEIO-FFMPEG
    # =====================================================

    print()
    print(
        "Converting video to browser-compatible H.264..."
    )


    try:

        ffmpeg_exe = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )


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
            stderr=subprocess.PIPE,
            text=True
        )


        # -----------------------------------------------
        # REMOVE TEMP FILE
        # -----------------------------------------------

        if os.path.exists(
            temp_output_path
        ):

            os.remove(
                temp_output_path
            )


        print(
            "H.264 conversion completed."
        )

        print(
            "Browser-compatible video:",
            output_path
        )


    except Exception as e:

        print(
            "FFmpeg conversion error:",
            e
        )


        # -----------------------------------------------
        # FALLBACK
        # -----------------------------------------------

        if os.path.exists(
            temp_output_path
        ):

            if os.path.exists(
                output_path
            ):

                os.remove(
                    output_path
                )


            os.replace(
                temp_output_path,
                output_path
            )


        print(
            "Fallback video:",
            output_path
        )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    print()
    print(
        "=============================================="
    )

    print(
        "VIDEO DETECTION COMPLETED"
    )

    print(
        "=============================================="
    )

    print(
        "Total vehicle detections:",
        total_vehicle_detections
    )

    print(
        "Unique vehicles tracked:",
        len(unique_vehicle_ids)
    )

    print(
        "Total helmet detections:",
        total_helmet_detections
    )

    print(
        "With Helmet detections:",
        with_helmet_count
    )

    print(
        "Without Helmet detections:",
        without_helmet_count
    )

    print(
        "Output video:",
        output_path
    )

    print(
        "Output exists:",
        os.path.exists(output_path)
    )

    print(
        "=============================================="
    )


    # =====================================================
    # RETURN
    # =====================================================

    return (
        output_path,
        total_vehicle_detections,
        len(unique_vehicle_ids),
        with_helmet_count,
        without_helmet_count
    )
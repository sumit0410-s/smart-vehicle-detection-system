import os
import subprocess

import streamlit as st
import imageio_ffmpeg

from src.detector import detect_vehicle
from src.helmet_detector import detect_helmet
from src.video_detector import detect_video
from src.number_plate import detect_number_plate


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Vehicle Detection System",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚦 Smart Vehicle Detection System")

st.write(
    "Vehicle Detection | Helmet Detection | "
    "Number Plate Recognition | Video Analysis"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs("data/input", exist_ok=True)
os.makedirs("data/output", exist_ok=True)
os.makedirs("data/evidence", exist_ok=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ System")

st.sidebar.write("### Available Features")

st.sidebar.write("🚗 Vehicle Detection")
st.sidebar.write("🪖 Helmet Detection")
st.sidebar.write("🔢 Number Plate Recognition")
st.sidebar.write("🎥 Video Detection")
st.sidebar.write("📊 Statistics")


# ============================================================
# IMAGE DETECTION
# ============================================================

st.header("📷 Image Detection")

uploaded_image = st.file_uploader(
    "Upload vehicle image",
    type=["jpg", "jpeg", "png"],
    key="image_uploader"
)


if uploaded_image is not None:

    # --------------------------------------------------------
    # SAVE UPLOADED IMAGE
    # --------------------------------------------------------

    image_path = os.path.join(
        "data/input",
        uploaded_image.name
    )

    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    st.success("Image uploaded successfully!")


    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    st.subheader("Original Image")

    st.image(
        image_path,
        use_container_width=True
    )


    # ========================================================
    # VEHICLE DETECTION
    # ========================================================

    st.subheader("🚗 Vehicle Detection")

    if st.button(
        "Detect Vehicles",
        key="vehicle_detection_button"
    ):

        with st.spinner(
            "Detecting vehicles..."
        ):

            # detect_vehicle returns a DICTIONARY
            vehicle_result = detect_vehicle(
                image_path
            )

        # ----------------------------------------------------
        # OUTPUT IMAGE PATH
        # ----------------------------------------------------

        output_path = (
            "data/output/detected_image.jpg"
        )

        # ----------------------------------------------------
        # TOTAL VEHICLES
        # ----------------------------------------------------

        vehicle_count = vehicle_result.get(
            "Total",
            vehicle_result.get(
                "total",
                0
            )
        )

        st.success(
            "Vehicle detection completed!"
        )

        # ----------------------------------------------------
        # VEHICLE STATISTICS
        # ----------------------------------------------------

        st.subheader(
            "📊 Vehicle Statistics"
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Cars",
                vehicle_result.get(
                    "car",
                    0
                )
            )

        with col2:
            st.metric(
                "Motorcycles",
                vehicle_result.get(
                    "motorcycle",
                    0
                )
            )

        with col3:
            st.metric(
                "Buses",
                vehicle_result.get(
                    "bus",
                    0
                )
            )

        with col4:
            st.metric(
                "Trucks",
                vehicle_result.get(
                    "truck",
                    0
                )
            )

        with col5:
            st.metric(
                "Total Vehicles",
                vehicle_count
            )

        # ----------------------------------------------------
        # SHOW DETECTED IMAGE
        # ----------------------------------------------------

        if os.path.exists(output_path):

            st.image(
                output_path,
                caption="Vehicle Detection Result",
                use_container_width=True
            )

        else:

            st.warning(
                "Vehicle detection completed, "
                "but output image was not found."
            )


    # ========================================================
    # HELMET DETECTION
    # ========================================================

    st.subheader("🪖 Helmet Detection")

    if st.button(
        "Detect Helmet",
        key="helmet_detection_button"
    ):

        with st.spinner(
            "Detecting helmets..."
        ):

            (
                helmet_output,
                with_helmet,
                without_helmet
            ) = detect_helmet(
                image_path
            )

        st.success(
            "Helmet detection completed!"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "With Helmet",
                with_helmet
            )

        with col2:

            st.metric(
                "Without Helmet",
                without_helmet
            )

        if os.path.exists(
            helmet_output
        ):

            st.image(
                helmet_output,
                caption="Helmet Detection Result",
                use_container_width=True
            )


    # ========================================================
    # NUMBER PLATE RECOGNITION
    # ========================================================

    st.subheader(
        "🔢 Number Plate Recognition (ANPR)"
    )

    st.write(
        "License Plate Detection + PaddleOCR"
    )

    if st.button(
        "Detect Number Plate",
        key="number_plate_button"
    ):

        with st.spinner(
            "Detecting license plates and "
            "reading numbers..."
        ):

            (
                plate_output,
                detected_plates
            ) = detect_number_plate(
                image_path
            )

        st.success(
            "Number plate detection completed!"
        )

        # ----------------------------------------------------
        # OUTPUT IMAGE
        # ----------------------------------------------------

        if os.path.exists(
            plate_output
        ):

            st.image(
                plate_output,
                caption="Number Plate Detection Result",
                use_container_width=True
            )


        # ----------------------------------------------------
        # DETECTED PLATES
        # ----------------------------------------------------

        if detected_plates:

            st.subheader(
                "📋 Detected Plates"
            )

            for index, plate in enumerate(
                detected_plates,
                start=1
            ):

                st.markdown(
                    f"### Plate {index}: "
                    f"`{plate['plate']}`"
                )

                col1, col2, col3 = st.columns(3)

                # --------------------------------------------
                # OCR CONFIDENCE
                # --------------------------------------------

                with col1:

                    st.metric(
                        "OCR Confidence",
                        f"{plate['ocr_confidence']:.2f}%"
                    )

                # --------------------------------------------
                # DETECTION CONFIDENCE
                # --------------------------------------------

                with col2:

                    st.metric(
                        "Detection Confidence",
                        f"{plate['detection_confidence']:.2f}%"
                    )

                # --------------------------------------------
                # STATUS
                # --------------------------------------------

                with col3:

                    st.write("**Status**")

                    if (
                        plate["status"]
                        == "HIGH CONFIDENCE"
                    ):

                        st.success(
                            plate["status"]
                        )

                    elif (
                        plate["status"]
                        == "MEDIUM CONFIDENCE"
                    ):

                        st.warning(
                            plate["status"]
                        )

                    else:

                        st.error(
                            plate["status"]
                        )

                st.divider()

        else:

            st.warning(
                "No license plate detected."
            )


# ============================================================
# VIDEO DETECTION
# ============================================================

st.header("🎥 Video Detection")

uploaded_video = st.file_uploader(
    "Upload vehicle video",
    type=["mp4", "avi", "mov"],
    key="video_uploader"
)


if uploaded_video is not None:

    # --------------------------------------------------------
    # SAVE VIDEO
    # --------------------------------------------------------

    video_path = os.path.join(
        "data/input",
        uploaded_video.name
    )

    with open(video_path, "wb") as f:
        f.write(
            uploaded_video.getbuffer()
        )

    st.success(
        "Video uploaded successfully!"
    )


    # --------------------------------------------------------
    # ORIGINAL VIDEO
    # --------------------------------------------------------

    st.subheader(
        "Original Video"
    )

    st.video(
        video_path
    )


    # ========================================================
    # VIDEO PROCESSING
    # ========================================================

    if st.button(
        "Detect Vehicles in Video",
        key="video_detection_button"
    ):

        with st.spinner(
            "Processing video... "
            "This may take some time on CPU."
        ):

            (
                detection_output_path,
                total_vehicle_detections,
                unique_vehicle_count,
                with_helmet_count,
                without_helmet_count
            ) = detect_video(
                video_path
            )

        st.success(
            "Video processing completed!"
        )


        # ====================================================
        # VIDEO STATISTICS
        # ====================================================

        st.subheader(
            "📊 Video Statistics"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Unique Vehicles",
                unique_vehicle_count
            )

        with col2:

            st.metric(
                "Vehicle Detections",
                total_vehicle_detections
            )

        with col3:

            st.metric(
                "With Helmet",
                with_helmet_count
            )

        with col4:

            st.metric(
                "Without Helmet",
                without_helmet_count
            )


        # ====================================================
        # CONVERT VIDEO FOR BROWSER
        # ====================================================

        browser_video_path = (
            "data/output/browser_video.mp4"
        )

        try:

            ffmpeg_exe = (
                imageio_ffmpeg.get_ffmpeg_exe()
            )

            command = [
                ffmpeg_exe,
                "-y",
                "-i",
                detection_output_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                browser_video_path
            ]

            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            st.subheader(
                "▶️ Processed Video"
            )

            st.video(
                browser_video_path
            )

        except Exception as e:

            st.warning(
                "Browser video conversion failed."
            )

            st.write(e)

            st.subheader(
                "Processed Video File"
            )

            st.write(
                detection_output_path
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Vehicle Detection System | "
    "YOLO + PaddleOCR + OpenCV + Streamlit"
)
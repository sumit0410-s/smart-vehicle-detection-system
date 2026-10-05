
import os
import streamlit as st

from src.detector import detect_vehicle
from src.helmet_detector import detect_helmet
from src.video_detector import detect_video
from src.number_plate import detect_number_plate
from src.video_number_plate import detect_video_number_plate
from src.vehicle_plate_anpr import detect_vehicle_plate_anpr
from src.violation_detector import detect_violations
from src.challan_generator import generate_challans
from src.database import save_challan, get_all_challans


st.set_page_config(
    page_title="Smart Vehicle Detection System",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Smart Vehicle Detection System")
st.caption("YOLO + PaddleOCR + OpenCV + MongoDB + Streamlit")

st.sidebar.markdown("""
<style>

.sidebar-title {
    font-size: 28px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.sidebar-subtitle {
    text-align: center;
    font-size: 13px;
    color: #888;
    margin-bottom: 20px;
}

.nav-link {
    display: block;
    padding: 11px 14px;
    margin: 7px 0;
    border-radius: 10px;
    background: #f5f7fa;
    color: #222 !important;
    text-decoration: none !important;
    font-size: 15px;
    font-weight: 600;
    border: 1px solid #e5e7eb;
    transition: all 0.2s ease;
}

.nav-link:hover {
    background: #e8f0fe;
    border-color: #4285f4;
    transform: translateX(4px);
}

.nav-section {
    font-size: 12px;
    font-weight: 700;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 15px 0 8px 5px;
}

</style>

<div class="sidebar-title">
    🚦 Smart Vehicle
</div>

<div class="sidebar-subtitle">
    Detection & Traffic Management
</div>

<div class="nav-section">
    DETECTION
</div>

<a class="nav-link" href="#vehicle-detection">
    🚗 Vehicle Detection
</a>

<a class="nav-link" href="#helmet-detection">
    🪖 Helmet Detection
</a>

<a class="nav-link" href="#number-plate-recognition">
    🔢 Number Plate Recognition
</a>

<a class="nav-link" href="#vehicle-helmet-video-detection">
    🎥 Vehicle + Helmet Video
</a>

<a class="nav-link" href="#vehicle-wise-video-anpr">
    🚘 Vehicle-wise ANPR
</a>

<div class="nav-section">
    TRAFFIC VIOLATION
</div>

<a class="nav-link" href="#automatic-violation-challan">
    🚨 Automatic Violation
</a>

<a class="nav-link" href="#challan-dashboard">
    🧾 Challan Management
</a>
""", unsafe_allow_html=True)

def save_upload(file, filename):
    os.makedirs("data/input", exist_ok=True)
    path = os.path.join("data", "input", filename)
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    return path


def show_image(path, caption):
    if path and os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)


def show_video(path):
    if path and os.path.exists(path):
        st.video(path)


# ============================================================
# 1. VEHICLE DETECTION
# ============================================================

st.header("🚗 Vehicle Detection")

file = st.file_uploader(
    "Upload vehicle image",
    type=["jpg", "jpeg", "png"],
    key="vehicle_upload"
)

if file:
    path = save_upload(file, "vehicle_image.jpg")

    if st.button("🚗 Detect Vehicles", key="vehicle_button"):
        try:
            result = detect_vehicle(path)

            if isinstance(result, dict):
                c1, c2, c3, c4, c5 = st.columns(5)

                c1.metric("Cars", result.get("car", result.get("Car", 0)))
                c2.metric(
                    "Motorcycles",
                    result.get("motorcycle", result.get("Motorcycle", 0))
                )
                c3.metric("Buses", result.get("bus", result.get("Bus", 0)))
                c4.metric("Trucks", result.get("truck", result.get("Truck", 0)))
                c5.metric(
                    "Total",
                    result.get("Total", result.get("total", 0))
                )

            show_image(
                "data/output/detected_image.jpg",
                "Detected Vehicles"
            )

        except Exception as e:
            st.error(f"Vehicle detection error: {e}")


# ============================================================
# 2. HELMET DETECTION
# ============================================================

st.divider()
st.header("🪖 Helmet Detection")

file = st.file_uploader(
    "Upload helmet image",
    type=["jpg", "jpeg", "png"],
    key="helmet_upload"
)

if file:
    path = save_upload(file, "helmet_image.jpg")

    if st.button("🪖 Detect Helmet", key="helmet_button"):
        try:
            result = detect_helmet(path)

            output = result
            with_helmet = 0
            without_helmet = 0

            if isinstance(result, tuple):
                output = result[0]
                if len(result) > 1:
                    with_helmet = result[1]
                if len(result) > 2:
                    without_helmet = result[2]

            c1, c2 = st.columns(2)
            c1.metric("With Helmet", with_helmet)
            c2.metric("Without Helmet", without_helmet)

            show_image(output, "Helmet Detection Result")

        except Exception as e:
            st.error(f"Helmet detection error: {e}")


# ============================================================
# 3. NUMBER PLATE RECOGNITION
# ============================================================

st.divider()
st.header("🔢 Number Plate Recognition")

file = st.file_uploader(
    "Upload vehicle / number plate image",
    type=["jpg", "jpeg", "png"],
    key="plate_upload"
)

if file:
    path = save_upload(file, "plate_image.jpg")

    if st.button("🔢 Read Number Plate", key="plate_button"):
        try:
            result = detect_number_plate(path)

            output = result
            plates = []

            if isinstance(result, tuple):
                output = result[0]
                if len(result) > 1:
                    plates = result[1]

            if plates:
                for i, plate in enumerate(plates, 1):
                    st.write(f"**Plate {i}:** {plate}")
            else:
                st.info("No number plate text detected.")

            show_image(output, "Number Plate Recognition Result")

        except Exception as e:
            st.error(f"Number plate error: {e}")


# ============================================================
# 4. VEHICLE + HELMET VIDEO
# ============================================================

st.divider()
st.header("🎥 Vehicle + Helmet Video Detection")

file = st.file_uploader(
    "Upload video",
    type=["mp4", "avi", "mov"],
    key="vehicle_helmet_video_upload"
)

if file:
    path = save_upload(file, "vehicle_helmet_video.mp4")

    if st.button(
        "🎥 Detect Vehicles + Helmet",
        key="vehicle_helmet_video_button"
    ):
        try:
            result = detect_video(path)
            output = result[0] if isinstance(result, tuple) else result
            show_video(output)
        except Exception as e:
            st.error(f"Video detection error: {e}")


# ============================================================
# 5. VIDEO NUMBER PLATE
# ============================================================

st.divider()
st.header("🔢 Video Number Plate Recognition")

file = st.file_uploader(
    "Upload video for number plate recognition",
    type=["mp4", "avi", "mov"],
    key="video_plate_upload"
)

if file:
    path = save_upload(file, "video_number_plate.mp4")

    if st.button(
        "🔢 Detect Plates in Video",
        key="video_plate_button"
    ):
        try:
            result = detect_video_number_plate(path)

            output = result
            plates = []

            if isinstance(result, tuple):
                output = result[0]
                if len(result) > 1:
                    plates = result[1]
            show_video(output)

        except Exception as e:
            st.error(f"Video number plate error: {e}")


# ============================================================
# 6. VEHICLE-WISE ANPR
# ============================================================

st.divider()
st.header("🚘 Vehicle-wise Video ANPR")

file = st.file_uploader(
    "Upload video for vehicle-wise ANPR",
    type=["mp4", "avi", "mov"],
    key="anpr_upload"
)

if file:
    path = save_upload(file, "vehicle_anpr_video.mp4")

    if st.button(
        "🚘 Run Vehicle-wise ANPR",
        key="anpr_button"
    ):
        try:
            output, vehicles = detect_vehicle_plate_anpr(path)

            show_video(output)

        except Exception as e:
            st.error(f"Vehicle-wise ANPR error: {e}")


# ============================================================
# 7. AUTOMATIC VIOLATION + CHALLAN
# ============================================================

st.divider()
st.header("🚨 Automatic Violation & Challan")

file = st.file_uploader(
    "Upload video for automatic violation detection",
    type=["mp4", "avi", "mov"],
    key="violation_upload"
)

if file:
    path = save_upload(file, "violation_video.mp4")

    if st.button(
        "🚨 Detect Violations & Generate Challans",
        key="violation_button"
    ):

        with st.spinner(
            "Detecting vehicles, helmets and number plates..."
        ):

            try:
                vehicles = detect_violations(path)
                output_video = "data/output/violation_detection.mp4"


                if os.path.exists(output_video):
                    st.subheader("🎥 Violation Detection Output Video")
                    st.video(output_video)
                

                total = len(vehicles)

                motorcycles = sum(
                    1
                    for v in vehicles
                    if v.get("vehicle_type") == "Motorcycle"
                )

                helmet_violations = sum(
                    1
                    for v in vehicles
                    if v.get("helmet_violation", False)
                )

                plates = sum(
                    1
                    for v in vehicles
                    if v.get("plate", "NOT READ")
                    not in ["NOT READ", "", None]
                )

                evidence = sum(
                    1
                    for v in vehicles
                    if v.get("evidence_saved", False)
                )

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Vehicles", total)
                c2.metric("Motorcycles", motorcycles)
                c3.metric("Helmet Violations", helmet_violations)
                c4.metric("Plates Read", plates)
                c5.metric("Evidence Saved", evidence)

                challans = generate_challans(vehicles)

                if not challans:
                    st.info(
                        "No violations detected in this video."
                    )
                else:
                    st.subheader(
                        "🧾 Generated Draft Challans"
                    )

                    saved = 0

                    for challan in challans:

                        try:
                            save_challan(challan)
                            saved += 1
                        except Exception as e:
                            st.error(
                                f"MongoDB save error: {e}"
                            )

                        with st.expander(
                            challan.get(
                                "challan_id",
                                "DRAFT CHALLAN"
                            )
                        ):
                            st.write(
                                "**Status:**",
                                challan.get("status", "DRAFT")
                            )
                            st.write(
                                "**Vehicle ID:**",
                                challan.get("vehicle_id", "N/A")
                            )
                            st.write(
                                "**Vehicle Type:**",
                                challan.get(
                                    "vehicle_type",
                                    "Unknown"
                                )
                            )
                            st.write(
                                "**Number Plate:**",
                                challan.get(
                                    "number_plate",
                                    "NOT READ"
                                )
                            )
                            st.write(
                                "**Helmet Status:**",
                                challan.get(
                                    "helmet_status",
                                    "NOT CHECKED"
                                )
                            )
                            st.write(
                                "**Violations:**",
                                ", ".join(
                                    challan.get(
                                        "violations",
                                        []
                                    )
                                )
                            )

                            evidence_data = challan.get("evidence")

                            if evidence_data:
                                image = evidence_data.get("image")

                                if image and os.path.exists(image):
                                    st.image(
                                        image,
                                        caption="Violation Evidence",
                                        use_container_width=True
                                    )

                    st.success(
                        f"{saved} draft challan(s) saved to MongoDB."
                    )

            except Exception as e:
                st.error(
                    f"Automatic violation/challan error: {e}"
                )


# ============================================================
# 8. CHALLAN DASHBOARD
# ============================================================

st.divider()
st.header("🧾 Challan Dashboard")

if st.button(
    "🔄 Load Challan Records",
    key="load_challans"
):

    try:
        challans = get_all_challans()

        if not challans:
            st.info(
                "MongoDB me abhi koi challan record nahi hai."
            )
        else:
            st.success(
                f"{len(challans)} challan record(s) found."
            )

            for challan in challans:

                challan_id = challan.get(
                    "challan_id",
                    "N/A"
                )

                with st.expander(
                    f"🧾 {challan_id}"
                ):

                    c1, c2 = st.columns(2)

                    with c1:
                        st.write(
                            "**Status:**",
                            challan.get(
                                "status",
                                "N/A"
                            )
                        )
                        st.write(
                            "**Vehicle ID:**",
                            challan.get(
                                "vehicle_id",
                                "N/A"
                            )
                        )
                        st.write(
                            "**Vehicle Type:**",
                            challan.get(
                                "vehicle_type",
                                "N/A"
                            )
                        )
                        st.write(
                            "**Number Plate:**",
                            challan.get(
                                "number_plate",
                                "NOT READ"
                            )
                        )

                    with c2:
                        st.write(
                            "**Helmet Status:**",
                            challan.get(
                                "helmet_status",
                                "N/A"
                            )
                        )
                        st.write(
                            "**Plate Status:**",
                            challan.get(
                                "plate_status",
                                "N/A"
                            )
                        )
                        st.write(
                            "**OCR Confidence:**",
                            f"{challan.get('ocr_confidence', 0.0) * 100:.2f}%"
                        )
                        st.write(
                            "**Created At:**",
                            challan.get(
                                "created_at",
                                "N/A"
                            )
                        )

                    st.subheader("🚨 Violations")

                    violations = challan.get(
                        "violations",
                        []
                    )

                    if violations:
                        for violation in violations:
                            st.error(
                                f"⚠️ {violation}"
                            )
                    else:
                        st.info(
                            "No violation information."
                        )

                    st.subheader(
                        "📸 Violation Evidence"
                    )

                    evidence_data = challan.get(
                        "evidence"
                    )

                    if evidence_data:
                        image = evidence_data.get(
                            "image"
                        )

                        if image and os.path.exists(image):
                            st.image(
                                image,
                                caption="Violation Evidence",
                                use_container_width=True
                            )
                        else:
                            st.warning(
                                "Evidence image file nahi mili."
                            )
                    else:
                        st.info(
                            "Is challan ke saath evidence available nahi hai."
                        )

    except Exception as e:
        st.error(
            f"Challan dashboard error: {e}"
        )


st.divider()

st.caption(
    "Smart Vehicle Detection System | "
    "YOLO + PaddleOCR + OpenCV + MongoDB + Streamlit"
)

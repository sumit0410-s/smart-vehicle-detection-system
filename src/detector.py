import cv2
from pathlib import Path
from ultralytics import YOLO
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "yolov8s.pt"
OUTPUT_DIR = BASE_DIR / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model = YOLO(str(MODEL_PATH))


def detect_vehicle(image_path):

    image_path = str(image_path)
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image could not be read.")
        return {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
            "Total": 0
        }
    vehicle_classes = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }
    results = model.predict(
        source=image,
        conf=0.15,
        iou=0.45,
        imgsz=1280,
        classes=[2, 3, 5, 7],
        verbose=False
    )

    result = results[0]
    counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())

            if class_id in vehicle_classes:

                vehicle_name = vehicle_classes[class_id]

                counts[vehicle_name] += 1

    total_vehicles = sum(counts.values())

    counts["Total"] = total_vehicles
    counts["total"] = total_vehicles

    # Draw detection boxes
    detected_image = result.plot(
        line_width=4,
        font_size=13,
        labels=True,
        conf=True

    )
    output_path = OUTPUT_DIR / "detected_image.jpg"

    saved = cv2.imwrite(
        str(output_path),
        detected_image
    )
    print("\nVehicle Detection Results")
    print("-------------------------")
    print("Cars:", counts["car"])
    print("Motorcycles:", counts["motorcycle"])
    print("Buses:", counts["bus"])
    print("Trucks:", counts["truck"])
    print("Total Vehicles:", counts["Total"])

    if saved:
        print("Detection image saved:", output_path)
    else:
        print("Error: Could not save detection image.")

    return counts
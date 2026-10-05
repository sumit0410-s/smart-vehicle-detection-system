import os
import cv2
import json
from datetime import datetime


# ============================================================
# EVIDENCE FOLDER
# ============================================================

EVIDENCE_FOLDER = "data/evidence"


# ============================================================
# CREATE EVIDENCE FOLDER
# ============================================================

os.makedirs(
    EVIDENCE_FOLDER,
    exist_ok=True
)


# ============================================================
# SAVE EVIDENCE IMAGE
# ============================================================

def save_evidence_image(
    frame,
    vehicle_id,
    vehicle_type,
    plate,
    violations
):
    """
    Save violation frame as evidence image.
    """

    if frame is None:
        return None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"vehicle_{vehicle_id}_"
        f"{timestamp}.jpg"
    )

    file_path = os.path.join(
        EVIDENCE_FOLDER,
        filename
    )

    success = cv2.imwrite(
        file_path,
        frame
    )

    if not success:
        print(
            "ERROR: Evidence image save nahi hui."
        )
        return None

    print(
        "Evidence saved:",
        file_path
    )

    return file_path


# ============================================================
# SAVE EVIDENCE INFORMATION
# ============================================================

def save_evidence_info(
    vehicle_id,
    vehicle_type,
    plate,
    violations,
    evidence_image
):
    """
    Save evidence details in JSON file.
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"vehicle_{vehicle_id}_"
        f"{timestamp}.json"
    )

    file_path = os.path.join(
        EVIDENCE_FOLDER,
        filename
    )

    evidence_data = {

        "vehicle_id": vehicle_id,

        "vehicle_type": vehicle_type,

        "number_plate": plate,

        "violations": violations,

        "evidence_image": evidence_image,

        "created_at":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "status": "DRAFT EVIDENCE"

    }

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evidence_data,
            file,
            indent=4
        )

    print(
        "Evidence information saved:",
        file_path
    )

    return file_path


# ============================================================
# SAVE COMPLETE EVIDENCE
# ============================================================

def save_complete_evidence(
    frame,
    vehicle_id,
    vehicle_type,
    plate,
    violations
):
    """
    Save both:
    1. Evidence image
    2. Evidence JSON
    """

    image_path = save_evidence_image(
        frame=frame,
        vehicle_id=vehicle_id,
        vehicle_type=vehicle_type,
        plate=plate,
        violations=violations
    )

    if image_path is None:
        return None

    json_path = save_evidence_info(
        vehicle_id=vehicle_id,
        vehicle_type=vehicle_type,
        plate=plate,
        violations=violations,
        evidence_image=image_path
    )

    return {
        "image": image_path,
        "json": json_path
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Evidence Manager Loaded Successfully"
    )

    print(
        "Evidence folder:",
        EVIDENCE_FOLDER
    )
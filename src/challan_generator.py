from datetime import datetime


def generate_challan_record(vehicle):

    violations = []

    # Helmet violation
    if vehicle.get("helmet_violation", False):
        violations.append("Riding Without Helmet")

    # Triple riding
    if vehicle.get("triple_riding", False):
        violations.append("Triple Riding")

    # Wrong side
    if vehicle.get("wrong_side", False):
        violations.append("Wrong Side Driving")

    # Red light
    if vehicle.get("red_light_violation", False):
        violations.append("Red Light Violation")

    # No violation -> no challan
    if not violations:
        return None

    challan_id = (
        "DRAFT-"
        + datetime.now().strftime("%Y%m%d%H%M%S")
        + "-"
        + str(vehicle.get("vehicle_id", "NA"))
    )

    # IMPORTANT:
    # Preserve the evidence object already created by
    # violation_detector.py instead of resetting it to None.
    evidence = vehicle.get("evidence")

    challan = {
        "challan_id": challan_id,
        "status": "DRAFT",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "vehicle_id": vehicle.get("vehicle_id", "NA"),
        "vehicle_type": vehicle.get("vehicle_type", "Unknown"),

        "number_plate": vehicle.get(
            "plate",
            "NOT READ"
        ),

        "helmet_status": vehicle.get(
            "helmet_status",
            "NOT CHECKED"
        ),

        "plate_status": vehicle.get(
            "plate_status",
            "NOT READ"
        ),

        "ocr_confidence": vehicle.get(
            "ocr_confidence",
            0.0
        ),

        "violations": violations,

        # Keep evidence image + JSON path in the challan.
        "evidence": evidence
    }

    return challan


def generate_challans(vehicles):

    challans = []

    for vehicle in vehicles:

        challan = generate_challan_record(
            vehicle
        )

        if challan is not None:
            challans.append(challan)

    return challans

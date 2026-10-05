from pymongo import MongoClient


# MongoDB connection
MONGO_URI = "mongodb://localhost:27017/"

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=3000
)

# Database
db = client["smart_vehicle_detection"]

# Challan collection
challan_collection = db["challans"]


def test_connection():

    try:
        client.admin.command("ping")
        return True

    except Exception:
        return False


def save_challan(challan):

    if not challan:
        return None

    result = challan_collection.insert_one(challan)

    return str(result.inserted_id)


def get_all_challans():

    challans = list(
        challan_collection.find(
            {},
            {"_id": 0}
        )
    )

    return challans
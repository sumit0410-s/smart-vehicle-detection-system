# 🚗 Smart Vehicle Detection System

An AI-powered Smart Vehicle Detection System developed using Python, YOLO, OpenCV, Streamlit, PaddleOCR and MongoDB.

The system can detect vehicles, identify helmet violations, recognize vehicle number plates and generate draft challan records with supporting evidence.

## ✨ Features

- 🚗 Vehicle Detection
- 🔢 Vehicle Counting
- 🏍️ Motorcycle Detection
- 🪖 Helmet Detection
- 🚫 Helmet Violation Detection
- 📷 Number Plate Detection
- 🔍 Automatic Number Plate Recognition (ANPR)
- 📝 OCR-based Number Plate Reading
- 🎥 Video Vehicle Detection
- 🚘 Vehicle-wise Video ANPR
- 📸 Violation Evidence Capture
- 🧾 Draft Challan Generation
- 🗄️ MongoDB Challan Storage
- 📊 Challan Dashboard
- ⚡ GPU-accelerated YOLO Detection
- 🌐 Streamlit Web Interface

## 🛠️ Technologies Used

- Python
- YOLO / Ultralytics
- OpenCV
- Streamlit
- PaddleOCR
- PyMongo
- MongoDB
- Pandas
- Plotly
- EasyOCR
- Tesseract OCR
- FFmpeg

## 📁 Project Structure

```text
SmartVehicleDetection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── best.pt
│   └── license_plate.pt
│
├── src/
│   ├── detector.py
│   ├── helmet_detector.py
│   ├── video_detector.py
│   ├── video_number_plate.py
│   ├── vehicle_plate_anpr.py
│   ├── violation_detector.py
│   ├── challan_generator.py
│   ├── database.py
│   └── evidence_manager.py
│
└── data/
    ├── input/
    ├── output/
    └── evidence/
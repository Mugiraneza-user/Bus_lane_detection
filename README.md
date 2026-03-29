# Bus Lane Enforcement System
A computer vision system that automatically detects vehicles illegally using bus lanes using YOLO (You Only Look Once) object detection and OpenCV lane detection.

Overview
This system monitors traffic in real-time to identify vehicles that are illegally driving in bus-only lanes. It combines:

YOLOv8 for vehicle detection (cars, buses, trucks, motorcycles)

OpenCV for lane detection and image processing

Custom logic to determine if a vehicle is in the restricted bus lane

When a violation is detected, the system:

Highlights the violating vehicle in red

Displays an on-screen alert

Saves an image as evidence with timestamp

Maintains a violation count
# Features
✅ Real-time vehicle detection using YOLOv8

✅ Automatic lane detection with OpenCV

✅ Bus lane identification (configurable)

✅ Violation detection (non-bus vehicles in bus lane)

✅ Visual alerts with bounding boxes

✅ Evidence saving (auto-saves violation images)

✅ Real-time webcam monitoring

✅ Video file processing

✅ Violation reports with statistics

✅ Works on Windows 10/11

nstallation Guide
Step 1: Install Python
Download Python 3.10 from python.org

IMPORTANT: Check ✅ "Add Python to PATH" during installation

Verify installation:

bash
python --version
# Should show: Python 3.10.x
Step 2: Create Project Directory
bash
# Create project folder
mkdir C:\Users\educa\bus_lane_system
cd C:\Users\educa\bus_lane_system

Step 3: Create Virtual Environment (Recommended)
bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# You should see (venv) in your prompt
Step 4: Install Required Packages
bash
# Install all required packages
pip install ultralytics opencv-python numpy matplotlib pillow ipykernel jupyter
Package versions that work well together:

bash
# If you want specific versions, use:
pip install ultralytics==8.0.200
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install matplotlib==3.7.2
pip install pillow==10.0.0
Step 5: Verify Installation
python
# Create test.py file with this content
import cv2
import numpy as np
from ultralytics import YOLO

print("✅ OpenCV:", cv2.__version__)
print("✅ NumPy:", np.__version__)
print("✅ YOLO:", YOLO.__version__)

# Test YOLO model download
model = YOLO('yolov8n.pt')
print("✅ YOLO model loaded successfully!")
Run the test:

bash
python test.py
Step 6: Launch Jupyter Notebook
bash
# From your activated venv
jupyter notebook
Your browser will open with Jupyter interface.

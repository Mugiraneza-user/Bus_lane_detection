from ultralytics import YOLO
import cv2
import numpy as np

# Vehicle class IDs from YOLO (COCO)
VEHICLE_CLASSES = [1, 2, 3, 5, 7]

# Labels
LABELS = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Define lanes 
lane_polygons = {
    "Bus Lane": np.array([
        [400, 180],  # top-left
        [560, 180],  # top-right
        [640, 720],  # bottom-right
        [320, 720]   # bottom-left
    ], np.int32).reshape((-1, 1, 2))
}

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to match polygon (if needed)
    frame = cv2.resize(frame, (640, 720))
    # overlay = frame.copy()

    # Draw lane polygons
    for lane_name, polygon in lane_polygons.items():
        
        cv2.polylines(frame, [polygon], True, (0, 255, 255), 2)
    # frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # YOLO expects RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)

    # Loop through detections
    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls in VEHICLE_CLASSES:
            vehicle_type = LABELS.get(cls, "Unknown")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Check if inside any lane
            inside_lane = False
            for lane_name, polygon in lane_polygons.items():
                 points = [
                        (x1, y1),  # top-left
                        (x2, y1),  # top-right
                        (x1, y2),  # bottom-left
                        (x2, y2)   # bottom-right
                ]
            for point in points:
               if cv2.pointPolygonTest(polygon, point, False) >= 0:
                  inside_lane = True
                  break

               if inside_lane:
                  break

            # Determine color & label
            if inside_lane  and cls != 5:
                color = (0, 0, 255)  # Red = violation
                label = f"{vehicle_type} (VIOLATION)"
            else:
                color = (0, 255, 0)
                label = vehicle_type

            # Draw rectangle, label, and center
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    cv2.imshow("Lane + Vehicle Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
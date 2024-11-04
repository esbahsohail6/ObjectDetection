import cv2
import pandas as pd
import numpy as np
import time
from ultralytics import YOLO
import cvzone
import Tracker as T  # Import your custom tracker module

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize video capture
cap = cv2.VideoCapture(r"C:\Users\Hashi\speed\video4.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read class names from coco.txt file
with open(r"C:\Users\Hashi\speed\coco.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

# Initialize trackers
tracker = T.Tracker()
tracker1 = T.Tracker()
tracker2 = T.Tracker()

# Variables for speed estimation
previous_positions = {}
car_speeds = {}
scale_meters_per_pixel = 0.08  # Adjusted to better reflect real-world distances
min_movement_threshold = 1.17  # Movement threshold in pixels for stationary detection
min_speed_kmh = 0  # Set minimum speed to 0 for stationary cars

# Function to calculate speed with improved smoothing and zero handling
def calculate_speed(pos1, pos2, time_elapsed, previous_speed=0, alpha=0.2, scale=1):
    if time_elapsed > 0:
        distance_in_pixels = np.linalg.norm(np.array(pos2) - np.array(pos1))

        # Detect if the vehicle is stationary
        if distance_in_pixels < min_movement_threshold:
            return 0  # Set speed to 0 for stationary vehicle

        distance_in_meters = distance_in_pixels * scale
        speed_mps = distance_in_meters / time_elapsed
        speed_kmh = speed_mps * 3.6 * 2  # Adjust the speed by a factor if needed

        # Smooth speed using EMA
        smoothed_speed = (alpha * speed_kmh) + ((1 - alpha) * previous_speed)
        return max(smoothed_speed, min_speed_kmh)  # Avoid negative speeds
    else:
        return previous_speed

# Function to check if the camera is stationary
def is_camera_stationary(frame1, frame2, threshold=2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    movements = [m.distance for m in matches]

    return np.mean(movements) < threshold  # Camera is stationary if movements are below threshold

# Main loop with updated speed calculation
try:
    previous_frame = None
    is_dynamic_camera = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame or end of video")
            break

        frame = cv2.resize(frame, (640, 360))

        # Check if the camera is stationary
        if previous_frame is not None:
            is_dynamic_camera = not is_camera_stationary(previous_frame, frame)

        # Run object detection
        results = model.predict(frame, conf=0.5, iou=0.4)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        cars, buses, trucks = [], [], []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])

            if 'car' in class_list[d]:
                cars.append([x1, y1, x2, y2])
            elif 'bus' in class_list[d]:
                buses.append([x1, y1, x2, y2])
            elif 'truck' in class_list[d]:
                trucks.append([x1, y1, x2, y2])

        # Update trackers
        bbox_idx_cars = tracker.update(cars)
        bbox_idx_buses = tracker1.update(buses)
        bbox_idx_trucks = tracker2.update(trucks)

        # Track position and calculate speed for cars
        current_time = time.time()
        for bbox in bbox_idx_cars:
            x3, y3, x4, y4, id1 = bbox
            cx3 = int((x3 + x4) / 2)
            cy3 = int(y4)
            center = (cx3, cy3)

            if id1 not in previous_positions:
                previous_positions[id1] = (center, current_time)
            else:
                prev_center, prev_time = previous_positions[id1]
                time_diff = current_time - prev_time

                if time_diff > 0:
                    previous_speed = car_speeds.get(id1, 0)
                    speed = calculate_speed(prev_center, center, time_diff, previous_speed,
                                            scale=scale_meters_per_pixel)
                    car_speeds[id1] = speed

                previous_positions[id1] = (center, current_time)

            # Display speed for each car
            speed_text = f"Speed: {car_speeds.get(id1, 0):.2f} km/h"
            cvzone.putTextRect(frame, speed_text, (x3, y4 + 15), scale=0.8, thickness=2, offset=3,
                               colorR=(0, 128, 0), colorT=(0, 0, 0))

            # Draw bounding box
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 128, 0), 4)
            cvzone.putTextRect(frame, f'Car {id1}', (x3, y3 - 10), scale=1.0, thickness=1)

        # Display the frame
        cv2.imshow("Vehicle Detection and Speed Estimation", frame)

        previous_frame = frame  # Store current frame for next iteration
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
            break

except Exception as e:
    print(f"Error: {str(e)}")

finally:
    cap.release()
    cv2.destroyAllWindows()
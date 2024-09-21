import cv2
from time import time
from ultralytics import YOLO
import random
import numpy as np

class ObjectDetection:
    def __init__(self):
        self.model = self.load_model()
        self.output_video = "output/output.avi"
        self.tracker = {}
    
    def load_model(self):
        model = YOLO("yolov8m.pt")  # Load a pretrained YOLOv8 model
        model.fuse()
        return model
    
    def detect_colors(self, class_list):
        detection_colors = []
        for _ in class_list:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            detection_colors.append((b, g, r))
        return detection_colors
    
    def process_frame(self, frame):
        results = self.model(frame)  # Get detection results
        annotated_frame = results[0].plot()  # Get annotated frame with bounding boxes
        return annotated_frame

    def plot_bboxes(self, Dnum, frame, detect, class_list, start_time):
        speed_kmh=0
        if len(Dnum) != 0:
            detection_colors = self.detect_colors(class_list)
            for i in range(len(detect[0])):
                boxes = detect[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                # Speed estimation for vehicles
                if class_list[int(clsID)] in ['car', 'truck', 'bus']:
                    center_x = (bb[0] + bb[2]) / 2
                    center_y = (bb[1] + bb[3]) / 2

                    vehicle_id = f"vehicle_{int(center_x)}"
                    if vehicle_id not in self.tracker:
                        self.tracker[vehicle_id] = {"last_position": (center_x, center_y), "last_time": start_time}

                    else:
                        last_position = self.tracker[vehicle_id]["last_position"]
                        last_time = self.tracker[vehicle_id]["last_time"]
                        current_time = start_time


                        distance_pixels = ((center_x - last_position[0])**2 + (center_y - last_position[1])**2) ** 0.5
                        time_elapsed = current_time - last_time
                        
                        if time_elapsed > 0:
                            speed_pixels_per_sec = distance_pixels / time_elapsed
                            speed_mps = speed_pixels_per_sec

                            speed_kmh = speed_mps * 3.6

                            self.tracker[vehicle_id]["last_position"] = (center_x, center_y)
                            self.tracker[vehicle_id]["last_time"] = current_time


                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                end_time = time()
                fps = 1 / (end_time - start_time)
                cv2.putText(
                    frame,
                    f"{class_list[int(clsID)]} ,{round(speed_kmh, 2)} km/h",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )
    
    def estimate_speed(self, bbox, frame, start_time):
        # Assuming the distance in pixels to meters is 1:1 for simplicity
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # Use a simple approach to identify unique vehicles
        vehicle_id = f"vehicle_{int(center_x)}"
        if vehicle_id not in self.tracker:
            self.tracker[vehicle_id] = {"last_position": (center_x, center_y), "last_time": start_time}

        # Calculate speed if this vehicle was seen before
        else:
            last_position = self.tracker[vehicle_id]["last_position"]
            last_time = self.tracker[vehicle_id]["last_time"]
            current_time = start_time
            
            # Calculate distance in pixels
            distance_pixels = ((center_x - last_position[0])**2 + (center_y - last_position[1])**2) ** 0.5
            time_elapsed = current_time - last_time
            
            if time_elapsed > 0:
                speed_pixels_per_sec = distance_pixels / time_elapsed
                speed_mps = speed_pixels_per_sec  # Speed in meters per second

                # Convert speed to km/h
                speed_kmh = speed_mps * 3.6

                # Update the last position and time
                self.tracker[vehicle_id]["last_position"] = (center_x, center_y)
                self.tracker[vehicle_id]["last_time"] = current_time

                # Display speed on the frame
                cv2.putText(frame, f"{round(speed_kmh, 2)} km/h", (int(center_x), int(center_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def open_video(self):
        cap = cv2.VideoCapture("highway_mini.mp4")
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap
    
    def open_dataset(self):
        with open("coco.txt") as my_file:
            data = my_file.read()
        class_list = data.split("\n")
        return class_list

    def __call__(self):
        cap = self.open_video()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_vid = 640
        frame_hyt = 480
        class_list = self.open_dataset()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        while True:
            start_time = time()
            ret, frame = cap.read()
            if not ret:
                break

            annot_frame = self.process_frame(frame)
            frame = cv2.resize(frame, (frame_vid, frame_hyt))
            detect = self.model.predict(source=[frame], conf=0.45, save=False)
            Dnum = detect[0].numpy()
            self.plot_bboxes(Dnum, frame, detect, class_list, start_time)
            out.write(annot_frame)

            cv2.imshow('objectdetection', frame)
            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

detector = ObjectDetection()
detector()

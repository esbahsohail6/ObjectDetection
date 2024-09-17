import cv2
from time import time
from ultralytics import YOLO
import random
import numpy as np


class ObjectDetection:
    def __init__(self):
        self.model=self.load_model()
        self.output_video="output/output.avi"
    def load_model(self):
        model = YOLO("yolov8m.pt")  # Load a pretrained YOLOv8 model
        model.fuse()
        return model
    
    def detect_colors(self,class_list):
        detection_colors=[]
        for i in range(len(class_list)):
          r=random.randint(0,255)
          g=random.randint(0,255)
          b=random.randint(0,255)
          detection_colors.append((b,g,r))
        return detection_colors
    
    def process_frame(self,frame):
        results = self.model(frame)  # Get detection results
        annotated_frame = results[0].plot()  # Get annotated frame with bounding boxes
        return annotated_frame



    def plot_bboxes(self, Dnum, frame, detect,class_list, start_time):
       if len(Dnum)!=0:
            detection_colors=self.detect_colors(class_list)
            for i in range(len(detect[0])):
                boxes = detect[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

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
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )
            return
       
    def open_video(self):
        cap = cv2.VideoCapture("highway_mini.mp4")
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap
    
    def open_dataset(self):
        my_file=open("coco.txt")
        data=my_file.read()
        class_list=data.split("\n")
        my_file.close()
        return class_list

    def __call__(self):
        cap=self.open_video()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_vid=640
        frame_hyt=480
        class_list=self.open_dataset()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video, fourcc,fps, (int(cap.get(3)), int(cap.get(4))))

        while True:
            start_time = time()
            
            ret, frame = cap.read()
            if not ret:
                break
            annot_frame=self.process_frame(frame)
            frame = cv2.resize(frame, (frame_vid, frame_hyt))
            
            detect=self.model.predict(source=[frame],conf=0.45,save=False)
            Dnum=detect[0].numpy()
            self.plot_bboxes(Dnum,frame,detect,class_list,start_time)
            out.write(annot_frame)
            
            cv2.imshow('objectdetection', frame)
            if cv2.waitKey(1) == ord("q"):
              break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
detector = ObjectDetection()
detector()

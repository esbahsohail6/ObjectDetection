import numpy as np
import cv2
from ultralytics import YOLO
import random

my_file=open("coco.txt")
data=my_file.read()
class_list=data.split("\n")
my_file.close()


detection_colors=[]
for i in range(len(class_list)):
  r=random.randint(0,255)
  g=random.randint(0,255)
  b=random.randint(0,255)
  detection_colors.append((b,g,r))


model=YOLO("yolov8n.pt")


frame_vid=640
frame_hyt=480
cap=cv2.VideoCapture("2053100-uhd_3840_2160_30fps.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    print(frame)
    if not ret:
      
      print("cant recieve frame (stream end?). Exiting...")
      break
    
    frame = cv2.resize(frame, (frame_vid, frame_hyt))

    cv2.imwrite("images/frame.png",frame)

    detect_params=model.predict(source="images/frame.png",conf=0.45,save=False)

    print(detect_params[0].numpy())

    DP=detect_params[0].numpy()

    if len(DP)!=0:
       for i in range(len(detect_params[0])):
            print(i)

            boxes = detect_params[0].boxes
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

            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    cv2.imshow('objectdetection', frame)
    
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
          

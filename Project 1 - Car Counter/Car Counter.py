from pydoc import classname

import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

#cap = cv2.VideoCapture(0) # For WebCam
#cap.set(3, 1280)
#cap.set(4, 720)
cap = cv2.VideoCapture("../videos/cars.mp4") # For Video



model = YOLO("../Yolo - Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask cars.png")

#Traking
tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)
limits = [400, 297, 673, 297]
count = []

while True:
    success, img = cap.read()
    Region = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    results = model(Region, stream = True)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    #img = cv2.flip(img, 1)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            #w, h = x2-x1, y2-y1
            #cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil(box.conf[0]*100)/100

            cls = int(box.cls[0])

            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "motorbike" or currentClass == "truck" or currentClass == "bus" and conf > 0.3 :
                #cvzone.putTextRect(img, f'{classNames[cls]}  {conf}', (max(0, x1), max(35, y1-20)), scale = 0.8, thickness = 1, offset = 5)
                #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))


    resultsTracker = tracker.update(detections)


    for results in resultsTracker:
        x1, y1, x2, y2, Id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(results)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cvzone.putTextRect(img, f'{Id}', (max(0, x1), max(35, y1 - 20)), scale = 0.8, thickness = 1, offset = 2)

        cx, cy = (x1+x2)/2 , (y1+y2)/2
        cx, cy = int(cx), int(cy)
        cv2.circle(img, (cx, cy), 5, (0, 255, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[3] + 20:
            if count.count(Id) == 0:
                count.append(Id)

    #cvzone.putTextRect(img, f'Count {len(count)}', (50, 50))
    cv2.putText(img, str(len(count)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 5)


    cv2.imshow("Image", img)
    #cv2.imshow("Region", Region)
    cv2.waitKey(1)

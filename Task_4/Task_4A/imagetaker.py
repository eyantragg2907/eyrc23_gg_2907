CAMERA_ID = 1 # 0 for internal, 1 for external as a basis

import cv2
from datetime import datetime
import time

filenames = "A.png B.png C.png D.png E.png".split()

num_of_frames_skip = 100
# Initialize the camera
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.open(CAMERA_ID):
    print("CAMERA NOT OPEN. ERROR")
else:
    # take a photo
    for i in range(num_of_frames_skip):
        ret, frame = cap.read()
    
    ret, frame = cap.read()
    frame = increase_brightness(frame, value=30)
    frame, pts, events = get_events(frame)
    # save the photo
    if ret is True:
        print("Photo 01 taken")

cap.release()


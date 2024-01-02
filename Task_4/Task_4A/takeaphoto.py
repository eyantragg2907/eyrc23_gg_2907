CAMERA_ID = 1  # 0 for internal, 1 for external (usually)

import cv2
from datetime import datetime

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

    # save the photo
    if ret is True:
        addr = f"temp_snap_{str(datetime.now().timestamp()).replace('.', '-')}.png"
        cv2.imwrite(addr, frame)
        cap.release()

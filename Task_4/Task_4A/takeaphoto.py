CAMERA_ID = 1 # 0 for internal, 1 for external as a basis

import cv2
from datetime import datetime

# Initialize the camera
cap = cv2.VideoCapture(CAMERA_ID)

# take a photo
ret, frame = cap.read()

# save the photo
cv2.imwrite(f"snapshot_{datetime.now().isoformat}.png", frame)
cap.release()


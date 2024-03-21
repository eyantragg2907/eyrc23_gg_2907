import cv2

CAMERA_ID = 0

capture = cv2.VideoCapture(CAMERA_ID)

while True:
    _, frame = capture.read()
    cv2.imshow("w", frame)
    cv2.waitKey(0)
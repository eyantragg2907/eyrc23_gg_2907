CAMERA_ID = 0  # 0 for internal, 1 for external as a basis

import cv2
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)


def get_aruco_data(frame):
    # plt.figure()
    # plt.imshow(frame)
    # plt.show()
    global detector
    c, i, r = detector.detectMarkers(frame)

    print(c)

    if len(c) == 0:
        raise Exception("No Aruco Markers Found")
    return c, i.flatten(), r


def get_points_from_aruco(frame):
    (
        corners,
        ids,
        _,
    ) = get_aruco_data(frame)
    reqd_ids = {4, 5, 6, 7}
    pt_A, pt_B, pt_C, pt_D = None, None, None, None

    for markerCorner, markerID in zip(corners, ids):
        if markerID not in reqd_ids:
            continue

        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners

        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        if markerID == 5:
            pt_A = topLeft
        elif markerID == 7:
            pt_B = bottomLeft
        elif markerID == 6:
            pt_C = bottomRight
        elif markerID == 4:
            pt_D = topRight
    return pt_A, pt_B, pt_C, pt_D


def transform_frame(frame):
    pt_A, pt_B, pt_C, pt_D = get_points_from_aruco(frame)

    if pt_A is None or pt_B is None or pt_C is None or pt_D is None:
        print(f"{pt_A=}, {pt_B=}, {pt_C=}, {pt_D=}")
        raise Exception("Corners not detected")

    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    s = min(maxHeight, maxWidth)
    input_pts = np.array([pt_A, pt_B, pt_C, pt_D], dtype=np.float32)
    output_pts = np.array([[0, 0], [0, s - 1], [s - 1, s - 1], [s - 1, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(frame, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    out = out[:s, :s]
    out = cv2.resize(out, (1024, 1024), interpolation = cv2.INTER_AREA)
    
    cv2.imwrite("temp_perspective.jpg", out)

    return out, s


def get_pts_from_frame(s):

    S = 1024
    Apts = (np.array([[980 / S, 896 / S], [294 / S, 212 / S]]) * s).astype(int)
    Bpts = (np.array([[696 / S, 766 / S], [780 / S, 695 / S]]) * s).astype(int)
    Cpts = (np.array([[489 / S, 574 / S], [693 / S, 776 / S]]) * s).astype(int)
    Dpts = (np.array([[485 / S, 570 / S], [196 / S, 280 / S]]) * s).astype(int)
    Epts = (np.array([[151 / S, 233 / S], [218 / S, 297 / S]]) * s).astype(int)

    return (Apts, Bpts, Cpts, Dpts, Epts)


def save_event_images(frame, pts, filenames):
    for p, f in zip(pts, filenames):
        print(p)
        print(frame)
        event = frame[p[0, 1] : p[0, 0], p[1, 1] : p[1, 0]]
        print(event)
        print("saving to", f)
        cv2.imwrite(f, event)


def get_events(frame, filenames):
    frame, side = transform_frame(frame)
    pts = get_pts_from_frame(side)
    
    save_event_images(frame, pts, filenames)

    return frame, None, None


def main():
    num_of_frames_skip = 100
    # Initialize the camera
    if sys.platform == "win32":
        cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # take a photo
    for i in range(num_of_frames_skip):
        ret, frame = cap.read()


    # frame = increase_brightness(frame, value=30)

    # set 02
    # A = "fire"
    # B = "destroyed_building"
    # C = "human_aid_rehabilitation"
    # D = "military_vehicles"
    # E = "combat"

    A = "empty0"
    B = "empty1"
    C = "empty2"
    D = "empty3"
    E = "empty4"

    SET = "NEWANDPRETTY_"

    FOLDER = "temp_pjrtrain"

    # while True:
    # # if ret:
    #     ret, frame = cap.read()
    #     cv2.imshow("frame", frame)
    #     if cv2.waitKey(1) == ord("q"):
    #         break
    
    # cv2.destroyAllWindows()

    c = 0
    while c < 5:
        ret, frame = cap.read()
        
        # save the photo
        if ret is True:
            print(f"Photo {c} taken")
            cv2.imwrite(f"temp_save.jpg", frame)
            filenames = f"{FOLDER}/{A}/{c}{SET}.png {FOLDER}/{B}/{c}{SET}.png {FOLDER}/{C}/{c}{SET}.png {FOLDER}/{D}/{c}{SET}.png {FOLDER}/{E}/{c}{SET}.png".split()
            frame, pts, events = get_events(frame, filenames)
            print(f"Photo {c} saved")
        c += 1
        print("Now we wait")
        if c == 5:
            break
        print("Next frame")
        time.sleep(5)

    print("Done")

    cap.release()


if __name__ == "__main__":
    main()

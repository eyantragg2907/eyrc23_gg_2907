CAMERA_ID = 1 # 0 for internal, 1 for external as a basis

import cv2
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

def get_aruco_data(frame):
    # plt.figure()
    # plt.imshow(frame)
    # plt.show()
    global detector
    c, i, r = detector.detectMarkers(frame)

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
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D]) # type: ignore
    output_pts = np.float32([[0, 0], [0, s - 1], [s - 1, s - 1], [s - 1, 0]]) # type: ignore
    M = cv2.getPerspectiveTransform(input_pts, output_pts) # type: ignore
    out = cv2.warpPerspective(frame, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    out = out[:s, :s]
    # out = cv2.resize(out, (1024,1024), interpolation = cv2.INTER_AREA)

    return out, s
def get_pts_from_frame(frame, s):
    S = 937
    Apts = (np.array([[811 / S, 887 / S], [194 / S, 269 / S]]) * s).astype(int)
    Bpts = (np.array([[628 / S, 705 / S], [620 / S, 695 / S]]) * s).astype(int)
    Cpts = (np.array([[444 / S, 519 / S], [628 / S, 701 / S]]) * s).astype(int)
    Dpts = (np.array([[440 / S, 516 / S], [183 / S, 260 / S]]) * s).astype(int)
    Epts = (np.array([[136 / S, 212 / S], [200 / S, 276 / S]]) * s).astype(int)

    return (Apts, Bpts, Cpts, Dpts, Epts)

def get_event_images(frame, pts, filenames):
    events = []
    for p, f in zip(pts, filenames):
        event = frame[p[0, 0] : p[0, 1], p[1, 0] : p[1, 1]]
        cv2.imwrite(f, event)
        events.append(event)
    return events

def get_events(frame, filenames):
    frame, side = transform_frame(frame)
    pts = get_pts_from_frame(frame, side)
    events = get_event_images(frame, pts, filenames)

    return frame, pts, events

def main():

    num_of_frames_skip = 100
    # Initialize the camera
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # take a photo
    for i in range(num_of_frames_skip):
        ret, frame = cap.read()

    # frame = increase_brightness(frame, value=30)
    
    # set 02
    A = "military_vehicles"
    B = "human_aid_rehabilitation"
    C = "fire"
    D = "destroyed_building"
    E = "combat"

    SET = "03"

    c = 1s
    while c < 5:
        ret, frame = cap.read()
        # save the photo
        if ret is True:
            print(f"Photo {c} taken")
            cv2.imwrite(f"temp_save.jpg", frame)
            filenames = f"temp_train/{A}/{c}{SET}.png temp_train/{B}/{c}{SET}.png temp_train/{C}/{c}{SET}.png temp_train/{D}/{c}{SET}.png temp_train/{E}/{c}{SET}.png".split()
            frame, pts, events = get_events(frame, filenames)
            print(f"Photo {c} saved")
        print("Now we wait")
        time.sleep(10) # 5 minutes
        print("Next frame")
        c += 1
    print("Done")
    
    cap.release()


if __name__ == "__main__":
    main()

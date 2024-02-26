CAMERA_ID = 1 # 0 for internal, 1 for external as a basis

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

    # print(c)

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
    out = cv2.resize(out, (1080, 1080), interpolation = cv2.INTER_AREA)
    
    cv2.imwrite("temp_perspective.jpg", out)

    return out, 1080

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
    return cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 0)

def get_pts_from_frame(s):
    S = 1080
    Apts = (np.array([[940 / S, 1026 / S], [222 / S, 308 / S]]) * s).astype(int)
    Bpts = (np.array([[729 / S, 816 / S], [717 / S, 802 / S]]) * s).astype(int)
    Cpts = (np.array([[513 / S, 601 / S], [725 / S, 811 / S]]) * s).astype(int)
    Dpts = (np.array([[509 / S, 597 / S], [206 / S, 293 / S]]) * s).astype(int)
    Epts = (np.array([[157 / S, 245 / S], [227 / S, 313 / S]]) * s).astype(int)

    return (Apts, Bpts, Cpts, Dpts, Epts)

COUNTER = 0
def save_event_images(frame, pts, filenames):
    global COUNTER
    for p, f in zip(pts, filenames):
        # print(p)
        # print(frame)
        event = frame[p[0, 0] : p[0, 1], p[1, 0] : p[1, 1]]
        # event = unsharp_mask(event)
        # print(event)
        dt = str(datetime.now().timestamp()).replace(".", "")
        f = f.split(".")[0] + dt + f"_{COUNTER}.png"
        print("saving to", f)
        cv2.imwrite(f, event)
        COUNTER += 1


def get_events(frame, filenames):
    frame, side = transform_frame(frame)
    pts = get_pts_from_frame(side)
    
    save_event_images(frame, pts, filenames)

    return frame, None, None


def main():
    num_of_frames_skip = 40
    # Initialize the camera
    if sys.platform == "win32":
        cap = cv2.VideoCapture(CAMERA_ID)
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


    classmap = [
        "combat",
        "building",
        "fire",
        "human",
        "vehicle",
        "None"
    ]

    A = classmap.index("vehicle")
    B = classmap.index("building")
    C = classmap.index("combat")
    D = classmap.index("fire")
    E = classmap.index("fire")

    SET = "NEWSET_"
    FOLDER = "overfit"

    # while True:
    # # if ret:
    #     ret, frame = cap.read()
    #     cv2.imshow("frame", frame)
    #     if cv2.waitKey(1) == ord("q"):
    #         break
    
    # cv2.destroyAllWindows()

    c = 0
    while c < 120:
        ret, frame = cap.read()
        
        # save the photo
        cx = c
        if ret is True:
            # cv2.imwrite(f"temp_save.jpg", frame)
            filenames = f"{FOLDER}/{A}/{cx}{SET}.png {FOLDER}/{B}/{cx}{SET}.png {FOLDER}/{C}/{cx}{SET}.png {FOLDER}/{D}/{cx}{SET}.png {FOLDER}/{E}/{cx}{SET}.png".split()
            frame, pts, events = get_events(frame, filenames)
            c += 1
        
        print("Now we wait")

    print("Done")

    cap.release()


if __name__ == "__main__":
    main()

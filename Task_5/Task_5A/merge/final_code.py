"""IMPORTANT"""
## LATEST ##
## TASK_4B FILE WITH QGIS AND LATEST DEBUGGING SYSTEM ##
"""IMPORTANT"""

# Team ID:			[ 2907 ]
# Author List:		[ Arnav Rustagi, Abhinav Lodha, Pranjal Rastogi]
# Filename:			task_4b.py

####################### IMPORT MODULES #######################

import cv2
import cv2.aruco as aruco
import numpy as np
import os
import sys
import csv
import pandas as pd
import socket
import threading

# from
import djikstra
import predictor

##############################################################

CAMERA_ID = 0  # camera ID for external camera

ARUCO_REQD_IDS = {4, 5, 6, 7}  # corners

ARUCO_ROBOT_ID = 100  # we chose this ID as it wasn't in the csv
IDEAL_MAP_SIZE = 1080  # map frame size

IP_ADDRESS = "192.168.187.144"  # IP of the Laptop on Hotspot

CHECK_FOR_ROBOT_AT_EVENT = True
OUT_FILE_LOC = "live_location.csv"

EVENT_FILENAMES = ["A.png", "B.png", "C.png", "D.png", "E.png"]

################# ADD UTILITY FUNCTIONS HERE #################

if not os.path.exists(OUT_FILE_LOC):
    with open(OUT_FILE_LOC, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon"])


def get_aruco_locs():
    return pd.read_csv("lat_long.csv", index_col="id")


def get_aruco_detector():
    dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    return detector


def cleanup(s):  # closes socket
    s.close()


def send_setup_robot(
    s: socket.socket, conn: socket.socket, command: str
):  # sends setup command to robot
    data = conn.recv(1024)
    data = data.decode("utf-8")  # decode the data from bytes to string

    if data == "ACK_REQ_FROM_ROBOT":
        pass
    else:
        print("Error in connection")
        cleanup(s)
        sys.exit(1)

    # conn.sendall(str.encode(COMMAND))
    conn.sendall(str.encode("START\n"))
    conn.sendall(str.encode(command))
    print(f"SENT START w/ {command}")

    # print(f"Sent command to robot: {COMMAND}")


def init_connection():  # initializes connection with robot
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((IP_ADDRESS, 8002))
        s.listen()
        conn, addr = s.accept()
        print(f"Connected by {addr}")
        return s, conn


def get_frame(video):  # gets frame from camera
    ret, frame = video.read()
    if ret:
        return frame
    else:
        raise Exception("Fatal camera error")


def get_image_pts_from_frame(side_len):
    s = side_len
    S = IDEAL_MAP_SIZE
    Apts = (np.array([[940 / S, 1026 / S], [222 / S, 308 / S]]) * s).astype(int)
    Bpts = (np.array([[729 / S, 816 / S], [717 / S, 802 / S]]) * s).astype(int)
    Cpts = (np.array([[513 / S, 601 / S], [725 / S, 811 / S]]) * s).astype(int)
    Dpts = (np.array([[509 / S, 597 / S], [206 / S, 293 / S]]) * s).astype(int)
    Epts = (np.array([[157 / S, 245 / S], [227 / S, 313 / S]]) * s).astype(int)

    return (Apts, Bpts, Cpts, Dpts, Epts)


def save_event_images(frame, pts, filenames):
    for p, f in zip(pts, filenames):
        event = frame[p[0, 0] : p[0, 1], p[1, 0] : p[1, 1]]
        cv2.imwrite(f, event)


def get_robot_coords_and_frame(
    capture, save_images=False
):  # updates the position of the robot
    frame = get_frame(capture)
    frame, side_len = transform_frame(frame)
    image_pts = get_image_pts_from_frame(side_len)

    if save_images:
        save_event_images(frame, image_pts, EVENT_FILENAMES)

    stopcoords = get_stopcoords(image_pts)
    pxcoords = get_robot_coords(frame)

    return frame, stopcoords, pxcoords


def get_stopcoords(pts):
    stopcoords = []
    for p in pts:
        stopcoords.append(
            [(p[1, 0] - 35, p[0, 0] - 100), (p[1, 1] + 35, p[0, 1] - 100)]
        )

    return stopcoords


def transform_frame(frame):  # transforms the frame to get
    """
    Purpose:
    ---
    Transforms the frame based on the markers

    Arguments:
    ---
    `frame` : { numpy.ndarray }
        the frame to transform

    Returns:
    ---
    `out` : { numpy.ndarray }
        the transformed frame
    `IDEAL_FRAME_SIZE` : { int }
        the length of the frame
    """

    pt_A, pt_B, pt_C, pt_D = get_points_from_aruco(frame)

    if pt_A is None or pt_B is None or pt_C is None or pt_D is None:
        print(f"{pt_A=}\n\n{pt_B=}\n\n{pt_C=}\n\n{pt_D=}")
        raise Exception("Corners not found correctly")

    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    s = min(maxHeight, maxWidth)
    input_pts = np.array([pt_A, pt_B, pt_C, pt_D], dtype=np.float32)
    output_pts = np.array(
        [[0, 0], [0, s - 1], [s - 1, s - 1], [s - 1, 0]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(frame, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    out = out[:s, :s]
    out = cv2.resize(
        out, (IDEAL_MAP_SIZE, IDEAL_MAP_SIZE), interpolation=cv2.INTER_AREA
    )

    return out, IDEAL_MAP_SIZE


prev_pt_A, prev_pt_B, prev_pt_C, prev_pt_D = None, None, None, None  # corners


def get_points_from_aruco(frame):
    """

    Purpose:
    ---
    This function returns the corners of the aruco markers

    Input Arguments:

    frame :   [ numpy array ]


    Returns:

    pt_A :   [ tuple ]

    pt_B :   [ tuple ]

    pt_C :   [ tuple ]

    pt_D :   [ tuple ]

    """
    global prev_pt_A, prev_pt_B, prev_pt_C, prev_pt_D

    corners, ids, _ = get_aruco_data(frame)

    pt_A, pt_B, pt_C, pt_D = None, None, None, None

    for markerCorner, markerID in zip(corners, ids):
        if markerID not in ARUCO_REQD_IDS:
            continue

        corners = markerCorner.reshape((4, 2))
        (top_left, top_right, bottom_right, bottom_left) = corners

        # convert to int
        top_right = tuple(map(int, top_right))
        bottom_right = tuple(map(int, bottom_right))
        bottom_left = tuple(map(int, bottom_left))
        top_left = tuple(map(int, top_left))

        # set the points for the corners, so we can transform later
        if markerID == 4:
            pt_D = top_right
        elif markerID == 5:
            pt_A = top_left
        elif markerID == 6:
            pt_C = bottom_right
        elif markerID == 7:
            pt_B = bottom_left

    if pt_A is None or pt_B is None or pt_C is None or pt_D is None:
        # use the previous frame's points
        return prev_pt_A, prev_pt_B, prev_pt_C, prev_pt_D
    else:
        # update the previous frame's points
        prev_pt_A, prev_pt_B, prev_pt_C, prev_pt_D = pt_A, pt_B, pt_C, pt_D

    return pt_A, pt_B, pt_C, pt_D


def get_aruco_data(frame, flatten=True):
    """
    Purpose:
    ---
    This function returns the aruco data

    Input Arguments:

    frame :   [ numpy array ]

    flatten :   [ boolean ]

    Returns:

    corners :   [ list ]

    ids :   [ list ]

    rrejected :   [ list ]

    """
    detector = get_aruco_detector()

    c, i, r = detector.detectMarkers(frame)

    if len(c) == 0:
        raise Exception("No Aruco Markers Found")
    if flatten:
        i = i.flatten()

    return c, i, r


def get_pxcoords(robot_id, ids, corners):
    """
    Purpose:

    This function returns the pixel coordinates of the robot

    Input Arguments:

    robot_id :   [ int ]

    ids :   [ list ]

    corners :   [ list ]

    Returns:

    coords :   [ numpy array ]

    """
    try:
        index = np.where(ids == robot_id)[0][0]
        coords = np.mean(corners[index].reshape((4, 2)), axis=0)
        return np.array([int(x) for x in coords])
    except:
        return []


def get_nearestmarker(robotcoords, corners, ids):
    global prev_closest_marker

    mindist = float("inf")
    closestmarker = None

    if len(robotcoords) == 0:
        return prev_closest_marker

    for markerCorner, markerID in zip(corners, ids):
        if markerID != ARUCO_ROBOT_ID:
            corners = markerCorner.reshape((4, 2))
            marker_center = np.mean(corners, axis=0)
            dist = np.linalg.norm(robotcoords - marker_center)
            if dist < mindist:  # type: ignore
                closestmarker = markerID
                mindist = dist

    prev_closest_marker = closestmarker

    return closestmarker


prev_closest_marker = None


def write_csv(loc, csv_name):
    with open(csv_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon"])
        writer.writerow(loc)


def update_qgis_position(robotcoords, corners, ids):
    arucolat_long = get_aruco_locs()
    nearest_marker = get_nearestmarker(robotcoords, corners, ids)
    if nearest_marker is None:
        return None
    try:
        coordinate = arucolat_long.loc[nearest_marker]

        write_csv(coordinate, OUT_FILE_LOC)

        return coordinate

    except:
        print("No lat long from this marker!!")
        return None


def get_robot_coords(frame):
    corners, ids, _ = get_aruco_data(frame)
    robot_pxcoords = get_pxcoords(ARUCO_ROBOT_ID, ids, corners)
    update_qgis_position(robot_pxcoords, corners, ids)

    return robot_pxcoords


def initialize_capture(frames_to_skip=100) -> cv2.VideoCapture:
    capture = None
    if sys.platform == "win32":
        capture = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(CAMERA_ID)

    # ensure 1920x1080 resolution
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    for _ in range(frames_to_skip):
        ret, frame = capture.read()

    return capture


def listen_and_print(s, conn: socket.socket):
    print("=" * 80)
    while True:
        data = conn.recv(4096)
        data = data.decode("utf-8")
        print(f"{data}")


###############	Main Function	#################
if __name__ == "__main__":
    capture = initialize_capture()

    frame, stopcoords, pxcoords = get_robot_coords_and_frame(capture, save_images=True)
    detected_events = predictor.run_predictor(EVENT_FILENAMES)
    path = djikstra.final_path(detected_events)
    command = "n" + path

    print("Initializing Connection")
    soc, conn = init_connection()
    print("Sending command..")
    send_setup_robot(soc, conn, command)

    lpt = threading.Thread(target=listen_and_print, args=(soc, conn))

    lpt.start()

    try:
        while True:
            # get a new frame, transform it and get the robots coordinates
            frame, stopcoords, pxcoords = get_robot_coords_and_frame(
                capture, save_images=False
            )

            if len(pxcoords) != 0:
                # robot is in frame
                for i in stopcoords:
                    # check if robot is at any of the stopcoords
                    if (i[0][0] < pxcoords[0] and i[1][0] > pxcoords[0]) and (
                        i[0][1] < pxcoords[1] and i[1][1] > pxcoords[1]
                    ):  # robot is at the event
                        conn.sendall(str.encode("ISTOP\n"))

    except KeyboardInterrupt:
        lpt.join()
        cleanup(soc)
        capture.release()
        # delete_images()
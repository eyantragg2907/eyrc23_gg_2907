"""IMPORTANT"""
## LATEST ##
## TASK_4B FILE WITH QGIS AND LATEST DEBUGGING SYSTEM ##
"""IMPORTANT"""

# Team ID:			[ 2907 ]
# Author List:		[ Arnav Rustagi, Abhinav Lodha]
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

##############################################################
OUT_FILE_LOC = "live_location.csv"
if not os.path.exists(OUT_FILE_LOC):
    with open(OUT_FILE_LOC, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon"])

CAMERA_ID = 1


ARUCO_REQD_IDS = {"4", "5", "6", "7"}

ARUCO_ROBOT_ID = 97
IDEAL_FRAME_SIZE = 1080

IP_ADDRESS = "192.168.54.92"  # IP of the Laptop on Hotspot
COMMAND = "nnrnlnrnrnnrnnlnn"  # the full cycle command

################# ADD UTILITY FUNCTIONS HERE #################


def get_aruco_locs():
    return pd.read_csv("lat_long.csv", index_col="id")


def get_aruco_detector():
    dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    return detector


def cleanup(s):
    if s is not None:   
        s.close()


def send_to_robot(s: socket.socket, conn: socket.socket):
    data = conn.recv(1024)
    data = data.decode("utf-8")

    if data == "ACK_REQ_FROM_ROBOT":
        pass
    else:
        print("Error in connection")
        cleanup(s)
        sys.exit(1)
    
    conn.sendall(str.encode(COMMAND))

    print(f"Sent command to robot: {COMMAND}")


def init_connection():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((IP_ADDRESS, 8002))
        s.listen()
        conn, addr = s.accept()
        print(f"Connected by {addr}")
        return s, conn


def get_frame(video):
    ret, frame = video.read()
    if ret:
        return frame
    else:
        raise Exception("Fatal camera error")


def update_qgis_position(capture):
    frame = get_frame(capture)
    frame, side_len = transform_frame(frame)
    get_robot_coords(frame)
    return frame


def transform_frame(frame):
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
        out, (IDEAL_FRAME_SIZE, IDEAL_FRAME_SIZE), interpolation=cv2.INTER_AREA
    )

    return out, IDEAL_FRAME_SIZE


prev_pt_A, prev_pt_B, prev_pt_C, prev_pt_D = None, None, None, None


def get_points_from_aruco(frame):
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
    detector = get_aruco_detector()

    c, i, r = detector.detectMarkers(frame)

    if len(c) == 0:
        raise Exception("No Aruco Markers Found")
    if flatten:
        i = i.flatten()

    return c, i, r


def get_pxcoords(robot_id, ids, corners):
    try:
        index = np.where(ids == robot_id)[0][0]
        coords = np.mean(corners[index].reshape((4, 2)), axis=0)
        return np.array([int(x) for x in coords])
    except:
        return []


prev_closest_marker = None


def get_nearestmarker(robot_id, ids, corners):
    global prev_closest_marker

    mindist = float("inf")
    closestmarker = None

    coords1 = get_pxcoords(robot_id, ids, corners)

    if len(coords1) == 0:
        return prev_closest_marker

    for markerCorner, markerID in zip(corners, ids):
        if markerID != robot_id:
            corners = markerCorner.reshape((4, 2))
            marker_center = np.mean(corners, axis=0)
            dist = np.linalg.norm(coords1 - marker_center)
            if dist < mindist:  # type: ignore
                closestmarker = markerID
                mindist = dist

    prev_closest_marker = closestmarker

    return closestmarker


def get_robot_coords(frame):
    corners, ids, _ = get_aruco_data(frame)
    nearest_marker = get_nearestmarker(ARUCO_ROBOT_ID, ids, corners)
    arucolat_long = get_aruco_locs()

    if nearest_marker is None:
        return None
    try:
        coordinate = arucolat_long.loc[nearest_marker]
       
        write_csv(coordinate, OUT_FILE_LOC)

        return coordinate
    
    except:
        print("No lat long from this marker!!")
        return None


def write_csv(loc, csv_name):

    with open(csv_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon"])
        writer.writerow(loc)


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
    if conn is None:
        return None
    data = conn.recv(1024)
    data = data.decode("utf-8")
    print(f"recv: {data}")

###############	Main Function	#################
if __name__ == "__main__":
    capture = initialize_capture()

    counter = 0
    soc, conn = None, None
    while True:
        # get a new frame, transform it and get the robots coordinates
        frame = update_qgis_position(capture)

        corners, ids, rejected = get_aruco_data(frame, flatten=False)
        aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.namedWindow("Arena Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Arena Feed", (750, 750))
        cv2.imshow("Arena Feed", cv2.resize(frame, (750, 750)))
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break
        if counter == 0:
            soc, conn = init_connection()
            send_to_robot(soc, conn)
        
        listen_and_print(soc, conn)

        counter += 1

    capture.release()
    cleanup(soc)

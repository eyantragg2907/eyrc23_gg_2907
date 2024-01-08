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

##############################################################
OUT_FILE_LOC = "live_location.csv" # outputs live locations
if not os.path.exists(OUT_FILE_LOC):
    with open(OUT_FILE_LOC, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon"])

CAMERA_ID = 1 # camera ID for external camera


ARUCO_REQD_IDS = {4, 5, 6, 7} # corners

ARUCO_ROBOT_ID = 97 # we chose this ID as it wasn't in the csv
IDEAL_FRAME_SIZE = 1080 # camera frame

IP_ADDRESS = "192.168.128.92"  # IP of the Laptop on Hotspot
COMMAND = "nnrnlnrnrnnrnnlnn"  # the path

################# ADD UTILITY FUNCTIONS HERE #################


def get_aruco_locs(): 
    """
    Purpose:
    ---
    This function reads the csv file containing the ArUco IDs and their corresponding latitudes and longitudes

    Arguments:
    ---
    None

    Returns:
    ---
    dataframe
    """
    return pd.read_csv("lat_long.csv", index_col="id")


def get_aruco_detector(): 
    """
    Purpose:
    ---
    This function returns the aruco detector object

    Arguments:
    ---
    None

    Returns:
    ---
    detector : aruco detector object
    """
    dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    return detector


def cleanup(s): # closes socket
    """
    Purpose:

    This function closes the socket connection.

    Input Arguments:

    s :   [ socket object ]

        socket object which needs to be closed.
    
    Returns:

    None
    """
    s.close()


def send_to_robot(s: socket.socket, conn: socket.socket): # sends command to robot
    """
    Purpose:
    
    This function sends the command to the robot.

    Input Arguments:

    s :   [ socket object ]


    conn :   [ socket object ]

    Returns:

    None

    """        
    data = conn.recv(1024)
    data = data.decode("utf-8") # decode the data from bytes to string

    if data == "ACK_REQ_FROM_ROBOT":
        pass
    else:
        print("Error in connection")
        cleanup(s)
        sys.exit(1)
    
    conn.sendall(str.encode(COMMAND))

    print(f"Sent command to robot: {COMMAND}")


def init_connection(): # initializes connection with robot
    """
    Purpose:

    This function initializes the connection with the robot.

    Input Arguments:

    None

    Returns:

    s :   [ socket object ]

        socket object 
    
    conn :   [ socket object ]

        socket object
    
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((IP_ADDRESS, 8002))
        s.listen()
        conn, addr = s.accept()
        print(f"Connected by {addr}")
        return s, conn


def get_frame(video): # gets frame from camera
    """
    Purpose:

    This function gets the frame from the camera.

    Input Arguments:

    video :   [ cv2.VideoCapture object ]
    
            cv2.VideoCapture object

    Returns:

    frame :   [ numpy array ]
    
                numpy array of the frame

    """
    ret, frame = video.read()
    if ret:
        return frame
    else:
        raise Exception("Fatal camera error")


def update_qgis_position(capture): # updates the position of the robot
    """
    
    Purpose:

    This function updates the position of the robot.

    Input Arguments:

    capture :   [ cv2.VideoCapture object ]
        
                    cv2.VideoCapture object

    Returns:

    frame :   [ numpy array ]
        
                    numpy array of the frame

    """
    frame = get_frame(capture)
    frame, side_len = transform_frame(frame)
    get_robot_coords(frame)
    return frame


def transform_frame(frame): # transforms the frame to get 
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
        out, (IDEAL_FRAME_SIZE, IDEAL_FRAME_SIZE), interpolation=cv2.INTER_AREA
    )

    return out, IDEAL_FRAME_SIZE


prev_pt_A, prev_pt_B, prev_pt_C, prev_pt_D = None, None, None,  # corners


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


prev_closest_marker = None


def get_nearestmarker(robot_id, ids, corners):
    """
    """
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

###############	Main Function	#################
if __name__ == "__main__":
    capture = initialize_capture()

    counter = 0   
    while True:
        # get a new frame, transform it and get the robots coordinates
        frame = update_qgis_position(capture)

        corners, ids, rejected = get_aruco_data(frame, flatten=False)
        # aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.namedWindow("Arena Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Arena Feed", (750, 750))
        cv2.imshow("Arena Feed", cv2.resize(frame, (750, 750)))
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break

        if counter == 0:
            soc, conn = init_connection()
            send_to_robot(soc, conn)
            cleanup(soc)

        counter += 1

    capture.release()

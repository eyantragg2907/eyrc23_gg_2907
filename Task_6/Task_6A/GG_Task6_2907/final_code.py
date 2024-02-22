"""IMPORTANT"""
## LATEST ##
## TASK_5A FILE WITH QGIS ##
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
import time
import threading
import sys
import djikstra
import pickle
import ast
##############################################################

CAMERA_ID = 0  # camera ID for external camera

ARUCO_REQD_IDS = {4, 5, 6, 7}  # corners

ARUCO_ROBOT_ID = 100  # we chose this ID as it wasn't in the csv
IDEAL_MAP_SIZE = 1080  # map frame size

IP_ADDRESS = "192.168.67.62"  # IP of the Laptop on Hotspot

CHECK_FOR_ROBOT_AT_EVENT = True
OUT_FILE_LOC = "live_location.csv"

EVENT_FILENAMES = ["A.png", "B.png", "C.png", "D.png", "E.png"]

EVENT_STOP_EXTRA_PIXELS = 25

# with open("map_corners.pkl", "rb") as f:
#     GLOBAL_ARUCO_CORNERS = pickle.load(f)
# with open("map_ids.pkl", "rb") as f:
#     GLOBAL_ARUCO_IDS = pickle.load(f)
# print(len(GLOBAL_ARUCO_IDS), len(GLOBAL_ARUCO_CORNERS))

df = pd.read_csv("corners.csv")

GLOBAL_ARUCO_CORNERS = [np.array(ast.literal_eval(x.replace('\n', ',').replace('.', ','))) for x in df["corners"].values]
GLOBAL_ARUCO_IDS = df["ids"].values
# print(GLOBAL_ARUCO_IDS, GLOBAL_ARUCO_CORNERS)
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
    conn.sendall(str.encode(command + "\n"))
    # print(f"SENT START w/ {command}")

    # print(f"Sent command to robot: {COMMAND}")


def init_connection():  # initializes connection with robot
    print("INIIT")
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

    return frame, stopcoords, pxcoords, image_pts


def get_stopcoords(pts):
    stopcoords = []
    for p in pts:
        stopcoords.append(
            [
                (p[1, 0] - EVENT_STOP_EXTRA_PIXELS, p[0, 0] - 100),
                (p[1, 1] + EVENT_STOP_EXTRA_PIXELS, p[0, 1] - 100),
            ]
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
    """
    Purpose:

    This function returns the pixel coordinates of the robot

    Input Arguments:

    robotcoords :   [ numpy array ]

    corners :   [ list ]

    ids :   [ list ]

    Returns:

    closestmarker :   [ int ]
    """
    global prev_closest_marker

    mindist = float("inf")
    closestmarker = None
    # print(f"Len corners: {len(corners)}")
    if len(robotcoords) == 0:
        return prev_closest_marker
    
    df = pd.DataFrame({"corners": corners, "ids": ids})
    # print(df)
    # print(df)
    # df.to_csv("corners.csv")
    # print(len(corners))
    for markerCorner, markerID in zip(GLOBAL_ARUCO_CORNERS, GLOBAL_ARUCO_IDS):
        if markerID != ARUCO_ROBOT_ID:
            corners_ = markerCorner.reshape((4, 2))
            marker_center = np.mean(corners_, axis=0)
            dist = np.linalg.norm(robotcoords - marker_center)
            if dist < mindist:  # type: ignore
                closestmarker = markerID
                mindist = dist
    # for corner, ID in zip(corners, ids): ADD FAKE ARUCOS
    #     if ID == ARUCO_ROBOT_ID:
    #         corners_ = corner.reshape((4, 2))
    #         print(corners_)
    # print(closestmarker)
    prev_closest_marker = closestmarker
    # print(closestmarker)
    return closestmarker


prev_closest_marker = None

def write_csv(loc, csv_name):
    """
    Write the given location data to a CSV file.

    Input Arguments:
        loc (list): The location data to be written to the CSV file.
        csv_name (str): The name of the CSV file.

    Returns:
        None
    """
    #print("INSIDE", csv_name, loc)
    # os.remove(csv_name)
    with open(csv_name, "w") as f:
        #print(f"writing to file, {loc}")
        f.write(f"lat, lon\n{loc[0]}, {loc[1]}")
    
    # with open(csv_name, "r") as f:
    #     print(f"{f.readlines()=}")


def update_qgis_position(robotcoords, corners, ids):
    """
    Update the position of the robot in QGIS based on the given robot coordinates.

    Input Arguments:
        robotcoords (list): The coordinates of the robot.
        corners (list): The corners of the markers.
        ids (list): The IDs of the markers.

    Returns:
        coordinate (tuple): The updated coordinate of the robot in QGIS.
        None: If no valid coordinate is found.
    """
    arucolat_long = get_aruco_locs()
    # print(len(GLOBAL_ARUCO_CORNERS))
    nearest_marker = get_nearestmarker(robotcoords, corners, ids)
    if nearest_marker is None:
        return None
    try:

        coordinate = arucolat_long.loc[nearest_marker]
        # print(coordinate)
        # print("upd qgis") 
        coordinate.reset_index()
        coordinate = tuple(coordinate)
        # print(coordinate)
        write_csv(coordinate, OUT_FILE_LOC)

        return coordinate

    except:
        # print("No lat long from this marker!!")
        return None


def get_robot_coords(frame):
    """
    Get the coordinates of the robot based on the given frame.

    Input Arguments:
        frame: The frame from which to extract the robot coordinates.

    Returns:
        robot_pxcoords: The pixel coordinates of the robot.
    """
    corners, ids, _ = get_aruco_data(frame)
    robot_pxcoords = get_pxcoords(ARUCO_ROBOT_ID, ids, corners)
    update_qgis_position(robot_pxcoords, corners, ids)

    return robot_pxcoords


def initialize_capture(frames_to_skip=20) -> cv2.VideoCapture:
    """
    Initialize the video capture device.

    Input Arguments:
        frames_to_skip: The number of frames to skip before starting the capture.

    Returns:
        capture: The initialized video capture device.
    """
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
    """
    Listen for incoming data from the socket connection and print it.

    Input Arguments:
        s: The socket object.
        conn: The socket connection.

    Returns:
        None
    """
    print("=" * 80)
    try:
        while True:
            data = conn.recv(4096)
            data = data.decode("utf-8")
            print(f"{data}")
            if "terminate" in data:
                cv2.destroyAllWindows()
                sys.exit(0)
    except KeyboardInterrupt:
        raise KeyboardInterrupt


def add_rect_labels(frame, pts, labels):
    for p, l in zip(pts, labels):
        frame = cv2.rectangle(
            frame, (p[1, 0], p[0, 0]), (p[1, 1], p[0, 1]), (0, 255, 0), 2
        )
        frame = cv2.putText(
            frame,
            str(l),
            (p[1, 0], p[0, 0] - 15),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    return frame


def get_detected_events():
    import predictor

    detected_events = predictor.run_predictor(EVENT_FILENAMES)
    detected_events_out = {k: v for k, v in detected_events.items() if v is not None}
    print(detected_events_out)
    return detected_events


def delete_images():
    for f in EVENT_FILENAMES:
        os.remove(f)


###############	Main Function	#################
if __name__ == "__main__":
    capture = initialize_capture()

    frame, stopcoords, pxcoords, img_pts = get_robot_coords_and_frame(
        capture, save_images=True
    )

    if (len(sys.argv) == 3) and (sys.argv[1] == "--command"):
        detected_events = {k: None for k in EVENT_FILENAMES}
        command = sys.argv[2]
        print(f"DEBUG: moving with command: {command}")
    else:
        detected_events = get_detected_events()
        frame = add_rect_labels(frame, img_pts, detected_events.values())

        # show bounding boxes
        cv2.namedWindow("image_with_boundingbox", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image_with_boundingbox", (750, 750))
        cv2.imshow("image_with_boundingbox", cv2.resize(frame, (750, 750)))
        cv2.moveWindow("image_with_boundingbox", 0, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # run djikstra to get the path between events, maintainig priority
        path = djikstra.final_path(detected_events)
        command = "n" + path
        
    # print(command)
    
    # send robot the command!
    soc, conn = init_connection()
    # command="nnn"
    # print("SENDING")
    
    send_setup_robot(soc, conn, command)
    # print("SENT")
    
    
    
    # DEBUG: listen to robot
    lpt = threading.Thread(target=listen_and_print, args=(soc, conn))
    lpt.daemon = True
    lpt.start()
    


    try:
        cv2.namedWindow("robot_moving", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("robot_moving", (750, 750))
        while True:
            # get a new frame, transform it and get the robots coordinates
            frame, stopcoords, pxcoords, img_pts = get_robot_coords_and_frame(
                capture, save_images=False
            )

            # if len(pxcoords) != 0:
            #     # robot is in frame
            #     for i in stopcoords:
            #         # check if robot is at any of the stopcoords
            #         if (i[0][0] < pxcoords[0] and i[1][0] > pxcoords[0]) and (
            #             i[0][1] < pxcoords[1] and i[1][1] > pxcoords[1]
            #         ):  # robot is at the event
            #             conn.sendall(str.encode("ISTOP\n"))

            cv2.imshow("robot_moving", cv2.resize(frame, (750, 750)))
            cv2.moveWindow("robot_moving", 0, 0)
            if cv2.waitKey(1) == ord("q"):
                break

    except KeyboardInterrupt:
        # cleanup(soc)
        capture.release()
        lpt.join()
        cv2.destroyAllWindows()
        delete_images()

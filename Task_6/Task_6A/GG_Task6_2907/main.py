""" 
* Team Id : 2907
* Author List : Arnav Rustagi, Abhinav Lodha, Pranjal Rastogi
* Filename: main.py
* Theme: GeoGuide (GG)
* Functions: get_aruco_gps_df, get_aruco_detector, cleanup, connect_and_move, init_connection, get_camera_frame,
            get_events_coords, save_event_images, get_robot_coords_and_frame, get_stopcoords, transform_frame,
            get_corners_map, get_aruco_map_data, get_robot_pxcoords, get_nearest_aruco, write_live_location_csv,
            update_qgis_position, get_robot_pxcoords_update_qgis, initialize_capture, listen_and_print,
            draw_event_labels, get_detected_events, delete_event_images
* Global Variables: CAMERA_ID, ARUCO_CORNER_IDS, ROBOT_ARUCO_ID, IDEAL_MAP_SIZE, HOST_IP_ADDRESS,
                    CLOSEST_ARUCO_OUTPUT_FILE_LOC, ARUCO_DATA_PATH, GPS_COORDS_DATA,
                    EVENT_FILENAMES, EVENT_STOP_EXTRA_PIXELS, GLOBAL_ARUCOS_DF, GLOBAL_ARUCO_CORNERS, GLOBAL_ARUCO_IDS,
                    prev_closest_aruco
"""

# Importing libraries
import cv2
import cv2.aruco as aruco
import numpy as np
import os
import sys
import csv
import pandas as pd
import socket
import threading
import sys
import ast


# Importing local modules
import predictor
import djikstra

# Importing types for Type Hinting
from cv2.typing import MatLike
from collections.abc import Sequence
from typing import Tuple, Union

# Camera ID has to be specified for using external camera
CAMERA_ID = 0

# IDs of corner AruCos
ARUCO_CORNER_IDS = {4, 5, 6, 7}

ROBOT_ARUCO_ID = 100

IDEAL_MAP_SIZE = 1080

# IP address of the computer (changed everytime)
HOST_IP_ADDRESS = "192.168.67.62"

# For QGIS live tracking
CLOSEST_ARUCO_OUTPUT_FILE_LOC = "live_location.csv"

# This file maps pixel coordinates to AruCo IDs for optimal live tracking
ARUCO_DATA_PATH = "corners.csv"

# This file contains the mapping between AruCo IDs and GPS coordinates
GPS_COORDS_DATA = "lat_long.csv"

EVENT_FILENAMES = ["A.png", "B.png", "C.png", "D.png", "E.png"]

EVENT_STOP_EXTRA_PIXELS = 35

# Reads the AruCo IDs and Corners Data
GLOBAL_ARUCOS_DF = pd.read_csv(ARUCO_DATA_PATH)
GLOBAL_ARUCO_CORNERS = [
    np.array(ast.literal_eval(x.replace("\n", ",").replace(".", ",")))
    for x in GLOBAL_ARUCOS_DF["corners"].values
]
GLOBAL_ARUCO_IDS = GLOBAL_ARUCOS_DF["ids"].values

# Used for tracking last closest AruCo marker to robot
prev_closest_aruco = None

# Creates file if not exists
if not os.path.exists(CLOSEST_ARUCO_OUTPUT_FILE_LOC):
    with open(CLOSEST_ARUCO_OUTPUT_FILE_LOC, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon"])

""" 
* Function Name: get_aruco_gps_df
* Input: None
* Output: pandas.DataFrame
* Logic: Returns the dataframe containing the mapping between AruCo IDs and GPS coordinates
* Example Call: get_aruco_gps_df() -> pandas.DataFrame
"""
def get_aruco_gps_df() -> pd.DataFrame:
    return pd.read_csv(GPS_COORDS_DATA, index_col="id")


""" 
* Function Name: get_aruco_detector
* Input: None
* Output: aruco.ArucoDetector
* Logic: Returns the aruco detector
* Example Call: get_aruco_detector() -> aruco.ArucoDetector
"""
def get_aruco_detector() -> aruco.ArucoDetector:
    dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    return detector


""" 
* Function Name: cleanup
* Input: s: socket.socket
* Output: None
* Logic: closes socket
* Example Call: cleanup(s) -> None
"""
def cleanup(s: socket.socket) -> None:
    s.close()


""" 
* Function Name: connect_and_move
* Input: s: socket.socket, conn: socket.socket, path: str
* Output: None
* Logic: Checks connection to the robot and sends the path
* Example Call: connect_and_move(s, conn, "nnn") -> None
"""
def connect_and_move(s: socket.socket, conn: socket.socket, path: str) -> None:
    data = conn.recv(1024)
    data = data.decode("utf-8")

    if data == "ACK_REQ_FROM_ROBOT":
        pass
    else:
        print("Error in connection")
        cleanup(s)
        sys.exit(1)

    # Send start ping and path to robot
    conn.sendall(str.encode("START\n"))
    conn.sendall(str.encode(path + "\n"))


""" 
* Function Name: init_connection
* Input: None
* Output: tuple[socket.socket, socket.socket]
* Logic: Initializes the connection with the robot
* Example Call: init_connection() -> (socket.socket, socket.socket)
"""
def init_connection() -> tuple[socket.socket, socket.socket]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # We are using port 8002 for our connections
        s.bind((HOST_IP_ADDRESS, 8002))

        s.listen()
        conn, addr = s.accept()
        return s, conn


""" 
* Function Name: get_camera_frame
* Input: capture: cv2.VideoCapture
* Output: numpy.ndarray
* Logic: Returns the frame from the camera
* Example Call: get_camera_frame(capture) -> numpy.ndarray
"""
def get_camera_frame(capture: cv2.VideoCapture) -> np.ndarray:
    ret, frame = capture.read()
    if ret:
        return frame
    else:
        raise Exception("Fatal Camera Error!")


""" 
* Function Name: get_events_coords
* Input: side_len: int
* Output: Tuple[numpy.ndarray, ...]
* Logic: Returns the pixel coordinates (bounding boxes) of the events
* Example Call: get_events_coords(1080) -> (numpy.ndarray, numpy.ndarray, 
                            numpy.ndarray, numpy.ndarray, numpy.ndarray)
"""
def get_events_coords(side_len: int) -> Tuple[np.ndarray, ...]:

    s = side_len
    S = IDEAL_MAP_SIZE
    Apts = (np.array([[940 / S, 1026 / S], [222 / S, 308 / S]]) * s).astype(int)
    Bpts = (np.array([[729 / S, 816 / S], [717 / S, 802 / S]]) * s).astype(int)
    Cpts = (np.array([[513 / S, 601 / S], [725 / S, 811 / S]]) * s).astype(int)
    Dpts = (np.array([[509 / S, 597 / S], [206 / S, 293 / S]]) * s).astype(int)
    Epts = (np.array([[157 / S, 245 / S], [227 / S, 313 / S]]) * s).astype(int)

    return (Apts, Bpts, Cpts, Dpts, Epts)


""" 
* Function Name: save_event_images
* Input: frame: numpy.ndarray, pts: Tuple[numpy.ndarray, ...],
            filenames: list[str]
* Output: None
* Logic: Saves the images of the events
* Example Call: save_event_images(frame, (Apts, Bpts, Cpts, Dpts, Epts),
                            ["A.png", "B.png", "C.png", "D.png", "E.png"]) -> None
"""
def save_event_images(
    frame: np.ndarray, pts: Tuple[np.ndarray, ...], filenames: list[str]
) -> None:
    for p, f in zip(pts, filenames):
        event = frame[p[0, 0] : p[0, 1], p[1, 0] : p[1, 1]]
        cv2.imwrite(f, event)


""" 
* Function Name: get_robot_coords_and_frame
* Input: capture: cv2.VideoCapture, save_images: bool
* Output: tuple[np.ndarray, list[np.ndarray], np.ndarray, Tuple[np.ndarray, ...]]
* Logic: Returns the frame, stopcoords, pxcoords and event_coords
* Example Call: get_robot_coords_and_frame(capture) -> (np.ndarray, list[np.ndarray],
                    np.ndarray, Tuple[np.ndarray, ...])
"""
def get_robot_coords_and_frame(
    capture: cv2.VideoCapture, save_images=False
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, Tuple[np.ndarray, ...]]:
    frame = get_camera_frame(capture)
    frame, side_len = transform_frame(frame)
    event_coords = get_events_coords(side_len)

    if save_images:
        save_event_images(frame, event_coords, EVENT_FILENAMES)

    stopcoords = get_stopcoords(event_coords)
    pxcoords = get_robot_pxcoords_update_qgis(frame)

    return frame, stopcoords, pxcoords, event_coords


""" 
* Function Name: get_stopcoords
* Input: event_coords: Tuple[numpy.ndarray, ...]
* Output: list[numpy.ndarray]
* Logic: Returns the stop coordinates for the robot to stop at the events
* Example Call: get_stopcoords(event_coords) -> list[numpy.ndarray]
"""
def get_stopcoords(event_coords: Tuple[np.ndarray, ...]) -> list[np.ndarray]:
    stopcoords = []
    for p in event_coords:
        stopcoords.append(
            [
                (p[1, 0] - EVENT_STOP_EXTRA_PIXELS, p[0, 0] - 100),
                (p[1, 1] + EVENT_STOP_EXTRA_PIXELS, p[0, 1] - 100),
            ]
        )

    return stopcoords


""" 
* Function Name: transform_frame
* Input: frame: numpy.ndarray
* Output: tuple[numpy.ndarray, int]
* Logic: Transforms the frame to a square and returns 
         the transformed frame and the side length of the square
* Example Call: transform_frame(frame) -> (numpy.ndarray, int)
"""
def transform_frame(frame: np.ndarray) -> tuple[np.ndarray, int]:

    pt_A, pt_B, pt_C, pt_D = get_corners_map(frame)

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

""" 
* Function Name: get_corners_map
* Input: frame: numpy.ndarray
* Output: Tuple[numpy.ndarray, ...]
* Logic: Returns the updated corners of the map (accounts of minute camera movements)
* Example Call: transform_frame(frame) -> (numpy.ndarray, int)
"""
def get_corners_map(
    frame: np.ndarray,
) -> Union[Tuple[Tuple[int, ...], ...], Tuple[None, ...]]:

    global prev_pt_A, prev_pt_B, prev_pt_C, prev_pt_D

    corners, ids, _ = get_aruco_map_data(frame)

    pt_A, pt_B, pt_C, pt_D = None, None, None, None

    for markerCorner, markerID in zip(corners, ids):
        if markerID not in ARUCO_CORNER_IDS:
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


""" 
* Function Name: get_aruco_data
* Input: frame: numpy.ndarray, flatten: bool
* Output: tuple[Sequence[MatLike], MatLike, Sequence[MatLike]]
* Logic: Detects the aruco markers and returns the corners, ids and rejected points
* Example Call: get_aruco_data(frame) -> (Sequence[MatLike], MatLike, Sequence[MatLike])
"""
def get_aruco_map_data(
    frame: np.ndarray, flatten: bool = True
) -> tuple[Sequence[MatLike], MatLike, Sequence[MatLike]]:
    detector = get_aruco_detector()
    c, i, r = detector.detectMarkers(frame)
    if len(c) == 0:
        cv2.imshow("wtf", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        raise Exception("No Aruco Markers Found")
    if flatten:
        i = i.flatten()
    return c, i, r


""" 
* Function Name: get_robot_pxcoords
* Input: robot_id: int, ids: MatLike, corners: Sequence[MatLike]
* Output: np.ndarray
* Logic: Returns the pixel coordinates of the robot
* Example Call: get_robot_pxcoords(100, ids, corners) -> np.ndarray
"""
def get_robot_pxcoords(
    robot_id: int, ids: MatLike, corners: Sequence[MatLike]
) -> np.ndarray:
    try:
        index = np.where(ids == robot_id)[0][0]
        coords = np.mean(corners[index].reshape((4, 2)), axis=0)
        return np.array([int(x) for x in coords])
    except:
        return np.array([])


""" 
* Function Name: get_nearest_aruco
* Input: robotcoords: np.ndarray
* Output: int | None
* Logic: Returns the nearest aruco marker to the robot 
        (uses GLOBAL_ARUCO_CORNERS and GLOBAL_ARUCO_IDS)
* Example Call: get_nearest_aruco(robotcoords) -> 4
"""
def get_nearest_aruco(robotcoords: np.ndarray) -> Union[int, None]:

    global prev_closest_aruco

    mindist = float("inf")
    closestmarker = None
    if len(robotcoords) == 0:
        return prev_closest_aruco

    for markerCorner, markerID in zip(GLOBAL_ARUCO_CORNERS, GLOBAL_ARUCO_IDS):
        if markerID != ROBOT_ARUCO_ID:
            corners_ = markerCorner.reshape((4, 2))
            marker_center = np.mean(corners_, axis=0)
            dist = np.linalg.norm(robotcoords - marker_center)
            if dist < mindist:
                closestmarker = markerID
                mindist = dist

    prev_closest_aruco = closestmarker
    return closestmarker


""" 
* Function Name: write_live_location_csv
* Input: loc: tuple[float, float], csv_name: str
* Output: None
* Logic: Writes the live location to a csv file
* Example Call: write_live_location_csv((28.7041, 77.1025), "live_location.csv") -> None
"""
def write_live_location_csv(loc, csv_name: str) -> None:
    with open(csv_name, "w") as f:
        f.write(f"lat, lon\n{loc[0]}, {loc[1]}")


""" 
* Function Name: update_qgis_position
* Input: robotcoords: np.ndarray
* Output: None
* Logic: Updates the QGIS position by writing to a csv which QGIS reads
* Example Call: update_qgis_position(robotcoords) -> None
"""
def update_qgis_position(robotcoords: np.ndarray) -> None:

    arucos_gps_data = get_aruco_gps_df()
    nearest_marker = get_nearest_aruco(robotcoords)
    if nearest_marker is not None:
        try:
            coordinate = arucos_gps_data.loc[nearest_marker]
            coordinate.reset_index()
            coordinate = tuple(coordinate)
            write_live_location_csv(coordinate, CLOSEST_ARUCO_OUTPUT_FILE_LOC)
        except:
            return None


""" 
* Function Name: get_robot_pxcoords_update_qgis
* Input: frame: numpy.ndarray
* Output: np.ndarray
* Logic: Updates the QGIS position and returns the pixel coordinates of the robot
* Example Call: get_robot_pxcoords_update_qgis(frame) -> np.ndarray
"""
def get_robot_pxcoords_update_qgis(frame: np.ndarray) -> np.ndarray:

    corners, ids, _ = get_aruco_map_data(frame)
    # epic=np.where(ids==97)[0][0]
    # print(corners,ids)
    # print(corners[epic])
    robot_pxcoords = get_robot_pxcoords(ROBOT_ARUCO_ID, ids, corners)
    update_qgis_position(robot_pxcoords)

    return robot_pxcoords


""" 
* Function Name: initialize_capture
* Input: frames_to_skip: int
* Output: cv2.VideoCapture
* Logic: Initializes the camera capture
* Example Call: initialize_capture() -> cv2.VideoCapture
"""
def initialize_capture(frames_to_skip: int = 30) -> cv2.VideoCapture:

    # Windows uses DirectShow API to access the camera
    if sys.platform == "win32":
        capture = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(CAMERA_ID)

    # ensure 1920x1080 resolution
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # skip the first few frames to allow the camera to adjust to the light
    for _ in range(frames_to_skip):
        ret, frame = capture.read()

    return capture


""" 
* Function Name: listen_and_print
* Input: conn: socket.socket
* Output: None
* Logic: Listens to the robot and prints the data received
* Example Call: listen_and_print(conn) -> None
"""
def listen_and_print(s: socket.socket, conn: socket.socket) -> None:
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


### TYPE HINT TO BE ADDED HERE
""" 
* Function Name: draw_event_labels
* Input: frame: np.ndarray, pts: np.ndarray, labels: list[str]
* Output: np.ndarray
* Logic: Draws the bounding boxes and labels on the frame
* Example Call: draw_event_labels(frame, pts, ["fire", "combat"]) -> np.ndarray
"""

def draw_event_labels(frame: np.ndarray, pts, labels: list[str]) -> np.ndarray:
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


""" 
* Function Name: get_detected_events
* Input: None
* Output: dict
* Logic: Returns the detected events using the predictor
* Example Call: get_detected_events() -> {"A": "fire", "B": "empty", "C": "combat"}
"""


def get_detected_events() -> dict:

    detected_events = predictor.run_predictor(EVENT_FILENAMES)

    # Removing None values (no events)
    detected_events_out = {k: v for k, v in detected_events.items() if v is not None}
    print(detected_events_out)
    return detected_events


""" 
* Function Name: delete_event_images
* Input: None
* Output: None
* Logic: Deletes the event images after use
* Example Call: delete_event_images() -> None
"""
def delete_event_images() -> None:
    for f in EVENT_FILENAMES:
        os.remove(f)


if __name__ == "__main__":

    capture = initialize_capture()

    frame, stopcoords, pxcoords, img_pts = get_robot_coords_and_frame(
        capture, save_images=True
    )

    # Debugging code for custom paths
    if (len(sys.argv) == 3) and (sys.argv[1] == "--command"):
        detected_events = {k: None for k in EVENT_FILENAMES}
        command = sys.argv[2]
        print(f"DEBUG: moving with command: {command}")
    else:
        detected_events = get_detected_events()
        frame = draw_event_labels(frame, img_pts, list(detected_events.values()))

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
    soc, conn = init_connection()

    connect_and_move(soc, conn, command)

    # DEBUG: listen to robot
    # lpt = threading.Thread(target=listen_and_print, args=(soc, conn))
    # lpt.daemon = True
    # lpt.start()

    try:
        cv2.namedWindow("robot_moving", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("robot_moving", (750, 750))
        while True:
            # get a new frame, transform it and get the robots coordinates
            frame, stopcoords, pxcoords, img_pts = get_robot_coords_and_frame(
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

            cv2.imshow("robot_moving", cv2.resize(frame, (750, 750)))
            cv2.moveWindow("robot_moving", 0, 0)
            if cv2.waitKey(1) == ord("q"):
                raise KeyboardInterrupt  # exit on q

    except KeyboardInterrupt:
        cleanup(soc)
        capture.release()
        # lpt.join()
        cv2.destroyAllWindows()
        delete_event_images()

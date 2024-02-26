""" 
* Team Id:          2907
* Author List:      Arnav Rustagi, Abhinav Lodha, Pranjal Rastogi, Subham Jalan
* Filename:         main.py
* Theme:            GeoGuide (GG)
* Functions:        get_aruco_gps_df, get_aruco_detector, cleanup, connect_and_move, init_connection, get_camera_frame,
                    get_events_coords, save_event_images, get_robot_coords_and_frame, get_stopcoords, transform_frame,
                    get_corners_map, get_aruco_map_data, get_robot_pxcoords, get_nearest_aruco, write_live_location_csv,
                    update_qgis_position, get_robot_pxcoords_update_qgis, initialize_capture, listen_and_print, draw_event_labels,
                    get_detected_events
* Global Variables: CAMERA_ID, ARUCO_CORNER_IDS, ROBOT_ARUCO_ID< IDEAL_MAP_SIZE, HOST_IP_ADDRESS, HOST_PORT,
                    CLOSEST_COORDINATE_OUTPUT_FILE_LOC, ARUCO_DATA_PATH, GPS_COORDS_DATA, EVENT_FILENAMES, EVENT_STOP_EXTRA_PIXELS,
                    aruco_df, GLOBAL_ARUCO_CORNERS, GLOBAL_ARUCO_IDS, prev_closest_aruco, prev_pt_A, prev_pt_B, prev_pt_C, prev_pt_D
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
import sys
import ast

# Importing local modules
import predictor
import djikstra

# Importing types for Type Hinting
from collections.abc import Sequence
from typing import Tuple, Union

# Constants:

# CAMERA_ID: Camera ID corresponding to the Lenovo Camera
CAMERA_ID = 1

# ARUCO_CORNER_IDS: IDs of corner AruCos
ARUCO_CORNER_IDS = {4, 5, 6, 7}

# ROBOT_ARUCO_ID: The Aruco ID for the robot
ROBOT_ARUCO_ID = 100

# IDEAL_MAP_SIZE: The map size that should be used
IDEAL_MAP_SIZE = 1080

# HOST_IP_ADDRESS: IP address of the computer (changed everytime)
HOST_IP_ADDRESS = "192.168.67.62"
HOST_PORT = 8002  # Port to listen on

# CLOSEST_ARUCO_OUTPUT_FILE_LOC: For QGIS live tracking, the location of the file where the code should write the closest coordinates
CLOSEST_COORDINATE_OUTPUT_FILE_LOC = "live_location.csv"

# ARUCO_DATA_PATH: This file maps pixel coordinates to AruCo IDs for optimal live tracking
ARUCO_DATA_PATH = "corners.csv"

# GPS_COORDS_DATA: This file contains the mapping between AruCo IDs and GPS coordinates
GPS_COORDS_DATA = "lat_long.csv"

# EVENT_FILENAMES: A list of file names used by the predictor
EVENT_FILENAMES = ["A.png", "B.png", "C.png", "D.png", "E.png"]

# EVENT_STOP_EXTRA_PIXELS: Extra pixels to be added to the bounding box of the event when trying to Interrupt Stop
EVENT_STOP_EXTRA_PIXELS = 35

aruco_df = pd.read_csv(ARUCO_DATA_PATH)  # we use this to load the pre-saved aruco data

# GLOBAL_ARUCO_CORNERS: Corner data of the Arucos w.r.t to our image
GLOBAL_ARUCO_CORNERS = [
    np.array(ast.literal_eval(x.replace("\n", ",").replace(".", ",")))
    for x in aruco_df["corners"].values
]

# GLOBAL_ARUCO_IDS: Which IDs exist.
GLOBAL_ARUCO_IDS = aruco_df["ids"].values

# Used for tracking last closest AruCo marker to robot
prev_closest_aruco = None

# previous iteration's corners, to be used if aruco fails for a certain frame.
prev_pt_A, prev_pt_B, prev_pt_C, prev_pt_D = None, None, None, None

""" 
* Function Name:    get_aruco_gps_df
* Input:            None
* Output:           GPS_COORDS_DATA pandas.DataFrame containing the mapping between AruCo IDs and GPS coordinates
* Logic:            Returns the dataframe containing the mapping between AruCo IDs and GPS coordinates
* Example Call:     get_aruco_gps_df()
"""
def get_aruco_gps_df() -> pd.DataFrame:
    return pd.read_csv(GPS_COORDS_DATA, index_col="id")


""" 
* Function Name:    get_aruco_detector
* Input:            None
* Output:           aruco.ArucoDetector object to enable detection of AruCo markers
* Logic:            Returns the aruco detector
* Example Call:     get_aruco_detector()
"""
def get_aruco_detector() -> aruco.ArucoDetector:
    dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    return detector


""" 
* Function Name:    cleanup
* Input:            s: socket.socket, the socket to be closed
* Output:           None
* Logic:            closes socket `s`
* Example Call:     cleanup(s)
"""
def cleanup(s: socket.socket) -> None:
    s.close()


""" 
* Function Name:    connect_and_move
* Input:            s: socket.socket, conn: socket.socket, path: str
* Output:           None
* Logic:            Checks connection to the robot over `conn` and sends the `path` to it
* Example Call:     connect_and_move(s, conn, "nnn")
"""
def connect_and_move(s: socket.socket, conn: socket.socket, path: str) -> None:

    # check if ACK received
    data = conn.recv(1024)
    data = data.decode("utf-8")

    if data != "ACK_REQ_FROM_ROBOT":
        print("Error in connection")
        cleanup(s)
        sys.exit(1)  # error in connection!

    # Else, send start ping and path to robot
    conn.sendall(str.encode("START\n"))
    conn.sendall(str.encode(path + "\n"))


"""
* Function Name:    init_connection
* Input:            None
* Output:           tuple[socket.socket, socket.socket]: s and conn required to communicate with the robot
* Logic:            Initializes the connection with the robot
* Example Call:     init_connection()
"""
def init_connection() -> tuple[socket.socket, socket.socket]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        s.bind((HOST_IP_ADDRESS, HOST_PORT))

        s.listen()

        # start listening for the robot
        conn, addr = s.accept()

        return s, conn


""" 
* Function Name:        get_camera_frame
* Input:                capture: cv2.VideoCapture object that captures the video from the camera
* Output:               numpy.ndarray representing the captured frame
* Logic:                Returns the frame from the camera
* Example Call:         get_camera_frame(capture)
"""
def get_camera_frame(capture: cv2.VideoCapture) -> np.ndarray:
    ret, frame = capture.read()
    if ret:
        return frame
    else:
        raise Exception("Fatal Camera Error!")


""" 
* Function Name:    get_events_coords
* Input:            side_len: int representing the ideal map size
* Output:           Tuple[numpy.ndarray, ...]. 5-tuple of numpy arrays, 
                    where each array represents the bounding boxes of each 
                    event A, B, C, D and E respectively.
* Logic:            Returns the pixel coordinates (bounding boxes) of the events
* Example Call:     get_events_coords(1080)
"""
def get_events_coords(side_len: int) -> Tuple[np.ndarray, ...]:

    s = side_len
    S = IDEAL_MAP_SIZE

    # crop!
    Apts = (np.array([[940 / S, 1026 / S], [222 / S, 308 / S]]) * s).astype(int)
    Bpts = (np.array([[729 / S, 816 / S], [717 / S, 802 / S]]) * s).astype(int)
    Cpts = (np.array([[513 / S, 601 / S], [725 / S, 811 / S]]) * s).astype(int)
    Dpts = (np.array([[509 / S, 597 / S], [206 / S, 293 / S]]) * s).astype(int)
    Epts = (np.array([[157 / S, 245 / S], [227 / S, 313 / S]]) * s).astype(int)

    return (Apts, Bpts, Cpts, Dpts, Epts)


""" 
* Function Name:    save_event_images
* Input:            frame: numpy.ndarray, pts: Tuple[numpy.ndarray, ...],
                        filenames: list[str]. frame is the image, pts is the bounding boxes 
                        of the events, and filenames is the list of filenames to save the images to.
* Output:           None
* Logic:            Saves the images of the events to the given filenames.
* Example Call:     save_event_images(frame, (Apts, Bpts, Cpts, Dpts, Epts),
                        ["A.png", "B.png", "C.png", "D.png", "E.png"]) -> None
"""
def save_event_images(
    frame: np.ndarray, pts: Tuple[np.ndarray, ...], filenames: list[str]
) -> None:
    for p, f in zip(pts, filenames):
        # clip event out
        event = frame[p[0, 0] : p[0, 1], p[1, 0] : p[1, 1]]
        cv2.imwrite(f, event)


""" 
* Function Name:        get_robot_coords_and_frame
* Input:                capture: cv2.VideoCapture, save_images: bool, where capture is the capture device.
* Output:               frame, stopcoords, pxcoords, and event_coords. frame is the image, stopcoords is the 
                            list of coordinates to "stop" at, pxcoords is the pixel coordinates of the robot, and event_coords
                            are the coordinates of the events.
* Logic:                Returns the frame, stopcoords, pxcoords and event_coords by calling other functions.
* Example Call:         get_robot_coords_and_frame(capture) or get_robot_coords_and_frame(capture, True)
"""
def get_robot_coords_and_frame(
    capture: cv2.VideoCapture, save_images=False
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, Tuple[np.ndarray, ...]]:
    frame = get_camera_frame(capture)
    frame, side_len = transform_frame(frame)  # transform to fit conditions
    event_coords = get_events_coords(side_len)  # get event coordinates from map

    if save_images:
        # good for running our CVmodels later.
        save_event_images(frame, event_coords, EVENT_FILENAMES)

    stopcoords = get_stopcoords(event_coords)
    pxcoords = get_robot_pxcoords_update_qgis(frame)

    return frame, stopcoords, pxcoords, event_coords


""" 
* Function Name:    get_stopcoords
* Input:            event_coords: Tuple[numpy.ndarray, ...] that represents the coordinates of the events.
* Output:           list[numpy.ndarray], that represents the coordinates of the stop points.
* Logic:            Returns the stop coordinates for the robot to stop at the events, based on the
                        event coordinates.
* Example Call:     get_stopcoords(event_coords)
"""
def get_stopcoords(event_coords: Tuple[np.ndarray, ...]) -> list[np.ndarray]:
    stopcoords = []
    for p in event_coords:
        # do some mathematics on every event_coord `p`, then append
        stopcoords.append(
            [
                (p[1, 0] - EVENT_STOP_EXTRA_PIXELS, p[0, 0] - 100),
                (p[1, 1] + EVENT_STOP_EXTRA_PIXELS, p[0, 1] - 100),
            ]
        )

    return stopcoords


""" 
* Function Name:    transform_frame
* Input:            frame: numpy.ndarray, image of the frame
* Output:           tuple[numpy.ndarray, int], tuple that represents the transformed frame 
                        and the side length of the square
* Logic:            Transforms the frame to a square and also perspective transform and then returns 
                        the transformed frame and the side length of the square
* Example Call:     transform_frame(frame)
"""
def transform_frame(frame: np.ndarray) -> tuple[np.ndarray, int]:

    pt_A, pt_B, pt_C, pt_D = get_corners_map(
        frame
    )  # returns updated corners of the map

    # the following if condition will be true only in the beginning.
    if pt_A is None or pt_B is None or pt_C is None or pt_D is None:
        print(f"{pt_A=}\n\n{pt_B=}\n\n{pt_C=}\n\n{pt_D=}")
        raise Exception("Corners not found correctly")

    # calculate width
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    # calculate height
    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    s = min(maxHeight, maxWidth)  # s: side_len

    # point calculation for perspective transforms
    input_pts = np.array([pt_A, pt_B, pt_C, pt_D], dtype=np.float32)
    output_pts = np.array(
        [[0, 0], [0, s - 1], [s - 1, s - 1], [s - 1, 0]], dtype=np.float32
    )

    # perspective transform
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(frame, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    # resize to IDEAL size
    out = out[:s, :s]
    out = cv2.resize(
        out, (IDEAL_MAP_SIZE, IDEAL_MAP_SIZE), interpolation=cv2.INTER_AREA
    )

    return out, IDEAL_MAP_SIZE


""" 
* Function Name:    get_corners_map
* Input:            frame: numpy.ndarray, the camera frame
* Output:           Tuple[numpy.ndarray, ...]
* Logic:            Returns the updated corners of the map (accounts for minute camera movements).
* Example Call:     transform_frame(frame)
"""
def get_corners_map(
    frame: np.ndarray,
) -> Union[Tuple[Tuple[int, ...], ...], Tuple[None, ...]]:

    global prev_pt_A, prev_pt_B, prev_pt_C, prev_pt_D

    # get the corners and ids
    corners, ids, _ = get_aruco_map_data(frame)

    pt_A, pt_B, pt_C, pt_D = None, None, None, None

    for markerCorner, markerID in zip(corners, ids):
        if markerID not in ARUCO_CORNER_IDS:
            continue

        # save the corners
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
* Function Name:    get_aruco_data
* Input:            frame: numpy.ndarray (camera frame), flatten: bool (whether to flatten the ids)
* Output:           tuple[Sequence[cv2.MatLike], cv2.MatLike, Sequence[cv2.MatLike]] where
                        c: Sequence[cv2.MatLike] is the corners, i: cv2.MatLike is the ids, and
                        r: Sequence[cv2.MatLike] is the rejected points
* Logic:            Detects the aruco markers and returns the corners, ids and rejected points that cv2 has given using aruco detector.
* Example Call:     get_aruco_data(frame) or get_aruco_data(frame, True)
"""
def get_aruco_map_data(frame: np.ndarray, flatten: bool = True):
    detector = get_aruco_detector()

    # run the detector
    c, i, r = detector.detectMarkers(frame)

    if len(c) == 0:
        # no frames detected: there was something wrong with the camera
        # show the image received to the user for debugging.
        cv2.imshow("No frame issue", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        raise Exception("No Aruco Markers Found")
    if flatten:
        i = i.flatten()
    return c, i, r


""" 
* Function Name:    get_robot_pxcoords
* Input:            robot_id: int, ids: cv2.MatLike, corners: Sequence[cv2.MatLike]
* Output:           np.ndarray representing the robot's pixel coordinates.
* Logic:            Returns the pixel coordinates of the robot.
* Example Call:     get_robot_pxcoords(100, ids, corners)
"""
def get_robot_pxcoords(robot_id: int, ids, corners: Sequence) -> np.ndarray:
    try:
        # trying to index to get the id
        index = np.where(ids == robot_id)[0][0]
        coords = np.mean(corners[index].reshape((4, 2)), axis=0)
        return np.array([int(x) for x in coords])
    except:
        # robot wasn't found!
        return np.array([])


""" 
* Function Name:    get_nearest_aruco
* Input:            robotcoords: np.ndarray (coordinates of the robot)
* Output:           int representing the ID of the nearest aruco marker or None
* Logic:            Returns the nearest aruco marker to the robot
                        (uses GLOBAL_ARUCO_CORNERS and GLOBAL_ARUCO_IDS)
* Example Call:     get_nearest_aruco(robotcoords)
"""
def get_nearest_aruco(robotcoords: np.ndarray) -> Union[int, None]:

    global prev_closest_aruco

    mindist = float("inf")
    closestmarker = None
    if len(robotcoords) == 0:
        return prev_closest_aruco

    # loop through and find closest aruco marker based on distance
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
* Function Name:    write_live_location_csv
* Input:            loc: tuple[float, float], csv_name: str
* Output:           None
* Logic:            Writes the live location to a csv file with name `csv_name` and location as `loc`. This is for QGIS.
* Example Call:     write_live_location_csv((28.7041, 77.1025), "live_location.csv")
"""
def write_live_location_csv(loc, csv_name: str) -> None:
    with open(csv_name, "w") as f:
        f.write(f"lat, lon\n{loc[0]}, {loc[1]}")


""" 
* Function Name:    update_qgis_position
* Input:            robotcoords: np.ndarray (the coordinates of the robot)
* Output:           None
* Logic:            Updates the QGIS position by writing to the csv which QGIS reads
* Example Call:     update_qgis_position(robotcoords)
"""
def update_qgis_position(robotcoords: np.ndarray) -> None:

    arucos_gps_data = get_aruco_gps_df()
    nearest_marker = get_nearest_aruco(robotcoords)
    if nearest_marker is not None:
        try:
            # write it down to the csv
            coordinate = arucos_gps_data.loc[nearest_marker]
            coordinate.reset_index()
            coordinate = tuple(coordinate)
            write_live_location_csv(coordinate, CLOSEST_COORDINATE_OUTPUT_FILE_LOC)
        except:
            return None


""" 
* Function Name:    get_robot_pxcoords_update_qgis
* Input:            frame: numpy.ndarray (camra frame)
* Output:           np.ndarray that represents the pixel coordinates of the robot
* Logic:            Updates the QGIS position and returns the pixel coordinates of the robot
* Example Call:     get_robot_pxcoords_update_qgis(frame)
"""
def get_robot_pxcoords_update_qgis(frame: np.ndarray) -> np.ndarray:

    # get aruco data from frame
    corners, ids, _ = get_aruco_map_data(frame)

    # get pixel coordinates
    robot_pxcoords = get_robot_pxcoords(ROBOT_ARUCO_ID, ids, corners)

    # update qgis
    update_qgis_position(robot_pxcoords)

    return robot_pxcoords


""" 
* Function Name:    initialize_capture
* Input:            frames_to_skip: int (how many frames to skip before the camera warms up)
* Output:           cv2.VideoCapture (the camera capture device)
* Logic:            Initializes the camera capture and discards the first `frames_to_skip` frames.
* Example Call:     initialize_capture()
"""
def initialize_capture(frames_to_skip: int = 40) -> cv2.VideoCapture:

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
* Function Name:    listen_and_print
* Input:            s: socket.socket, conn: socket.socket (sockets for robot connection)
* Output:           None
* Logic:            Listens to the robot and prints the data received, also exists the 
                        program if `terminate` is received.
* Example Call:     listen_and_print(s, conn). But, usually run this on a separate thread!
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


""" 
* Function Name:    draw_event_labels
* Input:            frame: np.ndarray (camera frame), 
                    pts: np.ndarray (pts to draw events at), 
                    labels: list[str] (which index is which event!)
* Output:           np.ndarray (frame with drawed events)
* Logic:            Draws the bounding boxes and adds labels on the frame
* Example Call:     draw_event_labels(frame, pts, ["fire", "combat"])
"""
def draw_event_labels(frame: np.ndarray, pts, labels: list[str]) -> np.ndarray:
    for p, l in zip(pts, labels):
        # add rectangle
        frame = cv2.rectangle(
            frame, (p[1, 0], p[0, 0]), (p[1, 1], p[0, 1]), (0, 255, 0), 2
        )
        # add text
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
* Function Name:    get_detected_events
* Input:            None
* Output:           dict containing the detected events
* Logic:            Returns the detected events using the predictor model
* Example Call:     get_detected_events()
"""
def get_detected_events() -> dict:

    # runs the predictor model on the filenames
    detected_events = predictor.run_predictor(EVENT_FILENAMES)

    # Removing None values (no events) and swapping the values for correct terminal output
    terminal_output_events_swap = {
        "combat": "Combat",
        "destroyed_buildings": "Destroyed Buildings",
        "fire": "Fire",
        "humanitarian_aid": "Humanitarian Aid and rehabilitation",
        "military_vehicles": "Military Vehicles",
    }
    detected_events_out = {
        k: terminal_output_events_swap[v]
        for k, v in detected_events.items()
        if v is not None
    }
    print(detected_events_out)
    return detected_events


""" 
* Function Name:    delete_event_images
* Input:            None
* Output:           None
* Logic:            Deletes the event images after use.
* Example Call:     delete_event_images()
"""
def delete_event_images() -> None:
    for f in EVENT_FILENAMES:
        os.remove(f)


""" 
Main function, called when the program starts.
"""
if __name__ == "__main__":

    # ensure the file for coordinate output exists so that QGIS can read it
    if not os.path.exists(CLOSEST_COORDINATE_OUTPUT_FILE_LOC):
        with open(CLOSEST_COORDINATE_OUTPUT_FILE_LOC, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["lat", "lon"])

    # start capture device
    capture = initialize_capture()

    # get robot coordinates and frame once
    frame, stopcoords, pxcoords, img_pts = get_robot_coords_and_frame(
        capture, save_images=True
    )

    # Debugging code for custom paths
    if (len(sys.argv) == 3) and (sys.argv[1] == "--command"):
        detected_events = {k: None for k in EVENT_FILENAMES}
        command = sys.argv[2]
        print(f"DEBUG: moving with command: {command}")
    else:
        # else, detect the events and get the path using djikstra
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

    # start connection to robot after getting the path
    soc, conn = init_connection()

    # send movement command
    connect_and_move(soc, conn, command)

    # DEBUG: uncomment the following to listen to the robot
    # lpt = threading.Thread(target=listen_and_print, args=(soc, conn))
    # lpt.daemon = True
    # lpt.start()

    try:
        # show robot moving window
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
                    ):  # if the robot is at the event, send ISTOP
                        conn.sendall(str.encode("ISTOP\n"))

            cv2.imshow("robot_moving", cv2.resize(frame, (750, 750)))
            cv2.moveWindow("robot_moving", 0, 0)
            if cv2.waitKey(1) == ord("q"):
                raise KeyboardInterrupt  # exit on q

    except KeyboardInterrupt:
        cleanup(soc)
        capture.release()
        # lpt.join() # uncomment this if you were listening to the robot
        cv2.destroyAllWindows()
        delete_event_images()

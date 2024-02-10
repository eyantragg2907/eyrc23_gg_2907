"""
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 4A of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
"""

# Team ID:			[ GG_2907 ]
# Author List:		[ Arnav Rustagi, Subham Jalan, Pranjal Rastogi ]
# Filename:			task_4a.py


####################### IMPORT MODULES #######################
import os
import cv2
import numpy as np
import sys
import logging
import threading

# suppress tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

import tensorflow as tf

##############################################################


################# ADD UTILITY FUNCTIONS HERE #################

# constants
CLASS_MAP = [
    "combat",
    "destroyed_buildings",
    "fire",
    "human_aid_rehabilitation",
    "military_vehicles",
]
FILENAMES = ["A.png", "B.png", "C.png", "D.png", "E.png"]

MODEL_PATH = "lr_net_arnav_1000.tf"  # model.tf should be a folder containing the model

CAMERA_ID = 1

ARUCO_REQD_IDS = {4, 5, 6, 7}  # the ids of the corners!

IDEAL_FRAME_SIZE = 1080

# global variables
model = None


# functions
def load_model() -> None:
    """
    Purpose:
    ---
    This function loads the model from the model's saved path into a global variable named `model`

    Input Arguments:
    ---
    None

    Returns:
    ---
    None
    """

    global model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # type: ignore
    if model is None:
        raise Exception("Model not found at path")
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # type: ignore
        metrics=["accuracy"],
    )


def classify_event(imagepath: str) -> str:
    """
    Purpose:
    ---
    This function classifies the event from the image at the given path

    Arguments:
    ---
    `imagepath` : { str }
        the path of the image to classify

    Returns:
    ---
    `event` : { str }
        the event
    """

    if model is None:
        raise Exception("Model is not loaded")

    """
    # the model is trained on 75x75 images
    img = tf.keras.preprocessing.image.load_img(imagepath, target_size=(75, 75))  # type: ignore
    """
    img = cv2.imread(imagepath)
    img = cv2.resize(img, (75, 75))
    img = np.array(img, dtype=np.float32)
    img = tf.expand_dims(img, axis=0)

    # predict the event
    prediction = model.predict(img, verbose=0)
    print(prediction)
    predicted_class = np.argmax(prediction[0], axis=-1)

    event = CLASS_MAP[predicted_class]

    return event


def get_aruco_detector() -> cv2.aruco.ArucoDetector:
    """
    Purpose:
    ---
    This function initializes and returns the aruco detector

    Arguments:
    ---
    None

    Returns:
    ---
    `detector` : { cv2.aruco.ArucoDetector }
        the aruco detector
    """

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    return detector


def get_clean_video_frame(frames_to_skip=100) -> np.ndarray:
    """
    Purpose:
    ---
    This function returns a clean frame from the camera. The frame is cleaned by skipping the first few frames.

    Arguments:
    ---
    `frames_to_skip` : { int }
        the number of frames to skip, default 100

    Returns:
    ---
    `frame` : { numpy.ndarray }
        the clean frame
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

    # now, its a clean frame
    ret, frame = capture.read()

    if not ret:
        raise Exception("Fatal camera error")

    capture.release()

    return frame


def get_aruco_data(frame):
    detector = get_aruco_detector()

    c, i, r = detector.detectMarkers(frame)

    if len(c) == 0:
        raise Exception("No Aruco Markers Found")

    return c, i.flatten(), r


def get_points_from_aruco(frame):
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

    return pt_A, pt_B, pt_C, pt_D


def transform_frame_based_on_markers(frame: np.ndarray) -> tuple[np.ndarray, int]:
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
    `side_length` : { int }
        the length of the frame
    """

    pt_A, pt_B, pt_C, pt_D = get_points_from_aruco(frame)

    if pt_A is None or pt_B is None or pt_C is None or pt_D is None:
        raise Exception("Corners not detected properly")

    # get the side length of the frame
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    max_width = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    max_height = max(int(height_AB), int(height_CD))

    s = min(max_height, max_width)

    # transform the perspective
    input_pts = np.array([pt_A, pt_B, pt_C, pt_D], dtype=np.float32)
    output_pts = np.array(
        [[0, 0], [0, s - 1], [s - 1, s - 1], [s - 1, 0]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(frame, M, (max_width, max_height), flags=cv2.INTER_LINEAR)
    out = out[:s, :s]

    # resize the frame to 1080x1080
    out = cv2.resize(
        out, (IDEAL_FRAME_SIZE, IDEAL_FRAME_SIZE), interpolation=cv2.INTER_AREA
    )

    return out, IDEAL_FRAME_SIZE


def get_image_pts_from_frame(side_len: int) -> tuple:
    """
    Purpose:
    ---
    Gets the points of the events from the frame, so that we can crop the events later.

    Arguments:
    ---
    `side_len` : { int }
        the length of the side of the frame

    Returns:
    ---
    `Apts` : { numpy.ndarray }
        the points of the A event
    `Bpts` : { numpy.ndarray }
        the points of the B event
    `Cpts` : { numpy.ndarray }
        the points of the C event
    `Dpts` : { numpy.ndarray }
        the points of the D event
    `Epts` : { numpy.ndarray }
        the points of the E event
    """

    # the points are hardcoded, as they are constant on a 1080x1080 frame
    Apts = (
        np.array([[940 / side_len, 1026 / side_len], [222 / side_len, 308 / side_len]])
        * side_len
    ).astype(int)
    Bpts = (
        np.array([[729 / side_len, 816 / side_len], [717 / side_len, 802 / side_len]])
        * side_len
    ).astype(int)
    Cpts = (
        np.array([[513 / side_len, 601 / side_len], [725 / side_len, 811 / side_len]])
        * side_len
    ).astype(int)
    Dpts = (
        np.array([[509 / side_len, 597 / side_len], [206 / side_len, 293 / side_len]])
        * side_len
    ).astype(int)
    Epts = (
        np.array([[157 / side_len, 245 / side_len], [227 / side_len, 313 / side_len]])
        * side_len
    ).astype(int)

    return (Apts, Bpts, Cpts, Dpts, Epts)


def unsharp_mask(image: np.ndarray) -> np.ndarray:
    """
    Purpose:
    ---
    Applies the unsharp mask to the image, for sharpening the image.

    Arguments:
    ---
    `image` : { numpy.ndarray }
        the image to sharpen

    Returns:
    ---
    `image` : { numpy.ndarray }
        the sharpened image
    """

    gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
    return cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 0)


def get_event_images(frame: np.ndarray, pts: tuple) -> list[np.ndarray]:
    """
    Purpose:
    ---
    Gets the event images from the frame (crops them out!)

    Arguments:
    ---
    `frame` : { numpy.ndarray }
        the frame to crop the events from
    `pts` : { tuple }
        the points of the events

    Returns:
    ---
    `events` : { list }
        the event images
    """

    events = []
    for p, f in zip(pts, FILENAMES):
        event = frame[p[0, 0] : p[0, 1], p[1, 0] : p[1, 1]]
        # unsharp mask is for preprocessing the image
        event = unsharp_mask(event)
        cv2.imwrite(f, event)  # required for classification later
        events.append(event)
    return events


def cleanup() -> None:
    """
    Purpose:
    ---
    Deletes all the files generated for cleaning up.

    Arguments:
    ---
    None

    Returns:
    ---
    None
    """

    for f in FILENAMES:
        os.remove(f)


def process_and_get_events_from_frame(
    frame: np.ndarray,
) -> tuple[np.ndarray, tuple, list[np.ndarray]]:
    """
    Purpose:
    ---
    Gets the events from the frame

    Arguments:
    ---
    `frame` : { numpy.ndarray }
        the frame to process

    Returns:
    ---
    `frame` : { numpy.ndarray }
        the frame after processing
    `image_pts` : { list }
        the points of the events
    `events` : { list }
        the events
    """
    frame, side_len = transform_frame_based_on_markers(frame)
    image_pts = get_image_pts_from_frame(side_len)
    events = get_event_images(frame, image_pts)

    return frame, image_pts, events


def classify_and_get_labels() -> tuple[list, dict]:
    """
    Purpose:
    ---
    Classifies the events and returns the labels

    Arguments:
    ---
    None

    Returns:
    ---
    `labels` : { list }
        the labels of the events
    `identified_labels` : { dictionary }
        the dictionary containing the labels of the events
    """

    identified_labels = {}
    global filenames
    labels = []

    # classify the events
    for key, img in zip("ABCDE", FILENAMES):
        label = classify_event(img)
        labels.append(label)
        identified_labels[key] = label

    return labels, identified_labels


def add_rects_to_frame(frame: np.ndarray, pts: list, labels: list) -> np.ndarray:
    """
    Purpose:
    ---
    Adds rectangles amd labels to the frame

    Arguments:
    ---
    `frame` : { numpy.ndarray }
        the frame to display
    `pts` : { list }
        the points of the events
    `labels` : { list }
        the labels of the events

    Returns:
    ---
    `frame` : { numpy.ndarray }
        the frame with the rectangles and labels
    """
    for p, l in zip(pts, labels):
        frame = cv2.rectangle(
            frame, (p[1, 0], p[0, 0]), (p[1, 1], p[0, 1]), (0, 255, 0), 2
        )
        frame = cv2.putText(
            frame,
            str(l),
            (p[1, 0], p[0, 0] + 120),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    return frame


def show_feed(frame: np.ndarray, pts: list, labels: list) -> None:
    """
    Purpose:
    ---
    Displays the video feed with the identified labels. This is a blocking function.

    Arguments:
    ---
    `frame` : { numpy.ndarray }
        the frame to display
    `pts` : { list }
        the points of the events
    `labels` : { list }
        the labels of the events

    Returns:
    ---
    None
    """
    frame = add_rects_to_frame(frame, pts, labels)

    cv2.namedWindow("Arena Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Arena Feed", (750, 750))
    frametoshow = cv2.resize(frame, (750, 750))

    cv2.imshow("Arena Feed", frametoshow)

    while True:
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break


##############################################################


def task_4a_return():
    """
    Purpose:
    ---
    Only for returning the final dictionary variable

    Arguments:
    ---
    You are not allowed to define any input arguments for this function. You can
    return the dictionary from a user-defined function and just call the
    function here

    Returns:
    ---
    `identified_labels` : { dictionary }
        dictionary containing the labels of the events detected
    """
    identified_labels = {}

    ##############	ADD YOUR CODE HERE	##############
    frame = get_clean_video_frame() if len(sys.argv) < 2 else cv2.imread("test.jpg")
    cv2.imwrite("vis.png", frame)

    frame, pts, _ = process_and_get_events_from_frame(frame)

    load_model()
    labels, identified_labels = classify_and_get_labels()

    thread_func = threading.Thread(target=show_feed, args=(frame, pts, labels))
    thread_func.start()

    cleanup()  # delete all the files generated

    ##################################################
    return identified_labels


###############	Main Function ################
if __name__ == "__main__":
    identified_labels = task_4a_return()
    print(identified_labels)

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
from datetime import datetime
import tensorflow as tf
import threading

##############################################################


################# ADD UTILITY FUNCTIONS HERE #################

"""
You are allowed to add any number of functions to this code.
"""

# constants
CLASS_MAP = [
    "combat",
    "destroyed_buildings",
    "fire",
    "human_aid_rehabilitation",
    "military_vehicles",
]
FILENAMES = ["A.png", "B.png", "C.png", "D.png", "E.png"]

MODEL_PATH = "model.tf"  # model.tf should be a folder containing the model

CAMERA_ID = 1

ARUCO_REQD_IDS = {4, 5, 6, 7}  # the ids of the corners!

IDEAL_FRAME_SIZE = 1080

model = None


def load_model():
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
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if model is None:
        raise Exception("Model not found at path")
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


def classify_event(imagepath):
    if model is None:
        raise Exception("Model is not loaded")

    img = tf.keras.preprocessing.image.load_img(imagepath, target_size=(75, 75))
    img = np.array(img, dtype=np.float32)
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0], axis=-1)

    event = CLASS_MAP[predicted_class]

    return event


def get_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    return detector


def get_clean_video_frame(frames_to_skip=100):
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

    # clean frame
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
        top_left, top_right, bottom_right, bottom_left = corners

        top_right = list(map(int, top_right))
        bottom_right = list(map(int, bottom_right))
        bottom_left = list(map(int, bottom_left))
        top_left = list(map(int, top_left))

        if markerID == 5:
            pt_A = top_left
        elif markerID == 6:
            pt_B = bottom_right
        elif markerID == 7:
            pt_C = bottom_left
        elif markerID == 4:
            pt_D = top_right

    return pt_A, pt_B, pt_C, pt_D


def transform_frame_based_on_markers(frame):
    pt_A, pt_B, pt_C, pt_D = get_points_from_aruco(frame)

    if pt_A is None or pt_B is None or pt_C is None or pt_D is None:
        raise Exception("Corners not detected properly")

    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    max_width = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    max_height = max(int(height_AB), int(height_CD))

    s = min(max_height, max_width)
    input_pts = np.array([pt_A, pt_B, pt_C, pt_D], dtype=np.float32)
    output_pts = np.array(
        [[0, 0], [0, s - 1], [s - 1, s - 1], [s - 1, 0]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(frame, M, (max_width, max_height), flags=cv2.INTER_LINEAR)
    out = out[:s, :s]
    out = cv2.resize(
        out, (IDEAL_FRAME_SIZE, IDEAL_FRAME_SIZE), interpolation=cv2.INTER_AREA
    )

    return out, IDEAL_FRAME_SIZE


def get_image_pts_from_frame(side_len):
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


def unsharp_mask(image):
    gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
    return cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 0)


def get_event_images(frame, pts):
    events = []
    for p, f in zip(pts, FILENAMES):
        event = frame[p[0, 0] : p[0, 1], p[1, 0] : p[1, 1]]
        # unsharp mask is for preprocessing the image
        event = unsharp_mask(event)
        cv2.imwrite(f, event)
        events.append(event)
    return events


def cleanup():
    # delete all the files
    for f in FILENAMES:
        os.remove(f)


def get_events_from_frame(frame):
    frame, side_len = transform_frame_based_on_markers(frame)
    image_pts = get_image_pts_from_frame(side_len)
    events = get_event_images(frame, image_pts)

    return frame, image_pts, events


def classify_and_get_labels(identified_labels):
    global filenames
    labels = []
    for key, img in zip("ABCDE", FILENAMES):
        label = classify_event(img)
        labels.append(label)
        identified_labels[key] = label

    return labels, identified_labels


def get_frame_with_rects(frame, pts, labels):
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


def show_feed(frame, pts, labels):
    frame = get_frame_with_rects(frame, pts, labels)

    cv2.namedWindow("Arena Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Arena Feed", (500, 500))
    frametoshow = cv2.resize(frame, (500, 500))

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
    frame = get_clean_video_frame()

    frame, pts, _ = get_events_from_frame(frame)

    labels, identified_labels = classify_and_get_labels(identified_labels)

    thread_func = threading.Thread(target=show_feed, args=(frame, pts, labels))
    thread_func.start()

    ##################################################
    return identified_labels


###############	Main Function ################
if __name__ == "__main__":
    identified_labels = task_4a_return()
    print(identified_labels)

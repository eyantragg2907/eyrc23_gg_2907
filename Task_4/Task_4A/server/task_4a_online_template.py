import cv2
import numpy as np
import tensorflow as tf
import sys
from datetime import datetime

import torch
import torchvision

import torch
import torchvision

DEBUG = True

classmap = [
    "combat",
    "destroyed_buildings",
    "fire",
    "human_aid_rehabilitation",
    "military_vehicles",
]


def model_load():
    # -->LOL<<--[[{{LOAD_MODEL}}]]-->>LOL<<--
    pass


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

filenames = "A__{{REPLACE THIS}}__.png B__{{REPLACE THIS}}__.png C__{{REPLACE THIS}}__.png D__{{REPLACE THIS}}__.png E__{{REPLACE THIS}}__.png".split()


def classify_event(imagepath):
    global classmap
    # -->LOL<<--[[{{CLASSIFY_EVENT}}]]-->>LOL<<--

    pass


def get_events(frame):
    frame, side = transform_frame(frame)
    pts = get_pts_from_frame(frame, side)
    events = get_event_images(frame, pts)

    return frame, pts, events


def transform_frame(frame):
    pt_A, pt_B, pt_C, pt_D = get_points_from_aruco(frame)

    if pt_A is None or pt_B is None or pt_C is None or pt_D is None:
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
    out = cv2.resize(out, (1080, 1080), interpolation=cv2.INTER_AREA)
    
    return out, 1080  # why is this hardcoded?


def increase_brightness(img, value=100):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img


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

        if DEBUG:
            print(f"{markerID=}\n")

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


def get_aruco_data(frame):
    global detector
    c, i, r = detector.detectMarkers(frame)

    if len(c) == 0:
        raise Exception("No Aruco Markers Found")
    return c, i.flatten(), r


def get_pts_from_frame(frame, s):
    S = 1080
    Apts = (np.array([[940 / S, 1026 / S], [222 / S, 308 / S]]) * s).astype(int)
    Bpts = (np.array([[729 / S, 816 / S], [717 / S, 802 / S]]) * s).astype(int)
    Cpts = (np.array([[513 / S, 601 / S], [725 / S, 811 / S]]) * s).astype(int)
    Dpts = (np.array([[509 / S, 597 / S], [206 / S, 293 / S]]) * s).astype(int)
    Epts = (np.array([[157 / S, 245 / S], [227 / S, 313 / S]]) * s).astype(int)

    return (Apts, Bpts, Cpts, Dpts, Epts)


def get_event_images(frame, pts):
    global filenames
    events = []
    for p, f in zip(pts, filenames):
        event = frame[p[0, 0] : p[0, 1], p[1, 0] : p[1, 1]]
        event = unsharp_mask(event)
        cv2.imwrite(f, event)
        events.append(event)
    return events

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
    return cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 0)


def add_rects_labels(frame, pts, labels):
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


##############################################################

def get_clean_video_frame(frames_to_skip=100):
    cap = None
    if sys.platform == "win32":
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    else:
        cap = cv2.VideoCapture(1)

    for _ in range(frames_to_skip):
        ret, frame = cap.read()

    ret, frame = cap.read()

    if not ret:
        raise Exception("No frame found")

    cap.release()

    return frame

def initialise_identified_labels(identified_labels):

    global filenames
    labels = []
    for key, img in zip("ABCDE", filenames):
        label = classify_event(img)
        labels.append(label)
        identified_labels[key] = label

    return labels, identified_labels

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

    frame, pts, events = get_events(frame)

    
    labels, identified_labels = initialise_identified_labels(identified_labels)
    
    frame = add_rects_labels(frame, pts, labels)

    frametoshow = cv2.resize(frame, (480, 480))

    cv2.imwrite("arena_with_labels__{{REPLACE THIS}}__.jpg", frametoshow)

    return identified_labels


###############	Main Function	#################
if __name__ == "__main__":
    identified_labels = task_4a_return()
    print(identified_labels)

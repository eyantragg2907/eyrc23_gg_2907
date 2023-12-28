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

# Team ID:			[ 2907 ]
# Author List:		[ Arnav Rustagi, Subham Jalan]
# Filename:			task_4a.py


####################### IMPORT MODULES #######################

import cv2
import numpy as np
import tensorflow as tf
import sys
from datetime import datetime

##############################################################


################# ADD UTILITY FUNCTIONS HERE #################c
DEBUG = True

classmap = [
    "combat",
    "destroyed_buildings",
    "fire",
    "human_aid_rehabilitation",
    "military_vehicles",
]

filenames = "A.png B.png C.png D.png E.png".split()

modelpath = None
model = None
detector = None

if len(sys.argv) < 1:
    modelpath = "model.tf"

    model = tf.keras.models.load_model(modelpath, compile=False)
    if model is None:
        raise Exception("Model not found at path")
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

if True:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)


def get_clean_video_frame(frames_to_skip=100):
    video = None
    if sys.platform == "win32":
        video = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    else:
        video = cv2.VideoCapture(1)

    frames_to_skip = 100
    for _ in range(frames_to_skip):
        ret, frame = video.read()

    ret, frame = video.read()
    if not ret:
        raise Exception("No frame found")

    return frame


def classify_event(image):
    global classmap, model
    """
    ADD YOUR CODE HERE
    """
    img = tf.keras.preprocessing.image.load_img(image, target_size=(75, 75))
    """
    if DEBUG:
        addr = f"temp_tomodelafterresize_{str(datetime.now().timestamp()).replace('.', '-')}.jpg"
        cv2.imwrite(addr, img.numpy())
    """

    img = np.array(img, dtype=np.float32)
    # print(img.shape)
    img = tf.expand_dims(img, axis=0)
    # print(img.shape)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0], axis=-1)

    event = classmap[predicted_class]

    return event


def get_events(frame):
    frame, side = transform_frame(frame)
    pts = get_pts_from_frame(frame, side)
    events = get_event_images(frame, pts)

    return frame, pts, events


def transform_frame(frame):
    pt_A, pt_B, pt_C, pt_D = get_points_from_aruco(frame)

    if DEBUG:
        print(f"{pt_A=}\n\n{pt_B=}\n\n{pt_C=}\n\n{pt_D=}")

    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    s = min(maxHeight, maxWidth)
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0], [0, s - 1], [s - 1, s - 1], [s - 1, 0]])
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(frame, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    out = out[:s, :s]
    out = cv2.resize(out, (1080, 1080), interpolation=cv2.INTER_AREA)

    return out, s


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
    S = 937
    Apts = (np.array([[811 / S, 887 / S], [194 / S, 269 / S]]) * s).astype(int)
    Bpts = (np.array([[628 / S, 705 / S], [620 / S, 695 / S]]) * s).astype(int)
    Cpts = (np.array([[444 / S, 519 / S], [628 / S, 701 / S]]) * s).astype(int)
    Dpts = (np.array([[440 / S, 516 / S], [183 / S, 260 / S]]) * s).astype(int)
    Epts = (np.array([[136 / S, 212 / S], [200 / S, 276 / S]]) * s).astype(int)

    return (Apts, Bpts, Cpts, Dpts, Epts)


def get_event_images(frame, pts):
    global filenames
    events = []
    for p, f in zip(pts, filenames):
        event = frame[p[0, 0] : p[0, 1], p[1, 0] : p[1, 1]]
        cv2.imwrite(f, event)
        events.append(event)
    return events


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


def initialise_identified_labels():
    global filenames
    labels = []
    for key, img in zip("ABCDE", filenames):
        label = classify_event(img)
        labels.append(label)
        identified_labels[key] = label


def show_feed_and_release_video(video, frame):
    video.release()
    cv2.namedWindow("Arena Feed", cv2.WINDOW_NORMAL)
    frametoshow = cv2.resize(frame, (960, 960))
    cv2.imshow("Arena Feed", frametoshow)
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
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
    frame = None
    if len(sys.argv) > 1 and "no-video" in sys.argv:
        frame = cv2.imread("arena.jpeg")
    else:
        frame = get_clean_video_frame()

    if len(sys.argv) > 1:
        if "aruco-only" in sys.argv:
            frame, pts, events = get_events(frame)
            cv2.imwrite("frame.png", frame)
            cv2.imshow("frame", frame)
            cv2.waitKey(0)
        elif "return_frame" in sys.argv:
            cv2.imwrite("frame.jpg", frame)
    frame, pts, events = get_events(frame)

    identified_labels = intialise_identified_labels()
    frame = add_rects_labels(frame, pts, labels)

    show_feed_and_release_video(video, frame)
    ##################################################
    return identified_labels


###############	Main Function	#################
if __name__ == "__main__":
    identified_labels = task_4a_return()
    print(identified_labels)

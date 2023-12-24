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


################# ADD UTILITY FUNCTIONS HERE #################

classmap = [
    "combat",
    "destroyedbuilding",
    "fire",
    "humanitarianaid",
    "militaryvehicles",
]
modelpath = r"model.h5"
model = tf.keras.models.load_model(modelpath, compile=False)
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


def classify_event(image):
    global classmap, model
    """
    ADD YOUR CODE HERE
    """
    img = tf.image.resize(image, (180, 180))
    addr = f"temp_{str(datetime.now().timestamp()).replace('.', '-')}.jpg"
    wr = cv2.imwrite(addr, img.numpy())
    print(wr)
    img = np.array(img, dtype=np.float32)
    img = tf.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0], axis=-1)

    event = classmap[predicted_class]

    return event


def modify_and_get_events(frame):
    frame, side = transform_frame(frame)
    pts = get_pts_from_frame(frame, side)
    events = get_event_images(frame, pts)
    modified_frame = draw_rects(frame, pts)

    return modified_frame, events


def transform_frame(frame):
    pt_A, pt_B, pt_C, pt_D = get_points_from_aruco(frame)

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
    # out = cv2.resize(out, (1024,1024), interpolation = cv2.INTER_AREA)

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
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

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
    events = []
    for p in pts:
        events.append(frame[p[0, 0] : p[0, 1], p[1, 0] : p[1, 1]])
    return events


def draw_rects(frame, pts):
    for p in pts:
        frame = cv2.rectangle(
            frame, (p[1, 0], p[0, 0]), (p[1, 1], p[0, 1]), (0, 255, 0), 2
        )
    return frame


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
    video = cv2.VideoCapture(1)
    num_of_frames_skip = 100
    for i in range(num_of_frames_skip):
        ret, frame = video.read()
    c = 0
    while True:
        _, frame = video.read()
        frame = increase_brightness(frame, value=30)
        if c == 0:
            cv2.imwrite("firstframe.jpg", frame)
        if len(sys.argv) > 1:
            frame = cv2.imread("arena.jpg")
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        frame, events = modify_and_get_events(frame)

        for key, img in zip("ABCDE", events):
            identified_labels[key] = classify_event(img)
        cv2.imshow("Arena Feed", frame)
        c += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            video.release()
            cv2.destroyAllWindows()
            break
    ##################################################
    return identified_labels


###############	Main Function	#################
if __name__ == "__main__":
    identified_labels = task_4a_return()
    print(identified_labels)

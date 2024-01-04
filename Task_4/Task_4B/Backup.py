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
# Author List:		[ Arnav Rustagi, Abhinav Lodha]
# Filename:			task_4b.py


####################### IMPORT MODULES #######################

import cv2
import cv2.aruco as aruco
import numpy as np
import sys
from datetime import datetime
import csv
import time
import pandas as pd
import socket
# import signal		

##############################################################
OUT_FILE_LOC = "live_location.csv"
arucolat_long=pd.read_csv("lat_long.csv",index_col="id")
dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

ip = "192.168.54.92"     # Enter IP address of laptop after connecting it to WIFI hotspot
commandsent = 0
command = "nnrnlnrnrnnrnnlnn"
command = "nnnn"
command = "nnrnlnrnrnnrnnlnn"
################# ADD UTILITY FUNCTIONS HERE #################


def cleanup(s):
    s.close()
    print("cleanup done")

    
def send_to_robot(s,conn):
    data = conn.recv(1024)
    print(data)
    print(command)
    conn.sendall(str.encode(str(command)))
    time.sleep(1)
    cleanup(s)
    
def give_s_conn():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((ip, 8002))
        s.listen()
        conn, addr = s.accept()
        print(f"Connected by {addr}")
        return s,conn

def get_frame(video):
    ret, frame = video.read()
    print(frame.shape)
    frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame

def update_position(video):
    frame = get_frame(video)
    frame, side = transform_frame(frame)
    get_robot_coords(frame)
    return frame

def transform_frame(frame):
    pt_A, pt_B, pt_C, pt_D = get_points_from_aruco(frame)

    if pt_A is None or pt_B is None or pt_C is None or pt_D is None: raise Exception("Corners not found correctly")
    print(f"{pt_A=}\n\n{pt_B=}\n\n{pt_C=}\n\n{pt_D=}")

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
    # out = cv2.resize(out, (1024,1024), interpolation = cv2.INTER_AREA)

    return out, s

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
    c, i, r = detector.detectMarkers(frame)

    if len(c) == 0:
        raise Exception("No Aruco Markers Found")
    return c, i.flatten(), r

def get_pxcoords(id,ids,corners):
    try:
        index= np.where(ids == id)[0][0]
        coords = np.mean(corners[index].reshape((4, 2)), axis=0)
        return np.array([ int(x) for x in coords])
    except:
        return []

prevclosestmarker = None
def get_nearestmarker(id, ids, corners):
    global prevclosestmarker
    mindist = float("inf")
    closestmarker = None
    coords1 = get_pxcoords(id, ids, corners)
    print(coords1)
    if len(coords1) == 0:
        return prevclosestmarker
    for markerCorner, markerID in zip(corners, ids):
        if markerID != 97:
            corners = markerCorner.reshape((4, 2))
            marker_center = np.mean(corners, axis=0)
            dist = np.linalg.norm(coords1 - marker_center)
            if dist < mindist:  # type: ignore
                closestmarker = markerID
                mindist = dist
    prevclosestmarker = closestmarker
    return closestmarker

def get_robot_coords(frame):
    corners, ids, _ = get_aruco_data(frame)    
    NearestMarker = get_nearestmarker(97,ids,corners)
    if NearestMarker == None:
        return None
    try:
        coordinate = arucolat_long.loc[NearestMarker]
        print(f'Nearest Marker ID: {NearestMarker} and Nearest marker lat_long: {coordinate}')
        write_csv(coordinate, OUT_FILE_LOC)
        return coordinate
    except:
        print("NO lat long from this marker!!")
        return None

def write_csv(loc, csv_name):

    # open csv (csv_name)
    # write column names "lat", "lon"
    # write loc ([lat, lon]) in respective columns
    with open(csv_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon"])
        writer.writerow(loc)



###############	Main Function	#################
if __name__ == "__main__":
    if sys.platform == "win32":
        video = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    else:
        video = cv2.VideoCapture(0)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    num_of_frames_skip = 100
    for i in range(num_of_frames_skip):
        ret, frame = video.read()
    while True:
        frame = update_position(
            video
        )  # gets the new frame updates, transforms it, gets the robot coords
        if commandsent == 0:
            s,conn=give_s_conn()
            send_to_robot(s,conn)
            commandsent = 1
        # corners, ids, rejected = detector.detectMarkers(frame)
        # aruco.drawDetectedMarkers(frame, corners, ids)
        # cv2.imshow("Arena Feed", frame)

        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
        time.sleep(0.25)

    video.release()
    # cv2.destroyAllWindows()

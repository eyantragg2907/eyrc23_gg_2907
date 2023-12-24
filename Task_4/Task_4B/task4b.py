import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd

def transform_frame(frame):
    pt_A, pt_B, pt_C, pt_D = get_points_from_aruco(frame)
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))
     
     
    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    s = min(maxHeight, maxWidth)
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
            [0, s - 1],
            [s - 1, s - 1],
            [s - 1, 0]])
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    out = cv2.warpPerspective(frame,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
    out = out[:s, :s]

    return out, s


def get_points_from_aruco(frame):
    corners, ids, _ = get_aruco_data(frame)
    reqd_ids = {4,5,6,7}
    pt_A,pt_B,pt_C,pt_D = None, None, None, None

    for (markerCorner, markerID) in zip(corners, ids):
        if markerID not in reqd_ids: continue

        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        offset = 15
        topRight = (int(topRight[0])+offset, int(topRight[1])-offset)
        bottomRight = (int(bottomRight[0])+offset, int(bottomRight[1])+offset)
        bottomLeft = (int(bottomLeft[0])-offset, int(bottomLeft[1])+offset)
        topLeft = (int(topLeft[0])-offset, int(topLeft[1])-offset)

        if markerID == 5:
            pt_A = topLeft
        elif markerID == 7:
            pt_B = bottomLeft
        elif markerID == 6:
            pt_C = bottomRight
        elif markerID == 4:
            pt_D = topRight
    return pt_A, pt_B, pt_C, pt_D

def get_pts_from_frame(frame, s):
    S = 937
    Apts = (np.array([[811/S,887/S],[194/S,269/S]])*s).astype(int)
    Bpts = (np.array([[628/S,705/S],[620/S,695/S]])*s).astype(int)
    Cpts = (np.array([[444/S,519/S],[628/S,701/S]])*s).astype(int)
    Dpts = (np.array([[440/S,516/S],[183/S,260/S]])*s).astype(int)
    Epts = (np.array([[136/S,212/S],[200/S,276/S]])*s).astype(int)

    return (Apts, Bpts, Cpts, Dpts, Epts)

def get_aruco_data(frame):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    c, i, r = detector.detectMarkers(frame)
    return c, i.flatten(), r

def get_pxcoords(id,ids,corners):
    index= np.where(ids == id)[0][0]
    coords = np.mean(corners[index].reshape((4, 2)), axis=0)
    return np.array([ int(x) for x in coords])

def get_nearestmarker(id,ids,corners):
    mindist= float('inf')
    closestmarker = None
    coords1 = get_pxcoords(id,ids,corners)
    for (markerCorner, markerID) in zip(corners, ids):
        if markerID != 97:
            corners = markerCorner.reshape((4, 2))
            marker_center = np.mean(corners, axis=0)
            dist = np.linalg.norm(coords1-marker_center)
            if dist < mindist:
                closestmarker = markerID
                mindist = dist
    return closestmarker

def get_robot_coords(frame):
    corners, ids, _ = get_aruco_data(frame)    
    robotpxcoords = get_pxcoords(97,ids,corners)
    # print(f'ID: {97}, marker center in px: {robotpxcoords[0],robotpxcoords[1]},marker center in lat_long: {arucolat_long.loc[97]}')
    NearestMarker = get_nearestmarker(97,ids,corners)
    # verticaldiff = pxcoords5[1]-pxcoords7[1]
    # horizontaldiff = pxcoords6[0]-pxcoords7[0]
    # changeby1x = [(arucolat_long.loc[6].iloc[0]-arucolat_long.loc[7].iloc[0])/horizontaldiff,(arucolat_long.loc[6].iloc[1]-arucolat_long.loc[7].iloc[1])/horizontaldiff]
    # changeby1y = [(arucolat_long.loc[4].iloc[0]-arucolat_long.loc[7].iloc[0])/verticaldiff, (arucolat_long.loc[4].iloc[1]-arucolat_long.loc[7].iloc[1])/verticaldiff]
    # print(f'Vertical diff in px: 0 and Horizontal diff in px: 1, changes lat by : {changeby1x[0]} and changes long by : {changeby1x[1]}')
    # print(f'Vertical diff in px: 1 and Horizontal diff 0, changes lat by : {changeby1y[0]} and changes long by : {changeby1y[1]}')
    # print(f'Nearest marker: {NearestMarker} and px coords for that: {NearestMarkerpxcoords}')
    # pxdiff = robotpxcoords-NearestMarkerpxcoords
    # print(f'px diff : {pxdiff} and px coords for that: {get_pxcoords(26,ids,corners)}')
    # print(f'latlong coords of robot: nearestMarkerlatlong {arucolat_long.loc[NearestMarker]} + change by cahnge in x px : {pxdiff[0]*np.array(changeby1x)} +change by change in y px :{pxdiff[1]*np.array(changeby1y)} ')
    # robotlatlong = arucolat_long.loc[NearestMarker]+pxdiff[0]*np.array(changeby1x)+pxdiff[1]*np.array(changeby1y)
    # print(f'latlong coords of robot: {robotlatlong} ')
    # return robotlatlong
    return arucolat_long.loc[NearestMarker]

if __name__ == "__main__":
    frame = cv2.imread("gg_arenanew.jpg")
    arucolat_long=pd.read_csv("lat_long.csv",index_col="id")
    frame,side = transform_frame(frame)
    robotcoords=get_robot_coords(frame)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejected = detector.detectMarkers(frame)
    image = frame.copy()
    aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imwrite("bounding.png", frame)


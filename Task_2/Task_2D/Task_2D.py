'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2D of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''
############################## FILL THE MANDATORY INFORMATION BELOW ###############################

# Team ID:			[ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			Task_2D.py
# Functions:	    [ Function names ]
###################################################################################################

# IMPORTS (DO NOT CHANGE/REMOVE THESE IMPORTS)
import csv
import time

# Additional Imports
'''
You can import your required libraries here

'''

# DECLARING VARIABLES (DO NOT CHANGE/REMOVE THESE VARIABLES)
path1 = [11, 14, 13, 18, 19, 20, 23, 21, 22, 33, 30, 35, 32, 31, 34, 40, 36, 38, 37, 39, 41, 50, 4, 6, 52, 7, 8, 1, 2, 11]
path2 = [11, 14, 13, 10, 9, 51, 53, 0, 39, 37, 38, 28, 25, 54, 5, 3, 19, 20, 17, 12, 15, 16, 27, 26, 24, 29, 40, 34, 31, 32, 35, 30, 33, 22, 21, 23, 20, 19, 18, 13, 14, 11]
###################################################################################################

# Declaring Variables
'''
You can delare the necessary variables here

'''

def read_csv(csv_name):
    lat_lon = {}

    # open csv file (lat_lon.csv)
    # read "lat_lon.csv" file 
    # store csv data in lat_lon dictionary as {id:[lat, lon].....}
    # return lat_lon

    '''
    ADD YOUR CODE HERE

    '''
    return lat_lon 

def write_csv(loc, csv_name):

    # open csv (csv_name)
    # write column names "lat", "lon"
    # write loc ([lat, lon]) in respective columns

    '''

    ADD YOUR CODE HERE

    '''

def tracker(ar_id, lat_lon):

    # find the lat, lon associated with ar_id (aruco id)
    # write these lat, lon to "live_data.csv"

    '''

    ADD YOUR CODE HERE

    '''
    # also return coordinate ([lat, lon]) associated with respective ar_id.
    return coordinate

# ADDITIONAL FUNCTIONS

'''
If there are any additonal functions that you're using, you shall add them here. 

'''

###################################################################################################
########################### DO NOT MAKE ANY CHANGES IN THE SCRIPT BELOW ###########################

def main():
    ###### reading csv ##############
    lat_lon = read_csv('lat_long.csv')
    print("###############################################")
    print(f"Received lat, lons : {len(lat_lon)}")
    if len(lat_lon) != 48:
        print(f"Incomplete coordinates received.")
        print("###############################################")
        exit()
    ###### Test case 1 ##############
    print("########## Executing first test case ##########")
    start = time.time()
    passed = 0
    traversedPath1 = []
    for i in path1:
        t_point = tracker(i, lat_lon)
        traversedPath1.append(t_point)
        time.sleep(0.5)
    end = time.time()
    if None in traversedPath1:
        print(f"Incorrect path travelled.")
        exit()
    print(f"{len(traversedPath1)} points traversed out of {len(path1)} points")
    print(f"Time taken: {int(end-start)} sec")
    if len(traversedPath1) != len(path1):
        print("Test case 1 failed. Travelled path is incomplete")
    else:
        print("Test case 1 passed !!!")
        passed = passed+1
    print("########## Executing second test case ##########")
    ###### Test case 2 ##############
    start = time.time()
    traversedPath2 = []
    for i in path2:
        t_point = tracker(i, lat_lon)
        traversedPath2.append(t_point)
        time.sleep(0.5)
    end = time.time()
    if None in traversedPath2:
        print(f"Incorrect path travelled.")
        exit()
    print(f"{len(traversedPath2)} points traversed out of {len(path2)} points")
    print(f"Time taken: {int(end-start)} sec")
    if len(traversedPath2) != len(path2):
        print("Test case 2 failed. Travelled path is incomplete")
    else:
        print("Test case 2 passed !!!")
        passed = passed+1
    print("###############################################")
    if passed==0:
        print("0 Test cases passed please check your code.")
    elif passed==1:
        print("Partialy correct, look for any logical erro ;(")
    else:
        print("Congratulations!")
        print("You've succesfully passed all the test cases \U0001f600")
    print("###############################################")
if __name__ == "__main__":
    main()

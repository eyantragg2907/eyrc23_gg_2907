## OUTDATED ##
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

import sys
import time
import pandas as pd
import socket
import signal		

##############################################################

ip = "192.168.229.144"     # Enter IP address of laptop after connecting it to WIFI hotspot
commandsent = 0
command = "nrnn"
command = "nnrnlnrnrnrln"
# command = "nn"
################# ADD UTILITY FUNCTIONS HERE #################


def cleanup(s):
    s.close()
    print("cleanup done")
    sys.exit(0)

    
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

###############	Main Function	#################
if __name__ == "__main__":
    if commandsent == 0:
        s,conn=give_s_conn()
        send_to_robot(s,conn)
        commandsent = 1

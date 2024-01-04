
"""IMPORTANT"""
## LATEST ##
## STANDARD FILE WITHOUT QGIS AND LATEST DEBUGGING SYSTEM ##
"""IMPORTANT"""

##############################################################

import os
import sys
import time
import pandas as pd
import socket

##############################################################

ip = "192.168.54.92"  # Enter IP address of laptop after connecting it to WIFI hotspot
commandsent = 0
# command = "nrnn"
command = "nnrnlnrnrnrnln"
command = "nrnn"
# command = "nnrnlnrnrnnrnnlnn"
# command = "nn"
command = "nnrnlnrnrnnrnnlnn"
################# ADD UTILITY FUNCTIONS HERE #################


def cleanup(s):
    s.close()
    print("cleanup done")
    sys.exit(0)


def send_to_robot(s, conn):
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
        return s, conn


###############	Main Function	#################
if __name__ == "__main__":
    if not os.path.exists("lat_long.csv"):
        with open("lat_long.csv", "w") as f:
            f.writelines(["id,lat,lon", "23,39.6128542,-74.3629792"])

    if commandsent == 0:
        s, conn = give_s_conn()
        send_to_robot(s, conn)
        commandsent = 1

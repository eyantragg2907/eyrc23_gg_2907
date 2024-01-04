
"""IMPORTANT"""
## LATEST ##
## STANDARD FILE WITHOUT QGIS AND LATEST DEBUGGING SYSTEM ##
"""IMPORTANT"""

##############################################################

import sys
import time
import socket

##############################################################

ip = "192.168.54.92"  # IP of the Laptop on Hotspot
command = "nnrnlnrnrnnrnnlnn"  # the full cycle command

################# ADD UTILITY FUNCTIONS HERE #################


def cleanup(s):
    s.close()


def send_to_robot(s: socket.socket, conn: socket.socket):
    data = conn.recv(1024)
    data = data.decode("utf-8")

    if data == "ACK_REQ_FROM_ROBOT":
        pass
    else:
        print("Error in connection")
        cleanup(s)
        sys.exit(1)
    
    conn.sendall(str.encode(command))

    print(f"Sent command to robot: {command}")


def init_connection():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as soc:
        soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        soc.bind((ip, 8002))
        soc.listen()

        conn, addr = soc.accept()
        print(f"Connected to {addr}")

        return soc, conn

def listen_and_print(s, conn):
    while True:
        data = conn.recv(1024)
        data = data.decode("utf-8")
        print(f"recv: {data}")
        time.sleep(1)


###############	Main Function	#################
if __name__ == "__main__":

    soc, conn = init_connection()
    send_to_robot(soc, conn)

    listen_and_print(soc, conn)

    cleanup(soc)
    
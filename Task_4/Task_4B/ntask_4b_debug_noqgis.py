
"""IMPORTANT"""
## LATEST ##
## STANDARD FILE WITHOUT QGIS AND LATEST DEBUGGING SYSTEM ##
"""IMPORTANT"""

##############################################################

import sys
import time
import socket

##############################################################

IP_ADDRESS = "192.168.128.144"  # IP of the Laptop on Hotspot
COMMAND = "nnrnlnrnrnnrnnlnn"  # the full cycle command

COMMAND = "nrnn"

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
    
    conn.sendall(str.encode(COMMAND))

    print(f"Sent command to robot: {COMMAND}")


def init_connection():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as soc:
        soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        soc.bind((IP_ADDRESS, 8002))
        soc.listen()

        conn, addr = soc.accept()
        print(f"Connected to {addr}")

        return soc, conn

def listen_and_print(s, conn: socket.socket):
    print("="*80)
    while True:
        try:
            data = conn.recv(4096)
            data = data.decode("utf-8")
            print(f"{data}")
        except KeyboardInterrupt:
    
            cleanup(s)
            sys.exit(0)


###############	Main Function	#################
if __name__ == "__main__":

    soc, conn = init_connection()
    send_to_robot(soc, conn)

    listen_and_print(soc, conn)

    cleanup(soc)
    
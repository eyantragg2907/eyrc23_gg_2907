"""IMPORTANT"""
## LATEST ##
## STANDARD FILE WITHOUT QGIS AND LATEST DEBUGGING SYSTEM ##
"""IMPORTANT"""

##############################################################

import sys
import socket
import select
import termios
import tty


old_settings = termios.tcgetattr(sys.stdin)

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

##############################################################

IP_ADDRESS = "192.168.128.144"  # IP of the Laptop on Hotspot

################# ADD UTILITY FUNCTIONS HERE #################


def cleanup(s):
    s.close()


def listen_and_teleop(s: socket.socket, conn: socket.socket):
    data = conn.recv(1024)
    data = data.decode("utf-8")

    if data == "ACK_REQ_FROM_ROBOT":
        pass
    else:
        print("Error in connection")
        cleanup(s)
        sys.exit(1)
    
    try:
        tty.setcbreak(sys.stdin.fileno())

        while 1:
            recv = conn.recv(1024)
            print(f"Received: {recv.decode('utf-8')}", end="")

            if isData():
                c = sys.stdin.read(1)
                if c == 'q':
                    break
                if c == "w":
                    conn.sendall(str.encode("F\n"))
                elif c == "a":
                    conn.sendall(str.encode("L\n"))
                elif c == "d":
                    conn.sendall(str.encode("R\n"))
                elif c == "s":
                    conn.sendall(str.encode("B\n"))
                elif c == " ":
                    conn.sendall(str.encode("S\n"))
                else:
                    pass
                print(f"Sent command to robot: {c}")
    finally:
        conn.sendall(str.encode("S\n"))
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    cleanup(s)
    sys.exit(1)


def init_connection():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as soc:
        soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        soc.bind((IP_ADDRESS, 8002))
        soc.listen()

        conn, addr = soc.accept()
        print(f"Connected to {addr}")

        return soc, conn


###############	Main Function	#################
if __name__ == "__main__":
    soc, conn = init_connection()
    listen_and_teleop(soc, conn)

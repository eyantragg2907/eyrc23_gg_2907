LEFT_TURN_TIME = 1
RIGHT_TURN_TIME = 2

FRONT, RIGHT, BACK, LEFT = 0, 1, 2, 3
TOTAL_POSES = 4
INITIAL_POSE = 0

GRAPH_DATA_FILENAME = "./map_data.txt"


def reverse_pose(pose):
    if pose == FRONT:
        return BACK
    if pose == RIGHT:
        return LEFT
    if pose == BACK:
        return FRONT
    if pose == LEFT:
        return RIGHT

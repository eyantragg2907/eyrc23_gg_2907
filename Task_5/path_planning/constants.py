LEFT_TURN_TIME = 0.5
RIGHT_TURN_TIME = 0.5

FRONT, RIGHT, BACK, LEFT = 0, 1, 2, 3
TOTAL_POSES = 4
INITIAL_POSE = 0

GRAPH_DATA_FILENAME = "./map_data.txt"

LEFT_INSTRUCTION = "L"
RIGHT_INSTRUCTION = "R"
FORWARD_INSTRUCTION = "N"

PRIORITY = {"Fire":1,"Destroyed Buildings":2,"Humanitarian Aid and rehabilitation":3,"Military Vehicles":4,"Combat":5}

def reverse_pose(pose):
    if pose == FRONT:
        return BACK
    if pose == RIGHT:
        return LEFT
    if pose == BACK:
        return FRONT
    if pose == LEFT:
        return RIGHT

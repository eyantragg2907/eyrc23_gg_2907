LEFT_TURN_TIME = 0.5
RIGHT_TURN_TIME = 0.5

FRONT, RIGHT, BACK, LEFT = 0, 1, 2, 3
TOTAL_POSES = 4
INITIAL_POSE = 0

GRAPH_DATA_FILENAME = "./map_data.txt"

LEFT_INSTRUCTION = "l"
RIGHT_INSTRUCTION = "r"
FORWARD_INSTRUCTION = "n"
U_TURN_INSTRUCTION = "R"
SPECIAL_FORWARD_INSTRUCTION = "x"

label2priority = {
    "fire": 1,
    "destroyed_buildings": 2,
    "humanitarian_aid_and_rehabilitation": 3,
    "human_aid_rehabilitation": 3,
    "military_vehicles": 4,
    "combat": 5,
}


def reverse_pose(pose):
    if pose == FRONT:
        return BACK
    if pose == RIGHT:
        return LEFT
    if pose == BACK:
        return FRONT
    if pose == LEFT:
        return RIGHT

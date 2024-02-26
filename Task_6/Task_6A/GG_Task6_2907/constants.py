""" 
* Team Id : 2907
* Author List : Arnav Rustagi
* Filename: constants.py
* Theme: GeoGuide (GG)
* Functions: reverse_pose
* Global Variables: LEFT_TURN_TIME, RIGHT_TURN_TIME, FRONT, RIGHT, BACK, LEFT, TOTAL_POSES, INITIAL_POSE, 
  GRAPH_DATA_FILENAME, LEFT_INSTRUCTION, RIGHT_INSTRUCTION, FORWARD_INSTRUCTION, U_TURN_INSTRUCTION, 
  SPECIAL_FORWARD_INSTRUCTION, label2priority
"""
# Turn times in milliseconds measured physically
LEFT_TURN_TIME = 1200
RIGHT_TURN_TIME = 1241

# Poses represented by integers
FRONT, RIGHT, BACK, LEFT = 0, 1, 2, 3
TOTAL_POSES = 4

# Initial pose of the bot (FRONT)
INITIAL_POSE = 0

# File to read the graph data from.
# Graph data contains the distance between the nodes/ events (measured in milliseconds), as well as the direction to 
# take to reach the node. 
# Also contains special cases where the robot's orientation changes without the robot taking an explicit turn (left/right)
# The nodes and events were named by us, and the graph data was collected by us.
GRAPH_DATA_FILENAME = "./map_data.txt"

# These represent the instruction/control that the bot recognizes
# We send a string of these instructions to the bot, and the bot executes them
LEFT_INSTRUCTION = "l"
RIGHT_INSTRUCTION = "r"
FORWARD_INSTRUCTION = "n"
SLOW_FORWARD = "d"
U_TURN_INSTRUCTION = "R"

# Special forward instruction is used when the bot needs to stop at an event after being pinged by the computer
# The computer uses distance from AruCo marker to determine if the robot is at an event
SPL_FWD_INSTRUCTION = "x"
SPL_SLOW_FWD= "X"

# This dictionary is used to assign priority to the labels of the events. Lower number is higher priority
label2priority = {
    "fire": 1,
    "destroyed_buildings": 2,
    "humanitarian_aid": 3,
    "military_vehicles": 4,
    "combat": 5,
}


""" 
* Function Name: reverse_pose
* Input: pose: int  (0, 1, 2, 3)
* Output: int (0, 1, 2, 3) 
* Logic: Returns the reverse of the pose
* Example Call: reverse_pose(0) -> 2
"""
def reverse_pose(pose: int) -> int:
    if pose == FRONT:
        return BACK
    elif pose == RIGHT:
        return LEFT
    elif pose == BACK:
        return FRONT
    else:
        return RIGHT



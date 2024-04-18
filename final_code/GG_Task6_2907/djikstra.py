""" 
* Team Id:              2907
* Author List:          Arnav Rustagi
* Filename:             djikstra.py
* Theme:                GeoGuide (GG)
* Functions:            get_shortest_path, get_pose_cost, path_plan_all_nodes, 
                            path_plan_based_on_events_detected, final_path
* Global Variables:     None
"""

# get all related files
from constants import *
from graph import Graph

# initialize our Graph class. Look at the Graph() class implementation for more details.
graph = Graph()


class Path:
    """
    * Class Name:       Path
    * Use:              A base class to help us store paths.
    """

    """ 
    * Function Name:    __init__
    * Input:            nodes: list (list of nodes), cost: int (cost it took), instructions: str (instructions), end_pose: int (last pose of the bot)
    * Output:           N/A (Constructor)
    * Logic:            Constructor for Path class.
    * Example Call:     Path()
    """
    def __init__(self, nodes=[], cost=0, instructions="", end_pose=INITIAL_POSE):
        self.nodes = nodes
        self.cost = cost
        self.instructions = instructions
        self.end_pose = end_pose

    """ 
    * Function Name:    __add__
    * Input:            data: tuple (node, cost, instructions, end_pose)
    * Output:           Added path, where we add the new node to the path.
    * Logic:            Adds the new node to the path, and updates the cost, instructions, and end_pose.
    * Example Call:     Path() + (node, cost, instructions, end_pose)
    """
    def __add__(self, data):
        (node, cost, instructions, end_pose) = data
        path = Path(self.nodes[:], self.cost, self.instructions, end_pose)
        path.nodes.append(node)
        path.cost += cost
        path.instructions += instructions

        return path

    """ 
    * Function Name:    __len__
    * Input:            None
    * Output:           int: length of the path
    * Logic:            Just the length of the node.
    * Example Call:     len(Path)
    """
    def __len__(self) -> int:
        return len(self.nodes)

    """ 
    * Function Name:    __str__
    * Input:            None
    * Output:           str: string representation of the path
    * Logic:            Just the string representation of the path.
    * Example Call:     str(Path)
    """
    def __str__(self):
        return f"Path < {self.instructions=} {self.cost=} >"

    """ 
    * Function Name:    __hash__
    * Input:            None
    * Output:           hash of the end_pose of the path
    * Logic:            Just the hash of the end_pose of the path.
    * Example Call:     hash(Path)
    """
    def __hash__(self):
        return hash(self.end_pose)

    """ 
    * Function Name:    has_goals
    * Input:            goal_nodes (set of goal nodes)
    * Output:           bool: True if the path has goals, False otherwise.
    * Logic:            Checks if the path has goals or not.
    * Example Call:     Path().has_goals()
    """
    def has_goals(self, goal_nodes):
        return any(i in goal_nodes for i in self.nodes[:-1])


""" 
* Function Name:    get_shortest_path
* Input:            graph, node_1, node_2, path (path class), visited (all nodes visited till now), robot_pose (current pose)
* Output:           list[Path]: sorted list of paths, shortest path in 0th position.
* Logic:            Get's the shortest path using the djikstra algorithm. This function is recursive.
* Example Call:     path_plan_based_on_events_detected(nodes)
"""
def get_shortest_path(
    graph: Graph,
    node_1: str,
    node_2: str,
    path: Path = Path(),
    visited: set = set(),
    robot_pose: int = INITIAL_POSE,
):

    # if the path is empty, add the node_1 directly to the path
    if len(path) == 0:
        path.nodes.append(node_1)

    # make local copy of visited set, and other variables
    visited = visited.copy()
    visited.add(node_1)
    children = graph.nodes[node_1]
    paths = []

    # for every child of the node_1, get the shortest path
    for node, start_pose, distance, end_pose in children:
        if node in visited:
            continue

        # the cost of the path is the distance + the pose cost
        cum_cost = distance + get_pose_cost(robot_pose, start_pose)

        # turns required to reach the node
        turns = robot_pose - start_pose

        # get the instructions required to reach the node
        instructions = (
            LEFT_INSTRUCTION * abs(turns)
            if turns > 0
            else RIGHT_INSTRUCTION * abs(turns)
        )

        if abs(turns) == 2:
            instructions = U_TURN_INSTRUCTION
        if abs(turns) == 3:
            instructions = LEFT_INSTRUCTION if turns < 0 else RIGHT_INSTRUCTION

        # the tricky nodes are the ones where the start_pose is not equal to the end_pose
        tricky_node = start_pose != end_pose
        forward_instruction = SLOW_FORWARD if tricky_node else FORWARD_INSTRUCTION

        if node.startswith("E_"):
            # event node
            if node == node_2:
                forward_instruction = (
                    SPL_SLOW_FWD if tricky_node else SPL_FWD_INSTRUCTION
                )
            else:
                forward_instruction = ""

        # update the path with the new node, cost, instructions, and end_pose
        updated_path = path + (
            node,
            cum_cost,
            instructions + forward_instruction,
            end_pose,
        )

        if node == node_2:
            return [updated_path]

        # get the shortest path for the new node, recursively
        paths += get_shortest_path(
            graph,
            node,
            node_2,
            path=updated_path,
            visited=visited,
            robot_pose=end_pose,
        )

    # return the paths
    return [
        i
        for i in sorted(paths, key=lambda x: x.cost)[::]
        if not i.has_goals(graph.goal_nodes)
    ]


""" 
* Function Name:    get_pose_cost
* Input:            initial_pose: int (which pose is currently there), reqd_pose: int (which pose is required)
* Output:           int: the cost of the pose given the initial pose and the required pose.
* Logic:            Get's the pose cost using the difference between the initial pose and the required pose.
* Example Call:     path_plan_based_on_events_detected(nodes)
"""
def get_pose_cost(initial_pose: int, reqd_pose: int) -> int:
    diff = abs(initial_pose - reqd_pose)

    if reqd_pose > initial_pose:
        return RIGHT_TURN_TIME * diff
    elif reqd_pose < initial_pose:
        return LEFT_TURN_TIME * diff
    else:
        return 0


""" 
* Function Name:    path_plan_all_nodes
* Input:            nodes: list (the nodes to visit), end_pose: int (the pose to have at the end), 
                        cum_path: Path (cumulative path)
* Output:           list[Path]: sorted list of paths, shortest path in 0th position.
* Logic:            One of the main path planning functions, does the path sorting and
                        returns the shortest path.
* Example Call:     path_plan_based_on_events_detected(nodes)
"""
def path_plan_all_nodes(
    nodes: list, end_pose: int = INITIAL_POSE, cum_path: Path = Path()
) -> list[Path]:
    global graph  # the graph variable created earlier

    cum_paths = [cum_path]

    # for every two nodes which are far apart in our list.
    for node_1, node_2 in zip(nodes[:-1], nodes[1:]):
        temp_cum_paths = []
        poses_covered = set()

        # for every potential path till now,
        for cum_path in cum_paths:
            # the shortest path which includes the new nodes should be added to temp_cum_paths
            paths = get_shortest_path(
                graph, node_1, node_2, path=cum_path, robot_pose=cum_path.end_pose
            )
            temp_cum_paths += paths

        cum_paths = []
        # sort the path based on cost
        temp_cum_paths = [
            i
            for i in sorted(temp_cum_paths, key=lambda x: x.cost)[::]
            if not i.has_goals(graph.goal_nodes)
        ]

        # add the new paths to the cum_paths list, and add the poses
        # covered to the poses_covered set.
        for i in temp_cum_paths:
            if i.end_pose not in poses_covered:
                cum_paths.append(i)
                poses_covered.add(i.end_pose)

    # return the sorted cum_paths, so the bot can take the best path.
    return sorted(cum_paths, key=lambda x: x.cost)[::]


""" 
* Function Name:    path_plan_based_on_events_detected
* Input:            dict events_detected, that contains the events detected by the model in {"A": "event"} format.
* Output:           str final.instructions, the final path to be taken by the bot.
* Logic:            Calculates the priority of the events, sorts the events based on the priority, 
                        and then figures out the path to be taken by the bot and returns it!
* Example Call:     path_plan_based_on_events_detected(events: )
"""
def path_plan_based_on_events_detected(events: dict) -> list[Path]:
    global label2priority

    # get the priority of the events
    priorities = ((k, label2priority[v]) for (k, v) in events.items() if bool(v))

    # sort the events based on priority, so the nodes visit correctly.
    nodes_to_visit = sorted(priorities, key=lambda x: x[1], reverse=False)[::]

    # add the prefix "E_" to the nodes to visit
    nodes_to_visit = [f"E_{x[0]}" for x in nodes_to_visit]

    # figure out the path to be taken by the bot and return it!
    return path_plan_all_nodes(["A"] + nodes_to_visit + ["A"])


""" 
* Function Name:    final_path
* Input:            dict events_detected, that contains the events detected by the model in {"A": "event"} format.
* Output:           str final.instructions, the final path to be taken by the bot.
* Logic:            Uses the functions above to get the final path to be taken by the bot.
* Example Call:     final_path({"D": "combat", "B": "fire", "C": "military_vehicles", "A": "fire", "E": "military_vehicles"})
"""
def final_path(events_detected: dict) -> str:
    paths = path_plan_based_on_events_detected(events_detected)
    final = paths[0]
    final.instructions += "p" if final.end_pose == LEFT else ""
    return final.instructions


if __name__ == "__main__":
    """Code to test the implementation of the djikstra.py module. This code is for testing purposes only."""

    events_detected = {
        "D": "combat",
        "B": "fire",
        "C": "military_vehicles",
        "A": "fire",
        "E": "military_vehicles",
    }  # our model detected these "events" (for testing only)

    paths = path_plan_based_on_events_detected(events_detected)

    print(paths[0].instructions)

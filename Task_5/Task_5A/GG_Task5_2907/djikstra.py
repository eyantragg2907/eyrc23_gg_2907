from constants import *
from graph import Graph
import numpy as np

from tqdm import tqdm

graph = Graph()


class Path:
    def __init__(self, nodes=[], cost=0, instructions="", end_pose=INITIAL_POSE):
        self.nodes = nodes
        self.cost = cost
        self.instructions = instructions
        self.end_pose = end_pose

    def __add__(self, data):
        (node, cost, instructions, end_pose) = data
        path = Path(self.nodes[:], self.cost, self.instructions, end_pose)
        path.nodes.append(node)
        path.cost += cost
        path.instructions += instructions

        return path

    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        return f"Path < {self.instructions=} {self.cost=} >"

    def __hash__(self):
        return hash(self.end_pose)

    def has_goals(self, goal_nodes):
        return any(i in goal_nodes for i in self.nodes[:-1])


def get_shortest_path(
    graph,
    node_1,
    node_2,
    path=Path(),
    visited=set(),
    robot_pose=INITIAL_POSE,
):
    if len(path) == 0:
        path.nodes.append(node_1)
    visited = visited.copy()
    visited.add(node_1)
    children = graph.nodes[node_1]
    paths = []

    for node, direction, distance, end_pose in children:
        if node in visited:
            continue

        cum_cost = distance + get_pose_cost(robot_pose, direction)
        turns = robot_pose - direction
        instructions = (
            LEFT_INSTRUCTION * abs(turns)
            if turns > 0
            else RIGHT_INSTRUCTION * abs(turns)
        )

        if abs(turns) == 2:
            instructions = U_TURN_INSTRUCTION
        if abs(turns) == 3:
            instructions = LEFT_INSTRUCTION if turns < 0 else RIGHT_INSTRUCTION
        forward_instruction = FORWARD_INSTRUCTION
        if node.startswith("E_"):
            if node == node_2:
                forward_instruction = SPECIAL_FORWARD_INSTRUCTION
            else:
                forward_instruction = ""
        updated_path = path + (
            node,
            cum_cost,
            instructions + forward_instruction,
            end_pose,
        )

        if node == node_2:
            return [updated_path]

        paths += get_shortest_path(
            graph,
            node,
            node_2,
            path=updated_path,
            visited=visited,
            robot_pose=end_pose,
        )

    return [
        i
        for i in sorted(paths, key=lambda x: x.cost)[::]
        if not i.has_goals(graph.goal_nodes)
    ]


def get_pose_cost(initial_pose, reqd_pose):
    diff = abs(initial_pose - reqd_pose)

    # XXX: REMOVE THIS SHIT LATER!!! WE SHOULD REALLY ALLOW U-TURNS, WHY AREN'T WE?
    # TODO: REMOVE, WHO THE F WROTE THIS?
    if diff == 2:
        return 99999999

    if reqd_pose > initial_pose:
        return RIGHT_TURN_TIME * diff
    elif reqd_pose < initial_pose:
        return LEFT_TURN_TIME * diff
    else:
        return 0


def get_unique_paths(cum_paths):
    if not cum_paths: return
    return cum_paths

def path_plan_all_nodes(nodes, end_pose=INITIAL_POSE, cum_path=Path()):
    global graph
    cum_paths = [cum_path]
    for node_1, node_2 in zip(nodes[:-1], nodes[1:]):
        temp_cum_paths = []
        poses_covered = set()
        for cum_path in tqdm(cum_paths):
            # print(node_1, node_2, str(cum_path))
            paths = get_shortest_path(
                graph, node_1, node_2, path=cum_path, robot_pose=cum_path.end_pose
            )
            temp_cum_paths += paths 

        cum_paths = []
        temp_cum_paths = [
            i
            for i in sorted(temp_cum_paths, key=lambda x: x.cost)[::]
            if not i.has_goals(graph.goal_nodes)
        ]

        for i in temp_cum_paths:
            if i.end_pose not in poses_covered:
                print(i)
                cum_paths.append(i)
                poses_covered.add(i.end_pose)
        print([str(i) for i in cum_paths])
        print(node_1,node_2)

    return sorted(cum_paths, key=lambda x: x.cost)[::]


def path_plan_based_on_events_detected(events):
    global label2priority
    # print(events)
    # print(bool("str"))

    priorities = ((k, label2priority[v]) for (k, v) in events.items() if bool(v))

    nodes_to_visit = sorted(priorities, key=lambda x: x[1], reverse=False)[::]

    nodes_to_visit = [f"E_{x[0]}" for x in nodes_to_visit]

    return path_plan_all_nodes(["A"] + nodes_to_visit + ["A"])


def final_path(events_detected):
    paths = path_plan_based_on_events_detected(events_detected)
    final = paths[0]
    final.instructions += "p" if final.end_pose == LEFT else ""
    return str(final)


if __name__ == "__main__":
    events_detected = {
        "D": "combat",
        "B": "fire",
        "C": "humanitarian_aid_and_rehabilitation",
        "A": "military_vehicles",
        "E": "destroyed_buildings",
    }  # our model detected this events

    paths = path_plan_based_on_events_detected(events_detected)

    print(str(paths[0]))

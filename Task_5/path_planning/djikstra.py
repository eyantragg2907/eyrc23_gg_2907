from constants import *
from graph import Graph

graph = Graph()


class Path(object):
    def __init__(self, nodes=[], cost=0, instructions="", end_pose=FRONT):
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
        return f"{self.nodes=}, {self.cost=}, {self.instructions=} {self.end_pose}"

    def __hash__(self):
        return hash(self.end_pose)


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

        # print("",path)
        cum_cost = distance + get_pose_cost(robot_pose, direction)
        turns = robot_pose - direction
        instructions = (
            LEFT_INSTRUCTION * abs(turns)
            if turns > 0
            else RIGHT_INSTRUCTION * abs(turns)
        )
        updated_path = path + (
            node,
            cum_cost,
            instructions + FORWARD_INSTRUCTION,
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

    return sorted(paths, key=lambda x: x.cost)[::]


def get_pose_cost(initial_pose, reqd_pose):
    diff = abs(initial_pose - reqd_pose)
    if reqd_pose > initial_pose:
        return RIGHT_TURN_TIME * diff
    elif reqd_pose < initial_pose:
        return LEFT_TURN_TIME * diff
    else:
        return 0


def path_plan_all_nodes(nodes, end_pose=INITIAL_POSE, cum_path=Path()):
    global graph
    cum_paths = [cum_path]
    for node_1, node_2 in zip(nodes[:-1], nodes[1:]):
        temp_cum_paths = []
        for cum_path in cum_paths:
            print(node_1, node_2, str(cum_path))
            paths = get_shortest_path(
                graph, node_1, node_2, path=cum_path, robot_pose=cum_path.end_pose
            )
            unique_end_poses = set()
            [unique_end_poses.add(i) for i in paths]
            temp_cum_paths += list(unique_end_poses)
        print(temp_cum_paths)
        cum_paths = temp_cum_paths

    return sorted(cum_paths, key=lambda x: x.cost)[::]


def path_plan_based_on_events_detected(events):
    global label2priority
    priorities = ((k, label2priority[v]) for (k, v) in events.items() if bool(v))
    nodes_to_visit = sorted(priorities, key=lambda x: -x[1])[::]
    nodes_to_visit = [f"E_{x[0]}" for x in nodes_to_visit]

    return path_plan_all_nodes(["A"] + nodes_to_visit)


if __name__ == "__main__":
    events_detected = {
        "A": "combat",
        "B": None,
        "C": None,
        "D": "military_vehicles",
        "E": "fire",
    }  # our model detected this events

    paths = path_plan_based_on_events_detected(events_detected)
    for path in paths[::-1]:
        print(str(path))

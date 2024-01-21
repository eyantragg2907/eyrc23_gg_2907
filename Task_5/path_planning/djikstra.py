from constants import *
from graph import Graph


class Path(object):
    def __init__(self, nodes=[], cost=0):
        self.nodes = nodes
        self.cost = cost

    def __add__(self, data):
        (node, cost) = data
        path = Path(self.nodes[:], self.cost)
        path.nodes.append(node)
        path.cost += cost

        return path

    def __len__(self):
        return len(self.nodes)
    
    def __str__(self):
        return f"{self.nodes=}, {self.cost=}"

def get_shortest_path(
    graph,
    node_1,
    node_2,
    path=Path(),
    visited=set(),
    robot_pose=INITIAL_POSE,
):
    if len(path) == 0: path.nodes.append(node_1)
    visited = visited.copy()
    visited.add(node_1)
    children = graph.nodes[node_1]
    paths = []

    for node, direction, distance, end_pose in children:
        if node in visited: continue

        print("",path)
        cum_cost = distance + get_pose_cost(robot_pose, direction)
        updated_path = path + (node, cum_cost)

        if node == node_2:  return [updated_path]

        paths += get_shortest_path(
            graph,
            node,
            node_2,
            path=updated_path,
            visited=visited,
            robot_pose=end_pose,
        )

    return paths


def get_pose_cost(initial_pose, reqd_pose):
    diff = abs(initial_pose - reqd_pose)
    if reqd_pose > initial_pose:
        return RIGHT_TURN_TIME * diff
    elif reqd_pose < initial_pose:
        return LEFT_TURN_TIME * diff
    else:
        return 0


if __name__ == "__main__":
    graph = Graph()
    print([str(a) for a in get_shortest_path(graph, "A", "D")])

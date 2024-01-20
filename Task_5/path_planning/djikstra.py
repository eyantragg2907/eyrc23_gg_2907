from constants import *
from graph import Graph


def get_shortest_path(graph, node_1, node_2, current_path=("", 0), visited=set(), current_pose=INITIAL_POSE):
    visited = visited.copy()
    visited.add(node_1)
    children = graph.nodes[node_1]
    paths = []

    for (node, pose, distance) in children:
        if node in visited: continue

        print(f"FROM {node_1}")
        print(current_path)
        print(node, pose, distance)
        print(visited)
        print(f"old_cost : {current_path[1]}")
        print(f"distance {node_1} {node} : {distance}")
        print(f"pose_cost {current_pose} {pose} : {get_pose_cost(current_pose, pose)}")
        print("\n\n")
        updated_current_path = (
            current_path[0] + node,
            current_path[1]
            + distance
            + get_pose_cost(current_pose, pose)
        )

        if node == node_2: return [updated_current_path]
        paths += get_shortest_path(graph, node, node_2, updated_current_path, visited, pose)

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
    print(get_shortest_path(graph, 'D','B', current_pose=RIGHT))

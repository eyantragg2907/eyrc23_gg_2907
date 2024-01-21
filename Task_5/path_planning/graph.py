from constants import GRAPH_DATA_FILENAME
from constants import FRONT, RIGHT, BACK, LEFT
from constants import reverse_pose


class Graph:
    def __init__(self, filename=GRAPH_DATA_FILENAME):
        self.nodes = {}
        self.goal_nodes = set()
        self.load_from_file(filename)

    def load_from_file(self, filename):
        for line in open(filename).read().splitlines():
            words = line.split()
            N = len(words)
            if N not in (2, 4, 5):
                continue
            if N == 5:
                node_1, node_2, direction, distance, pose = words
                self.add_edge(node_1, node_2, eval(direction), float(distance), eval(pose))
            if N == 4:  # initialising an edge
                node_1, node_2, direction, distance = words
                self.add_edge(node_1, node_2, eval(direction), float(distance))
                self.add_edge(
                    node_2, node_1, reverse_pose(eval(direction)), float(distance)
                )
            if N == 2:  # initialising goal node
                self.goal_nodes.add(words[0])

    def add_edge(self, node_1, node_2, direction, distance, pose=None):
        if pose is None:
            pose = direction
        if node_1 not in self.nodes:
            self.nodes[node_1] = []
        self.nodes[node_1].append((node_2, direction, distance, pose))

    def __len__(self):
        return len(self.nodes)


if __name__ == "__main__":
    graph = Graph()
    print(graph.nodes)

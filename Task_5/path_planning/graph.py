from constants import GRAPH_DATA_FILENAME
from constants import FRONT, RIGHT, BACK, LEFT
from constants import reverse_pose

class Graph():
    def __init__(self, filename=GRAPH_DATA_FILENAME):
        self.nodes = {}
        self.goal_nodes = set()
        self.load_from_file(filename)

    def load_from_file(self, filename):
        for line in open(filename).read().splitlines():
            words = line.split()
            N = len(words)
            if N > 4 or N < 2: continue
            if N == 4: # initialising an edge
                node_1, node_2, pose, distance = words
                self.add_edge(node_1, node_2, eval(pose), float(distance))
                self.add_edge(node_2, node_1, reverse_pose(eval(pose)), float(distance))
            if N == 2: # initialising goal node
                self.goal_nodes.add(words[0])

    def add_edge(self, node_1, node_2, pose, distance):
        if node_1 not in self.nodes: self.nodes[node_1] = []
        self.nodes[node_1].append((node_2, pose, distance))

    def __len__ (self):
        return len(self.nodes)

if __name__ == "__main__":
    graph = Graph()
    print(graph.nodes)
